import os
from typing import Callable, Collection
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from tts_audiobook_tool.sound_segment_util import SoundSegment, SoundSegmentUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.text_normalizer import TextNormalizer
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.validate_util import ValidateUtil

class ProjectSoundSegments:
    """
    Keeps cached catalog of the project's sound segment files
    List gets invalidated and refreshed using directory watcher
    """

    def __init__(self, project: Project):
        self.project = project # TODO: circular reference, ng
        self._sound_segments_map: dict[int, list[SoundSegment]] = {}
        self._dirty = True
        self._segments_dir: str | None = None

        event_handler = DirHandler(self.on_dir_contents_change)
        self.observer = Observer()

        if project.dir_path:
            self._segments_dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
            # Watch the project directory (parent) instead of segments/ directly.
            # This is more reliable: watchdog can miss self-deletion events for the
            # watched directory itself, but child events (segments dir being deleted,
            # files created/deleted within segments/) are reliably delivered.
            self.observer.schedule(event_handler, project.dir_path, recursive=True)
            self.observer.start()

    def force_invalidate(self) -> None:
        # TODO: add logic "if am at some important point like GenMenu etc, and is WSL or MacOS, call this"
        self._dirty = True

    def on_dir_contents_change(self, event):
        if not self._segments_dir:
            return

        # The parent project dir is being watched, so events for the segments/
        # subdirectory and its contents come through here.
        # We only care about things inside or equal to the segments/ directory path.

        # --- Directory-level events (segments/ itself) ---
        # Detect when the segments/ subdirectory is created, deleted, or moved.
        if event.is_directory:
            # Disappeared (deleted or moved/renamed away)
            if event.event_type in ('deleted', 'moved') and event.src_path == self._segments_dir:
                self._sound_segments_map = {}
                self._dirty = False
                return
            # Appeared (created or moved/renamed back to "segments")
            if event.event_type in ('created', 'moved'):
                dest = getattr(event, 'dest_path', None)
                if event.src_path == self._segments_dir or dest == self._segments_dir:
                    self._dirty = True
                    return

        # --- File-level events within segments/ ---
        # Eg: FileCreatedEvent(src_path='/xyz.flac', dest_path='', event_type='created', is_directory=False, is_synthetic=False)
        if not event.src_path:
            return
        if event.is_directory:
            # Not interested in sub-subdirectory creation etc.
            return
        # Only handle events inside the segments/ directory
        if not event.src_path.startswith(self._segments_dir + os.sep):
            return

        # Don't make dirty if new file is not a sound segment file (ie, debug files)
        path = Path(event.src_path)
        if path.suffix != ".flac":
            return
        if not bool(SoundSegment.from_file_name(event.src_path)):
            return

        self._dirty = True
        
    @property
    def sound_segments_map(self) -> dict[int, list[SoundSegment]]:
        if self._dirty:
            printt(f"{COL_DIM}{Ansi.ITALICS}Project directory contents have changed. Scanning...", end="\r")
            self._sound_segments_map = SoundSegmentUtil.make_sound_segments_map(self.project)
            printt(f"{Ansi.ERASE_REST_OF_LINE}", end="\r")
            self._dirty = False
        return self._sound_segments_map
    
    def get_filenames_for(self, index: int) -> list[str]:
        sound_segments = self.sound_segments_map.get(index, [])
        return [item.file_name for item in sound_segments]
    
    def get_best_item_for(self, index: int) -> SoundSegment | None:
        """ For the given index, returns the sound segment with the least number of word fails """
        sound_segments = self.sound_segments_map.get(index, [])
        best_sound_segment = None
        best_fails = 9999 + 1
        for item in sound_segments:
            item_fails = item.num_errors if item.num_errors != -1 else 9999
            if item_fails < best_fails:
                best_sound_segment = item
                best_fails = item_fails    
        return best_sound_segment
    
    def get_best_file_for(self, index: int) -> str:
        item = self.get_best_item_for(index)
        return item.file_name if item else ""

    # ---

    def num_generated(self) -> int:
        return len( self.sound_segments_map.keys() )

    def num_generated_in_current_range(self) -> int:
        sett = self.project.get_indices_to_generate()
        dic = self.sound_segments_map
        count = 0
        for key in dic.keys():
            if key in sett:
                count += 1
        return count

    def get_existing_indices(self) -> set[int]:
        """
        Returns the set of indices of existing sound segments
        """
        return set(self.sound_segments_map.keys())

    def is_segment_failed(self, index: int, item: SoundSegment) -> bool:
        """
        Determines whether a sound segment should be considered 'failed' 
        based on its num_errors and the project's current strictness setting.
        
        Unknown error count (num_errors == -1) is treated as not-failed.
        """
        if item.num_errors == -1:
            # Unknown error count — treat as not-failed
            return False
        if item.num_errors == 99:
            # Music-fail sentinel — always considered failed
            return True
        phrase_group = self.project.phrase_groups[index]
        normalized_source = TextNormalizer.normalize_source(phrase_group.text, self.project.language_code)
        num_words = TextUtil.get_word_count(normalized_source, vocalizable_only=True)
        threshold = ValidateUtil.compute_threshold(num_words, self.project.strictness)
        return item.num_errors > threshold

    def get_failed_indices_in_generate_range(self) -> set[int]:
        """ 
        Within the project's defined 'generate range', 
        returns the indices of sound segments that exist but are all failed.
        
        Uses dynamic failure detection based on num_errors and project strictness.
        """
        all_indices = self.project.get_indices_to_generate()
        failed_indices = set()
        for index in all_indices:
            items = self.sound_segments_map.get(index, [])
            if not items:
                continue
            for item in items:
                if self.is_segment_failed(index, item):
                    failed_indices.add(index)
                    break
        return failed_indices
    
    def get_word_error_counts_in_generate_range(self) -> dict[int, int]:
        """ 
        Within the project's defined 'generate range', 
        returns the fail counts of existing sound segment items
        """
        all_indices = self.project.get_indices_to_generate()
        fail_counts = {}
        for index in all_indices:
            items = self.sound_segments_map.get(index, [])
            if not items:
                continue
            for item in items:
                fail_counts[index] = item.num_errors
        return fail_counts
    
    def delete_all(self) -> None:
        for sound_segments in self.sound_segments_map.values():
            for item in sound_segments:
                sound_file_path = Path(self.project.sound_segments_path) / item.file_name
                delete_silently(str(sound_file_path))
                timing_file_path = sound_file_path.with_suffix(".json")
                if timing_file_path.exists():
                    delete_silently(str(timing_file_path))

    def delete_redundants_for(self, index: int) -> int:
        """ Keeps the item with the least word fails and deletes the rest """        
        items = self.sound_segments_map.get(index, [])
        if not items:
            return 0
        
        best_item = self.get_best_item_for(index)
        if not best_item:
            return 0
        
        num_deleted = 0
        for item in items:
            if item != best_item:    
                path = Path(os.path.join(self.project.sound_segments_path, item.file_name))
                delete_silently(str(path))
                # And also timing json and debug json if exists
                path = path.with_suffix(".json")
                delete_silently(str(path))
                path = path.with_suffix(".debug.json")
                delete_silently(str(path))
                num_deleted += 1
        return num_deleted
    
    def delete_by_indices(self, indices: Collection[int]) -> None:
        """ Returns num deleted and num failed """
        for index in indices:
            items = self._sound_segments_map[index]
            for item in items:
                sound_file_path = Path(self.project.sound_segments_path) / item.file_name
                delete_silently(str(sound_file_path))
                timing_file_path = sound_file_path.with_suffix(".json")
                delete_silently(str(timing_file_path))

# ---

class DirHandler(FileSystemEventHandler):

    def __init__(self, callback: Callable):
        self.callback = callback

    def on_created(self, event):
        self.callback(event)

    def on_deleted(self, event):
        self.callback(event)

    def on_moved(self, event):
        self.callback(event)

    def on_modified(self, event):
        # No need to call back here for our use case atm
        # FYI, just 'touching' file modifies file
        pass

