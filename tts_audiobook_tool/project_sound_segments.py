import os
from typing import Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from tts_audiobook_tool.sound_segment_util import SoundSegment, SoundSegmentUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class ProjectSoundSegments:
    """
    Keeps cached catalog of the project's sound segment files
    List gets invalidated and refreshed using directory watcher
    """

    def __init__(self, project: Project):
        self.project = project # TODO: circular reference, ng
        self._sound_segments_map: dict[int, list[SoundSegment]] = {}
        self._dirty = True

        event_handler = DirHandler(self.on_dir_contents_change)
        self.observer = Observer()

        if project.dir_path:
            dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
            self.observer.schedule(event_handler, dir, recursive=False)
            self.observer.start()

    def force_invalidate(self) -> None:
        # TODO: add logic "if am at some important point like GenMenu etc, and is WSL or MacOS, call this"
        self._dirty = True

    def on_dir_contents_change(self, event):        
        # Eg: FileCreatedEvent(src_path='/xyz.flac', dest_path='', event_type='created', is_directory=False, is_synthetic=False)
        if event.src_path:
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

    def count_num_generated_in(self, set_: set[int]) -> int:
        dic = self.sound_segments_map
        count = 0
        for key in dic.keys():
            if key in set_:
                count += 1
        return count

    def get_failed_indices_in_generate_range(self) -> set[int]:
        """ 
        Within the project's defined 'generate range', 
        returns the indices of sound segments that exist but are all tagged as fails.
        """
        all_indices = self.project.get_indices_to_generate()
        failed_indices = set()
        for index in all_indices:
            items = self.sound_segments_map.get(index, [])
            if not items:
                continue
            is_fail = False
            for item in items:
                if item.is_fail:
                    is_fail = True
                    break
            if is_fail:
                failed_indices.add(index)
        return failed_indices
    
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

