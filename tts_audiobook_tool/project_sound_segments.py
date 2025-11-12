import os
from typing import Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class ProjectSoundSegments:
    """
    Keeps cached catalog of the project's sound segment files
    List gets invalidated and refreshed using directory watcher
    """

    def __init__(self, project: Project):
        self.project = project
        self._sound_segments: dict[int, str] = {}
        self._dirty = True

        event_handler = DirHandler(self.on_dir_change)
        observer = Observer()

        if project.dir_path:
            dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
            observer.schedule(event_handler, dir, recursive=False)
            observer.start()

    def force_invalidate(self) -> None:
        # TODO: add logic "if am at some important point like GenSubMenu etc, and is WSL or MacOS, call this"
        self._dirty = True

    def on_dir_change(self):
        self._dirty = True

    @property
    def sound_segments(self) -> dict[int, str]:
        if self._dirty:
            printt(f"{COL_DIM}{Ansi.ITALICS}Project directory contents have changed. Scanning...", end="\r")
            self._sound_segments = SoundSegmentUtil.get_project_sound_segments(self.project)
            printt(f"{Ansi.ERASE_REST_OF_LINE}", end="\r")
            self._dirty = False
        return self._sound_segments

    # ---

    def num_generated(self) -> int:
        return len( self.sound_segments.keys() )

    def count_num_generated_in(self, set_: set[int]) -> int:
        dic = self.sound_segments
        count = 0
        for key in dic.keys():
            if key in set_:
                count += 1
        return count

    def get_sound_segments_with_tag(self, tag: str) -> dict[int, str]:
        """
        "tag" / so-called "bracket tag" is app nomenclature for strings like "[fail]" inserted in filename
        """
        tag = tag.lstrip("[")
        tag = tag.rstrip("]")
        tag = f"[{tag}]"
        result = { index: path for index, path in self._sound_segments.items() if tag in path }
        return result

    def get_failed_in_generate_range(self) -> dict[int, str]:
        all_failed = self.get_sound_segments_with_tag("fail")
        all_indices = self.project.get_indices_to_generate()
        subset: dict[int, str] = {}
        for index in all_indices:
            if index in all_failed:
                subset[index] = all_failed[index]
        return subset

# ---

class DirHandler(FileSystemEventHandler):

    def __init__(self, callback: Callable):
        self.callback = callback

    def on_modified(self, event):
        # No need to call back here for our use case atm
        # FYI, just 'touching' file modifies file
        pass

    def on_created(self, event):
        self.callback()

    def on_deleted(self, event):
        self.callback()

    def on_moved(self, event):
        self.callback()
