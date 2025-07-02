import os
import time
from typing import Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from tts_audiobook_tool.sound_segment_file_util import SoundSegmentFileUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class ProjectSoundSegments:
    """
    Keeps catalog of the project's sound segment files

    List is cached, gets invalidated using directory watcher
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

    def on_dir_change(self):
        self._dirty = True

    @property
    def sound_segments(self) -> dict[int, str]:
        if self._dirty:
            start = time.time()
            self._sound_segments = SoundSegmentFileUtil.get_project_sound_segments(self.project)
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
        "tag"
            is a so-called "bracket tag"
            (app nomenclature for strings like "[fail]" inserted in filename)
        """
        tag = tag.lstrip("[")
        tag = tag.rstrip("]")
        tag = f"[{tag}]"

        result = { index: path for index, path in self._sound_segments.items() if tag in path }
        return result

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
