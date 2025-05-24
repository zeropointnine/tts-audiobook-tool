from __future__ import annotations

from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class State:
    """
    Holds app's full state ("bad!")
    Viz: project object and app prefs/settings object
    """

    def __init__(self):

        self.prefs = Prefs.load()

        if not self.prefs.project_dir:
            self.project = Project()
        else:
            self.project = Project.load(self.prefs.project_dir)

    def set_project(self, path: str) -> None:
        self.prefs.project_dir = path
        self.project = Project.load(path)

    def reset(self):
        self.prefs = Prefs.new_and_save()
        self.project = Project()
