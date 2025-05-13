from __future__ import annotations

import outetts
from outetts.version.interface import InterfaceHF

from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

from tts_audiobook_tool.model_config import MODEL_CONFIG
try:
    from .model_config_dev import MODEL_CONFIG # type: ignore
except ImportError:
    pass

class State:
    """
    """

    def __init__(self):

        self.interface: InterfaceHF = outetts.Interface(config=MODEL_CONFIG)

        self.prefs = Prefs.load()
        if not self.prefs.project_dir:
            self.project = Project()
        else:
            self.project = Project.load(self.prefs.project_dir, self.interface)

    def set_project(self, path: str) -> None:
        self.prefs.project_dir = path
        self.project = Project.load(path, self.interface)

    def reset(self):
        self.prefs = Prefs.new_and_save()
        self.project = Project()
