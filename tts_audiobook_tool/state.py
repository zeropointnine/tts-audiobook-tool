from __future__ import annotations

import json
import os

import outetts
from outetts.version.interface import InterfaceHF

from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

from tts_audiobook_tool.model_config import MODEL_CONFIG
try:
    from .model_config_dev import MODEL_CONFIG # type: ignore
except ImportError:
    pass

class State:
    """
    Stores/manages program state.
    Is a flat list of properties, though there are rly 3 categories of things
    Not rly worth formalizing any further, imo

        project
            dir
            voice
            text_segments
        prefs
            temp
            play on generate
        model config
    """

    def __init__(self):

        self.interface: InterfaceHF = outetts.Interface(config=MODEL_CONFIG)

        self.project_dir = ""
        self.voice: dict | None = None
        self.text_segments: list[str] = []

        self.temperature = DEFAULT_TEMPERATURE
        self.play_on_generate = False

        self._restore_on_start()

    def _restore_on_start(self):
        """
        Restores state on app startup. Prints messages on unexpected errors.
        """

        # Project dir path
        from tts_audiobook_tool.project_settings_util import ProjectSettingsUtil
        err = ProjectSettingsUtil.load_from_disk(self)
        if err:
            printt(err, "error")
        if not self.project_dir:
            return

        # Project directory-derived state
        err = self.set_existing_project_dir(self.project_dir)
        if err:
            printt(err, "error")

        # Prefs
        from tts_audiobook_tool.prefs_util import PrefsUtil
        err = PrefsUtil.load_from_disk(self)
        if err:
            printt(err, "error")


    def load_voice_from_project_dir_and_set(self) -> str:
        """ Returns user-facing error message on fail """
        path = os.path.join(self.project_dir, PROJECT_VOICE_FILE_NAME)
        if not os.path.exists(path):
            return ""
        try:
            self.voice = self.interface.load_speaker(path)
            return ""
        except Exception as e:
            return f"Error loading project voice file: {e}"

    def load_text_from_project_dir_and_set(self) -> str:
        """ Returns user-facing error message on fail """

        file_path = os.path.join(self.project_dir, PROJECT_TEXT_FILE_NAME)
        if not os.path.exists(file_path):
            return ""

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                strings = json.load(file)
        except Exception as e:
            return f"Error loading project text: {e}"

        if not isinstance(strings, list):
            return "Error loading project text: Bad format"

        if len(strings) == 0:
            return "Project text has no data"

        self.text_segments = strings
        return ""

    def set_new_project_dir(self, s: str) -> None:
        """ Dir is expected to exist and be empty. """
        self.reset()
        self.project_dir = s
        self.save_project_dir_setting()

    def set_existing_project_dir(self, s: str) -> str:
        """
        Dir is expected to exist and generally be a valid project directory
        Returns error message or empty string for success
        """
        self.reset()

        self.project_dir = s

        err = self.load_voice_from_project_dir_and_set()
        if err:
            self.reset()
            return err

        err = self.load_text_from_project_dir_and_set()
        if err:
            self.reset()
            return err

        self.save_project_dir_setting()
        return ""

    def reset(self):
        self.project_dir = ""
        self.voice = None
        self.text_segments = []

    def save_project_dir_setting(self):
        from tts_audiobook_tool.project_settings_util import ProjectSettingsUtil
        ProjectSettingsUtil.save_to_disk(self)

    def save_pref_settings(self):
        from tts_audiobook_tool.prefs_util import PrefsUtil
        PrefsUtil.save_to_disk(self)
