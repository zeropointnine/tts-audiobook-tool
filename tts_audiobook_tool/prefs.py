from __future__ import annotations

import json
from pathlib import Path

from tts_audiobook_tool.l import L

from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class Prefs:
    """
    User settings that persist to file
    - project dir
    - should_normalize
    - play_on_generate
    """

    def __init__(
            self,
            project_dir: str = "",
            should_normalize: bool = True,
            play_on_generate: bool = False
    ) -> None:
        self._project_dir = project_dir
        self._should_normalize = should_normalize
        self._play_on_generate = play_on_generate
        self._has_shown_player_reminder = False

    @staticmethod
    def new_and_save() -> Prefs:
        prefs = Prefs()
        prefs.save()
        return prefs

    @staticmethod
    def load() -> Prefs:
        """
        Returns Prefs instance and error message if any
        """
        if not os.path.exists(Prefs.get_file_path()):
            return Prefs.new_and_save()

        try:
            with open(Prefs.get_file_path(), 'r', encoding='utf-8') as f:
                prefs_dict = json.load(f)
                if not isinstance(prefs_dict, dict):
                    L.e(f"Bad type for prefs: {type(prefs_dict)}")
                    return Prefs.new_and_save()
        except Exception as e:
            L.e(f"Prefs file error: {e}")
            return Prefs.new_and_save()

        dirty = False

        project_dir = prefs_dict.get("project_dir", "")
        if not isinstance(project_dir, str):
            project_dir = ""
            dirty = True

        should_normalize = prefs_dict.get("should_normalize", DEFAULT_SHOULD_NORMALIZE)
        if not isinstance(should_normalize, bool):
            should_normalize = DEFAULT_SHOULD_NORMALIZE
            dirty = True

        play_on_generate = prefs_dict.get("play_on_generate", False)
        if not isinstance(play_on_generate, bool):
            play_on_generate = False
            dirty = True

        has_shown_player_reminder = prefs_dict.get("has_shown_player_reminder", False)
        if not isinstance(has_shown_player_reminder, bool):
            has_shown_player_reminder = False
            dirty = True

        prefs = Prefs(project_dir, should_normalize, play_on_generate)
        prefs._has_shown_player_reminder = has_shown_player_reminder
        if dirty:
            prefs.save()
        return prefs

    @property
    def project_dir(self) -> str:
        return self._project_dir

    @project_dir.setter
    def project_dir(self, value: str):
        self._project_dir = value
        self.save()

    @property
    def should_normalize(self) -> bool:
        return self._should_normalize

    @should_normalize.setter
    def should_normalize(self, value: bool):
        self._should_normalize = value
        self.save()

    @property
    def play_on_generate(self) -> bool:
        return self._play_on_generate

    @play_on_generate.setter
    def play_on_generate(self, value: bool):
        self._play_on_generate = value
        self.save()

    @property
    def has_shown_player_reminder(self) -> bool:
        return self._has_shown_player_reminder

    @has_shown_player_reminder.setter
    def has_shown_player_reminder(self, value: bool):
        self._has_shown_player_reminder = value
        self.save()

    def save(self) -> None:
        dic = {
            "project_dir": self._project_dir,
            "should_normalize": self._should_normalize,
            "play_on_generate": self._play_on_generate,
            "has_shown_player_reminder": self._has_shown_player_reminder
        }
        try:
            with open(Prefs.get_file_path(), 'w', encoding='utf-8') as f:
                json.dump(dic, f, indent=4)
        except Exception as e:
            L.e(f"{e}")

    @staticmethod
    def get_file_path() -> str:
        dir = Path.home()
        return os.path.join(dir, PREFS_FILE_NAME)
