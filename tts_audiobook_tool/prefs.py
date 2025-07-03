from __future__ import annotations

import json
from pathlib import Path

from tts_audiobook_tool.l import L

from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *

class Prefs:
    """
    User settings that persist to file
    - project dir
    - should_normalize
    - play_on_generate

    hint flags: "player"; "line_breaks"; "real_time"
    """

    def __init__(
            self,
            project_dir: str = "",
            hints: dict = {},
            should_normalize: bool = PREFS_DEFULT_SHOULD_NORMALIZE,
            play_on_generate: bool = PREFS_DEFAULT_PLAY_ON_GENERATE
    ) -> None:
        self._project_dir = project_dir
        self._hints = hints
        self._should_normalize = should_normalize
        self._play_on_generate = play_on_generate

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

        should_normalize = prefs_dict.get("should_normalize", PREFS_DEFULT_SHOULD_NORMALIZE)
        if not isinstance(should_normalize, bool):
            should_normalize = PREFS_DEFULT_SHOULD_NORMALIZE
            dirty = True

        play_on_generate = prefs_dict.get("play_on_generate", PREFS_DEFAULT_PLAY_ON_GENERATE)
        if not isinstance(play_on_generate, bool):
            play_on_generate = PREFS_DEFAULT_PLAY_ON_GENERATE
            dirty = True

        hints = prefs_dict.get("hints", None) or {}

        prefs = Prefs(
            project_dir=project_dir,
            should_normalize=should_normalize,
            play_on_generate=play_on_generate,
            hints=hints
        )

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

    def get_hint(self, key: str) -> bool:
        return bool(self._hints.get(key, False))

    def set_hint_true(self, key: str) -> None:
        self._hints[key] = True
        self.save()

    def save(self) -> None:
        dic = {
            "project_dir": self._project_dir,
            "hints": self._hints,
            "should_normalize": self._should_normalize,
            "play_on_generate": self._play_on_generate
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
