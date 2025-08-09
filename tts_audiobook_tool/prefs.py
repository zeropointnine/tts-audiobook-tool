from __future__ import annotations

import json

from tts_audiobook_tool.app_types import NormalizationType
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
    - has-shown-hints dict

    TODO: singleton or smth
    """

    def __init__(
            self,
            project_dir: str = "",
            hints: dict = {},
            normalization_type: str = PREFS_DEFAULT_NORMALIZATION_LEVEL,
            play_on_generate: bool = PREFS_DEFAULT_PLAY_ON_GENERATE
    ) -> None:
        self._project_dir = project_dir
        self._hints = hints
        self._play_on_generate = play_on_generate
        self._normalization_type: str = normalization_type

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

        if not "normalization_type" in prefs_dict and "should_normalize" in prefs_dict:
            # Legacy
            norm_type = "default" if bool(prefs_dict["should_normalize"]) else "none"
            dirty = True
        else:
            if not "normalization_type" in prefs_dict:
                norm_type = "default"
                dirty = True
            else:
                norm_type = prefs_dict["normalization_type"]
                if not norm_type in NormalizationType.all_json_values():
                    norm_type = "default"
                    dirty = True

        play_on_generate = prefs_dict.get("play_on_generate", PREFS_DEFAULT_PLAY_ON_GENERATE)
        if not isinstance(play_on_generate, bool):
            play_on_generate = PREFS_DEFAULT_PLAY_ON_GENERATE
            dirty = True

        hints = prefs_dict.get("hints", None) or {}

        prefs = Prefs(
            project_dir=project_dir,
            normalization_type=norm_type,
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
    def normalization_type(self) -> NormalizationType:
        if self._normalization_type == "default":
            return NormalizationType.DEFAULT
        elif self._normalization_type == "strong":
            return NormalizationType.STRONGER
        else:
            return NormalizationType.DISABLED

    def set_normalization_type_value(self, value: str) -> None:
        if not value in NormalizationType.all_json_values():
            raise ValueError(f"Bad value: {value}")
        self._normalization_type = value
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

    def reset_hints(self) -> None:
        self._hints = {}
        self.save()

    def save(self) -> None:
        dic = {
            "project_dir": self._project_dir,
            "hints": self._hints,
            "normalization_type": self._normalization_type,
            "play_on_generate": self._play_on_generate
        }
        try:
            with open(Prefs.get_file_path(), 'w', encoding='utf-8') as f:
                json.dump(dic, f, indent=4)
        except Exception as e:
            L.e(f"{e}")

    @staticmethod
    def get_file_path() -> str:
        from tts_audiobook_tool.app_util import AppUtil # ugh
        dir = AppUtil.get_app_user_dir()
        return os.path.join(dir, PREFS_FILE_NAME)
