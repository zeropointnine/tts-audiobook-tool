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
    TODO: singleton
    """

    def __init__(
            self,
            project_dir: str = "",
            hints: dict = {},
            normalization_type: NormalizationType = NormalizationType.DEFAULT,
            play_on_generate: bool = PREFS_DEFAULT_PLAY_ON_GENERATE
    ) -> None:
        self._project_dir = project_dir
        self._hints = hints
        self._play_on_generate = play_on_generate
        self._normalization_type: NormalizationType = normalization_type

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
                    printt(f"Bad type for prefs: {type(prefs_dict)}")
                    return Prefs.new_and_save()
        except Exception as e:
            printt(f"Prefs file error: {e}")
            return Prefs.new_and_save()

        dirty = False

        project_dir = prefs_dict.get("project_dir", "")
        if not isinstance(project_dir, str):
            project_dir = ""
            dirty = True

        if not "normalization_type" in prefs_dict:
            s = "default"
            dirty = True
        else:
            s = prefs_dict["normalization_type"]
            if not s in NormalizationType.all_json_values():
                s = "default"
                dirty = True
        normalization_type = NormalizationType.from_json_value(s)
        if not normalization_type:
            normalization_type = NormalizationType.DEFAULT

        play_on_generate = prefs_dict.get("play_on_generate", PREFS_DEFAULT_PLAY_ON_GENERATE)
        if not isinstance(play_on_generate, bool):
            play_on_generate = PREFS_DEFAULT_PLAY_ON_GENERATE
            dirty = True

        hints = prefs_dict.get("hints", None) or {}

        prefs = Prefs(
            project_dir=project_dir,
            normalization_type=normalization_type,
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
        return self._normalization_type

    def set_normalization_type_using(self, json_value: str) -> None:
        value = NormalizationType.from_json_value(json_value)
        if not value:
            value = NormalizationType.DEFAULT
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
            "normalization_type": self._normalization_type.value.json_value,
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
