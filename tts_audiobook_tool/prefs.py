from __future__ import annotations

import json

from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *

class Prefs:
    """
    User-configurable app settings that persist to file
    """

    def __init__(
            self,
            project_dir: str = "",
            hints: dict[str, bool] = {},
            stt_variant: SttVariant = list(SttVariant)[0],
            stt_config: SttConfig | None = None,
            tts_force_cpu: bool = False,
            max_retries: int = PREFS_MAX_RETRIES_DEFAULT,
            last_voice_dir: str = "",
            last_project_dir: str = "",
            play_on_generate: bool = PREFS_DEFAULT_PLAY_ON_GENERATE
    ) -> None:
        self._project_dir = project_dir
        self._hints = hints
        self._stt_variant = stt_variant
        self._stt_config = stt_config if stt_config else SttConfig.get_default()
        self._tts_force_cpu = tts_force_cpu
        self._max_retries = max_retries
        self._last_voice_dir = last_voice_dir
        self._last_project_dir = last_project_dir
        self._play_on_generate = play_on_generate

    @staticmethod
    def new_and_save() -> Prefs:
        prefs = Prefs()
        prefs.save()
        return prefs

    @staticmethod
    def load(save_if_dirty: bool=True) -> Prefs:
        """
        Loads and parses prefs file, and returns Prefs instance

        save_if_dirty:
            If any pref value is missing or invalid and therefore gets set to default value, 
            saves updated prefs file.
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

        # Migration-related (properties which used to but no longer exist in Preferences)
        if save_if_dirty:            
            migrated_properties = ["segmentation_strategy", "max_words", "normalization_type", "use_section_sound_effect"]
            has = [item for item in migrated_properties if prefs_dict.get(item) is not None]
            if has:
                from tts_audiobook_tool.hint import Hint                
                hint = Hint("", "Properties have changed", MIGRATED_MESSAGE.replace("%1", ", ".join(has)))
                Hint.show_hint(hint, and_prompt=True)
                dirty = True

        # Project dir
        project_dir = prefs_dict.get("project_dir", "")
        if not isinstance(project_dir, str):
            project_dir = ""
            dirty = True

        # Hints
        hints = prefs_dict.get("hints", None) or {}

        # Speech-to-text variant
        s = prefs_dict.get("stt_variant", "")
        if not s:
            stt_variant = list(SttVariant)[0]
            dirty = True
        else:
            result = SttVariant.get_by_id(s)
            if result is not None:
                stt_variant = result
            else:
                stt_variant = list(SttVariant)[0]
                dirty = True

        # STT config (device + quantization)
        s = prefs_dict.get("stt_config", "")
        stt_config = SttConfig.from_id(s)
        if not stt_config:
            stt_config = SttConfig.get_default()
            dirty = True

        # TTS force
        tts_force_cpu = prefs_dict.get("tts_force_cpu", False)
        if not isinstance(tts_force_cpu, bool):
            tts_force_cpu = False
            dirty = True

        # Max retries
        max_retries = prefs_dict.get("max_retries", PREFS_MAX_RETRIES_DEFAULT)
        if not isinstance(max_retries, int) or not (PREFS_MAX_RETRIES_MIN <= max_retries <= PREFS_MAX_RETRIES_MAX):
            max_retries = PREFS_MAX_RETRIES_DEFAULT
            dirty = True

        # Last voice dir
        last_voice_dir = prefs_dict.get("last_voice_dir", "")
        if not isinstance(last_voice_dir, str):
            last_voice_dir = ""
            dirty = True
        elif last_voice_dir and not os.path.exists(last_voice_dir):
            last_voice_dir = ""
            dirty = True

        # Last project dir
        last_project_dir = prefs_dict.get("last_project_dir", "")
        if not isinstance(last_project_dir, str):
            last_project_dir = ""
            dirty = True
        elif last_project_dir and not os.path.exists(last_project_dir):
            last_project_dir = ""
            dirty = True

        # Play on generate
        play_on_generate = prefs_dict.get("play_on_generate", PREFS_DEFAULT_PLAY_ON_GENERATE)
        if not isinstance(play_on_generate, bool):
            play_on_generate = PREFS_DEFAULT_PLAY_ON_GENERATE
            dirty = True

        # Make prefs instance
        prefs = Prefs(
            project_dir=project_dir,
            stt_variant=stt_variant,
            stt_config=stt_config,
            tts_force_cpu=tts_force_cpu,
            max_retries=max_retries,
            last_voice_dir=last_voice_dir,
            last_project_dir=last_project_dir,
            play_on_generate=play_on_generate,
            hints=hints
        )

        if dirty and save_if_dirty:
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

    @property
    def stt_variant(self) -> SttVariant:
        return self._stt_variant

    @stt_variant.setter
    def stt_variant(self, value: SttVariant) -> None:        
        self._stt_variant = value
        self.save()
        # Sync static value
        from tts_audiobook_tool.stt import Stt
        Stt.set_variant(value)

    @property
    def stt_config(self) -> SttConfig:
        return self._stt_config

    @stt_config.setter
    def stt_config(self, value: SttConfig) -> None:
        self._stt_config = value
        self.save()
        # Sync static value
        from tts_audiobook_tool.stt import Stt
        Stt.set_config(value)

    @property
    def tts_force_cpu(self) -> bool:
        return self._tts_force_cpu

    @tts_force_cpu.setter
    def tts_force_cpu(self, value: bool) -> None:
        self._tts_force_cpu = value
        self.save()
        # Sync static value
        from tts_audiobook_tool.tts import Tts
        Tts.set_force_cpu(value)

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self._max_retries = value
        self.save()

    @property
    def last_voice_dir(self) -> str:
        return self._last_voice_dir

    @last_voice_dir.setter
    def last_voice_dir(self, value: str) -> None:
        self._last_voice_dir = value
        self.save()

    @property
    def last_project_dir(self) -> str:
        return self._last_project_dir

    @last_project_dir.setter
    def last_project_dir(self, value: str) -> None:
        self._last_project_dir = value
        self.save()

    @property
    def is_validation_disabled(self) -> bool:
        # When so-called stt variant is 'disabled', it is implied that validation-after-generation is disabled
        return (self._stt_variant == SttVariant.DISABLED)

    def save(self) -> None:
        dic = {
            "project_dir": self._project_dir,
            "hints": self._hints,
            "stt_variant": self._stt_variant.id,
            "stt_config": self._stt_config.id,
            "tts_force_cpu": self._tts_force_cpu,
            "max_retries": self._max_retries,
            "last_voice_dir": self._last_voice_dir,
            "last_project_dir": self._last_project_dir,
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

# ---

PREFS_FILE_NAME = "tts-audiobook-tool-prefs.json"

MIGRATED_MESSAGE = \
f"""The following values that used to be stored as app preferences 
are now stored as part of the project, and have been reset:
    %1
You may want to review them in this and any other pre-existing projects you may have."""
