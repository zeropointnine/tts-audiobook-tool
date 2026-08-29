from __future__ import annotations

from datetime import datetime, timezone

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.app_types import Hint, Saveable, SttConfig, SttVariant
from tts_audiobook_tool.l import L
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *

class Prefs(Saveable):
    """
    User-configurable app settings.

    Changes remain in memory until ``save()`` is called explicitly.
    """

    def __init__(
            self,
            project_dir: str = "",
            hints: dict[str, bool] = {},
            stt_variant: SttVariant = SttVariant.get_default(),
            stt_config: SttConfig | None = None,
            tts_force_cpu: bool = False,
            sgl_omni_type: TtsModelType | None = None,
            sgl_omni_url: str = "",
            aac_bitrate: str = AAC_BITRATE_DEFAULT,
            llm_url: str = "",
            llm_api_key: str = "",
            llm_model: str = "",
            llm_system_prompt: str = "",
            system_prompt_preset: str = CHAT_SYSTEM_PROMPTS[0][0],
            llm_extra_params: dict = {},
            last_voice_dir: str = "",
            last_project_dir: str = "",
            last_text_dir: str = "",
            chat_input_mode: str = PREFS_DEFAULT_CHAT_INPUT_MODE,
            chat_save: bool = PROJECT_DEFAULT_CHAT_SAVE,
            chat_save_mic: bool = PROJECT_DEFAULT_CHAT_SAVE_MIC,
            save_debug_files: bool = False,
            save_gen_log: bool = False,
            play_on_generate: bool = PREFS_DEFAULT_PLAY_ON_GENERATE,
    ) -> None:
        self._project_dir = project_dir
        self._hints = hints
        self._stt_variant = stt_variant
        self._stt_config = stt_config if stt_config else SttConfig.get_default()
        self._tts_force_cpu = tts_force_cpu

        # When in "sgl-omni mode", this is the active TTS type
        # When value is None, it autodetects based on server model id
        self._sgl_omni_type: TtsModelType | None = sgl_omni_type
        
        self._sgl_omni_url = sgl_omni_url.strip()
        
        self._aac_bitrate = aac_bitrate
        self._llm_url = llm_url
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_system_prompt = llm_system_prompt
        self._system_prompt_preset = system_prompt_preset
        self._llm_extra_params = llm_extra_params
        self._last_voice_dir = last_voice_dir
        self._last_project_dir = last_project_dir
        self._last_text_dir = last_text_dir
        self._chat_input_mode = chat_input_mode if chat_input_mode in CHAT_INPUT_MODES else PREFS_DEFAULT_CHAT_INPUT_MODE
        self._chat_save = chat_save
        self._chat_save_mic = chat_save_mic
        self._save_debug_files = save_debug_files
        self._save_gen_log = save_gen_log
        self._play_on_generate = play_on_generate

    @staticmethod
    def new_and_save() -> Prefs:
        prefs = Prefs()
        prefs.save()
        return prefs

    @staticmethod
    def quarantine_file(file_path: str) -> str:
        """Atomically move an unusable prefs file to a unique, preserved path."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base_path = f"{file_path}.{timestamp}"
        quarantine_path = f"{base_path}.corrupt"
        collision_index = 1

        while os.path.exists(quarantine_path):
            quarantine_path = f"{base_path}-{collision_index}.corrupt"
            collision_index += 1

        os.replace(file_path, quarantine_path)
        return quarantine_path

    @staticmethod
    def recover_from_malformed_file(file_path: str, reason: str) -> Prefs:
        """Preserve malformed input, replace it with defaults, and continue startup."""
        try:
            quarantine_path = Prefs.quarantine_file(file_path)
        except OSError as exception:
            message = (
                f"Preferences file is malformed, but it could not be preserved before "
                f"recovery: {make_error_string(exception)}\n"
                f"Preferences file: {file_path}"
            )
            if hasattr(L, "logger"):
                L.e(message)
            printt(f"\n{COL_ERROR}{message}\n")
            raise RuntimeError(message) from exception

        prefs = Prefs()
        save_error = prefs.save()
        message = (
            "Preferences recovery warning:\n"
            f"Could not load preferences: {reason}\n"
            f"The original file was preserved at: {quarantine_path}\n"
        )
        if save_error:
            message += (
                "The app is continuing with in-memory defaults, but a new preferences "
                f"file could not be saved at: {file_path}"
            )
        else:
            message += f"Defaults were saved to: {file_path}"
        if hasattr(L, "logger"):
            L.e(message)
        printt(f"\n{COL_ERROR}{message}\n")
        return prefs

    @staticmethod
    def load(save_if_dirty: bool=True) -> Prefs:
        """
        Loads and parses prefs file, and returns Prefs instance

        save_if_dirty:
            If any pref value is missing or invalid and therefore gets set to default value, 
            saves updated prefs file.
        """
        from tts_audiobook_tool.app_support import hints

        file_path = Prefs.get_file_path()
        if not os.path.exists(file_path):
            return Prefs.new_and_save()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prefs_dict = json.load(f)
        except (json.JSONDecodeError, UnicodeError, RecursionError) as exception:
            return Prefs.recover_from_malformed_file(file_path, make_error_string(exception))
        except OSError as exception:
            message = (
                f"Error reading preferences file: {make_error_string(exception)}\n"
                f"Preferences file: {file_path}"
            )
            if hasattr(L, "logger"):
                L.e(message)
            printt(f"\n{COL_ERROR}{message}\n")
            raise RuntimeError(message) from exception

        if not isinstance(prefs_dict, dict):
            reason = f"expected a JSON object, found {type(prefs_dict).__name__}"
            return Prefs.recover_from_malformed_file(file_path, reason)

        dirty = False

        # Mandatory migration: menus now always use the full-screen UI. The legacy
        # key is omitted from the Prefs instance and from the normalized JSON payload.
        had_legacy_menu_preference = "menu_clears_screen" in prefs_dict
        legacy_menu_did_not_clear = prefs_dict.get("menu_clears_screen") is False
        if had_legacy_menu_preference:
            dirty = True

        # Migration-related (properties which used to but no longer exist in Preferences)
        if save_if_dirty:            
            migrated_properties = ["segmentation_strategy", "max_words", "normalization_type", "use_break_sound_effect"]
            has = [item for item in migrated_properties if prefs_dict.get(item) is not None]
            if has:
                hint = Hint("", "Properties have changed", MIGRATED_MESSAGE.replace("%1", ", ".join(has)))
                hints.show_hint(hint, and_prompt=True)
                dirty = True

        # Project dir
        project_dir = prefs_dict.get("project_dir", "")
        if not isinstance(project_dir, str):
            project_dir = ""
            dirty = True

        # Hints
        hint_prefs = prefs_dict.get("hints", None) or {}
        if not isinstance(hint_prefs, dict) or not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in hint_prefs.items()
        ):
            hint_prefs = {}
            dirty = True

        # Speech-to-text variant
        s = prefs_dict.get("stt_variant", "")
        if not s:
            stt_variant = SttVariant.get_default()
            dirty = True
        else:
            result = SttVariant.get_by_id(s)
            if result is not None:
                stt_variant = result
            else:
                stt_variant = SttVariant.get_default()
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

        # SGL-Omni base url; empty = not set
        # (the program uses SGL_OMNI_URL_DEFAULT at request time when unset)
        sgl_omni_url = prefs_dict.get("sgl_omni_url", "")
        if not isinstance(sgl_omni_url, str) or not sgl_omni_url.strip():
            sgl_omni_url = ""
            dirty = True

        # SGL-Omni TTS type
        s = prefs_dict.get("sgl_omni_type", "")
        if s is None or s == "":
            sgl_omni_type = None
        elif isinstance(s, str):
            sgl_omni_type = TtsModelType.get_by_id(s)
            if not TtsModelType.is_valid_sgl_omni_type(sgl_omni_type):
                sgl_omni_type = None
                dirty = True
        else:
            sgl_omni_type = None
            dirty = True

        # AAC/M4B bitrate
        # Back-compat: support legacy key "aac_bitrate"
        aac_bitrate = prefs_dict.get("aac_bitrate", prefs_dict.get("aac_m4b_bitrate", AAC_BITRATE_DEFAULT))
        if not isinstance(aac_bitrate, str) or aac_bitrate not in AAC_BITRATES:
            aac_bitrate = AAC_BITRATE_DEFAULT
            dirty = True

        # LLM config
        llm_url = prefs_dict.get("llm_url", "")
        if not isinstance(llm_url, str):
            llm_url = ""
            dirty = True

        # Back-compat: support legacy key "api_key"
        llm_api_key = prefs_dict.get("llm_api_key", prefs_dict.get("api_key", ""))
        if not isinstance(llm_api_key, str):
            llm_api_key = ""
            dirty = True

        llm_model = prefs_dict.get("llm_model", "")
        if not isinstance(llm_model, str):
            llm_model = ""
            dirty = True

        llm_system_prompt = prefs_dict.get("llm_system_prompt", "")
        if not isinstance(llm_system_prompt, str):
            llm_system_prompt = ""
            dirty = True

        system_prompt_preset_default = CHAT_SYSTEM_PROMPTS[0][0]
        system_prompt_preset_files = [file_name for file_name, _ in CHAT_SYSTEM_PROMPTS]
        system_prompt_preset = prefs_dict.get("system_prompt_preset", system_prompt_preset_default)
        if not isinstance(system_prompt_preset, str):
            system_prompt_preset = system_prompt_preset_default
            dirty = True
        elif system_prompt_preset and system_prompt_preset not in system_prompt_preset_files:
            system_prompt_preset = system_prompt_preset_default
            dirty = True

        llm_extra_params = prefs_dict.get("llm_extra_params", {})
        if not isinstance(llm_extra_params, dict):
            llm_extra_params = {}
            dirty = True

        # Max retries
        max_retries = prefs_dict.get("max_retries", PROJECT_MAX_RETRIES_DEFAULT)
        if not isinstance(max_retries, int) or not (PROJECT_MAX_RETRIES_MIN <= max_retries <= PROJECT_MAX_RETRIES_MAX):
            max_retries = PROJECT_MAX_RETRIES_DEFAULT
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

        # Last text dir
        last_text_dir = prefs_dict.get("last_text_dir", "")
        if not isinstance(last_text_dir, str):
            last_text_dir = ""
            dirty = True
        elif last_text_dir and not os.path.exists(last_text_dir):
            last_text_dir = ""
            dirty = True

        # Chat input mode
        chat_input_mode = prefs_dict.get("chat_input_mode", PREFS_DEFAULT_CHAT_INPUT_MODE)
        if not isinstance(chat_input_mode, str) or chat_input_mode not in CHAT_INPUT_MODES:
            chat_input_mode = PREFS_DEFAULT_CHAT_INPUT_MODE
            dirty = True

        # Chat save
        chat_save = prefs_dict.get("chat_save", PROJECT_DEFAULT_CHAT_SAVE)
        if not isinstance(chat_save, bool):
            chat_save = PROJECT_DEFAULT_CHAT_SAVE
            dirty = True

        # Chat save mic
        chat_save_mic = prefs_dict.get("chat_save_mic", PROJECT_DEFAULT_CHAT_SAVE_MIC)
        if not isinstance(chat_save_mic, bool):
            chat_save_mic = PROJECT_DEFAULT_CHAT_SAVE_MIC
            dirty = True

        # Play on generate
        save_debug_files = prefs_dict.get("save_debug_files", False)
        if not isinstance(save_debug_files, bool):
            save_debug_files = False
            dirty = True

        # Save generation log
        save_gen_log = prefs_dict.get("save_gen_log", False)
        if not isinstance(save_gen_log, bool):
            save_gen_log = False
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
            sgl_omni_type=sgl_omni_type,
            sgl_omni_url=sgl_omni_url,
            aac_bitrate=aac_bitrate,
            llm_url=llm_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_system_prompt=llm_system_prompt,
            system_prompt_preset=system_prompt_preset,
            llm_extra_params=llm_extra_params,
            last_voice_dir=last_voice_dir,
            last_project_dir=last_project_dir,
            last_text_dir=last_text_dir,
            chat_input_mode=chat_input_mode,
            chat_save=chat_save,
            chat_save_mic=chat_save_mic,
            save_debug_files=save_debug_files,
            save_gen_log=save_gen_log,
            play_on_generate=play_on_generate,
            hints=hint_prefs,
        )

        # This removed preference must be cleaned during startup even when the
        # caller suppresses ordinary default/validation normalization.
        if dirty and (save_if_dirty or had_legacy_menu_preference):
            prefs.save()

        if legacy_menu_did_not_clear:
            from tts_audiobook_tool.constants_hints import HINT_FULL_SCREEN_UI
            hints.show_hint(HINT_FULL_SCREEN_UI, and_prompt=True)

        return prefs

    @property
    def project_dir(self) -> str:
        return self._project_dir

    @project_dir.setter
    def project_dir(self, value: str):
        self._project_dir = value

    @property
    def save_debug_files(self) -> bool:
        return self._save_debug_files

    @save_debug_files.setter
    def save_debug_files(self, value: bool):
        self._save_debug_files = value

    @property
    def save_gen_log(self) -> bool:
        return self._save_gen_log

    @save_gen_log.setter
    def save_gen_log(self, value: bool):
        self._save_gen_log = value

    @property
    def play_on_generate(self) -> bool:
        return self._play_on_generate

    @play_on_generate.setter
    def play_on_generate(self, value: bool):
        self._play_on_generate = value

    def get_hint(self, key: str) -> bool:
        return bool(self._hints.get(key, False))

    def set_hint_true(self, key: str) -> None:
        self._hints[key] = True

    def reset_hints(self) -> None:
        self._hints = {}

    @property
    def stt_variant(self) -> SttVariant:
        return self._stt_variant

    @stt_variant.setter
    def stt_variant(self, value: SttVariant) -> None:        
        self._stt_variant = value
        # Sync static value
        from tts_audiobook_tool.stt import Stt
        Stt.set_variant(value)

    @property
    def stt_config(self) -> SttConfig:
        return self._stt_config

    @stt_config.setter
    def stt_config(self, value: SttConfig) -> None:
        self._stt_config = value
        # Sync static value
        from tts_audiobook_tool.stt import Stt
        Stt.set_config(value)

    @property
    def tts_force_cpu(self) -> bool:
        return self._tts_force_cpu

    @tts_force_cpu.setter
    def tts_force_cpu(self, value: bool) -> None:
        self._tts_force_cpu = value
        # Sync static value
        from tts_audiobook_tool.tts import Tts
        Tts.set_force_cpu(value)

    @property
    def sgl_omni_type(self) -> TtsModelType | None:
        return self._sgl_omni_type

    @sgl_omni_type.setter
    def sgl_omni_type(self, value: TtsModelType | None) -> None:
        from tts_audiobook_tool.tts import Tts
        Tts.set_sgl_omni_type(value)
        # Mirror the normalized runtime value so prefs persists it
        # (Tts.set_sgl_omni_type() is the single source of validation)
        self._sgl_omni_type = Tts._sgl_omni_type

    @property
    def sgl_omni_url(self) -> str:
        return self._sgl_omni_url

    @sgl_omni_url.setter
    def sgl_omni_url(self, value: str) -> None:
        self._sgl_omni_url = value.strip()

    @property
    def aac_bitrate(self) -> str:
        return self._aac_bitrate

    @aac_bitrate.setter
    def aac_bitrate(self, value: str) -> None:
        if value not in AAC_BITRATES:
            value = AAC_BITRATE_DEFAULT
        self._aac_bitrate = value

    @property
    def llm_url(self) -> str:
        return self._llm_url

    @llm_url.setter
    def llm_url(self, value: str) -> None:
        self._llm_url = value

    @property
    def llm_api_key(self) -> str:
        return self._llm_api_key

    @llm_api_key.setter
    def llm_api_key(self, value: str) -> None:
        self._llm_api_key = value

    @property
    def llm_model(self) -> str:
        return self._llm_model

    @llm_model.setter
    def llm_model(self, value: str) -> None:
        self._llm_model = value

    @property
    def llm_system_prompt(self) -> str:
        return self._llm_system_prompt

    @llm_system_prompt.setter
    def llm_system_prompt(self, value: str) -> None:
        self._llm_system_prompt = value

    @property
    def system_prompt_preset(self) -> str:
        return self._system_prompt_preset

    @system_prompt_preset.setter
    def system_prompt_preset(self, value: str) -> None:
        if not isinstance(value, str):
            value = ""
        self._system_prompt_preset = value

    @property
    def llm_extra_params(self) -> dict:
        return self._llm_extra_params

    @llm_extra_params.setter
    def llm_extra_params(self, value: dict) -> None:
        if not isinstance(value, dict):
            value = {}
        self._llm_extra_params = value

    @property
    def last_voice_dir(self) -> str:
        return self._last_voice_dir

    @last_voice_dir.setter
    def last_voice_dir(self, value: str) -> None:
        self._last_voice_dir = value

    @property
    def last_project_dir(self) -> str:
        return self._last_project_dir

    @last_project_dir.setter
    def last_project_dir(self, value: str) -> None:
        self._last_project_dir = value

    @property
    def last_text_dir(self) -> str:
        return self._last_text_dir

    @last_text_dir.setter
    def last_text_dir(self, value: str) -> None:
        self._last_text_dir = value

    @property
    def chat_input_mode(self) -> str:
        return self._chat_input_mode

    @chat_input_mode.setter
    def chat_input_mode(self, value: str) -> None:
        if value not in CHAT_INPUT_MODES:
            value = PREFS_DEFAULT_CHAT_INPUT_MODE
        self._chat_input_mode = value

    @property
    def chat_save(self) -> bool:
        return self._chat_save

    @chat_save.setter
    def chat_save(self, value: bool) -> None:
        self._chat_save = value

    @property
    def chat_save_mic(self) -> bool:
        return self._chat_save_mic

    @chat_save_mic.setter
    def chat_save_mic(self, value: bool) -> None:
        self._chat_save_mic = value

    @property
    def is_validation_disabled(self) -> bool:
        # When so-called stt variant is 'disabled', it is implied that validation-after-generation is disabled
        return (self._stt_variant == SttVariant.DISABLED)

    def save(self) -> str:
        def make_payload() -> dict:
            return {
                "project_dir": self._project_dir,
                "hints": self._hints,
                "stt_variant": self._stt_variant.id,
                "stt_config": self._stt_config.id,
                "tts_force_cpu": self._tts_force_cpu,
                "sgl_omni_type": "" if self._sgl_omni_type is None else self._sgl_omni_type.value.id,
                "sgl_omni_url": self._sgl_omni_url,
                "aac_bitrate": self._aac_bitrate,
                "llm_url": self._llm_url,
                "llm_api_key": self._llm_api_key,
                "llm_model": self._llm_model,
                "llm_system_prompt": self._llm_system_prompt,
                "system_prompt_preset": self._system_prompt_preset,
                "llm_extra_params": self._llm_extra_params,
                "last_voice_dir": self._last_voice_dir,
                "last_project_dir": self._last_project_dir,
                "last_text_dir": self._last_text_dir,
                "chat_input_mode": self._chat_input_mode,
                "chat_save": self._chat_save,
                "chat_save_mic": self._chat_save_mic,
                "save_debug_files": self._save_debug_files,
                "save_gen_log": self._save_gen_log,
                "play_on_generate": self._play_on_generate,
            }

        err = JsonSaveUtil.save(
            JsonArtifactType.PREFS,
            Prefs.get_file_path(),
            make_payload,
        )
        if err:
            if hasattr(L, "logger"):
                L.e(err)
            printt(f"\n{COL_ERROR}{err}\n")
        else:
            if hasattr(L, "logger"):
                L.d("saved")
        return err

    @staticmethod
    def get_file_path() -> str:
        from tts_audiobook_tool.app_support import app_paths
        dir = app_paths.get_app_user_dir()
        return os.path.join(dir, PREFS_FILE_NAME)

# ---

PREFS_FILE_NAME = "tts-audiobook-tool-prefs.json"

MIGRATED_MESSAGE = \
f"""The following values that used to be stored as app preferences 
are now stored as part of the project, and have been reset:
    %1
You may want to review them in this and any other pre-existing projects you may have."""
