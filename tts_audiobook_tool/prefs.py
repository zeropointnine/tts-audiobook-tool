from __future__ import annotations

import os
from datetime import datetime, timezone

from tts_audiobook_tool.app_support import path_norm
from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.app_types import Hint, Saveable, SttConfig, SttVariant
from tts_audiobook_tool.conversation.conversation_types import ChatInputMode
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *

def _load_saved_dir_pref(prefs_dict: dict, key: str) -> tuple[str, bool]:
    """
    Reads a saved directory preference, returning the value to use and whether
    the stored value had to be replaced.

    A directory recorded by another operating system cannot name a directory
    here, and `os.path.exists` on such a string only asks whether something
    with backslashes in its name exists, which is not the question being asked.
    Values that cannot be used here are dropped. The drop is logged rather than
    shown to the user: these preferences only decide where a file dialog opens.
    """
    value = prefs_dict.get(key, "")
    if not isinstance(value, str):
        return "", True
    if not value:
        return "", False

    native_value = path_norm.normalize_native_path(value)
    if path_norm.looks_foreign(native_value) or not os.path.exists(native_value):
        L.w(f"Prefs: dropping saved directory for {key!r}, which is not a directory here: {value}")
        return "", True

    return native_value, native_value != value


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
            remote_tts_url: str = "",
            aac_bitrate: str = AAC_BITRATE_DEFAULT,
            llm_url: str = "",
            llm_api_key_env_var: str = PREFS_DEFAULT_LLM_API_KEY_ENV_VAR,
            llm_model: str = "",
            llm_system_prompt: str = "",
            system_prompt_preset: str = CHAT_SYSTEM_PROMPTS[0][0],
            llm_extra_params: dict = {},
            last_voice_dir: str = "",
            last_project_dir: str = "",
            last_text_dir: str = "",
            last_enhanced_dir: str = "",
            enhance_audio_path: str = "",
            chat_input_mode: ChatInputMode = PREFS_DEFAULT_CHAT_INPUT_MODE,
            chat_echo_override: bool = PREFS_DEFAULT_CHAT_ECHO_OVERRIDE,
            chat_save: bool = PROJECT_DEFAULT_CHAT_SAVE,
            chat_save_mic: bool = PROJECT_DEFAULT_CHAT_SAVE_MIC,
            save_debug_files: bool = False,
            save_gen_log: bool = False,
            startup_version_code: int | None = None,
    ) -> None:
        self._project_dir = project_dir
        # Copied: the signature default is a shared mutable object, so
        # storing it by reference would let one Prefs instance's
        # set_hint_true() leak into every other instance built without an
        # explicit dict.
        self._hints = dict(hints)
        self._stt_variant = stt_variant
        self._stt_config = stt_config if stt_config else SttConfig.get_default()
        self._tts_force_cpu = tts_force_cpu

        # Connection settings are machine-global; model selection belongs to Project.
        self._remote_tts_url = remote_tts_url.strip().rstrip("/")
        
        self._aac_bitrate = aac_bitrate
        self._llm_url = llm_url
        self._llm_api_key_env_var = llm_api_key_env_var

        # Transitory, not persisted: set by load() when an insecure legacy
        # API key value was found and scrubbed from the prefs file.
        self.legacy_llm_api_key_removed = False
        self._llm_model = llm_model
        self._llm_system_prompt = llm_system_prompt
        self._system_prompt_preset = system_prompt_preset
        self._llm_extra_params = llm_extra_params
        self._last_voice_dir = last_voice_dir
        self._last_project_dir = last_project_dir
        self._last_text_dir = last_text_dir
        self._last_enhanced_dir = last_enhanced_dir
        self._enhance_audio_path = enhance_audio_path
        if not isinstance(chat_input_mode, ChatInputMode):
            parsed = ChatInputMode.from_id(chat_input_mode) if isinstance(chat_input_mode, str) else None
            chat_input_mode = parsed or PREFS_DEFAULT_CHAT_INPUT_MODE
        self._chat_input_mode = chat_input_mode
        self._chat_echo_override = chat_echo_override
        self._chat_save = chat_save
        self._chat_save_mic = chat_save_mic
        self._save_debug_files = save_debug_files
        self._save_gen_log = save_gen_log

        # The startup-messaging version code the app last ran with.
        # None means "no history" (brand-new user, or a prefs file from before
        # this property existed); in that case no startup messages are shown.
        self._startup_version_code = startup_version_code

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
        elif project_dir and path_norm.looks_foreign(project_dir):
            # Deliberately left as saved. The app reports the directory it
            # cannot open, which tells the user what to replace; silently
            # starting with no project at all would hide where the value came
            # from. A path written in another operating system's grammar can
            # only ever fail here, so the reason is logged.
            L.w(f"Prefs: saved project_dir was written by another operating system and cannot be opened here: {project_dir}")

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

        remote_tts_url = prefs_dict.get("remote_tts_url", "")
        if not isinstance(remote_tts_url, str):
            remote_tts_url = ""
            dirty = True
        remote_tts_url = remote_tts_url.strip().rstrip("/")

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

        # Name of the environment variable read at request time for the API key.
        llm_api_key_env_var = prefs_dict.get("llm_api_key_env_var", PREFS_DEFAULT_LLM_API_KEY_ENV_VAR)
        if not isinstance(llm_api_key_env_var, str) or not llm_api_key_env_var:
            llm_api_key_env_var = PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
            dirty = True

        # Insecure legacy values ("llm_api_key", older "api_key") are no longer
        # stored. They are scrubbed from the file via the immediate re-save;
        # the warning to the user is deferred to the first main-menu show.
        had_legacy_llm_api_key = bool(prefs_dict.get("llm_api_key", prefs_dict.get("api_key", "")))
        if had_legacy_llm_api_key:
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
        last_voice_dir, last_voice_dir_changed = _load_saved_dir_pref(prefs_dict, "last_voice_dir")
        if last_voice_dir_changed:
            dirty = True

        # Last project dir
        last_project_dir, last_project_dir_changed = _load_saved_dir_pref(prefs_dict, "last_project_dir")
        if last_project_dir_changed:
            dirty = True

        # Last text dir
        last_text_dir, last_text_dir_changed = _load_saved_dir_pref(prefs_dict, "last_text_dir")
        if last_text_dir_changed:
            dirty = True

        # Last enhanced dir
        last_enhanced_dir, last_enhanced_dir_changed = _load_saved_dir_pref(prefs_dict, "last_enhanced_dir")
        if last_enhanced_dir_changed:
            dirty = True

        # Enhance audiobook selection. Unlike last-used directory preferences,
        # stale paths are retained so the resumable enhance menu can display and
        # explicitly clear a selection whose file was moved or removed.
        enhance_audio_path = prefs_dict.get("enhance_audio_path", "")
        if not isinstance(enhance_audio_path, str):
            enhance_audio_path = ""
            dirty = True

        # Chat input mode
        s = prefs_dict.get("chat_input_mode", PREFS_DEFAULT_CHAT_INPUT_MODE.id)
        chat_input_mode = ChatInputMode.from_id(s) if isinstance(s, str) else None
        if chat_input_mode is None:
            chat_input_mode = PREFS_DEFAULT_CHAT_INPUT_MODE
            dirty = True

        # Chat echo override
        chat_echo_override = prefs_dict.get(
            "chat_echo_override", PREFS_DEFAULT_CHAT_ECHO_OVERRIDE
        )
        if not isinstance(chat_echo_override, bool):
            chat_echo_override = PREFS_DEFAULT_CHAT_ECHO_OVERRIDE
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

        # Startup version code (None is a valid value: no history)
        startup_version_code = prefs_dict.get("startup_version_code", None)
        if startup_version_code is not None and (
            isinstance(startup_version_code, bool) or not isinstance(startup_version_code, int)
        ):
            startup_version_code = None
            dirty = True

        # Make prefs instance
        prefs = Prefs(
            project_dir=project_dir,
            stt_variant=stt_variant,
            stt_config=stt_config,
            tts_force_cpu=tts_force_cpu,
            remote_tts_url=remote_tts_url,
            aac_bitrate=aac_bitrate,
            llm_url=llm_url,
            llm_api_key_env_var=llm_api_key_env_var,
            llm_model=llm_model,
            llm_system_prompt=llm_system_prompt,
            system_prompt_preset=system_prompt_preset,
            llm_extra_params=llm_extra_params,
            last_voice_dir=last_voice_dir,
            last_project_dir=last_project_dir,
            last_text_dir=last_text_dir,
            last_enhanced_dir=last_enhanced_dir,
            enhance_audio_path=enhance_audio_path,
            chat_input_mode=chat_input_mode,
            chat_echo_override=chat_echo_override,
            chat_save=chat_save,
            chat_save_mic=chat_save_mic,
            save_debug_files=save_debug_files,
            save_gen_log=save_gen_log,
            startup_version_code=startup_version_code,
            hints=hint_prefs,
        )
        prefs.legacy_llm_api_key_removed = had_legacy_llm_api_key

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
    def startup_version_code(self) -> int | None:
        return self._startup_version_code

    @startup_version_code.setter
    def startup_version_code(self, value: int | None) -> None:
        self._startup_version_code = value

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
    def remote_tts_url(self) -> str:
        return self._remote_tts_url

    @remote_tts_url.setter
    def remote_tts_url(self, value: str) -> None:
        value = value.strip().rstrip("/")
        self._remote_tts_url = value
        from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
        RemoteTtsDiscovery.set_base_url(value)
        from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
        from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
        SglOmniUtil.set_base_url(value)
        AudioCppUtil.set_base_url(value)

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
        """
        API key value, resolved from the configured environment variable at
        read time. Empty string when the variable is unset (or set to "").
        """
        if self._llm_api_key_env_var:
            return os.environ.get(self._llm_api_key_env_var, "")
        return ""

    @property
    def llm_api_key_env_var(self) -> str:
        return self._llm_api_key_env_var

    @llm_api_key_env_var.setter
    def llm_api_key_env_var(self, value: str) -> None:
        self._llm_api_key_env_var = value

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
    def last_enhanced_dir(self) -> str:
        return self._last_enhanced_dir

    @last_enhanced_dir.setter
    def last_enhanced_dir(self, value: str) -> None:
        self._last_enhanced_dir = value

    @property
    def enhance_audio_path(self) -> str:
        return self._enhance_audio_path

    @enhance_audio_path.setter
    def enhance_audio_path(self, value: str) -> None:
        self._enhance_audio_path = value

    @property
    def chat_input_mode(self) -> ChatInputMode:
        return self._chat_input_mode

    @chat_input_mode.setter
    def chat_input_mode(self, value: ChatInputMode) -> None:
        if not isinstance(value, ChatInputMode):
            parsed = ChatInputMode.from_id(value) if isinstance(value, str) else None
            value = parsed or PREFS_DEFAULT_CHAT_INPUT_MODE
        self._chat_input_mode = value

    @property
    def chat_echo_override(self) -> bool:
        return self._chat_echo_override

    @chat_echo_override.setter
    def chat_echo_override(self, value: bool) -> None:
        self._chat_echo_override = value

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
                "remote_tts_url": self._remote_tts_url,
                "aac_bitrate": self._aac_bitrate,
                "llm_url": self._llm_url,
                "llm_api_key_env_var": self._llm_api_key_env_var,
                "llm_model": self._llm_model,
                "llm_system_prompt": self._llm_system_prompt,
                "system_prompt_preset": self._system_prompt_preset,
                "llm_extra_params": self._llm_extra_params,
                "last_voice_dir": self._last_voice_dir,
                "last_project_dir": self._last_project_dir,
                "last_text_dir": self._last_text_dir,
                "last_enhanced_dir": self._last_enhanced_dir,
                "enhance_audio_path": self._enhance_audio_path,
                "chat_input_mode": self._chat_input_mode.id,
                "chat_echo_override": self._chat_echo_override,
                "chat_save": self._chat_save,
                "chat_save_mic": self._chat_save_mic,
                "save_debug_files": self._save_debug_files,
                "save_gen_log": self._save_gen_log,
                "startup_version_code": self._startup_version_code,
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
