from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.constants import TOP_K_MAX_DEFAULT, TOP_P_MAX_DEFAULT
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class MossBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.MOSS.value

    MAX_NEW_TOKENS = 512

    # Rem, when temp, audio_top_p, or audio_top_k are too low,
    # model can fail to emit termination token, etc.

    # MOSS-TTS does not expose a structured supported-languages object in its
    # remote processor/config code. The processor accepts `language` as a
    # free-form prompt tag: build_user_message() interpolates the value into the
    # user message under "- Language:" without validation. These names are copied
    # from the MOSS-TTS-v1.5 model-card table so project language codes can still
    # steer generation with the documented language tags. Keep this app-owned map
    # in sync with the upstream README if MOSS adds or changes supported tags.
    LANGUAGE_NAMES_BY_CODE = {
        "zh": "Chinese",
        "yue": "Cantonese",
        "en": "English",
        "ar": "Arabic",
        "cs": "Czech",
        "da": "Danish",
        "nl": "Dutch",
        "fi": "Finnish",
        "fr": "French",
        "de": "German",
        "el": "Greek",
        "he": "Hebrew",
        "hi": "Hindi",
        "hu": "Hungarian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "mk": "Macedonian",
        "ms": "Malay",
        "fa": "Persian (Farsi)",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "es": "Spanish",
        "sw": "Swahili",
        "sv": "Swedish",
        "tl": "Tagalog",
        "th": "Thai",
        "tr": "Turkish",
        "vi": "Vietnamese",
    }

    @staticmethod
    def get_language_name(language_code: str) -> str:
        return MossBaseModel.LANGUAGE_NAMES_BY_CODE.get(language_code.strip().lower(), "")

    def clear_continuation(self) -> None:
        """
        Clears cached MOSS rolling-continuation state.

        Concrete MOSS implementations should override-and-super to reset any
        cached continuation transcript/audio data between caller-defined
        boundaries, such as paragraph or section breaks.
        """
        super().clear_continuation()

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        b = project.moss_batch_size > 1 and project.moss_mode == MossVoiceCloneMode.ROLLING_CONTINUATION
        if b:
            return [
                ReadinessIssue(
                    "batch size 1 for rolling cont",
                    "Rolling continuation cannot be used in combination with batch size > 1"
                )
            ]
        return []

    def get_warning_issues(self, project: Project) -> list[str]:
        # This is informational more than it is a warning
        language = MossBaseModel.get_language_name(project.language_code)
        return [f"Using MOSS-TTS language value: {language or 'None'}"]

    @classmethod
    def should_trim_trailing_token_noise(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> bool:
        return MossConfigs.get_by_target(project.moss_target) == MossConfigs.LOCAL

# ---

@dataclass(frozen=True)
class MossConfig:
    repo_id: str
    revision: str
    arch_name: str
    desc_extra: str
    temperature_default: float
    temperature_min: float
    temperature_max: float
    top_p_default: float
    top_p_min: float
    top_p_max: float
    top_k_default: int
    top_k_min: int
    top_k_max: int

class MossConfigs(Enum):

    # MOSS hf models use `trust_remote_code=True`. 
    # Therefore using pinned commits as a security precaution.

    DELAY = MossConfig(
        repo_id="OpenMOSS-Team/MOSS-TTS-v1.5",
        revision="cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        arch_name="MossTTSDelay",
        desc_extra="9B params",

        temperature_default=1.7,
        temperature_min=0.8,
        temperature_max=3.0,
        top_p_default=0.8,
        top_p_min=0.5,
        top_p_max=TOP_P_MAX_DEFAULT,
        top_k_default=25,
        top_k_min=10,
        top_k_max=TOP_K_MAX_DEFAULT,
    )

    LOCAL = MossConfig(
        repo_id="OpenMOSS-Team/MOSS-TTS-Local-Transformer", 
        revision="12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e",
        arch_name="MossTTSLocal", 
        desc_extra="1.7B params",

        temperature_default=1.0,
        temperature_min=0.8,
        temperature_max=3.0,
        top_p_default=0.95,
        top_p_min=0.5,
        top_p_max=TOP_P_MAX_DEFAULT,
        top_k_default=50,
        top_k_min=10,
        top_k_max=TOP_K_MAX_DEFAULT,
    )

    @staticmethod
    def get_default() -> "MossConfigs":
        return MossConfigs.DELAY

    @staticmethod
    def get_default_repo_id() -> str:
        return MossConfigs.get_default().value.repo_id

    @staticmethod
    def get_preset_by_target(target: str) -> "MossConfigs | None":
        target = target.strip()
        for config in MossConfigs:
            if target == config.value.repo_id:
                return config
        return None

    @staticmethod
    def get_by_target(target: str) -> "MossConfigs":
        target = target.strip()
        if not target:
            return MossConfigs.get_default()
        config = MossConfigs.get_preset_by_target(target)
        if config:
            return config
        return MossConfigs.LOCAL if "local" in target.lower() else MossConfigs.DELAY

    @property
    def preset_description(self) -> str:
        parts = [self.value.arch_name]
        if self.value.desc_extra:
            parts.append(self.value.desc_extra)
        return ", ".join(parts)

# ---

class MossVoiceCloneMode(tuple[str, str, str], Enum):    
    CLONE = (
        "clone", 
        "Clone", 
        "Use reference audio as a speaker/timbre prompt."
    )
    CONTINUATION = (
        "continuation", 
        "Continuation (experimental)", 
        "Continue from the reference audio by prepending its transcript to the prompt."
    )
    ROLLING_CONTINUATION = (
        "rolling_continuation", 
        "Rolling Continuation (experimental)", 
        "Within a paragraph, uses each generated segment as context for the next\n      while keeping the reference audio as a speaker/timbre prompt."
    )

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]

    @property
    def description(self) -> str:
        return self.value[2]

    @staticmethod
    def get_default() -> "MossVoiceCloneMode":
        return MossVoiceCloneMode.CLONE

    @staticmethod
    def normalize(mode: object) -> "MossVoiceCloneMode":
        if isinstance(mode, MossVoiceCloneMode):
            return mode
        return MossVoiceCloneMode.get_default()

    @staticmethod
    def get_by_id(id: str) -> "MossVoiceCloneMode | None":
        for item in list(MossVoiceCloneMode):
            if id == item.id:
                return item
        return None
