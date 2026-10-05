from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.constants import TOP_K_MAX_DEFAULT, TOP_P_MAX_DEFAULT
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING

from tts_audiobook_tool import util
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class MossArchType(Enum):
    LOCAL = "local"
    DELAY = "delay"
    UNKNOWN = "unknown"


class MossBaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("moss_local").value

    MAX_NEW_TOKENS = 1024
    ROLLING_CONTINUATION_MAX_LENGTH = 3

    # Rem, when temperature, audio_top_p, or audio_top_k are too low
    # respectively, model can fail to emit termination token, etc.

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

    @staticmethod
    def get_seed_setting_name(config: "MossConfigs") -> str:
        return f"{config.value.setting_prefix}_seed"

    @staticmethod
    def get_generation_params(project: Project, config: "MossConfigs") -> tuple[float, float, int]:
        """Resolve the saved-or-default sampling parameters for an architecture."""
        prefix = config.value.setting_prefix
        temperature = project.get_model_setting('moss_local', f'{prefix}_temperature')
        audio_top_p = project.get_model_setting('moss_local', f'{prefix}_top_p')
        audio_top_k = project.get_model_setting('moss_local', f'{prefix}_top_k')

        values = config.value
        return (
            values.temperature_default if temperature == -1 else temperature,
            values.audio_top_p_default if audio_top_p == -1 else audio_top_p,
            values.audio_top_k_default if audio_top_k == -1 else audio_top_k,
        )

    def get_loaded_arch_type(self) -> MossArchType:
        raise NotImplementedError()

    @classmethod
    def can_hallucinate_music(cls, project: Project, instance: TtsBaseModel | None=None) -> bool:

        if instance is not None:
            assert isinstance(instance, MossBaseModel)
            is_local = (instance.get_loaded_arch_type() == MossArchType.LOCAL)
            return is_local

        return MossConfigs.get_by_target(project.get_model_setting('moss_local', 'target')).is_local_arch

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        b = project.get_model_setting('moss_local', 'batch_size') > 1 and project.get_model_setting('moss_local', 'rolling_cont') > 0
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
        return MossConfigs.get_by_target(project.get_model_setting('moss_local', 'target')).is_local_arch

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        s = project.get_model_setting('moss_local', 'target') or MossConfigs.get_default_repo_id()
        s = s.removeprefix("OpenMOSS-Team/")
        s = util.ellipsize_path_for_menu(s)
        return s

    @classmethod
    def get_output_sample_rate(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> int:
        return MossConfigs.get_by_target(project.get_model_setting('moss_local', 'target')).value.output_sample_rate

# ---

TEMPERATURE_MIN = 0.8
TEMPERATURE_MAX = 3.0

TEMPERATURE_DEFAULT_DELAY = 1.7
TEMPERATURE_DEFAULT_LOCAL_V15 = 1.7
TEMPERATURE_DEFAULT_LOCAL = 1.2 # FYI, upstream default is 1.0, but I still got runaway gens with that value

TOP_P_MIN = 0.5
TOP_P_MAX = TOP_P_MAX_DEFAULT

TOP_P_DEFAULT_DELAY = 0.8
TOP_P_DEFAULT_LOCAL_V15 = 0.8
TOP_P_DEFAULT_LOCAL = 0.95

TOP_K_MIN = 10
TOP_K_MAX = TOP_K_MAX_DEFAULT

TOP_K_DEFAULT_DELAY = 25
TOP_K_DEFAULT_LOCAL_V15 = 25
TOP_K_DEFAULT_LOCAL = 50


@dataclass(frozen=True)
class MossConfig:
    repo_id: str
    revision: str
    arch_name: str
    desc_extra: str
    # Prefix of this preset's private project settings ("delay_temperature",
    # "local_top_p", "local_v15_seed", ...). Each preset owns its own
    # sampling values and seed so switching presets never cross-contaminates.
    setting_prefix: str
    temperature_default: float
    temperature_min: float
    temperature_max: float
    audio_top_p_default: float
    audio_top_p_min: float
    audio_top_p_max: float
    audio_top_k_default: int
    audio_top_k_min: int
    audio_top_k_max: int
    output_sample_rate: int

class MossConfigs(Enum):

    # MOSS hf models use `trust_remote_code=True`.
    # Therefore using pinned commits as a security precaution.

    DELAY = MossConfig(
        repo_id="OpenMOSS-Team/MOSS-TTS-v1.5",
        revision="cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        arch_name="MossTTSDelay",
        desc_extra="9B params",
        setting_prefix="delay",

        temperature_default=TEMPERATURE_DEFAULT_DELAY,
        temperature_min=TEMPERATURE_MIN,
        temperature_max=TEMPERATURE_MAX,
        audio_top_p_default=TOP_P_DEFAULT_DELAY,
        audio_top_p_min=TOP_P_MIN,
        audio_top_p_max=TOP_P_MAX,
        audio_top_k_default=TOP_K_DEFAULT_DELAY,
        audio_top_k_min=TOP_K_MIN,
        audio_top_k_max=TOP_K_MAX,
        output_sample_rate=MossBaseModel.INFO.default_output_sample_rate,
    )

    LOCAL_V15 = MossConfig(
        repo_id="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        revision="be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
        arch_name="MossTTSLocal v1.5",
        desc_extra="1.7B params, stereo codec",
        setting_prefix="local_v15",

        # Recommended sampling params per the upstream v1.5 model card.
        temperature_default=TEMPERATURE_DEFAULT_LOCAL_V15,
        temperature_min=TEMPERATURE_MIN,
        temperature_max=TEMPERATURE_MAX,
        audio_top_p_default=TOP_P_DEFAULT_LOCAL_V15,
        audio_top_p_min=TOP_P_MIN,
        audio_top_p_max=TOP_P_MAX,
        audio_top_k_default=TOP_K_DEFAULT_LOCAL_V15,
        audio_top_k_min=TOP_K_MIN,
        audio_top_k_max=TOP_K_MAX,
        output_sample_rate=48_000,
    )

    LOCAL = MossConfig(
        repo_id="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        revision="12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e",
        arch_name="MossTTSLocal",
        desc_extra="1.7B params",
        setting_prefix="local",

        temperature_default=TEMPERATURE_DEFAULT_LOCAL,
        temperature_min=TEMPERATURE_MIN,
        temperature_max=TEMPERATURE_MAX,
        audio_top_p_default=TOP_P_DEFAULT_LOCAL,
        audio_top_p_min=TOP_P_MIN,
        audio_top_p_max=TOP_P_MAX,
        audio_top_k_default=TOP_K_DEFAULT_LOCAL,
        audio_top_k_min=TOP_K_MIN,
        audio_top_k_max=TOP_K_MAX,
        output_sample_rate=48_000,
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
    def is_local_arch(self) -> bool:
        """Whether this preset uses the Local Transformer architecture."""
        return self in (MossConfigs.LOCAL, MossConfigs.LOCAL_V15)

    @property
    def preset_description(self) -> str:
        parts = [self.value.arch_name]
        if self.value.desc_extra:
            parts.append(self.value.desc_extra)
        return ", ".join(parts)
