from __future__ import annotations
from enum import Enum

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class Qwen3BaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("qwen3tts_local").value

    PRESET_REPO_IDS = [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "zeropointnine/Darwin-TTS-1.7B-Cross-Qwen3Tokenizer"
    ]
    DEFAULT_REPO_ID = PRESET_REPO_IDS[0]

    TEMPERATURE_FALLBACK_DEFAULT = 0.9 # (Real default is embedded in the loaded model)
    TEMPERATURE_MIN = 0.01
    TEMPERATURE_MAX = 3.0 # sane max IMO; at high values, gens can fail to terminate for a very long time
    TOP_K_DEFAULT = 50
    TOP_P_DEFAULT = 1.0
    REPETITION_PENALTY_DEFAULT = 1.05

    ROLLING_CONTINUATION_MAX_LENGTH = 3 # fyi, quality degrades quickly as segments goes up with this model

    def clear_voice(self) -> None:
        ...

    @property
    def model_target(self) -> str:
        ...

    @property
    def model_type(self) -> str:
        ...

    @property
    def supported_languages(self) -> list[str]:
        ...

    @property
    def supported_speakers(self) -> list[str]:
        ...

    def get_resolved_speaker_info(self, speaker_id: str) -> tuple[str, bool]:
        """
        Returns the resolved speaker id for a voice's stored speaker id
        (`Voice.presets`), and if it is valid for the currently loaded model.
        A single-speaker checkpoint always uses its only speaker.
        """
        if self.model_type != "custom_voice":
            return "", False
        if len(self.supported_speakers) == 1:
            return self.supported_speakers[0], True
        return speaker_id, speaker_id in self.supported_speakers

    @staticmethod
    def get_voice_speaker_ids(project: Project) -> list[str]:
        """Each voice's stored speaker id ("" when unset), by voice slot."""
        return [voice.preset_for("qwen3tts_local") for voice in project.voice_references]

    @property
    def generate_defaults(self) -> dict[str, Any]:
        ...

    @property
    def is_model_type_supported(self) -> bool:
        return self.model_type in ["base", "custom_voice", "voice_design"] # ie, all current types

    def resolve_language_value(self, language_code: str) -> tuple[str, str]:
        """
        Returns qwen model's language value, and a note detail ("" when the code maps).
        Project language code is expected to be a 2-letter ISO-mumblemumble language code, but
        Qwen uses english names for language value.
        """

        # This hardcoded map may need updating if model's supported languages changes with future updates.
        # Or, consider expanding this with the assumption that future models or finetunes will also
        # use english word values.
        MAP = {
            'zh': 'chinese',
            'en': 'english',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'ja': 'japanese',
            'ko': 'korean',
            'pt': 'portuguese',
            'ru': 'russian',
            'es': 'spanish'
        }

        language_code = (language_code or "").lower()

        if language_code.startswith("zh"):
            result = "chinese", ""
        elif language_code in MAP:
            result = MAP[language_code], ""
        elif language_code:
            result = "auto", "no matching Qwen3-TTS language"
        else:
            result = "auto", ""

        return result

    # ---

    @classmethod
    def can_instructions(cls, project: Project, instance: TtsBaseModel | None = None) -> bool:
        if instance is not None:
            assert isinstance(instance, Qwen3BaseModel)
            model_type = instance.model_type
        else:
            model_type = project.get_model_setting('qwen3tts_local', 'model_type')
        return model_type in ("custom_voice", "voice_design")

    @classmethod
    def can_voice_presets(cls, project: Project, instance: TtsBaseModel | None = None) -> bool:
        # CustomVoice checkpoints have built-in speakers, chosen per voice
        if instance is not None:
            assert isinstance(instance, Qwen3BaseModel)
            model_type = instance.model_type
        else:
            model_type = project.get_model_setting('qwen3tts_local', 'model_type')
        return model_type == "custom_voice"

    @classmethod
    def requires_voice_preset(cls, project: Project) -> bool:
        # Can't tell a single-speaker checkpoint without an instance; Set
        # speaker saves its only speaker to the voice.
        return project.get_model_setting('qwen3tts_local', 'model_type') == "custom_voice"

    @classmethod
    def can_voice_clone(cls, project: Project, instance: TtsBaseModel | None = None) -> bool:
        # Only the Base checkpoint clones from voice samples (an unset model_type means Base)
        if instance is not None:
            assert isinstance(instance, Qwen3BaseModel)
            model_type = instance.model_type
        else:
            model_type = project.get_model_setting('qwen3tts_local', 'model_type')
        return model_type not in ("custom_voice", "voice_design")

    @classmethod
    def expects_voice_instructions(cls, project: Project) -> bool:
        # VoiceDesign without a description gets an arbitrary voice; CustomVoice
        # without instructions is just the speaker's default delivery.
        return project.get_model_setting('qwen3tts_local', 'model_type') == "voice_design"

    @classmethod
    def uses_reference_transcript(cls, project: Project) -> bool:
        # Only the Base checkpoint clones from voice samples (an unset
        # model_type means Base); CustomVoice and VoiceDesign ignore them.
        return project.get_model_setting('qwen3tts_local', 'model_type') not in ("custom_voice", "voice_design")

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:

        if instance:
            assert(isinstance(instance, Qwen3BaseModel))

        items = []

        if instance and not instance.is_model_type_supported:
            return [ ReadinessIssue("supported model type", f"Model type {instance.model_type} is unsupported") ]

        if project.get_model_setting('qwen3tts_local', 'rolling_cont') > 0:
            if instance and instance.model_type != "base":
                items.append(
                    ReadinessIssue(
                        "model type \"base\" for rolling cont",
                        "Rolling continuation requires Qwen3-TTS model type \"base\""
                    )
                )
            if project.get_model_setting('qwen3tts_local', 'batch_size') > 1:
                items.append(
                    ReadinessIssue(
                        "batch size 1 for rolling cont",
                        "Rolling continuation requires batch size 1"
                    )
                )

        match project.get_model_setting('qwen3tts_local', 'model_type'):
            case "custom_voice":
                # A missing speaker is checked per used voice (voice pre-flight).
                # Validating a set one needs the checkpoint's speaker inventory.
                if instance and instance.model_type == "custom_voice":
                    invalid = [
                        f"Voice {index + 1} ({speaker_id})"
                        for index, speaker_id in enumerate(cls.get_voice_speaker_ids(project))
                        if speaker_id and not instance.get_resolved_speaker_info(speaker_id)[1]
                    ]
                    if invalid:
                        items.append( ReadinessIssue(
                            "valid speaker id",
                            "Invalid speaker for this model: " + ", ".join(invalid) + " (see Voices menu)",
                        ) )
            case "voice_design":
                ... # per-voice instructions do not add model-global requirements
            case "base" | _:
                ... # voice-clone state is validated lazily (validate_voices)

        return items

    def get_warning_issues(self, project: Project) -> list[str]:

        warnings = super().get_warning_issues(project)

        from tts_audiobook_tool.tts_models.language_value_note import make_language_value_note
        language, detail = self.resolve_language_value(project.language_code)
        warnings.append(make_language_value_note(language, project.language_code, detail))

        return warnings

    # ---

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
) -> str:
        s = project.get_model_setting('qwen3tts_local', 'target') or Qwen3BaseModel.DEFAULT_REPO_ID
        s = s.removeprefix("Qwen/")
        s = ellipsize_path_for_menu(s)
        s = f"{cls.INFO.ui['proper_name']} {COL_DIM}({s})"
        return s

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:

        match project.get_model_setting('qwen3tts_local', 'model_type'):
            case "custom_voice":
                speaker_ids = cls.get_voice_speaker_ids(project)
                if speaker_ids and speaker_ids[0]:
                    return cls.get_voice_tag_for_preset(speaker_ids[0])
                else:
                    return "speaker" # good enough
            case "voice_design":
                return "voice_design"
            case "base" | _:
                return super().get_voice_tag(project)

# ---

class Qwen3ModelType(str, Enum):
    BASE = "base"
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    UNKNOWN = "unknown"
