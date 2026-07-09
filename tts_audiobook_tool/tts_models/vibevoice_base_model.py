from abc import ABC, abstractmethod

from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class VibeVoiceBaseModel(TtsBaseModel, ABC):

    INFO = TtsModelType.VIBEVOICE.value

    DEFAULT_REPO_ID = "microsoft/VibeVoice-1.5b"
    PRESET_REPO_IDS = [
        "microsoft/VibeVoice-1.5b",
        "vibevoice/VibeVoice-7B",
    ]

    # nb, their gradio demo default is 1.3, which is IMO much too low
    CFG_DEFAULT = 2.0
    CFG_MIN = 1.0
    CFG_MAX = 7.0

    DEFAULT_NUM_STEPS = 10 # from vibevoice library code

    # Must accommodate worst-case prompt size (app limit 80 words)
    MAX_TOKENS = 250
    # Slightly above the library default to reduce end truncation on longer prompts
    MAX_LENGTH_TIMES = 2.5

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
) -> str:
        target = project.vibevoice_target or VibeVoiceBaseModel.DEFAULT_REPO_ID
        target = target.removeprefix("vibevoice/")
        target = target.removeprefix("microsoft/")
        target = ellipsize_path_for_menu(target)
        s = f"{cls.INFO.ui['proper_name']} {COL_DIM}({target})"
        return s

    @property
    @abstractmethod
    def has_lora(self) -> bool:
        ...

    @classmethod
    def can_hallucinate_music(cls, project: Project, instance: TtsBaseModel | None=None) -> bool:
        return True

    @classmethod
    def get_blocking_issues(cls, project: Project, instance: TtsBaseModel | None) -> list[ReadinessIssue]:
        errors = []
        if instance:
            assert isinstance(instance, VibeVoiceBaseModel)
            if project.vibevoice_lora_target and not instance.has_lora:
                errors.append( ReadinessIssue("valid LoRA", "Couldn't load LoRA") )
        return errors

    def get_warning_issues(self, project: Project) -> list[str]:
        warnings = []
        if not project.vibevoice_voice_file_name and not project.vibevoice_lora_target:
            warning = "Model may generate random voices because no voice sample or lora has been defined"
            warnings.append(warning) 
        return warnings

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:

        def get_lora_value() -> str:
            value = Path(project.vibevoice_lora_target).stem
            value = app_text.sanitize_for_filename(value[:30])
            return value

        match (bool(project.vibevoice_voice_file_name), bool(project.vibevoice_lora_target)):
            case (False, False):
                return "none"
            case (True, True):
                # use lora as the more 'relevant' identifier
                return get_lora_value()
            case (True, False):
                return super().get_voice_tag(project)
            case (False, True):
                return get_lora_value()

    @classmethod
    def get_voice_display_info(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[str, str]:

        match (bool(project.vibevoice_voice_file_name), bool(project.vibevoice_lora_target)):
            case (False, False):
                prefix = "current voice clone"
                value = COL_ERROR + "none"
            case (True, True):
                prefix = "currently"
                value = COL_ACCENT + "lora + voice clone"
            case (True, False):
                prefix = "current voice clone"
                value = COL_ACCENT + ProjectVoiceUtil.get_voice_label(project)
            case (False, True):
                prefix = "current lora"
                value = COL_ACCENT + ellipsize_path_for_menu(project.vibevoice_lora_target)
        return prefix, value

    @classmethod
    def get_strictness_warning(cls, strictness: Strictness, project: Project, instance: TtsBaseModel | None) -> str:

        WARNING = "Not recommended with VibeVoice 1.5B model (w/o use of lora)"

        if strictness >= Strictness.HIGH:
            if not instance:
                return WARNING
            else:
                if project.vibevoice_target or project.vibevoice_lora_target:
                    # Custom model or lora means can't make the "not recommended" assumption
                    return ""
                else:
                    return WARNING
        else:
            return ""
