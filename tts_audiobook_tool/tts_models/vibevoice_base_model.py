from abc import ABC, abstractmethod

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.local_definition import require_local_definition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class VibeVoiceBaseModel(TtsBaseModel, ABC):

    INFO = TtsModelType.require_by_id("vibevoice_local").value

    DEFAULT_REPO_ID = "microsoft/VibeVoice-1.5b"
    PRESET_REPO_IDS = [
        "microsoft/VibeVoice-1.5b",
        "vibevoice/VibeVoice-7B",
    ]

    # CFG and steps defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`.
    # nb, their gradio demo CFG default is 1.3, which is IMO much too low
    LOCAL = require_local_definition("vibevoice_local")

    # Must accommodate worst-case prompt size (app limit 80 words)
    MAX_TOKENS = 250
    # Slightly above the library default to reduce end truncation on longer prompts
    MAX_LENGTH_TIMES = 2.5

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
) -> str:
        target = project.get_model_setting('vibevoice_local', 'target') or VibeVoiceBaseModel.DEFAULT_REPO_ID
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
            if project.get_model_setting('vibevoice_local', 'lora_target') and not instance.has_lora:
                errors.append( ReadinessIssue("valid LoRA", "Couldn't load LoRA") )
        return errors

    @classmethod
    def has_fixed_default_voice(cls, project: Project) -> bool:
        # A LoRA gives sample-less voices a fixed speaker, so the "voices may vary"
        # run note (voice_variation_warning) does not apply
        return bool(project.get_model_setting('vibevoice_local', 'lora_target'))

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:

        def get_lora_value() -> str:
            value = Path(project.get_model_setting('vibevoice_local', 'lora_target')).stem
            value = app_text.sanitize_for_filename(value[:30])
            return value

        match (bool(ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.require_by_id("vibevoice_local"))), bool(project.get_model_setting('vibevoice_local', 'lora_target'))):
            case (False, False):
                return "none"
            case (True, True):
                # use lora as the more 'relevant' identifier
                return get_lora_value()
            case (True, False):
                return super().get_voice_tag(project)
            case (False, True):
                return get_lora_value()

# ---

def is_15b(project: Project) -> bool:
    # Best-guess using target value only
    target = project.get_model_setting('vibevoice_local', 'target').lower()
    if not target:
        return True # 1.5B is the default
    if "1.5b" in target or "15b" in target:
        return True
    return False

