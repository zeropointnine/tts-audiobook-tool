from abc import ABC

from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.local_definition import require_local_definition
from tts_audiobook_tool.tts_models.model_support import VoiceLoraMode
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

    @classmethod
    def can_hallucinate_music(cls, project: Project, instance: TtsBaseModel | None=None) -> bool:
        return True

    @classmethod
    def voice_lora_mode(cls, project: Project) -> VoiceLoraMode:
        # PEFT-format per-voice language model adapters, merged and switched in place (vibevoice_lora)
        return "merged"

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:
        # The project-level tag follows voice 1, whose LoRA is the more salient identifier,
        # even with a sample; per-line tags follow each voice's adapter
        voices = project.voice_references
        lora = ProjectVoiceUtil.active_voice_lora(project, voices[0]) if voices else None
        if lora is None:
            return super().get_voice_tag(project)
        return ProjectVoiceUtil.make_voice_tag(lora.target.display_name())

# ---

def is_15b(project: Project) -> bool:
    # Best-guess using target value only
    target = project.get_model_setting('vibevoice_local', 'target').lower()
    if not target:
        return True # 1.5B is the default
    if "1.5b" in target or "15b" in target:
        return True
    return False

