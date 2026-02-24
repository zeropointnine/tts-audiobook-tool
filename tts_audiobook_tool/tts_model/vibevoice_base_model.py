from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class VibeVoiceBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.VIBEVOICE.value

    DEFAULT_REPO_ID = "microsoft/VibeVoice-1.5b"

    # nb, their gradio demo default is 1.3, which is IMO much too low
    CFG_DEFAULT = 3.0
    CFG_MIN = 1.0
    CFG_MAX = 7.0

    DEFAULT_NUM_STEPS = 10 # from vibevoice library code

    # Must accommodate worst-case prompt size (app limit 80 words)
    MAX_TOKENS = 250

    @property
    def has_lora(self) -> bool:
        ...

    def get_prereq_warnings(self, project: Project) -> list[str]:
        warnings = []
        if not project.vibevoice_voice_file_name and not project.vibevoice_lora_target:
            warning = "Model may generate random voices because no voice sample or lora has been defined"
            warnings.append(warning) 
        return warnings

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:

        def get_lora_value() -> str:
            value = Path(project.vibevoice_lora_target).stem
            value = TextUtil.sanitize_for_filename(value[:30])
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
                value = COL_ACCENT + ellipsize_path_for_menu(project.vibevoice_voice_file_name)
            case (False, True):
                prefix = "current lora"
                value = COL_ACCENT + ellipsize_path_for_menu(project.vibevoice_lora_target)
        return prefix, value

