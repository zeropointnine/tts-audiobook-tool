from tts_audiobook_tool.app_types import VoiceDisplayInfo
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class IndexTts2BaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("indextts2_local").value

    DEFAULT_EMO_VOICE_ALPHA = 0.65 # project gradio demo default
    DEFAULT_TEMPERATURE = 0.8 # project api default
    DEFAULT_USE_FP16 = False # project api default
    DEFAULT_TOP_P = 0.8
    DEFAULT_TOP_K = 30

    @classmethod
    def get_max_words_range_reco(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[int, int, str]:
        return (40, 60, "")


    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        s = cls.INFO.ui.get("proper_name") or ""
        s += f" {COL_DIM}(fp16: {project.get_model_setting('indextts2_local', 'use_fp16')})"
        return s

    @classmethod
    def get_voice_display_info(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> VoiceDisplayInfo | None:

        display_info = super().get_voice_display_info(project, instance)
        if display_info is None:
            return None
        value = display_info.value

        if ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.require_by_id("indextts2_local")):
            if project.get_model_setting('indextts2_local', 'emo_vector') or project.get_model_setting('indextts2_local', 'emo_voice'):
                value += " + emotion"

        return display_info._replace(value=value)
