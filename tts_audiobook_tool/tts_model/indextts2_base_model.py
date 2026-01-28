from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class IndexTts2BaseModel(TtsBaseModel):

    INFO = TtsModelInfos.INDEXTTS2.value

    DEFAULT_EMO_VOICE_ALPHA = 0.65 # project gradio demo default
    DEFAULT_TEMPERATURE = 0.8 # project api default
    DEFAULT_USE_FP16 = False # project api default

    @classmethod
    def get_voice_display_info(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[str, str]:

        prefix, value = super().get_voice_display_info(project, instance)

        if project.indextts2_voice_file_name:
            if project.indextts2_emo_vector or project.indextts2_emo_voice_file_name:
                value += " + emotion"

        return prefix, value
