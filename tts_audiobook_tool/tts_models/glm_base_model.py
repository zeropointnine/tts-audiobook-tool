from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class GlmBaseModel(TtsBaseModel):
    
    INFO = TtsModelType.GLM.value
    
    SAMPLE_RATES = [24000, 32000]

    @classmethod
    def get_output_sample_rate(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> int:
        # GLM's output samplerate is a selectable project setting (see the
        # voice menu's "Model samplerate" option). A live instance is
        # constructed from that same setting, so the project value is
        # authoritative here too.
        sr = int(project.glm_sr)
        if sr not in GlmBaseModel.SAMPLE_RATES:
            sr = cls.INFO.default_output_sample_rate
        return sr

    @classmethod 
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        s = cls.INFO.ui.get("proper_name") or ""
        s += f" {COL_DIM}(sr: {project.glm_sr})"
        return s
