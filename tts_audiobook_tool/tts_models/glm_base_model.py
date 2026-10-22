from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.local_definition import require_local_definition

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class GlmBaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("glm_local").value
    LOCAL = require_local_definition("glm_local")
    # Selectable output sample rates (catalog `sr` choices)
    SAMPLE_RATES: list[int] = list(LOCAL.choice("sr").values)

    @classmethod
    def get_max_words_range_reco(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[int, int, str]:
        return (20, 40, "")

    @classmethod
    def get_output_sample_rate(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> int:
        # GLM's output samplerate is a selectable project setting (see the
        # voice menu's "Model samplerate" option). A live instance is
        # constructed from that same setting, so the project value is
        # authoritative here too.
        return GlmBaseModel.LOCAL.resolve_int(project, "sr")

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        s = cls.INFO.ui.get("proper_name") or ""
        s += f" {COL_DIM}(sr: {project.get_model_setting('glm_local', 'sr')})"
        return s
