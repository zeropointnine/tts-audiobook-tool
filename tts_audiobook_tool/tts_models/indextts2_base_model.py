from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.local_definition import require_local_definition

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class IndexTts2BaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("indextts2_local").value

    # Parameter defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`
    LOCAL = require_local_definition("indextts2_local")

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
