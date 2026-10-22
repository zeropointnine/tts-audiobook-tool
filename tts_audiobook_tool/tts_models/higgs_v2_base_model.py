from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.local_definition import require_local_definition

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class HiggsV2BaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("higgs_v2_local").value

    # Parameter defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`
    LOCAL = require_local_definition("higgs_v2_local")

    @classmethod
    def get_max_words_range_reco(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[int, int, str]:
        return (40, 40, "")

