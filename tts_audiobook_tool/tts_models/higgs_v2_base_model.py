from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class HiggsV2BaseModel(TtsBaseModel):

    INFO = TtsModelType.HIGGS_V2.value

    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 0.95

    @classmethod
    def get_max_words_range_reco(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[int, int, str]:
        return (40, 40, "")

