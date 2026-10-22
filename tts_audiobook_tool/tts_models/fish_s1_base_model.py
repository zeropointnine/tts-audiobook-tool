from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.local_definition import require_local_definition

class FishS1BaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("fish_s1_local").value

    # Parameter defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`
    LOCAL = require_local_definition("fish_s1_local")

    def clear_voice_clone(self) -> None:
        ...
