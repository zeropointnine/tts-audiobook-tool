from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.local_definition import require_local_definition


class MiraBaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("mira_local").value

    # Parameter defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`
    LOCAL = require_local_definition("mira_local")
    MAX_NEW_TOKENS = 2048 # default is 1024, which is enough for ~60 words

    def clear_voice_clone(self) -> None:
        ...

    def set_params(self, temperature: float, max_new_tokens: int, top_k: int=-1, top_p: float=-1, repetition_penalty: float=-1) -> None:
        ...
