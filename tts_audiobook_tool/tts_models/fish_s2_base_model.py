from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class FishS2BaseModel(TtsBaseModel):

    INFO = TtsModelType.FISH_S2.value

    DEFAULT_COMPILE_ENABLED = True

    DEFAULT_TEMPERATURE = 0.7 # from gradio demo
    MIN_TEMPERATURE = 0.01
    MAX_TEMPERATURE = 1.99 # library asserts temperature < 2
    DEFAULT_TOP_K = 30 # library function default value
    DEFAULT_TOP_P = 0.9 # library function default value

    ROLLING_CONTINUATION_MAX_LENGTH = 3

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...
