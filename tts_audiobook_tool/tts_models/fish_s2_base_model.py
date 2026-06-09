from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class FishS2BaseModel(TtsBaseModel):

    INFO = TtsModelType.FISH_S2.value

    DEFAULT_COMPILE_ENABLED = True

    TEMPERATURE_DEFAULT = 0.7 # from their gradio demo; note that library function default is 1.0
    TEMPERATURE_MIN = 0.01
    TEMPERATURE_MAX = 1.99 # library asserts temperature < 2

    TOP_K_DEFAULT = 30 # library function default value
    TOP_P_DEFAULT = 0.9 # library function default value
    REPETITION_PENALTY_DEFAULT = 1.1

    ROLLING_CONTINUATION_MAX_LENGTH = 3

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...
