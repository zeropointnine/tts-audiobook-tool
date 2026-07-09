from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

class FishS1BaseModel(TtsBaseModel):

    INFO = TtsModelType.FISH_S1.value

    DEFAULT_COMPILE_ENABLED = True

    DEFAULT_TEMPERATURE = 0.8 # from fish gradio demo
    DEFAULT_TOP_P = 0.8 # library function default value
    DEFAULT_REPETITION_PENALTY = 1.1 # library function default value

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...
