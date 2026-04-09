from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class FishS2BaseModel(TtsBaseModel):

    INFO = TtsModelInfos.FISH_S2.value

    DEFAULT_COMPILE_ENABLED = True

    DEFAULT_TEMPERATURE = 0.7 # from gradio demo
    DEFAULT_TOP_K = 30 # library function default value
    DEFAULT_TOP_P = 0.9 # library function default value
    DEFAULT_REPETITION_PENALTY = 1.1 # library function default value

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...


