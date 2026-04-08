from typing import Protocol

from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class FishS1BaseModel(TtsBaseModel):

    INFO = TtsModelInfos.FISH_S1.value

    DEFAULT_COMPILE_ENABLED = True

    DEFAULT_TEMPERATURE = 0.8 # from fish gradio demo

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...


