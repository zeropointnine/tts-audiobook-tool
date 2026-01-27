from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class FishProtocol(Protocol):

    DEFAULT_TEMPERATURE = 0.8 # from fish gradio demo

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...

    def clear_voice_clone(self) -> None:
        ...

    def generate(self, text: str, temperature: float, seed: int) -> Sound | str:
        ...

class FishModelProtocol(TtsModel, FishProtocol):
    ...


