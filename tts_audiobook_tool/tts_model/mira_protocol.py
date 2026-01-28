from typing import Protocol

from tts_audiobook_tool.tts_model.tts_model import TtsModel


class MiraProtocol(Protocol):

    TEMPERATURE_DEFAULT = 0.7
    TEMPERATURE_MIN = 0.0
    TEMPERATURE_MAX = 2.0

    MAX_NEW_TOKENS = 2048 # default is 1024, which is enough for ~60 words

class MiraModelProtocol(TtsModel, MiraProtocol):

    def set_voice_clone(self, path: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...
    def set_params(self, temperature: float, max_new_tokens: int) -> None:
        ...
