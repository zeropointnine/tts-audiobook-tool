from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class OuteProtocol(Protocol):

    DEFAULT_TEMPERATURE = 0.4 # from model library code

    def create_speaker(self, path: str) -> dict | str:
        ...

    def generate(
            self,
            prompt: str,
            voice: dict,
            temperature: float = -1
    ) -> Sound | str:
        ...

class OuteModelProtocol(TtsModel, OuteProtocol):
        ...


