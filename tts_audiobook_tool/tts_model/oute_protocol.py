from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class OuteProtocol(Protocol):

    DEFAULT_TEMPERATURE = 0.4 # from model library code


class OuteModelProtocol(TtsModel, OuteProtocol):

    def create_speaker(self, path: str) -> dict | str:
        ...


