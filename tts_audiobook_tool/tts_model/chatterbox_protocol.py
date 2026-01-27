from __future__ import annotations
from enum import Enum
from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class ChatterboxProtocol(Protocol):

    DEFAULT_EXAGGERATION = 0.5 # from chatterbox library code
    DEFAULT_CFG = 0.5
    DEFAULT_TEMPERATURE = 0.8

    def generate(
            self,
            text: str,
            voice_path: str,
            exaggeration: float,
            cfg: float,
            temperature: float,
            seed: int,  
            language_id: str = ""
    ) -> Sound | str:
        ...

class ChatterboxModelProtocol(TtsModel, ChatterboxProtocol):
        ...

class ChatterboxType(tuple[str, str, str], Enum):
    MULTILINGUAL = "multilingual", "Chatterbox-Multilingual", "Supports multiple languages"
    TURBO = "turbo", "Chatterbox-Turbo", "Distilled, en only"

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]
    
    @property
    def description(self) -> str:
        return self.value[2]

    @staticmethod
    def get_by_id(id: str) -> ChatterboxType | None:
        for item in list(ChatterboxType):
            if id == item.id:
                return item
        return None


