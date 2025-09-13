from abc import ABC, abstractmethod
from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model_info import TtsModelInfo


class TtsModel(ABC):
    """
    Base class for a TTS model
    """

    def __init__(self, info: TtsModelInfo):
        self.info = info

    @abstractmethod
    def kill(self) -> None:
        """
        Performs any clean-up (nulling out local member variables, etc)
        that might help with garbage collection.
        """
        ...

    def preprocess_text(self, text: str) -> str:
        """
        Applies text transformations from `self.info.substitutions`
        to address any model-specific idiosyncrasies, etc.
        Concrete class may want to override-and-super with extra logic.
        """
        for before, after in self.info.substitutions:
            text = text.replace(before, after)
        return text

# ---

class OuteGenerateProtocol(Protocol):
    def create_speaker(self, path: str) -> dict | str:
        ...

    def generate(
            self,
            prompt: str,
            voice: dict,
            temperature: float = -1
    ) -> Sound | str:
        ...

class OuteModelProtocol(TtsModel, OuteGenerateProtocol):
        ...


class ChatterboxGenerateProtocol(Protocol):
    def generate(
            self,
            text: str,
            voice_path: str,
            exaggeration: float,
            cfg: float,
            temperature: float
    ) -> Sound | str:
        ...

class ChatterboxModelProtocol(TtsModel, ChatterboxGenerateProtocol):
        ...


class FishGenerateProtocol(Protocol):
    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...

    def clear_voice_clone(self) -> None:
        ...

    def generate(self, text: str, temperature: float) -> Sound | str:
        ...

class FishModelProtocol(TtsModel, FishGenerateProtocol):
    ...


class HiggsGenerateProtocol(Protocol):
    def generate(
            self,
            p_voice_path: str,
            p_voice_transcript: str,
            text: str,
            seed: int,
            temperature: float
    ) -> Sound | str:
        ...

class HiggsModelProtocol(TtsModel, HiggsGenerateProtocol):
    ...


class VibeVoiceGenerateProtocol(Protocol):
    def generate(
            self,
            text: str,
            voice_path: str,
            cfg_scale: float,
            num_steps: int
    ) -> Sound | str:
        ...

class VibeVoiceModelProtocol(TtsModel, VibeVoiceGenerateProtocol):
    ...
