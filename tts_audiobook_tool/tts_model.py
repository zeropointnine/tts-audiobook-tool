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

class OuteProtocol(Protocol):

    DEFAULT_TEMPERATURE = 0.4 # from oute library code

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
            temperature: float
    ) -> Sound | str:
        ...

class ChatterboxModelProtocol(TtsModel, ChatterboxProtocol):
        ...


class FishProtocol(Protocol):

    DEFAULT_TEMPERATURE = 0.8 # from fish gradio demo

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...

    def clear_voice_clone(self) -> None:
        ...

    def generate(self, text: str, temperature: float) -> Sound | str:
        ...

class FishModelProtocol(TtsModel, FishProtocol):
    ...


class HiggsProtocol(Protocol):

    # from higgs project README example code
    # (nb, higgs library code uses a default of 1.0 which is much too high for narration)
    DEFAULT_TEMPERATURE = 0.3

    def generate(
            self,
            p_voice_path: str,
            p_voice_transcript: str,
            text: str,
            seed: int,
            temperature: float
    ) -> Sound | str:
        ...

class HiggsModelProtocol(TtsModel, HiggsProtocol):
    ...


class VibeVoiceProtocol(Protocol):

    DEFAULT_MODEL_PATH = "microsoft/VibeVoice-1.5b"
    DEFAULT_MODEL_NAME = "VibeVoice 1.5B"

    # nb, their gradio demo default is 1.3, which is much too low; library code is 1.0 even
    DEFAULT_CFG = 3.0

    DEFAULT_NUM_STEPS = 10 # from vibevoice library code

    # Setting this explicitly low b/c hallucinations can oftentimes
    # expand to fill up the entire context (!),
    # and app opts not to use model's long context feature anyway
    MAX_TOKENS = 250

    def generate(
            self,
            text: str,
            voice_path: str,
            cfg_scale: float,
            num_steps: int
    ) -> Sound | str:
        ...

class VibeVoiceModelProtocol(TtsModel, VibeVoiceProtocol):
    ...
