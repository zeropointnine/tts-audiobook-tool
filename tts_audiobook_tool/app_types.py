from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cache
import platform
from typing import NamedTuple, Protocol

from numpy import ndarray

from tts_audiobook_tool.constants_config import *

"""
Some common app types
"""

class SingletonBase:
    """
    Base class for singleton classes. Works.
    """
    _instances = {}
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class Sound(NamedTuple):
    data: ndarray
    sr: int

    @property
    def duration(self) -> float:
        return len(self.data) / self.sr

class Word(Protocol):
    """
    Duck-typed data structure for use with the `Word` objects returned by faster-whisper `generate()`
    """
    start: float
    end: float
    word: str
    probability: float

class ConcreteWord(Word):
    """
    Instantiatable "Word"-compatible object
    """
    def __init__(self, start: float, end: float, word: str, probability: float):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability

# ---

@dataclass
class ValidationResult(ABC):
    """ Base class for a validation result """
    # dummy: str = ""
    @abstractmethod
    def get_ui_message(self) -> str:
        return ""

@dataclass
class TranscriptResult(ValidationResult, ABC):
    """ Base class for a ValidationResult that has transcript data """
    transcript_words: list[Word]

@dataclass
class PassResult(TranscriptResult):
    num_word_fails: int
    word_fail_threshold: int
    transcript_words: list[Word]
    def get_ui_message(self) -> str:
        return f"Passed validation tests"

@dataclass
class FailResult(TranscriptResult):
    message: str
    num_word_fails: int
    word_fail_threshold: int
    transcript_words: list[Word]
    def get_ui_message(self) -> str:
        return self.message

@dataclass
class TrimmableResult(TranscriptResult):
    """ The sound can be trimmed at either/both ends to create a valid sound """
    start_time: float | None
    end_time: float | None
    duration: float
    transcript_words: list[Word]

    def __post_init__(self):
        if self.start_time is None and self.end_time is None:
            raise ValueError("start or end must be a float")
        if self.end_time == 0.0:
            raise ValueError("end time must be None or must be greater than zero")

    def get_ui_message(self) -> str:
        s = "Excess words detected at "
        if self.start_time and self.end_time:
            s += f"start and end; will remove 0s to {self.start_time:.2f}s, {self.end_time:.2f}s to end"
        elif self.start_time:
            s += f"start; will remove 0s to {self.start_time:.2f}s"
        else:
            s += f"end; will remove from {self.end_time:.2f}s to end"
        return s

@dataclass
class SkippedResult(ValidationResult):
    message: str
    def get_ui_message(self) -> str:
        return self.message

# ---

class NormalizationSpecs(NamedTuple):
    id: str
    label: str
    i: float
    lra: float
    tp: float

NORMALIZATION_SPECS_DEFAULT = NormalizationSpecs(id="default", label="ACX standard", i=-19.0, lra=9.0, tp=-3.0) # Values approximate 'ACX standard'
NORMALIZATION_SPECS_STRONGER = NormalizationSpecs(id="stronger", label="Stronger", i=-17.0, lra=7.0, tp=-2.5)
NORMALIZATION_SPECS_DISABLED = NormalizationSpecs(id="none", label="Disabled", i=0, lra=0, tp=0)

class NormalizationType(Enum):
    DEFAULT = NORMALIZATION_SPECS_DEFAULT
    STRONGER = NORMALIZATION_SPECS_STRONGER
    DISABLED = NORMALIZATION_SPECS_DISABLED

    @staticmethod
    @cache
    def all_ids() -> set[str]:
         return { item.value.id for item in NormalizationType }

    @staticmethod
    def from_id(s: str) -> NormalizationType | None:
        for item in NormalizationType:
            if s == item.value.id:
                return item
        return None

# ---

class SttVariant(tuple[str, str], Enum):

    LARGE_V3 = ("large-v3", "Best accuracy") # default
    LARGE_V3_TURBO = ("large-v3-turbo", "Less memory, faster")
    DISABLED = ("disabled", "Skips validation step when generating audio, adds no extra memory")

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def description(self) -> str:
        return self.value[1]

    @staticmethod
    def get_by_id(id: str) -> SttVariant | None:
        for item in list(SttVariant):
            if id == item.id:
                return item
        return None

# ---

class SttConfig(tuple[str, str, str], Enum):
    """
    Supported combinations of device + compute_type for the faster-whisper model
    Rem, no MPS, I think
    """

    CPU_INT8FLOAT32 = ("cpu", "int8_float32", "CPU, int8_float32") # default
    CUDA_FLOAT16 = ("cuda", "float16", f"CUDA, float16")

    @property
    def device(self) -> str:
        return self.value[0]

    @property
    def compute_type(self) -> str:
        return self.value[1]

    @property
    def description(self) -> str:
        return self.value[2]

    @property
    def id(self) -> str:
        return self.device + "_" + self.compute_type

    @staticmethod
    def from_id(id: str) -> SttConfig | None:
        for item in list(SttConfig):
            if id == item.id:
                return item
        return None

    @staticmethod
    def get_default() -> SttConfig:
        import torch
        if torch.cuda.is_available():
            if platform.system() == "Linux":
                return SttConfig.CUDA_FLOAT16
            else:
                return SttConfig.CUDA_FLOAT16
        else:
            return SttConfig.CPU_INT8FLOAT32

# ---

SS_NORMAL_DESC = \
"""    Text is segmented by paragraph, and within each paragraph, by sentence.
    This produces predictable caesuras between sentences."""

SS_MAX_LEN_DESC = \
"""    Text is segmented by paragraph, and within each paragraph, segmented by 
    'max words per segment' at the nearest phrase boundary.
    This maximizes text length per TTS generation."""

class SegmentationStrategy(tuple[str, str, str], Enum):

    NORMAL = "normal", "Normal", SS_NORMAL_DESC
    MAX_LEN = "max_len", "Maximized word count", SS_MAX_LEN_DESC

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
    def from_id(s: str) -> SegmentationStrategy | None:
        for item in list(SegmentationStrategy):
            if s == item.id:
                return item
        return None

# ---

class ExportType(tuple[str, str, str], Enum):

    AAC = ("m4a", "AAC/M4A", ".m4a") # default
    FLAC = ("flac", "FLAC", ".flac")

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]

    @property
    def suffix(self) -> str:
        return self.value[2]

    @staticmethod
    def get_by_id(id: str) -> ExportType | None:
        for item in list(ExportType):
            if id == item.id:
                return item
        return None

# ---

class RealTimeMenuState:
    """ Values related to the real-time playback feature """
    from tts_audiobook_tool.phrase import PhraseGroup
    custom_phrase_groups: list[PhraseGroup] = [] # ie, PhraseGroups
    line_range: tuple[int, int] | None = None
