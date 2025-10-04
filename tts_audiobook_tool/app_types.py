from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cache
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
    Duck-typed data structure for use with the "Word" objects returned by faster-whisper generate()
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

class Hint:
    def __init__(self, key: str, heading: str, text: str):
        self.key: str = key
        self.heading: str = heading
        self.text: str = text

# ---

class ValidationResult(ABC):
    """ Base class for a validation result """
    pass

    @abstractmethod
    def get_ui_message(self) -> str:
        return ""

@dataclass
class PassResult(ValidationResult):
    def get_ui_message(self) -> str:
        return f"Passed validation tests"

@dataclass
class TrimmableResult(ValidationResult):
    base_message: str
    start_time: float | None
    end_time: float | None
    duration: float

    def __post_init__(self):
        if self.start_time is None and self.end_time is None:
            raise ValueError("start or end must be a float")
        if self.end_time == 0.0:
            raise ValueError("end time must be None or must be greater than zero")

    def get_ui_message(self) -> str:

        start_string = f"{self.start_time:.2f}" if self.start_time is not None else ""
        end_string = f"{self.end_time:.2f}" if self.end_time is not None else ""
        duration_string = f"{self.duration:.2f}"

        if start_string and end_string:
            message = f"Will remove 0 to {start_string} and {end_string} to end {duration_string}"
        elif start_string:
            message = f"Will remove 0 to {start_string}"
        else: # end
            message = f"Will remove {end_string} to end {duration_string}"
        return self.base_message + ". " + message

@dataclass
class FailResult(ValidationResult):
    message: str

    def get_ui_message(self) -> str:
        return self.message

# ---

class NormalizationSpecs(NamedTuple):
    json_value: str
    label: str
    i: float
    lra: float
    tp: float

NORMALIZATION_SPECS_DEFAULT = NormalizationSpecs(json_value="default", label="Default - ACX standard", i=-19.0, lra=9.0, tp=-3.0) # Tracks with 'ACX standard'
NORMALIZATION_SPECS_STRONGER = NormalizationSpecs(json_value="stronger", label="Stronger", i=-17.0, lra=7.0, tp=-2.5)
NORMALIZATION_SPECS_DISABLED = NormalizationSpecs(json_value="none", label="Disabled", i=0, lra=0, tp=0)

class NormalizationType(Enum):
    DEFAULT = NORMALIZATION_SPECS_DEFAULT
    STRONGER = NORMALIZATION_SPECS_STRONGER
    DISABLED = NORMALIZATION_SPECS_DISABLED

    @staticmethod
    @cache
    def all_json_values() -> set[str]:
         return { item.value.json_value for item in NormalizationType }

    @staticmethod
    def from_json_value(s: str) -> NormalizationType | None:
        for item in NormalizationType:
            if s == item.value.json_value:
                return item
        return None

# ---

class SttVariant(str, Enum):
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"

