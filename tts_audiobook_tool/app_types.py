from __future__ import annotations
from enum import Enum
from functools import cache
from typing import NamedTuple

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

class Hint:
    def __init__(self, key: str, heading: str, text: str):
        self.key: str = key
        self.heading: str = heading
        self.text: str = text

# ---

class NormalizationSpecs(NamedTuple):
    json_value: str
    label: str
    i: float
    lra: float
    tp: float

NORMALIZATION_SPECS_DEFAULT = NormalizationSpecs(json_value="default", label="Default - ACX standard", i=-19.0, lra=9.0, tp=-3.0) # Tracks with 'ACX standard' basically
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
