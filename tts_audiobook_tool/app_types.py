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

class TtsInfo(NamedTuple):
    """
    App properties of a supported TTS model
    """
    # bucket of ui-related strings and values
    ui: dict
    # identifier used in file names
    file_tag: str
    # the model's output sample rate
    sample_rate: int
    # Should semantic trim return end time stamp if is last word
    # Doing so is generally redundant and risks unintended partial cropping of end of last word,
    # but can be useful for chopping off hallucinated noises past last word (Chatterbox)
    semantic_trim_last: bool
    # Model does not respect em-dashes in terms of 'prosody', so replace with some other puncutation which it will
    em_dash_replace: str

OUTE_SPECS = TtsInfo(
    ui = {
        "proper_name": "Oute TTS",
        "short_name": "Oute",
        "voice_path_console": "Enter voice clone audio clip file path (up to 15s) (wav, flac, or mp3): ", # no m4a
        "voice_path_requestor": "Select voice clone audio clip (up to 15s) (wav, flac, or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".mp3"]
    },
    file_tag="oute",
    sample_rate=44100,
    semantic_trim_last=False,
    em_dash_replace=", " # helps maybe
)
CHATTERBOX_SPECS = TtsInfo(
    ui = {
        "proper_name": "Chatterbox TTS",
        "short_name": "Chatterbox",
        "voice_path_console": "Enter voice clone audio clip (wav, flac, m4a or mp3): ",
        "voice_path_requestor": "Select voice clone audio clip (wav, flac, m4a or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".m4a", ".mp3"]
    },
    file_tag="chatterbox",
    sample_rate=24000,
    semantic_trim_last=True,
    em_dash_replace=": " # helps
)
FISH_SPECS = TtsInfo(
    ui = {
        "proper_name": "Fish S1-mini",
        "short_name": "S1-mini",
        "voice_path_console": "Enter voice clone audio clip file path (up to 10s) (wav, flac, or mp3): ", # no m4a
        "voice_path_requestor": "Select voice clone audio clip (up to 10s) (wav, flac, or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".mp3"]
    },
    file_tag="s1-mini",
    sample_rate=44100,
    semantic_trim_last=False,
    em_dash_replace=""
)

class TtsType(Enum):

    NONE = TtsInfo({}, "none", 0, False, "")
    OUTE = OUTE_SPECS
    CHATTERBOX = CHATTERBOX_SPECS
    FISH = FISH_SPECS

    @staticmethod
    @cache
    def all_file_tags() -> set[str]:
         return { item.value.file_tag for item in TtsType }

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
