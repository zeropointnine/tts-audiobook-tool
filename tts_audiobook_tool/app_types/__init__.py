"""
Shared structural data types for the application

Foundational value-like types, lightweight domain structures 
used across subsystems.

Intentionally excludes heavier stateful/persistent orchestrators such as
Project, Prefs, and State; those own lifecycle, IO, runtime coordination,
and application behavior rather than acting as lightweight shared structures.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cache
import platform
from typing import Any, Callable, NamedTuple, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray
else:
    ndarray = Any

from tts_audiobook_tool.constants import *

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types.phrase import PhraseGroup

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

class Saveable(Protocol):
    """
    An object that can save its own data (`Project`, `Prefs`)
    """    
    def save(self) -> str:
        """ Returns error string if any """
        ...

class Sound(NamedTuple):
    data: ndarray
    sr: int

    @property
    def duration(self) -> float:
        return len(self.data) / self.sr


StreamChunkCallback = Callable[[ndarray], None]
StreamEndCallback = Callable[[], None]

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

class Segment(Protocol):
    start: float
    end: float
    text: str
    words: Sequence[Word]

class ConcreteSegment(Segment):
    """
    Instantiatable `Segment`-compatible object for app-owned Whisper adapters.
    """
    def __init__(self, start: float, end: float, text: str, words: Sequence[Word]):
        self.start = start
        self.end = end
        self.text = text
        self.words = words

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

class HighShelfEq(tuple[str, float, float, float], Enum):
    """
    Preset values for high-shelf EQ settings
    For use with function SoundUtil.high_shelf_eq()
    Value shape: (id, strength, boost_start_hz, q_like)
    """

    DISABLED = ("disabled", 0.0, 2800.0, 0.8)
    MODERATE = ("moderate", 0.6, 4200.0, 1.1)
    STRONGER = ("stronger", 1.1, 3600.0, 1.2)

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def strength(self) -> float:
        return self.value[1]

    @property
    def boost_start_hz(self) -> float:
        return self.value[2]

    @property
    def q_like(self) -> float:
        return self.value[3]

    @staticmethod
    def get_by_id(id: str) -> HighShelfEq | None:
        for item in list(HighShelfEq):
            if id == item.id:
                return item
        return None

# ---

class SttVariant(tuple[str, str], Enum):
    """ Applies to both whisper implementations (faster-whisper and mlx-whisper) """

    LARGE_V3 = ("large-v3", "Reference accuracy") # default
    LARGE_V3_TURBO = ("large-v3-turbo", "Less memory, faster")
    DISABLED = ("disabled", "Skips validation step when generating audio, adds no extra memory")

    @staticmethod
    def get_default() -> SttVariant:
        return SttVariant.LARGE_V3_TURBO

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
    Preferred runtime configuration for STT backends that expose
    device/precision-style execution controls.

    Currently this is used by faster-whisper and ignored when mlx-whisper
    is active on Apple Silicon.
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

class Strictness(tuple[str, int, str], Enum):

    LOW = ("low", 1, "Loose") 
    MODERATE = ("moderate", 2, "Moderate")
    HIGH = ("high", 3, "Strict")
    INTOLERANT = ("intolerant", 4, "Intolerant")

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def level(self) -> int:
        return self.value[1]

    @property
    def label(self) -> str:
        return self.value[2]

    @staticmethod
    def get_by_id(id: str) -> Strictness | None:
        for item in list(Strictness):
            if id == item.id:
                return item
        return None
    
    @staticmethod
    def get_recommended_default(language_code: str) -> Strictness:
        if language_code == "en":
            return Strictness.MODERATE
        else:
            return Strictness.LOW
    
# ---

@dataclass
class ReadinessIssue:
    """
    User-facing issue found while checking whether an action is ready to run.

    Convention:
    - Blockers are readiness issues that prevent a desired action from starting.
    - Warnings are non-blocking readiness issues or messages that may be shown
      before/around an action but do not prevent it.

    The same structure can represent either category. API names communicate the
    category, eg `get_generate_blockers()` versus `get_warning_issues()`.
    """

    # Very short noun phrase naming what is missing or invalid.
    # Eg, "voice clone", "microphone", "valid speaker id".
    short: str

    # Complete user-facing sentence explaining the issue.
    # Eg, "A voice clone sample is required".
    verbose: str

# ---

FILES_DESC = "Each section marker defines the start of a new, separate audio file."
METADATA_DESC = (
    "Section markers are used for M4B chapter metadata and player bookmark metadata.\n"
    "      Always outputs to a single file."
)

class SectionMarkerMode(tuple[str, str, str], Enum):

    FILES = ("files", "Splits into files", FILES_DESC)
    BOOKMARKS = ("metadata", "Adds metadata", METADATA_DESC)

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
    def get_by_id(id: str) -> SectionMarkerMode | None:
        for item in list(SectionMarkerMode):
            if id == item.id:
                return item
        return None
    
# ---

SS_SENTENCE_DESC = \
"""    Text is segmented by sentence. Produces predictable caesuras between sentences. 
      Relatively short resulting word length may help some models maintain 
      a more natural speaking pace."""

SS_SENTENCE_PLUS_DESC = \
"""    Text is segmented by sentence, but short sentences are joined to
      neighboring sentences within a paragraph. This is done to avoid very
      short prompts, which many TTS models tend to struggle with."""

SS_MULTI_DESC = \
"""    Within a paragraph, text is grouped by sentences up to \"max words per segment.\"
      May preserve better continuity of timbre between those sentences compared to
      single-sentence segmentation."""

SS_MAX_LEN_DESC = \
"""    Text is segmented by paragraph, and within each paragraph, segmented by 
      \"max words per segment\" to the nearest sentence or phrase boundary.
      This maximizes text length per TTS generation."""

class SegmentationStrategy(tuple[str, str, str], Enum):

    SENTENCE = "sentence", "Sentence", SS_SENTENCE_DESC
    SENTENCE_PLUS = "normal", "Sentence+", SS_SENTENCE_PLUS_DESC 
    MULTI_SENTENCE = "multi", "Multiple sentences", SS_MULTI_DESC
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

@dataclass
class BookSection:
    """
    Structural subdivision of a Book, such as an imported EPUB chapter or a plain-text
    section. Owns the phrase groups for that section plus an optional display title.
    """
    phrase_groups: list[PhraseGroup]
    title: str = ""


class BookSegmentationSettings(NamedTuple):
    """
    Records the segmentation-time inputs used to produce a Book's current phrase groups.
    Replaces the applied segmentation fields that previously lived directly on Project.
    """
    language_code: str = ""
    max_words_per_segment: int = 0
    strategy: SegmentationStrategy = SegmentationStrategy.SENTENCE_PLUS


@dataclass
class Book:
    """
    Canonical segmented text model for an audiobook. Used as TTS generation input,
    forced-alignment reference text, and ABR/player reader text source.
    """
    sections: list[BookSection]
    title: str = ""
    text_source_kind: str = ""
    audio_source_kind: str = ""
    segmentation_settings: BookSegmentationSettings = BookSegmentationSettings()
    flat_phrase_groups: list[PhraseGroup] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rebuild_flat_phrase_groups()

    def rebuild_flat_phrase_groups(self) -> None:
        result: list[PhraseGroup] = []
        for section in self.sections:
            result.extend(section.phrase_groups)
        self.flat_phrase_groups = result

    @property
    def phrase_groups(self) -> list[PhraseGroup]:
        return self.flat_phrase_groups

    def section_start_indices(self) -> list[int]:
        result: list[int] = []
        index = 0
        for section in self.sections:
            result.append(index)
            index += len(section.phrase_groups)
        return result

    def section_ranges(self) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        index = 0
        for section in self.sections:
            end = index + len(section.phrase_groups)
            result.append((index, end))
            index = end
        return result

@dataclass
class ModelWarmUpResult:
    did_interrupt: bool = False
    error: str = ""

    @property
    def should_stop(self) -> bool:
        return self.did_interrupt or bool(self.error)

# ---

class ExportType(tuple[str, str, str], Enum):

    AAC = ("m4a", "AAC/M4B", ".m4b") # default
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

@dataclass
class Hint:

    # TODO: move Hint to app_types.py and rename rest as "hint_util.py" non-class module

    key: str
    heading: str
    text: str

    @staticmethod
    def make_using(source: Hint, value1: str, value2: str="") -> Hint:
        """ Does a string replace on the source Hint text using '%1' and optionally '%2' """
        text = source.text
        text = text.replace("%1", value1)
        text = text.replace("%2", value2)
        new_hint = replace(source, text=text)
        return new_hint

# ---

class RealTimeMenuState:
    """ Values related to the real-time playback feature """
    custom_phrase_groups: list[PhraseGroup] = [] # ie, PhraseGroups
    custom_text_line_range: tuple[int, int] | None = None
    project_text_line_range: tuple[int, int] | None = None

# ---

class SoundSegment(NamedTuple):
    """
    Contains the extracted data from the tags from a sound segment filename
    """

    # The pre-existing file_name from which the SoundSegment fields were extracted
    file_name: str

    # Zero-based index
    idx: int 
    # Hash of the source text used to generate the sound file
    hash: str
    num_errors: int
    model: str
    voice: str

class DeviceType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
