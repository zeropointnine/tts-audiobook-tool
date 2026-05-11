from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tts_audiobook_tool.app_types import Sound, Word
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.silence_util import SilenceGapTrim


@dataclass
class ValidationResult(ABC):
    """ 
    Base class for a validation result.
    A validation result contains a Sound, which may or may not be transformed.
    """
    
    sound: Sound
    intra_sample_silence_trims: list[SilenceGapTrim] = field(default_factory=list, kw_only=True)

    @property
    @abstractmethod
    def is_fail(self) -> bool:
        ...
    
    @abstractmethod
    def get_ui_message(self) -> str:
        """ User-facing description, including color code formatting """
        ...

    def get_intra_sample_silence_ui_message(self) -> str:
        if not self.intra_sample_silence_trims:
            return ""

        parts = [
            f"{item.original_duration:.1f}->{item.new_duration:.1f}"
            for item in self.intra_sample_silence_trims
        ]
        total_removed = sum(item.removed_duration for item in self.intra_sample_silence_trims)
        parts.append(f"{total_removed:.2}s total")
        return f"{COL_DEFAULT}Trimmed excess silence gaps: {COL_DIM}{', '.join(parts)}"

    def get_ui_message_with_post_processing(self) -> str:
        message = self.get_ui_message()
        extra = self.get_intra_sample_silence_ui_message()
        if extra:
            message += "\n" + extra
        return message

@dataclass
class TranscriptResult(ValidationResult, ABC):
    """ 
    Base class for a ValidationResult that has transcript data 
    (ie, anything other than SkippedResult)
    """
    transcript_words: list[Word]

@dataclass
class WordErrorResult(TranscriptResult):
    """ A ValidationResult that has transcript data and has calculated word error list """

    errors: list[str]
    threshold: int

    @property
    def num_errors(self) -> int:
        return len(self.errors)
    
    @property
    def is_fail(self) -> bool:
        return self.num_errors > self.threshold

    def get_ui_message(self) -> str:
        base_message = f"{COL_ERROR}Word error fail" if self.is_fail else "Passed"
        return f"{base_message} {COL_DIM}(word_errors={COL_DEFAULT}{self.num_errors}{COL_DIM}, threshold={self.threshold})"

@dataclass
class TrimmedResult(TranscriptResult):
    """ The sound has been trimmed at either/both ends """

    # Values stored for reporting purposes
    start_time: float | None 
    end_time: float | None
    original_duration: float
    
    def __post_init__(self):
        if self.start_time is None and self.end_time is None:
            raise ValueError("start or end must be a float")
        if self.end_time == 0.0:
            raise ValueError("end time must be None or must be greater than zero")

    @property
    def is_fail(self) -> bool:
        return False

    def get_ui_message(self) -> str:
        s = f"{COL_DEFAULT}Trimmed excess audio: "
        if self.start_time and self.end_time:
            s += f"{COL_DIM}0s to {self.start_time:.2f}s, {self.end_time:.2f}s to end {self.original_duration:.2f}s"
        elif self.start_time:
            s += f"{COL_DIM}0s to {self.start_time:.2f}s"
        else:
            s += f"{COL_DIM}{self.end_time:.2f}s to end {self.original_duration:.2f}s"
        return s

@dataclass
class MusicFailResult(TranscriptResult):
    """
    """
    @property
    def is_fail(self) -> bool:
        return True

    def get_ui_message(self) -> str:
        return f"{COL_ERROR}Music detected"

@dataclass
class SkippedResult(ValidationResult):

    message: str

    @property
    def is_fail(self) -> bool:
        return False

    def get_ui_message(self) -> str:
        return f"Skipped validation: {COL_DIM}{self.message}"
