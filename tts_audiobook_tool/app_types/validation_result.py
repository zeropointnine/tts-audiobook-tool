from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tts_audiobook_tool.app_types import Sound, Word
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound.silence_util import SilenceGapTrim


@dataclass
class ValidationResult(ABC):
    """ 
    Base class for a validation result.
    A validation result contains a Sound, which may or may not be already-transformed.
    """
    
    sound: Sound
    intra_sample_silence_trims: list[SilenceGapTrim] = field(default_factory=list, kw_only=True)
    generated_start_trim_time: float | None = field(default=None, kw_only=True)
    generated_end_trim_time: float | None = field(default=None, kw_only=True)
    generated_trim_original_duration: float | None = field(default=None, kw_only=True)
    trailing_token_noise_trim_time: float | None = field(default=None, kw_only=True)

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

    def get_generated_trim_ui_message(self) -> str:
        start_time = self.generated_start_trim_time
        end_time = self.generated_end_trim_time
        original_duration = self.generated_trim_original_duration
        if start_time is None and end_time is None:
            return ""

        s = f"{COL_DEFAULT}Trimmed excess audio: "
        if start_time is not None and end_time is not None and original_duration is not None:
            s += f"{COL_DIM}0s to {start_time:.2f}s, {end_time:.2f}s to end {original_duration:.2f}s"
        elif start_time is not None:
            s += f"{COL_DIM}0s to {start_time:.2f}s"
        elif end_time is not None and original_duration is not None:
            s += f"{COL_DIM}{end_time:.2f}s to end {original_duration:.2f}s"
        else:
            return ""
        return s

    def get_trailing_token_noise_trim_ui_message(self) -> str:
        trim_time = self.trailing_token_noise_trim_time
        if not trim_time:
            return ""
        return f"{COL_DEFAULT}Trimmed trailing token noise: {COL_DIM}{trim_time:.2f}s"

    def get_ui_message_with_post_processing(self) -> str:
        message = self.get_ui_message()
        extras = [
            self.get_generated_trim_ui_message(),
            self.get_trailing_token_noise_trim_ui_message(),
            self.get_intra_sample_silence_ui_message(),
        ]
        extras = [item for item in extras if item]
        if extras:
            message += "\n" + "\n".join(extras)
        return message

# ----------
# Subclasses

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
    num_words: int
    threshold: int

    @property
    def num_errors(self) -> int:
        return len(self.errors)
    
    @property
    def is_fail(self) -> bool:
        return self.num_errors > self.threshold

    def get_ui_message(self) -> str:
        base_message = f"{COL_ERROR}Word error fail" if self.is_fail else "Passed"
        return f"{base_message} {COL_DIM}(word_errors={COL_DEFAULT}{self.num_errors}{COL_DIM}, words={COL_DEFAULT}{self.num_words}{COL_DIM}, threshold={COL_DEFAULT}{self.threshold}{COL_DIM})"

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
