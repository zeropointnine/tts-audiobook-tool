from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types import Sound, Word
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings, ValidationInvalidReason
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound.silence_util import SilenceGapTrim
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer


POSSIBLE_TRUNCATION_UI_MESSAGE = f"Audio ends abruptly, last word may be truncated {COL_DIM}(+1 word error)"


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
    voice_tag: str = field(default="", kw_only=True)
    findings: ValidationFindings = field(default_factory=ValidationFindings, kw_only=True)

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

    def get_possible_truncation_ui_message(self) -> str:
        if not getattr(self, "possible_truncation", False):
            return ""
        return POSSIBLE_TRUNCATION_UI_MESSAGE

    def get_ui_message_with_extras(self) -> str:
        message = self.get_ui_message()
        extras = [
            self.get_possible_truncation_ui_message(),
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

    @property
    def possible_truncation(self) -> bool:
        return self.findings.possible_truncation

@dataclass
class WordErrorResult(TranscriptResult):
    """A runtime validation outcome with a threshold for transcript findings."""

    num_words: int
    threshold: int

    @property
    def num_errors(self) -> int:
        return self.findings.effective_word_error_count
    
    @property
    def is_fail(self) -> bool:
        return self.findings.is_failed(self.threshold)

    def get_ui_message(self) -> str:
        return self.findings.make_status_message(self.num_words, self.threshold)

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
    def __post_init__(self):
        if self.findings.invalid_reason is None:
            self.findings.invalid_reason = ValidationInvalidReason.MUSIC_DETECTED

    @property
    def is_fail(self) -> bool:
        return self.findings.is_hard_invalid

    def get_ui_message(self) -> str:
        return f"{COL_ERROR}Music detected"

@dataclass
class ExcessiveDurationResult(TranscriptResult):
    """
    Represents a excessively long sound generation for the number of words it contains.
    (Presumably contains many spurious words or excess noise, etc)
    """

    duration: float

    def __post_init__(self):
        if self.findings.invalid_reason is None:
            self.findings.invalid_reason = ValidationInvalidReason.EXCESSIVE_DURATION

    @staticmethod
    def is_excessively_long(source_text: str, language_code: str, sound_duration: float) -> bool:
        
        normalized_source = TextNormalizer.normalize_source(source_text, language_code)
        words = app_text.get_words(normalized_source, vocalizable_only=True)

        # Outlier: Skip if any word has 3+ digits (prevent false positives)
        if any(sum(char.isdigit() for char in word) >= 3 for word in words):
            return False

        threshold = 1.5 + len(words) * 0.75 
        return sound_duration > threshold

    @property
    def is_fail(self) -> bool:
        return self.findings.is_hard_invalid

    def get_ui_message(self) -> str:
        return f"{COL_ERROR}Sound duration is excessively long (duration: {self.duration:.2f}s, num words: {len(self.transcript_words)})"

@dataclass
class SkippedResult(ValidationResult):

    message: str

    @property
    def is_fail(self) -> bool:
        return False

    def get_ui_message(self) -> str:
        return f"Skipped {COL_DIM}({self.message})"
