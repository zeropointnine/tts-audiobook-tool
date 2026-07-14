from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM, COL_ERROR

class ValidationInvalidReason(str, Enum):
    """A validation condition that fails independently of word-error tolerance."""

    MUSIC_DETECTED = "music_detected"
    EXCESSIVE_DURATION = "excessive_sound_duration"


@dataclass
class ValidationFindings:
    """
    Durable observations from validation and the rules derived from them.

    Transcript mismatches, the possible-truncation diagnostic, and hard
    invalidation are deliberately separate.  
    
    The legacy filename score is a compatibility projection, not a validation fact.
    """

    transcript_errors: list[str] = field(default_factory=list)
    possible_truncation: bool = False
    invalid_reason: ValidationInvalidReason | None = None

    LEGACY_INVALID_SCORE = 99

    @property
    def transcript_word_error_count(self) -> int:
        return len(self.transcript_errors)

    @property
    def effective_word_error_count(self) -> int:
        """Count evaluated against the configured word-error threshold."""
        return self.transcript_word_error_count + int(self.possible_truncation)

    @property
    def is_hard_invalid(self) -> bool:
        return self.invalid_reason is not None

    def is_failed(self, threshold: int) -> bool:
        return self.is_hard_invalid or self.effective_word_error_count > threshold

    @property
    def legacy_filename_score(self) -> int:
        """Lossy compatibility score stored in existing segment filenames."""
        if self.is_hard_invalid:
            return self.LEGACY_INVALID_SCORE
        return self.effective_word_error_count

    @classmethod
    def is_legacy_filename_score_failed(cls, score: int, threshold: int) -> bool:
        """Evaluates the compatibility score available from a filename alone."""
        return score == cls.LEGACY_INVALID_SCORE or score > threshold

    def make_status_message(self, num_words: int, threshold: int) -> str:
        """Returns the shared terminal status line for these findings."""
        if self.invalid_reason == ValidationInvalidReason.MUSIC_DETECTED:
            return f"{COL_ERROR}Music detected"
        if self.invalid_reason == ValidationInvalidReason.EXCESSIVE_DURATION:
            return f"{COL_ERROR}Sound duration is excessively long"

        base_message = f"{COL_ERROR}Word error fail" if self.is_failed(threshold) else "Passed"
        return (
            f"{base_message} {COL_DIM}(words={COL_DEFAULT}{num_words}{COL_DIM}, "
            f"word_errors={COL_DEFAULT}{self.effective_word_error_count}{COL_DIM}, "
            f"threshold={COL_DEFAULT}{threshold}{COL_DIM})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript_errors": self.transcript_errors,
            "possible_truncation": self.possible_truncation,
            "invalid_reason": self.invalid_reason.value if self.invalid_reason else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidationFindings | str:
        errors = payload.get("transcript_errors", [])
        if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
            return "Missing or invalid validation transcript_errors"

        invalid_reason_value = payload.get("invalid_reason")
        try:
            invalid_reason = (
                ValidationInvalidReason(invalid_reason_value)
                if invalid_reason_value is not None else None
            )
        except ValueError:
            return f"Unsupported validation invalid_reason: {invalid_reason_value}"

        return cls(
            transcript_errors=errors,
            possible_truncation=bool(payload.get("possible_truncation", False)),
            invalid_reason=invalid_reason,
        )
