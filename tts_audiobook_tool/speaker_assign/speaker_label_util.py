from __future__ import annotations

from collections.abc import Set
from enum import Enum

from tts_audiobook_tool.speaker_assign.speaker_assign_types import NOT_SPEECH, OTHER, UNKNOWN


class SpeakerKind(Enum):
    CHARACTER = "character"    # a roster character id
    OTHER = "other"            # OTHER or OTHER:<name>
    UNKNOWN = "unknown"
    NOT_SPEECH = "not_speech"


class SpeakerLabelUtil:
    """
    Speaker labels are the stored attribution answers: a roster character id
    ("BARON"), "OTHER" / "OTHER:<short name>", "UNKNOWN" or "NOT_SPEECH".
    """

    @staticmethod
    def normalize(raw: object, roster_ids: Set[str]) -> str:
        """An LLM answer as a speaker label. Roster ids and reserved answers are
        matched case-insensitively; anything else is kept, for review, as
        OTHER:<answer>."""
        text = str(raw).strip()
        upper = text.upper()
        if upper in roster_ids:
            return upper
        if upper.replace(" ", "_") == NOT_SPEECH:
            return NOT_SPEECH
        if upper == UNKNOWN:
            return UNKNOWN
        if upper.startswith(OTHER):
            return text
        return f"{OTHER}:{text}"

    @staticmethod
    def kind(label: str, roster_ids: Set[str]) -> SpeakerKind:
        if label in roster_ids:
            return SpeakerKind.CHARACTER
        if label == NOT_SPEECH:
            return SpeakerKind.NOT_SPEECH
        if label.upper().startswith(OTHER):
            return SpeakerKind.OTHER
        return SpeakerKind.UNKNOWN

    @staticmethod
    def counting_id(label: str) -> str:
        """The label for counting speakers: OTHER:<name> counts as OTHER."""
        return OTHER if label.upper().startswith(OTHER) else label

    @staticmethod
    def other_name(label: str) -> str:
        """The <name> of an OTHER:<name> label, normalized ("" if none)."""
        if not label.upper().startswith(OTHER) or ":" not in label:
            return ""
        return label.split(":", 1)[1].strip().lower()
