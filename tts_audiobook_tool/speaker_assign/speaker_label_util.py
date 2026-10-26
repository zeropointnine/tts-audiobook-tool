from __future__ import annotations

from collections.abc import Mapping, Set
from enum import Enum

from tts_audiobook_tool.speaker_assign.speaker_assign_types import (
    NOT_SPEECH,
    OTHER,
    RESERVED_SPEAKER_IDS,
    RESERVED_SPEAKER_ORDER,
    UNKNOWN,
)


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
    def sorted_counts(counts: Mapping[str, int]) -> list[tuple[str, int]]:
        """(id, count) pairs for display: non-reserved ids by count (most
        first, then by id), followed by the reserved ids in
        RESERVED_SPEAKER_ORDER. Zero counts are omitted."""
        regular = sorted(
            ((label, count) for label, count in counts.items() if count and label not in RESERVED_SPEAKER_IDS),
            key=lambda item: (-item[1], item[0]))
        reserved = [(label, counts[label]) for label in RESERVED_SPEAKER_ORDER if counts.get(label)]
        return regular + reserved

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
