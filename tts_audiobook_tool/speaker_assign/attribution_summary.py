from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.speaker_label_util import SpeakerLabelUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


# An OTHER:<name> speaker this frequent is worth adding to the roster
RECURRING_OTHER_MIN_COUNT = 3


@dataclass(frozen=True)
class AttributionSummary:
    """
    Saved, still-valid attribution results over a book's lines [start, end):
    speaker counts by counting id (OTHER:<name> counts as OTHER), dialog
    utterances without results, utterances the LLM flagged uncertain, and
    recurring OTHER:<name> speakers (roster candidates).
    """
    counts: Counter[str]
    unattributed: int
    uncertain: int
    recurring_others: Counter[str]

    @property
    def attributed(self) -> int:
        return sum(self.counts.values())

    @staticmethod
    def make(attributions: AttributionsFile, book: Book, start: int = 0, end: int | None = None) -> AttributionSummary:
        end = len(book.flat_phrase_groups) if end is None else end
        current = {line: u for line, u in attributions.by_first_line(book).items() if start <= line < end}
        counts = Counter(SpeakerLabelUtil.counting_id(u.speaker) for u in current.values())
        unattributed = sum(1 for u in UtteranceUtil.find_utterances(book, start, end) if u.first_line not in current)
        others = Counter(name for u in current.values() if (name := SpeakerLabelUtil.other_name(u.speaker)))
        return AttributionSummary(
            counts=counts,
            unattributed=unattributed,
            uncertain=sum(1 for u in current.values() if u.uncertain),
            recurring_others=Counter(
                {name: count for name, count in others.items() if count >= RECURRING_OTHER_MIN_COUNT}),
        )

    @staticmethod
    def counts_text(counts: Counter[str]) -> str:
        """eg "BARON 20, NARRATOR 15, OTHER 3" (most common first)."""
        return ", ".join(f"{label} {count}" for label, count in counts.most_common())
