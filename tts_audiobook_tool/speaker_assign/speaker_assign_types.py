"""
Shared speaker assignment vocabulary.

Line conventions: a "line" is a phrase group, numbered book-wide from 0
(`Book.flat_phrase_groups` order). Line ranges are half-open, (start, end),
like `Book.section_ranges()`.
"""
from __future__ import annotations

from dataclasses import dataclass


# Attribution answers other than a roster character id. They're reserved, so no
# roster character may use them as its id.
NOT_SPEECH = "NOT_SPEECH"  # the quoted text isn't spoken dialog (a title, sign, written text, ...)
OTHER = "OTHER"            # a speaker who isn't in the roster; answers may be "OTHER:<short name>"
UNKNOWN = "UNKNOWN"        # the speaker can't be determined

RESERVED_SPEAKER_IDS = frozenset({NOT_SPEECH, OTHER, UNKNOWN})


@dataclass(frozen=True)
class Utterance:
    """
    One quotation's dialog lines: a dialog line (is_dialog) that starts
    with an opening quote mark, plus any directly following dialog lines that
    don't (continuation pieces of the same quotation). See UtteranceUtil.
    """
    lines: tuple[int, ...]  # book-wide line indices, ascending and contiguous
    text: str               # the lines' text, concatenated

    @property
    def first_line(self) -> int:
        return self.lines[0]
