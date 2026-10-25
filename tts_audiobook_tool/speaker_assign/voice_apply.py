"""
Step 3's apply: sets the voice of each attributed dialog passage whose
speaker is a roster character with a chosen voice (see VoiceMapFile).
Everything else is left as is: narration, characters without a chosen voice,
OTHER, UNKNOWN and NOT_SPEECH passages, and passages whose saved results no
longer match the project text.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Set
from dataclasses import dataclass

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.speaker_label_util import SpeakerLabelUtil
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile


@dataclass(frozen=True)
class VoiceApplyPlan:
    """
    line_voices: the new voice index by book-wide line, for every line of
        every passage being applied (including lines that already have it)
    applied: passages being applied, by character id
    unchanged: passages left as is, by counting id (eg OTHER:<name> as OTHER)
    skipped: passages in saved results whose text has since changed
    """
    line_voices: dict[int, int]
    applied: Counter[str]
    unchanged: Counter[str]
    skipped: int

    @staticmethod
    def make(
            book: Book, roster_ids: Set[str], attributions: AttributionsFile, voice_map: VoiceMapFile,
    ) -> VoiceApplyPlan:
        line_voices: dict[int, int] = {}
        applied: Counter[str] = Counter()
        unchanged: Counter[str] = Counter()
        for utterance in attributions.by_first_line(book).values():
            voice_index = voice_map.get(utterance.speaker) if utterance.speaker in roster_ids else None
            if voice_index is None:
                unchanged[SpeakerLabelUtil.counting_id(utterance.speaker)] += 1
                continue
            applied[utterance.speaker] += 1
            # Valid ranges lie within the book, so their lines do too
            for line in utterance.lines:
                line_voices[line] = voice_index
        skipped = sum(len(r.utterances) for r in attributions.ranges if not r.is_valid_for(book))
        return VoiceApplyPlan(line_voices, applied, unchanged, skipped)

    def changed_lines(self, book: Book) -> list[int]:
        """The lines whose voice would change, in book order."""
        groups = book.flat_phrase_groups
        return sorted(line for line, voice in self.line_voices.items() if groups[line].voice_index != voice)

    def apply(self, book: Book) -> dict[int, int]:
        """Sets the lines' voices in place; returns the previous voice index of
        each changed line (for rolling back if saving fails)."""
        groups = book.flat_phrase_groups
        previous = {line: groups[line].voice_index for line in self.changed_lines(book)}
        for line in previous:
            groups[line].voice_index = self.line_voices[line]
        return previous

    @staticmethod
    def restore(book: Book, previous: dict[int, int]) -> None:
        groups = book.flat_phrase_groups
        for line, voice_index in previous.items():
            groups[line].voice_index = voice_index
