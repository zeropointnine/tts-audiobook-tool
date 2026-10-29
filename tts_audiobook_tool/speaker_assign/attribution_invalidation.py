"""Preview and persist text-dependent dialog assignment cleanup."""
from __future__ import annotations

from dataclasses import dataclass

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil


@dataclass(frozen=True)
class AttributionInvalidation:
    """A cleanup planned against the incoming text, without changing disk.

    Only commit after the new text has been saved successfully. Roster and
    voice mappings are intentionally outside this text-dependent cache.
    """
    project_dir: str
    kept: AttributionsFile
    removed_count: int
    # Distinct utterance lines in the removed ranges, for the save confirmation
    removed_line_count: int = 0

    @staticmethod
    def plan(project_dir: str, book: Book) -> AttributionInvalidation | str:
        saved = SpeakerAssignFileUtil.load_attributions_file(project_dir)
        if isinstance(saved, str):
            return saved
        kept = AttributionsFile(tuple(saved.valid_ranges(book)))
        kept_keys = {r.key for r in kept.ranges}
        removed = [r for r in saved.ranges if r.key not in kept_keys]
        removed_lines = {line for r in removed for u in r.utterances for line in u.lines}
        return AttributionInvalidation(project_dir, kept, len(removed), len(removed_lines))

    def commit(self) -> str:
        if not self.removed_count:
            return ""
        return SpeakerAssignFileUtil.save_attributions_file(self.project_dir, self.kept)
