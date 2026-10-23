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

    @staticmethod
    def plan(project_dir: str, book: Book) -> AttributionInvalidation | str:
        saved = SpeakerAssignFileUtil.load_attributions_file(project_dir)
        if isinstance(saved, str):
            return saved
        kept = AttributionsFile(tuple(saved.valid_ranges(book)))
        return AttributionInvalidation(project_dir, kept, len(saved.ranges) - len(kept.ranges))

    @property
    def warning_text(self) -> str:
        if not self.removed_count:
            return ""
        chunks = "chunk" if self.removed_count == 1 else "chunks"
        return (
            f"This change will clear dialog assignments for {self.removed_count} affected {chunks}. "
            'Run "Assign dialog to characters" again for those passages. '
            "Your character roster and voice mappings will be kept."
        )

    def commit(self) -> str:
        if not self.removed_count:
            return ""
        return SpeakerAssignFileUtil.save_attributions_file(self.project_dir, self.kept)
