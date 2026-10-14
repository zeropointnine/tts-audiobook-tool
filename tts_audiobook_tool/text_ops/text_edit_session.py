from copy import deepcopy
from dataclasses import dataclass
import json
import re

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.book_serialization import book_to_json_dict
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason


_PRESERVED_END_REASONS = {Reason.PHRASE_QUOTE_END, Reason.SECTION_BREAK}


def _select_edited_end_reason(
    original_reason: Reason,
    trailing_whitespace: str,
    segmented_reason: Reason,
) -> Reason:
    """Select the edited group's final reason without punctuation heuristics.

    Semantic reasons which cannot be recovered from isolated spot-edit text are
    preserved. An original trailing line break remains authoritative for the
    group's external boundary. Otherwise, use the reason produced by the
    canonical ``PhraseSegmenter`` so language-specific sentence handling stays
    centralized there.
    """
    if original_reason in _PRESERVED_END_REASONS:
        return original_reason

    num_lf = trailing_whitespace.count("\n")
    if num_lf >= 3:
        return Reason.SPACE_BREAK
    if num_lf >= 1:
        return Reason.PARAGRAPH
    return segmented_reason


@dataclass(frozen=True)
class PhraseGroupSplitPoint:
    """A split between two existing phrases, expressed as a list boundary."""

    phrase_boundary: int


@dataclass
class StagedPhraseGroup:
    """A detached phrase group with stable editor identity and source lineage."""

    item_id: int
    phrase_group: PhraseGroup
    original_index: int


@dataclass
class StagedBookSection:
    """A detached book section which owns staged phrase groups."""

    item_id: int
    title: str
    phrase_groups: list[StagedPhraseGroup]


@dataclass(frozen=True)
class TextEditMutationResult:
    """Describes a staged structural mutation for UI projection and commit."""

    changed: bool = False
    focus_item_id: int | None = None
    earliest_affected_original_index: int | None = None
    deleted_count: int = 0


class TextEditSession:
    """Pure, detached staging model for structural edits to a Book."""

    def __init__(self, book: Book) -> None:
        staged_book = deepcopy(book)
        self.book_title = staged_book.title
        self.text_source_kind = staged_book.text_source_kind
        self.audio_source_kind = staged_book.audio_source_kind
        self.segmentation_settings = staged_book.segmentation_settings
        self.original_snapshot = self.make_snapshot(book)
        self.next_item_id = 1
        self.earliest_affected_original_index: int | None = None
        self.edited_original_indices: set[int] = set()
        self.did_structural_change = False

        next_original_index = 0
        self.sections: list[StagedBookSection] = []
        for section in staged_book.sections:
            staged_phrase_groups: list[StagedPhraseGroup] = []
            for phrase_group in section.phrase_groups:
                staged_phrase_groups.append(
                    StagedPhraseGroup(
                        item_id=self.take_item_id(),
                        phrase_group=phrase_group,
                        original_index=next_original_index,
                    )
                )
                next_original_index += 1
            self.sections.append(
                StagedBookSection(
                    item_id=self.take_item_id(),
                    title=section.title,
                    phrase_groups=staged_phrase_groups,
                )
            )

    def take_item_id(self) -> int:
        item_id = self.next_item_id
        self.next_item_id += 1
        return item_id

    @staticmethod
    def make_snapshot(book: Book) -> str:
        """Return a deterministic value snapshot suitable for stale checks."""
        return json.dumps(book_to_json_dict(book), sort_keys=True, separators=(",", ":"))

    @property
    def phrase_groups(self) -> list[StagedPhraseGroup]:
        return [
            phrase_group
            for section in self.sections
            for phrase_group in section.phrase_groups
        ]

    @property
    def has_changes(self) -> bool:
        return self.make_snapshot(self.to_book()) != self.original_snapshot

    def to_book(self) -> Book:
        """Materialize a detached canonical Book from the staged hierarchy."""
        return Book(
            sections=[
                BookSection(
                    title=section.title,
                    phrase_groups=[
                        staged_phrase_group.phrase_group
                        for staged_phrase_group in section.phrase_groups
                    ],
                )
                for section in self.sections
            ],
            title=self.book_title,
            text_source_kind=self.text_source_kind,
            audio_source_kind=self.audio_source_kind,
            segmentation_settings=self.segmentation_settings,
        )

    def get_phrase_group(self, item_id: int) -> StagedPhraseGroup | None:
        return next(
            (
                phrase_group
                for phrase_group in self.phrase_groups
                if phrase_group.item_id == item_id
            ),
            None,
        )

    def record_affected_index(self, original_index: int) -> None:
        if (
            self.earliest_affected_original_index is None
            or original_index < self.earliest_affected_original_index
        ):
            self.earliest_affected_original_index = original_index

    def delete_phrase_groups(self, item_ids: set[int]) -> TextEditMutationResult:
        """Delete selected phrase groups and prune sections emptied by the deletion."""
        phrase_groups = self.phrase_groups
        extant_ids = {item.item_id for item in phrase_groups}
        deletion_ids = item_ids & extant_ids
        if not deletion_ids:
            return TextEditMutationResult()

        self.did_structural_change = True
        first_deleted_position = min(
            index
            for index, phrase_group in enumerate(phrase_groups)
            if phrase_group.item_id in deletion_ids
        )
        earliest_index = min(
            phrase_group.original_index
            for phrase_group in phrase_groups
            if phrase_group.item_id in deletion_ids
        )

        sections_to_remove: set[int] = set()
        for section in self.sections:
            previous_count = len(section.phrase_groups)
            section.phrase_groups = [
                phrase_group
                for phrase_group in section.phrase_groups
                if phrase_group.item_id not in deletion_ids
            ]
            if previous_count > 0 and not section.phrase_groups:
                sections_to_remove.add(section.item_id)
        self.sections = [
            section
            for section in self.sections
            if section.item_id not in sections_to_remove
        ]

        remaining_phrase_groups = self.phrase_groups
        focus_item_id = (
            remaining_phrase_groups[
                min(first_deleted_position, len(remaining_phrase_groups) - 1)
            ].item_id
            if remaining_phrase_groups
            else None
        )
        self.record_affected_index(earliest_index)
        return TextEditMutationResult(
            changed=True,
            focus_item_id=focus_item_id,
            earliest_affected_original_index=earliest_index,
            deleted_count=len(deletion_ids),
        )

    def delete_section(self, item_id: int) -> TextEditMutationResult:
        """Delete one section and its phrase groups."""
        section_index = next(
            (
                index
                for index, section in enumerate(self.sections)
                if section.item_id == item_id
            ),
            None,
        )
        if section_index is None:
            return TextEditMutationResult()

        section = self.sections[section_index]
        if section.phrase_groups:
            return self.delete_phrase_groups(
                {phrase_group.item_id for phrase_group in section.phrase_groups}
            )

        self.did_structural_change = True
        del self.sections[section_index]
        remaining_phrase_groups = self.phrase_groups
        focus_item_id = (
            remaining_phrase_groups[
                min(
                    sum(
                        len(current_section.phrase_groups)
                        for current_section in self.sections[:section_index]
                    ),
                    len(remaining_phrase_groups) - 1,
                )
            ].item_id
            if remaining_phrase_groups
            else None
        )
        return TextEditMutationResult(
            changed=True,
            focus_item_id=focus_item_id,
        )

    def split_phrase_group(
        self,
        item_id: int,
        split_point: PhraseGroupSplitPoint,
    ) -> TextEditMutationResult:
        """Split one group at an existing Phrase boundary."""
        for section in self.sections:
            for index, staged_phrase_group in enumerate(section.phrase_groups):
                if staged_phrase_group.item_id != item_id:
                    continue

                phrase_group = staged_phrase_group.phrase_group
                boundary = split_point.phrase_boundary
                if not 0 < boundary < len(phrase_group.phrases):
                    return TextEditMutationResult(focus_item_id=item_id)

                self.did_structural_change = True
                left_group = PhraseGroup(
                    phrases=phrase_group.phrases[:boundary],
                    voice_index=phrase_group.voice_index,
                )
                right_group = PhraseGroup(
                    phrases=phrase_group.phrases[boundary:],
                    voice_index=phrase_group.voice_index,
                )
                right_item = StagedPhraseGroup(
                    item_id=self.take_item_id(),
                    phrase_group=right_group,
                    original_index=staged_phrase_group.original_index,
                )
                staged_phrase_group.phrase_group = left_group
                section.phrase_groups[index : index + 1] = [
                    staged_phrase_group,
                    right_item,
                ]
                self.record_affected_index(staged_phrase_group.original_index)
                return TextEditMutationResult(
                    changed=True,
                    focus_item_id=right_item.item_id,
                    earliest_affected_original_index=staged_phrase_group.original_index,
                )

        return TextEditMutationResult()

    def update_phrase_group_text(
        self,
        item_id: int,
        new_text: str,
        max_words: int,
        pysbd_lang: str,
    ) -> TextEditMutationResult:
        """Re-create one phrase group's text using the app's canonical segmentation.

        The edited text is passed through ``PhraseSegmenter.text_to_phrases``,
        the same phrase-building layer automatic import uses, and the resulting
        phrases are wrapped in a single new ``PhraseGroup``. This re-creates
        only the associated phrase group; its structure (voice_index) and the
        original group's trailing whitespace are preserved.

        Preserves:
        - voice_index of the PhraseGroup.
        - ``SECTION_BREAK`` and ``PHRASE_QUOTE_END`` on the original group's
          last phrase, because these semantic reasons cannot be recovered from
          the isolated spot-edit text.
        - The original group's trailing whitespace: trailing whitespace is
          stripped from the edited text and the original trailing whitespace is
          re-applied to the re-created result (the last phrase).

        Except for those preserved semantic reasons and trailing paragraph or
        space breaks, the last phrase keeps the reason assigned by the canonical
        ``PhraseSegmenter``. Edited whitespace before a line break is normalized
        using the same rule as automatic import.
        """
        from tts_audiobook_tool.app_support import app_text
        from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter

        for section in self.sections:
            for staged_phrase_group in section.phrase_groups:
                if staged_phrase_group.item_id != item_id:
                    continue

                original_group = staged_phrase_group.phrase_group
                voice_index = original_group.voice_index
                original_end_reason = original_group.last_reason

                original_norm = app_text.normalize_line_terminators(
                    original_group.text
                )
                trailing_whitespace = original_norm[len(original_norm.rstrip()):]

                edited_normalized = app_text.normalize_line_terminators(new_text)
                edited_normalized = re.sub(r"[ \t]+\n", "\n", edited_normalized)
                edited_stripped = edited_normalized.rstrip()

                phrases = PhraseSegmenter.text_to_phrases(
                    edited_stripped,
                    max_words=max_words,
                    pysbd_lang=pysbd_lang,
                )

                # Guard against an empty result so the phrase group stays legal.
                if not phrases:
                    phrases = [Phrase("", Reason.UNDEFINED)]

                # Re-apply the original trailing whitespace to the new result.
                # Preserve semantic boundary metadata which isolated spot-edit
                # text cannot recover; otherwise retain the canonical
                # segmenter's reason unless the preserved suffix is a paragraph
                # or space break.
                phrases[-1].text += trailing_whitespace
                phrases[-1].reason = _select_edited_end_reason(
                    original_end_reason,
                    trailing_whitespace,
                    phrases[-1].reason,
                )

                new_group = PhraseGroup(
                    phrases=phrases,
                    voice_index=voice_index,
                )

                # A change which only affected trailing whitespace re-creates
                # the original text; nothing meaningful is staged.
                if new_group.text == original_norm:
                    return TextEditMutationResult(
                        changed=False,
                        focus_item_id=item_id,
                    )

                staged_phrase_group.phrase_group = new_group
                self.edited_original_indices.add(staged_phrase_group.original_index)
                self.record_affected_index(staged_phrase_group.original_index)
                return TextEditMutationResult(
                    changed=True,
                    focus_item_id=item_id,
                    earliest_affected_original_index=None,
                )

        return TextEditMutationResult()