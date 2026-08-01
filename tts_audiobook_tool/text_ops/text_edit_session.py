from copy import deepcopy
from dataclasses import dataclass
import json

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.book_serialization import book_to_json_dict
from tts_audiobook_tool.app_types.phrase import PhraseGroup


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
