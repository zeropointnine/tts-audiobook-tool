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

    def update_phrase_group_text(
        self,
        item_id: int,
        new_text: str,
    ) -> TextEditMutationResult:
        """Edit the text of a phrase group, preserving structure where possible.

        Preserves:
        - voice_index of the PhraseGroup
        - Line breaks the user typed (e.g. Shift+Enter), encoded the same way
        automatic segmentation encodes them: as trailing "\\n" characters on
        a Phrase's text, not as separate empty-text Phrases.
        - The reason each resulting phrase would have received had it come
        from automatic segmentation, derived from its trailing line-break
        count via the same convention phrase_segmenter.py uses for the last
        phrase of a sentence (0 -> SENTENCE, 1-2 -> PARAGRAPH, 3+ ->
        SPACE_BREAK). Reason.PHRASE never applies here, since manual edits
        don't go through the delimiter-based sub-splitting (commas,
        semicolons, parentheses, etc.) that produces it - every line here is
        effectively the sole/last phrase of its own sentence.

        Does NOT preserve:
        - The exact original reason of a phrase whose text didn't change
        (e.g. SECTION_BREAK from an EPUB import): a free-form text edit
        can't reliably tell which lines are untouched, so reason is always
        recomputed from the edited text's own trailing line breaks.
        - The number of phrases (one per non-blank line): a line exceeding
        the project's max_words_per_segment is split into multiple phrases,
        matching how automatic segmentation enforces the same limit
        elsewhere in the app.
        """
        from tts_audiobook_tool.app_types.phrase import Phrase, Reason
        from tts_audiobook_tool.app_support import app_text

        for section in self.sections:
            for staged_phrase_group in section.phrase_groups:
                if staged_phrase_group.item_id != item_id:
                    continue

                # Preserve the original PhraseGroup voice_index
                voice_index = staged_phrase_group.phrase_group.voice_index

                # Split the edited text into lines, keeping line terminators
                # attached instead of discarding them. keepends=False would
                # silently drop every "\n" the user typed (e.g. a single
                # trailing Shift+Enter would vanish entirely and go
                # undetected by has_changes), and PhraseGroup.text later
                # concatenates phrase texts with no separator, so a dropped
                # "\n" also means separate lines get jammed together with
                # nothing between them once the phrase group is reloaded.
                raw_lines = new_text.splitlines(keepends=True)

                # Fold a blank line (pure "\n" / "\r\n") onto the end of the
                # previous line's text instead of turning it into its own
                # empty-text Phrase. This mirrors how automatic segmentation
                # encodes a paragraph/section break as trailing "\n"
                # characters on one Phrase's text (see the "p"/"x"/"xx"
                # reason codes) rather than as a separate empty Phrase - so
                # e.g. two trailing Shift+Enters produce one Phrase ending in
                # "\n\n", not a second Phrase with text "".
                merged_lines: list[str] = []
                for raw_line in raw_lines:
                    is_blank_line = raw_line.strip("\r\n") == ""
                    if is_blank_line and merged_lines:
                        merged_lines[-1] += raw_line
                    else:
                        merged_lines.append(raw_line)

                # Build new Phrases. massage_post_normalize() collapses all
                # \s+ (which includes "\n"), so it is applied only to each
                # line's non-whitespace content and the trailing whitespace
                # (line terminator) is reattached afterward, unmodified, to
                # avoid stripping the break back out.
                #
                # reason is derived the same way phrase_segmenter.py derives
                # it for the last phrase of a sentence: by the number of
                # trailing line breaks on the *original* (pre-normalize)
                # line, since normalization would erase that count.
                built_phrases: list[Phrase] = []
                for line in merged_lines:
                    content = line.rstrip()
                    trailing_whitespace = line[len(content):]

                    num_lf = app_text.num_trailing_line_breaks(line)
                    match num_lf:
                        case 0:
                            reason = Reason.SENTENCE
                        case 1:
                            reason = Reason.PARAGRAPH
                        case 2:
                            reason = Reason.PARAGRAPH
                        case _:  # >= 3
                            reason = Reason.SPACE_BREAK

                    built_phrases.append(
                        Phrase(
                            app_text.massage_post_normalize(content) + trailing_whitespace,
                            reason,
                        )
                    )

                # Guard against an empty result: always keep at least one
                # (possibly empty) phrase so the phrase group stays valid.
                # There is no trailing-line-break information to classify
                # here, so this synthetic phrase falls back to UNDEFINED.
                if not built_phrases:
                    built_phrases.append(Phrase("", Reason.UNDEFINED))

                # Enforce the project's configured segment length. Phrases
                # within the limit (including the empty-text guard above) pass
                # through unchanged; oversized ones are split using the same
                # word-count logic the app already uses for automatic
                # segmentation, so manually edited text stays consistent with
                # how the rest of the app defines "one segment". A falsy
                # max_words (e.g. None or 0) disables the limit entirely.
                max_words = self.segmentation_settings.max_words_per_segment
                new_phrases: list[Phrase] = []
                for phrase in built_phrases:
                    if max_words and phrase.num_words > max_words:
                        from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter

                        new_phrases.extend(
                            PhraseSegmenter.long_phrase_to_phrases(phrase, max_words)
                        )
                    else:
                        new_phrases.append(phrase)

                # Replace all phrases with the newly built ones
                staged_phrase_group.phrase_group = PhraseGroup(
                    phrases=new_phrases,
                    voice_index=voice_index,
                )

                self.record_affected_index(staged_phrase_group.original_index)
                return TextEditMutationResult(
                    changed=True,
                    focus_item_id=item_id,
                    earliest_affected_original_index=staged_phrase_group.original_index,
                )

        return TextEditMutationResult()