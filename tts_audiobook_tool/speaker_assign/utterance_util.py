from __future__ import annotations

from collections.abc import Callable
import hashlib

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.speaker_assign_types import Utterance


OPENING_QUOTES = ('"', "“", "„")


class UtteranceUtil:
    """
    Dialog utterances and line-range text of a Book. Line ranges are
    half-open (start, end) over book-wide line indices.
    """

    @staticmethod
    def find_utterances(book: Book, start: int, end: int) -> list[Utterance]:
        """
        The utterances whose lines lie within [start, end). Dialog lines are
        phrase groups flagged is_dialog (by the app's import-time dialog
        segmentation). A new utterance starts at a dialog line that begins
        with an opening quote mark or follows a non-dialog line; other dialog
        lines continue the previous utterance. Utterances never continue
        across a section boundary.
        """
        groups = book.flat_phrase_groups
        section_starts = set(book.section_start_indices())
        result: list[tuple[list[int], str]] = []
        previous_was_dialog = False
        for line in range(start, end):
            group = groups[line]
            if line in section_starts:
                previous_was_dialog = False
            is_dialog = group.is_dialog
            if is_dialog:
                starts_quote = group.text.lstrip().startswith(OPENING_QUOTES)
                if previous_was_dialog and not starts_quote:
                    result[-1][0].append(line)
                    result[-1] = (result[-1][0], result[-1][1] + group.text)
                else:
                    result.append(([line], group.text))
            previous_was_dialog = is_dialog
        return [Utterance(tuple(lines), text) for lines, text in result]

    @staticmethod
    def annotate_range(
            book: Book,
            start: int,
            end: int,
            open_tag: Callable[[int, Utterance], str | None],
            section_headers: bool,
    ) -> str:
        """
        The text of lines [start, end) with each utterance wrapped in
        open_tag(number, utterance) + "</q>", where number is the utterance's
        1-based position in the range (eg <q>, <q id="3">, <q speaker="BARON">).
        An utterance whose open_tag is None is left untagged. Trailing
        whitespace stays outside the tag. With section_headers, a header line
        is inserted where a new section begins after `start`.
        """
        groups = book.flat_phrase_groups
        section_starts = set(book.section_start_indices())
        numbered = {u.first_line: (number, u)
                    for number, u in enumerate(UtteranceUtil.find_utterances(book, start, end), 1)}
        parts: list[str] = []
        line = start
        while line < end:
            if section_headers and line != start and line in section_starts:
                section_index = UtteranceUtil.section_of_line(book, line)
                parts.append(f"\n===== {UtteranceUtil.section_label(book, section_index)} =====\n")
            entry = numbered.get(line)
            tag = open_tag(*entry) if entry else None
            if entry is None or tag is None:
                parts.append(groups[line].text)
                line += 1
                continue
            utterance = entry[1]
            body = utterance.text.rstrip()
            parts.append(f"{tag}{body}</q>{utterance.text[len(body):]}")
            line = utterance.lines[-1] + 1
        return "".join(parts)

    @staticmethod
    def range_text(book: Book, start: int, end: int) -> str:
        return "".join(group.text for group in book.flat_phrase_groups[start:end])

    @staticmethod
    def range_text_hash(book: Book, start: int, end: int) -> str:
        """A short hash of the range's text, for detecting later text changes."""
        text = UtteranceUtil.range_text(book, start, end)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def section_text_hashes(book: Book) -> tuple[str, ...]:
        return tuple(UtteranceUtil.range_text_hash(book, s, e) for s, e in book.section_ranges())

    @staticmethod
    def section_of_line(book: Book, line: int) -> int:
        """Index of the section containing `line`."""
        return max(i for i, section_start in enumerate(book.section_start_indices()) if section_start <= line)

    @staticmethod
    def section_label(book: Book, section_index: int) -> str:
        """eg "Section 4: Chapter III" (1-based section number)."""
        title = book.sections[section_index].title.strip() or "(untitled)"
        return f"Section {section_index + 1}: {title}"

    @staticmethod
    def range_label(book: Book, start: int, end: int) -> str:
        """eg "Section 4: Chapter III (lines 251-480)"; line numbers are shown
        0-based and inclusive."""
        first = UtteranceUtil.section_of_line(book, start)
        last = UtteranceUtil.section_of_line(book, end - 1)
        lines = f"{start}-{end - 1}"
        if first == last and (start, end) == book.section_ranges()[first]:
            return f"{UtteranceUtil.section_label(book, first)} (lines {lines})"
        if first == last:
            return f"Lines {lines} (in {UtteranceUtil.section_label(book, first)})"
        return (f"Lines {lines} ({UtteranceUtil.section_label(book, first)}"
                f" to {UtteranceUtil.section_label(book, last)})")
