"""
Re-applies display formatting from an EPUB to an existing project's text.

Unlike re-importing, this changes only `Phrase.fmt`: text, phrase groups,
sections, generated audio, markers and dialog assignments are untouched,
because display formatting is not part of any audio or text identity.
See docs-dev/text-formatting-layer.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.text_ops.epub_extractor import (
    EpubChapterTextExtractor,
    EpubExtractor,
    EpubTextChapter,
)
from tts_audiobook_tool.text_ops.format_attach import attach_formatting_to_groups
from tts_audiobook_tool.text_ops.text_format import StyledText


@dataclass
class EpubFormatReapplyResult:
    num_sections: int = 0
    num_paired_sections: int = 0
    # Sections with no matching EPUB section; their formatting is left as-is
    unpaired_section_titles: list[str] = field(default_factory=list)
    num_formatted_phrases: int = 0
    num_words: int = 0
    # Words of paired sections that matched no EPUB word, and so are unformatted
    num_unmatched_words: int = 0
    significant_warnings: list[str] = field(default_factory=list)


def pair_sections(
        sections: list[BookSection],
        chapters: list[EpubTextChapter],
) -> list[tuple[BookSection, EpubTextChapter | None]]:
    """
    Pairs project sections with the EPUB sections import would create
    (import makes one section per kept EPUB section, with its title).
    Sections pair in order by identical title, so deleted sections or a changed
    extractor degrade to unpaired sections rather than misplaced formatting.
    """
    if len(sections) == len(chapters) and all(
        section.title == chapter.title for section, chapter in zip(sections, chapters)
    ):
        return list(zip(sections, chapters))

    result: list[tuple[BookSection, EpubTextChapter | None]] = []
    cursor = 0
    for section in sections:
        match_index = next(
            (index for index in range(cursor, len(chapters)) if chapters[index].title == section.title),
            None,
        )
        if match_index is None:
            result.append((section, None))
        else:
            result.append((section, chapters[match_index]))
            cursor = match_index + 1
    return result


def reapply_epub_formatting(
        book: Book,
        epub_path: str,
        extractor: EpubChapterTextExtractor | None = None,
) -> EpubFormatReapplyResult | str:
    """
    Sets `fmt` on the phrases of every section that pairs with an EPUB section,
    replacing existing values. Mutates `book` in place; the caller persists it.
    Returns an error string when the EPUB yields no readable sections.
    """
    extraction = EpubExtractor.extract_import_chapters(epub_path, extractor)
    if not extraction.chapters:
        details = "; ".join(extraction.significant_warnings)
        return "No readable EPUB sections found" + (f": {details}" if details else "")

    result = EpubFormatReapplyResult(
        num_sections=len(book.sections),
        significant_warnings=list(extraction.significant_warnings),
    )
    for section, chapter in pair_sections(book.sections, extraction.chapters):
        if chapter is None:
            result.unpaired_section_titles.append(section.title)
            continue
        source = chapter.styled if chapter.styled is not None else StyledText.plain(chapter.text)
        alignment = attach_formatting_to_groups(section.phrase_groups, source)
        result.num_paired_sections += 1
        result.num_words += alignment.num_words
        result.num_unmatched_words += alignment.num_unmatched_words

    result.num_formatted_phrases = sum(
        1
        for group in book.phrase_groups
        for phrase in group.phrases
        if phrase.fmt is not None
    )
    return result
