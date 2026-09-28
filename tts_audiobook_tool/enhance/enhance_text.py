from __future__ import annotations

"""Canonical Book adaptation for the enhance-audiobook workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_audiobook_tool.app_types import (
    Book,
    BookSection,
    BookSegmentationSettings,
    Word,
)
from tts_audiobook_tool.app_types.app_metadata import AppMetadataSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.constants import COL_DEFAULT
from tts_audiobook_tool.constants_config import (
    ENHANCE_DIALOG_SEGMENTATION,
    ENHANCE_MAX_WORDS_PER_SEGMENT,
    ENHANCE_SEGMENTATION_STRATEGY,
)
from tts_audiobook_tool.enhance import enhance_alignment
from tts_audiobook_tool.menus.epub_menu_util import EpubMenuUtil
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.util import make_error_string, printt


PLAIN_TEXT_SOURCE_KIND = "plain_text"
EPUB_SOURCE_KIND = "epub"
PRE_EXISTING_AUDIO_SOURCE_KIND = "pre_existing"


@dataclass(frozen=True)
class FlattenedEnhanceBook:
    phrases: list[Phrase]
    section_ranges: list[tuple[int, int]]
    section_titles: list[str]


def segmentation_settings_from_state(state: Any) -> BookSegmentationSettings:
    """
    Enhance-flow segmentation policy. Strategy, max words, and dialog
    segmentation come from dedicated constants rather than the current
    project's TTS-flow segmentation settings. Only the language code is
    project-owned, since it drives language-aware sentence segmentation
    (and transcription) and must stay consistent with the audio.
    """
    return BookSegmentationSettings(
        language_code=state.project.language_code,
        max_words_per_segment=ENHANCE_MAX_WORDS_PER_SEGMENT,
        strategy=ENHANCE_SEGMENTATION_STRATEGY,
        dialog_segmentation=ENHANCE_DIALOG_SEGMENTATION,
    )


def plain_text_to_book(raw_text: str, state: Any) -> Book | str:
    if not raw_text.strip():
        return "File has no content."
    settings = segmentation_settings_from_state(state)
    try:
        groups = PhraseGrouper.text_to_groups(
            raw_text,
            max_words=settings.max_words_per_segment,
            strategy=settings.strategy,
            pysbd_lang=settings.language_code,
            dialog_segmentation=settings.dialog_segmentation,
        )
    except Exception as exception:
        return f"Error segmenting source text: {make_error_string(exception)}"
    if not groups or not PhraseGroup.flatten_groups(groups):
        return "Source text produced no segments."
    return Book(
        sections=[BookSection(phrase_groups=groups)],
        text_source_kind=PLAIN_TEXT_SOURCE_KIND,
        audio_source_kind=PRE_EXISTING_AUDIO_SOURCE_KIND,
        segmentation_settings=settings,
    )


def epub_result_to_book(result: EpubImportResult, state: Any) -> Book | str:
    if not result.raw_text.strip():
        return "EPUB import produced no text."
    if not result.phrase_groups or not PhraseGroup.flatten_groups(result.phrase_groups):
        return "EPUB import produced no text segments."
    settings = segmentation_settings_from_state(state)
    return ProjectBookUtil.make_book_from_flat_compatibility_fields(
        phrase_groups=result.phrase_groups,
        section_start_indices=result.section_start_indices,
        segmentation_settings=settings,
        text_source_kind=EPUB_SOURCE_KIND,
        audio_source_kind=PRE_EXISTING_AUDIO_SOURCE_KIND,
        title=result.book_title,
        section_titles=[chapter.title for chapter in result.chapters],
    )


def import_source_book(state: Any, source_path: str | Path) -> Book | str | None:
    """Import .txt/.epub; ``None`` means the EPUB importer was cancelled/failed."""
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception as exception:
            return f"Error reading text file: {make_error_string(exception)}"
        return plain_text_to_book(raw_text, state)
    if suffix != ".epub":
        return "Source text file must have a .txt or .epub suffix."

    result = EpubMenuUtil.import_epub(
        epub_path=str(path),
        max_words=ENHANCE_MAX_WORDS_PER_SEGMENT,
        segmentation_strategy=ENHANCE_SEGMENTATION_STRATEGY,
        language_code=state.project.language_code,
        dialog_segmentation=ENHANCE_DIALOG_SEGMENTATION,
    )
    if result is None:
        return None
    EpubMenuUtil.print_import_info(result)
    return epub_result_to_book(result, state)


def flatten_book(book: Book) -> FlattenedEnhanceBook:
    phrases: list[Phrase] = []
    ranges: list[tuple[int, int]] = []
    titles: list[str] = []
    for section in book.sections:
        start = len(phrases)
        phrases.extend(PhraseGroup.flatten_groups(section.phrase_groups))
        end = len(phrases)
        if end > start:
            ranges.append((start, end))
            titles.append(section.title)
    return FlattenedEnhanceBook(
        phrases=phrases,
        section_ranges=ranges,
        section_titles=titles,
    )


def align_book(
    book: Book,
    words: list[Word],
) -> tuple[list[TimedPhrase], bool]:
    flattened = flatten_book(book)
    if not flattened.phrases:
        return [], False

    alignment_state = enhance_alignment.AlignmentState()
    timed_phrases: list[TimedPhrase] = []
    Interrupts().set("thinking")
    try:
        for section_index, (start, end) in enumerate(flattened.section_ranges):
            section_phrases = flattened.phrases[start:end]
            if enhance_alignment.PRINT_VERBOSE and len(flattened.section_ranges) > 1:
                title = flattened.section_titles[section_index]
                title_suffix = f": {title}" if title else ""
                printt(
                    f"Aligning EPUB section "
                    f"{section_index + 1}/{len(flattened.section_ranges)}"
                    f"{title_suffix}{COL_DEFAULT}"
                )
                printt()
            section_timed, alignment_state, did_interrupt = (
                enhance_alignment.align_phrases_with_state(
                    section_phrases,
                    words,
                    alignment_state,
                    line_offset=start,
                    total_lines=len(flattened.phrases),
                )
            )
            if did_interrupt:
                return [], True
            timed_phrases.extend(section_timed)
        if not enhance_alignment.PRINT_VERBOSE:
            # Terminate the in-place progress line and add a blank line
            print("\n")
        return timed_phrases, False
    finally:
        Interrupts().clear()


def make_app_metadata_sections(
    book: Book,
    text_segment_count: int,
) -> list[AppMetadataSection]:
    flattened = flatten_book(book)
    sections: list[AppMetadataSection] = []
    for index, (start, end) in enumerate(flattened.section_ranges):
        if start >= text_segment_count:
            continue
        section_end = min(end, text_segment_count)
        if section_end <= start:
            continue
        sections.append(
            AppMetadataSection(
                title=flattened.section_titles[index],
                start_index=start,
                end_index=section_end,
            )
        )
    return sections
