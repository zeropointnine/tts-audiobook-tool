from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import (
    Book,
    BookSection,
    ConcreteWord,
    SegmentationStrategy,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.constants_config import (
    ENHANCE_DIALOG_SEGMENTATION,
    ENHANCE_MAX_WORDS_PER_SEGMENT,
    ENHANCE_SEGMENTATION_STRATEGY,
)
from tts_audiobook_tool.enhance.enhance_text import (
    align_book,
    epub_result_to_book,
    flatten_book,
    import_source_book,
    make_app_metadata_sections,
    plain_text_to_book,
)
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult, EpubTextChapter


def make_state():
    return SimpleNamespace(project=SimpleNamespace(
        max_words=55,
        segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
        language_code="en",
        dialog_segmentation=True,
    ))


def phrase(text: str) -> Phrase:
    return Phrase(text, Reason.SENTENCE)


def test_plain_text_uses_dedicated_enhance_segmentation_settings() -> None:
    # Project segmentation fields must be ignored in favor of the dedicated
    # enhance constants; only language_code is project-owned.
    state = make_state()
    groups = [PhraseGroup([phrase("First."), phrase(" Second.")])]
    with patch(
        "tts_audiobook_tool.enhance.enhance_text.PhraseGrouper.text_to_groups",
        return_value=groups,
    ) as segment:
        book = plain_text_to_book("First. Second.", state)

    assert isinstance(book, Book)
    segment.assert_called_once_with(
        "First. Second.",
        max_words=ENHANCE_MAX_WORDS_PER_SEGMENT,
        strategy=ENHANCE_SEGMENTATION_STRATEGY,
        pysbd_lang="en",
        dialog_segmentation=ENHANCE_DIALOG_SEGMENTATION,
    )
    assert book.text_source_kind == "plain_text"
    assert book.audio_source_kind == "pre_existing"
    assert book.segmentation_settings.max_words_per_segment == ENHANCE_MAX_WORDS_PER_SEGMENT
    assert book.segmentation_settings.strategy == ENHANCE_SEGMENTATION_STRATEGY
    assert book.segmentation_settings.dialog_segmentation == ENHANCE_DIALOG_SEGMENTATION
    assert book.segmentation_settings.language_code == "en"
    assert len(book.sections) == 1


def test_plain_text_rejects_blank_and_empty_segmentation() -> None:
    assert plain_text_to_book("  \n", make_state()) == "File has no content."
    with patch(
        "tts_audiobook_tool.enhance.enhance_text.PhraseGrouper.text_to_groups",
        return_value=[],
    ):
        assert plain_text_to_book("Words", make_state()) == "Source text produced no segments."


def test_epub_preserves_title_sections_and_phrase_order() -> None:
    groups = [
        PhraseGroup([phrase("A."), phrase(" B.")]),
        PhraseGroup([phrase("C.")]),
        PhraseGroup([phrase("D."), phrase(" E.")]),
    ]
    result = EpubImportResult(
        phrase_groups=groups,
        raw_text="A B C D E",
        section_start_indices=[1, 2],
        chapters=[
            EpubTextChapter("One", "1.xhtml", "A B"),
            EpubTextChapter("Two", "2.xhtml", "C"),
            EpubTextChapter("Three", "3.xhtml", "D E"),
        ],
        book_title="Novel",
    )

    book = epub_result_to_book(result, make_state())

    assert isinstance(book, Book)
    assert book.title == "Novel"
    assert book.text_source_kind == "epub"
    assert [section.title for section in book.sections] == ["One", "Two", "Three"]
    assert [[item.text for item in PhraseGroup.flatten_groups(section.phrase_groups)] for section in book.sections] == [
        ["A.", " B."], ["C."], ["D.", " E."],
    ]


def test_flatten_book_maps_sections_in_phrase_not_group_coordinates() -> None:
    book = Book(
        sections=[
            BookSection([PhraseGroup([phrase("A"), phrase("B")])], "One"),
            BookSection([
                PhraseGroup([phrase("C")]),
                PhraseGroup([phrase("D"), phrase("E")]),
            ], "Two"),
        ],
        text_source_kind="epub",
        audio_source_kind="pre_existing",
    )

    flattened = flatten_book(book)

    assert [item.text for item in flattened.phrases] == ["A", "B", "C", "D", "E"]
    assert flattened.section_ranges == [(0, 2), (2, 5)]
    assert flattened.section_titles == ["One", "Two"]
    sections = make_app_metadata_sections(book, 4)
    assert [(item.title, item.start_index, item.end_index) for item in sections] == [
        ("One", 0, 2), ("Two", 2, 4),
    ]


def test_section_alignment_carries_cursor_and_global_line_offset() -> None:
    book = Book(
        sections=[
            BookSection([PhraseGroup([phrase("Alpha")])], "One"),
            BookSection([PhraseGroup([phrase("Beta")])], "Two"),
        ],
        text_source_kind="epub",
        audio_source_kind="pre_existing",
    )
    words = [
        ConcreteWord(0.0, 0.5, "alpha", 1.0),
        ConcreteWord(1.0, 1.5, "beta", 1.0),
    ]

    with patch("tts_audiobook_tool.enhance.enhance_text.printt"):
        timed, interrupted = align_book(book, words)

    assert not interrupted
    assert [(item.time_start, item.time_end) for item in timed] == [(0.0, 1.0), (1.0, 1.5)]


def test_alignment_clears_interrupt_mode_when_alignment_raises() -> None:
    book = Book(
        sections=[BookSection([PhraseGroup([phrase("Alpha")])])],
        text_source_kind="plain_text",
        audio_source_kind="pre_existing",
    )
    with patch(
        "tts_audiobook_tool.enhance.enhance_text.enhance_alignment.align_phrases_with_state",
        side_effect=RuntimeError("alignment failed"),
    ), pytest.raises(RuntimeError, match="alignment failed"):
        align_book(book, [])
    assert Interrupts()._mode == ""


def test_import_source_book_rejects_unsupported_suffix(tmp_path) -> None:
    path = tmp_path / "book.md"
    path.write_text("text", encoding="utf-8")
    result = import_source_book(make_state(), path)
    assert result == "Source text file must have a .txt or .epub suffix."
