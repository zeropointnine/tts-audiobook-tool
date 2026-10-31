import os
import tempfile

from tts_audiobook_tool.app_types import Book, BookSection, SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.epub_extractor import (
    BeautifulSoupEpubChapterTextExtractor,
    EpubExtractor,
    EpubSourceChapter,
    EpubTextChapter,
)
from tts_audiobook_tool.text_ops.epub_format_reapply import pair_sections, reapply_epub_formatting
from tts_audiobook_tool.text_ops.format_attach import attach_formatting_to_groups
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.text_ops import text_format


def extract(html: str):
    chapter = EpubSourceChapter(title="T", href="t.xhtml", media_type="application/xhtml+xml", html=html)
    return BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)


def all_fmts(groups: list[PhraseGroup]) -> list[tuple[str, str | None]]:
    return [(phrase.text, phrase.fmt) for group in groups for phrase in group.phrases]


def write_epub(path: str, chapter_bodies: list[tuple[str, str]]) -> None:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("formatting-test")
    book.set_title("Formatting Test")
    book.set_language("en")
    chapters = []
    for index, (title, body) in enumerate(chapter_bodies):
        chapter = epub.EpubHtml(uid=f"c{index}", file_name=f"Text/c{index}.xhtml", title=title)
        chapter.content = f"<html><body>{body}</body></html>"
        book.add_item(chapter)
        chapters.append(chapter)
    nav = epub.EpubItem(uid="nav", file_name="Navigation/nav.xhtml", media_type="application/xhtml+xml")
    nav.properties = ["nav"]
    nav.content = '<html><body><nav epub:type="toc"><ol>' + "".join(
        f'<li><a href="../Text/c{index}.xhtml">{title}</a></li>'
        for index, (title, _) in enumerate(chapter_bodies)
    ) + "</ol></nav></body></html>"
    book.add_item(nav)
    book.spine = chapters
    epub.write_epub(path, book)


CHAPTERS = [
    ("One", "<h1>One</h1><p>It was <em>very</em> cold &amp; <b>dark</b>.</p>"),
    ("Two", '<p><span style="font-style: italic">Styled</span> and <u>lined</u>.</p>'),
]


def test_extractor_captures_tag_and_inline_css_styles() -> None:
    # Semantic tags, inline CSS (including turning a style back off) and
    # headings are recorded; the extracted plain text is unchanged.
    result = extract(
        "<h2>Title</h2>"
        "<p>A <strong>bold</strong> <i>it <span style='font-style:normal'>up</span></i> "
        "<span style='font-weight: 700; text-decoration: underline'>both</span>.</p>"
    )
    assert result.text == "Title\n\nA bold it up both."
    assert result.styled is not None

    alignment = text_format.align_masks(result.styled, result.text)
    assert alignment.is_exact
    assert text_format.render(result.text, alignment.masks) == (
        "<h>Title</h>\n\nA <b>bold</b> <i>it</i> up <b><u>both</u></b>."
    )


def test_extractor_reports_no_styles_for_plain_html() -> None:
    assert extract("<p>Nothing <span>special</span>.</p>").styled is None


def test_formatting_survives_segmentation_with_dialog_splitting() -> None:
    # Attaching after the full segmentation pipeline (including dialog
    # splitting) lands every style on the right phrase.
    result = extract('<p>"<i>Go</i> now," she said. He <b>ran</b>.</p>')
    groups = PhraseGrouper.text_to_groups(
        result.text, max_words=40, strategy=SegmentationStrategy.SENTENCE,
        pysbd_lang="en", dialog_segmentation=True,
    )
    assert result.styled is not None
    assert attach_formatting_to_groups(groups, result.styled).is_exact
    formatted = [fmt for _, fmt in all_fmts(groups) if fmt is not None]
    assert formatted == ['"<i>Go</i> now," ', "He <b>ran</b>."]


def test_import_epub_attaches_formatting_without_changing_text() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "book.epub")
        write_epub(path, CHAPTERS)
        result = EpubExtractor.import_epub(
            epub_path=path, max_words=40,
            segmentation_strategy=SegmentationStrategy.SENTENCE, language_code="en",
        )

    assert "<" not in result.raw_text
    fmts = [fmt for _, fmt in all_fmts(result.phrase_groups)]
    assert fmts == [
        "<h>One</h>\n\n",
        "It was <i>very</i> cold &amp; <b>dark</b>.",
        "<i>Styled</i> and <u>lined</u>.",
    ]
    for text, fmt in all_fmts(result.phrase_groups):
        assert fmt is None or text_format.is_valid_for(fmt, text)


def test_reapply_restores_formatting_without_touching_text_and_skips_edits() -> None:
    # Re-applying sets formatting on the current phrases from the EPUB: an
    # edited word stays unformatted, everything else regains its styles, and
    # phrase text is never modified.
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "book.epub")
        write_epub(path, CHAPTERS)
        imported = EpubExtractor.import_epub(
            epub_path=path, max_words=40,
            segmentation_strategy=SegmentationStrategy.SENTENCE, language_code="en",
        )
        sections = [
            BookSection(title="One", phrase_groups=imported.phrase_groups[:2]),
            BookSection(title="Two", phrase_groups=imported.phrase_groups[2:]),
        ]
        book = Book(sections=sections, text_source_kind="epub")
        for group in book.phrase_groups:
            for phrase in group.phrases:
                phrase.fmt = None
        edited = book.phrase_groups[1].phrases[0]
        edited.text = "It was very cold & gloomy."
        texts_before = [phrase.text for group in book.phrase_groups for phrase in group.phrases]

        result = reapply_epub_formatting(book, path)

    assert not isinstance(result, str)
    assert result.num_paired_sections == 2
    assert result.num_unmatched_words == 1
    assert [phrase.text for group in book.phrase_groups for phrase in group.phrases] == texts_before
    assert edited.fmt == "It was <i>very</i> cold &amp; gloomy."
    assert book.phrase_groups[2].phrases[0].fmt == "<i>Styled</i> and <u>lined</u>."


def test_pair_sections_falls_back_to_ordered_titles_when_counts_differ() -> None:
    # A deleted project section must not shift formatting onto its neighbors.
    def section(title: str) -> BookSection:
        return BookSection(title=title, phrase_groups=[PhraseGroup([Phrase("x", Reason.SENTENCE)])])

    chapters = [EpubTextChapter(title, f"{title}.xhtml", "x") for title in ("A", "B", "C")]
    sections = [section("A"), section("C"), section("Z")]

    pairs = pair_sections(sections, chapters)

    assert [(s.title, c.title if c else None) for s, c in pairs] == [("A", "A"), ("C", "C"), ("Z", None)]


def test_phrase_fmt_persists_and_is_dropped_when_stale_or_text_changes() -> None:
    # fmt is written only when set, survives a save/load round trip, is dropped
    # silently on load when it no longer matches the text, and is cleared when
    # the text is changed in place. It never affects phrase equality.
    phrase = Phrase("It was cold. ", Reason.SENTENCE, fmt="It was <i>cold</i>. ")
    plain = Phrase("Plain. ", Reason.SENTENCE)
    assert "fmt" not in plain.to_json_dict()
    assert phrase == Phrase("It was cold. ", Reason.SENTENCE)

    loaded = Phrase.phrases_from_json_dicts([
        phrase.to_json_dict(),
        {"text": "Edited text. ", "reason": "s", "fmt": "It was <i>cold</i>. "},
    ])
    assert not isinstance(loaded, str)
    assert loaded[0].fmt == "It was <i>cold</i>. "
    assert loaded[1].fmt is None

    loaded[0].text = "It was colder. "
    assert loaded[0].fmt is None


def test_flattened_group_joins_phrase_formatting() -> None:
    group = PhraseGroup([
        Phrase("It was very, ", Reason.PHRASE, fmt="It was <i>very,</i> "),
        Phrase("very cold. ", Reason.SENTENCE, fmt="<i>very</i> cold. "),
    ])
    assert group.as_flattened_phrase().fmt == "It was <i>very, very</i> cold. "
    assert PhraseGroup([Phrase("Plain.", Reason.SENTENCE)]).as_flattened_phrase().fmt is None
