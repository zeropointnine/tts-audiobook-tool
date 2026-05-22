import unittest
from unittest.mock import patch

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.text_ops.epub_extractor import BeautifulSoupEpubChapterTextExtractor, EpubExtractor, EpubSourceChapter, EpubTextExtractionResult
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason


class StubEpubChapterTextExtractor:
    def extract_text(self, chapter: EpubSourceChapter) -> EpubTextExtractionResult:
        return EpubTextExtractionResult(text=chapter.html)


class StubWarningEpubChapterTextExtractor:
    def extract_text(self, chapter: EpubSourceChapter) -> EpubTextExtractionResult:
        warning = (
            f"{BeautifulSoupEpubChapterTextExtractor.INLINE_WHITESPACE_REPAIR_WARNING_PREFIX} 10 times "
            f"in {chapter.href}. Some inline spacing in this EPUB needed cleanup during import; "
            "review imported text if spacing looks unusual."
        )
        return EpubTextExtractionResult(
            text=chapter.html,
            warnings=[warning],
            significant_warnings=[warning],
        )


class StubEpubBook:
    def __init__(self, metadata_values):
        self.metadata_values = metadata_values

    def get_metadata(self, namespace, name):
        if namespace == "DC" and name == "title":
            return self.metadata_values
        return []


class StubTocItem:
    def __init__(self, title: str, href: str, subitems=None):
        self.title = title
        self.href = href
        self.subitems = subitems or []


class StubEpubSpineItem:
    def __init__(self, item_id: str, file_name: str, content: str, media_type: str = "application/xhtml+xml"):
        self.id = item_id
        self.file_name = file_name
        self.media_type = media_type
        self.content = content

    def get_content(self):
        return self.content.encode("utf-8")


class StubEpubBookWithSpine:
    def __init__(self, items: list[StubEpubSpineItem], toc=None, metadata_values=None):
        self.items_by_id = {item.id: item for item in items}
        self.spine = [(item.id, "yes") for item in items]
        self.toc = toc or []
        self.metadata_values = metadata_values or []

    def get_item_with_id(self, item_id: str):
        return self.items_by_id.get(item_id)

    def get_metadata(self, namespace, name):
        if namespace == "DC" and name == "title":
            return self.metadata_values
        return []


class TestEpubExtractor(unittest.TestCase):
    def test_import_epub_marks_each_chapter_last_phrase_as_section(self):
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter one. Still chapter one."),
            EpubSourceChapter("Chapter 2", "chapter2.xhtml", "application/xhtml+xml", "Chapter two."),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual(result.section_start_indices, [1])
        self.assertEqual(len(result.phrase_groups), 2)
        self.assertEqual(result.phrase_groups[0].last_reason, Reason.SECTION_BREAK)
        self.assertEqual(result.phrase_groups[1].last_reason, Reason.SECTION_BREAK)
        self.assertTrue(result.phrase_groups[0].phrases[-1].text.endswith("one."))
        self.assertTrue(result.phrase_groups[1].phrases[-1].text.endswith("two."))

    def test_import_epub_downgrades_leading_section_after_previous_spine_boundary(self):
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter one prose."),
            EpubSourceChapter(
                "Chapter 2",
                "chapter2.xhtml",
                "application/xhtml+xml",
                "Chapter 2\n\n\nThe Beginning\n\n\nChapter two prose.",
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual(result.section_start_indices, [1])
        self.assertEqual(result.phrase_groups[0].last_reason, Reason.SECTION_BREAK)
        self.assertEqual(result.phrase_groups[1].text, "Chapter 2\n\n")
        self.assertEqual(result.phrase_groups[1].last_reason, Reason.PARAGRAPH)
        self.assertEqual(result.phrase_groups[2].last_reason, Reason.PARAGRAPH)
        self.assertEqual(result.phrase_groups[3].last_reason, Reason.SECTION_BREAK)

    def test_import_epub_single_group_chapter_still_ends_as_section(self):
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter one prose."),
            EpubSourceChapter("Chapter 2", "chapter2.xhtml", "application/xhtml+xml", "Chapter two prose."),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual(len(result.phrase_groups), 2)
        self.assertEqual(result.phrase_groups[1].last_reason, Reason.SECTION_BREAK)
        self.assertTrue(result.phrase_groups[1].phrases[-1].text.endswith("prose."))

    def test_import_epub_keeps_metadata_book_title_without_inserting_title_chapter(self):
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter 1\n\nThe story begins."),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "Example Book", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual(result.book_title, "Example Book")
        self.assertEqual(len(result.chapters), 1)
        self.assertEqual(result.chapters[0].href, "chapter1.xhtml")
        self.assertEqual(result.raw_text, "Chapter 1\n\nThe story begins.")
        self.assertEqual(result.section_start_indices, [])
        self.assertEqual(result.phrase_groups[0].phrases[0].text, "Chapter 1\n\n")
        self.assertEqual(result.warnings, [])

    def test_import_epub_uses_real_spine_chapter_boundaries_without_inserted_title_chapter(self):
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter 1\n\nThe story begins."),
            EpubSourceChapter("Chapter 2", "chapter2.xhtml", "application/xhtml+xml", "Chapter 2\n\nThe story continues."),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "Example Book", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual([chapter.href for chapter in result.chapters], ["chapter1.xhtml", "chapter2.xhtml"])
        self.assertEqual(result.section_start_indices, [2])
        self.assertEqual(result.phrase_groups[0].phrases[0].text, "Chapter 1\n\n")
        self.assertEqual(result.phrase_groups[0].last_reason, Reason.PARAGRAPH)

    def test_import_epub_reports_inline_whitespace_repair_warning_once(self):
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter one."),
            EpubSourceChapter("Chapter 2", "chapter2.xhtml", "application/xhtml+xml", "Chapter two."),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                extractor=StubWarningEpubChapterTextExtractor(),
            )

        repair_warnings = [
            warning for warning in result.significant_warnings
            if EpubExtractor.is_inline_whitespace_repair_warning(warning)
        ]
        self.assertEqual(len(repair_warnings), 1)
        self.assertIn("chapter1.xhtml", repair_warnings[0])

    def test_extract_book_title_uses_first_non_empty_dc_title(self):
        book = StubEpubBook([("  \n  ", {}), (" Example Book&nbsp; ", {})])

        title = EpubExtractor.extract_book_title(book)

        self.assertEqual(title, "Example Book")

    def test_get_ebook_title_uses_existing_book_title_extraction(self):
        with patch.object(EpubExtractor, "load_source_chapters", return_value=([], "Example Book", [], [])):
            title = EpubExtractor.get_ebook_title("book.epub")

        self.assertEqual(title, "Example Book")

    def test_extract_toc_title_by_href_collects_nested_titles_and_strips_fragments(self):
        book = StubEpubBookWithSpine(
            items=[],
            toc=[
                StubTocItem("Part One", "Text/part0001.xhtml#chapter"),
                StubTocItem("Parent", "Text/parent.xhtml", [
                    StubTocItem("Chapter Two", "./Text/chapter%2002.xhtml#start"),
                ]),
            ],
        )

        title_by_href = EpubExtractor.extract_toc_title_by_href(book)

        self.assertEqual(title_by_href["Text/part0001.xhtml"], "Part One")
        self.assertEqual(title_by_href["Text/chapter 02.xhtml"], "Chapter Two")

    def test_load_source_chapters_prefers_toc_title_over_html_heading(self):
        book = StubEpubBookWithSpine(
            items=[
                StubEpubSpineItem(
                    "chapter1",
                    "text/part0005.html",
                    "<html><body><h1>Generated Heading</h1><p>Body text.</p></body></html>",
                ),
            ],
            toc=[StubTocItem("CHAPTER ONE", "text/part0005.html#start")],
        )

        with patch("os.path.exists", return_value=True), \
                patch("importlib.import_module") as import_module, \
                patch.object(EpubExtractor, "extract_title", return_value="Visible Heading"):
            import_module.return_value.read_epub.return_value = book
            source_chapters, _, warnings, significant_warnings = EpubExtractor.load_source_chapters("book.epub")

        self.assertEqual(warnings, [])
        self.assertEqual(significant_warnings, [])
        self.assertEqual(source_chapters[0].title, "CHAPTER ONE")

    def test_load_source_chapters_falls_back_to_html_title_without_toc_match(self):
        book = StubEpubBookWithSpine(
            items=[
                StubEpubSpineItem(
                    "chapter1",
                    "text/part0005.html",
                    "<html><body><h1>Visible Heading</h1><p>Body text.</p></body></html>",
                ),
            ],
        )

        with patch("os.path.exists", return_value=True), \
                patch("importlib.import_module") as import_module, \
                patch.object(EpubExtractor, "extract_title", return_value="Visible Heading"):
            import_module.return_value.read_epub.return_value = book
            source_chapters, _, warnings, significant_warnings = EpubExtractor.load_source_chapters("book.epub")

        self.assertEqual(warnings, [])
        self.assertEqual(significant_warnings, [])
        self.assertEqual(source_chapters[0].title, "Visible Heading")

    def test_mark_last_phrase_as_section_handles_empty_groups(self):
        phrase_groups = [PhraseGroup()]

        EpubExtractor.mark_last_phrase_as_section(phrase_groups)

        self.assertEqual(phrase_groups[0].phrases, [])

    def test_mark_last_phrase_as_section_normalizes_trailing_linefeeds_to_three(self):
        phrase_groups = [PhraseGroup([Phrase("Section end.\n\n\n\n", Reason.PARAGRAPH)])]

        EpubExtractor.mark_last_phrase_as_section(phrase_groups)

        phrase = phrase_groups[0].phrases[0]
        self.assertEqual(phrase.text, "Section end.\n\n\n\n")
        self.assertEqual(phrase.reason, Reason.SECTION_BREAK)

    def test_extract_text_does_not_make_image_only_chapter_warning_significant(self):
        chapter = EpubSourceChapter(
            title="insert001",
            href="Text/insert001.xhtml",
            media_type="application/xhtml+xml",
            html='<html><body><div><img src="../Images/Art_insert001.jpg" alt="Book Title Page"/></div></body></html>',
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.text, "")
        self.assertEqual(result.warnings, ["No readable body text found in Text/insert001.xhtml"])
        self.assertEqual(result.significant_warnings, [])

    def test_extract_text_preserves_chapter_number_and_title_headings_without_heading_count_warning(self):
        chapter = EpubSourceChapter(
            title="1",
            href="Text/chapter001_a.xhtml",
            media_type="application/xhtml+xml",
            html="""
            <html><body>
                <h1 class="chapter-number">1</h1>
                <h1 class="chapter-title">The Chaos Begins</h1>
                <p>The entire area was one massive graveyard.</p>
            </body></html>
            """,
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.warnings, [])
        self.assertEqual(result.significant_warnings, [])
        self.assertIn("The Chaos Begins", result.text)

    def test_extract_text_does_not_warn_for_multiple_major_headings_in_one_spine_document(self):
        chapter = EpubSourceChapter(
            title="Chapter 1",
            href="Text/chapter001.xhtml",
            media_type="application/xhtml+xml",
            html="""
            <html><body>
                <h1>Chapter 1</h1>
                <h1>Chapter 2</h1>
                <p>The story continued.</p>
            </body></html>
            """,
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.warnings, [])
        self.assertEqual(result.significant_warnings, [])
        self.assertIn("Chapter 1", result.text)
        self.assertIn("Chapter 2", result.text)

    def test_extract_text_preserves_normal_paragraph_spacing(self):
        chapter = EpubSourceChapter(
            title="Chapter 1",
            href="Text/chapter001.xhtml",
            media_type="application/xhtml+xml",
            html="""
            <html><body>
                <p>Before.</p>
                <p>After.</p>
            </body></html>
            """,
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.text, "Before.\n\nAfter.")

    def test_extract_text_repairs_inline_newline_separator_spans_as_spaces(self):
        chapter = EpubSourceChapter(
            title="Chapter 1",
            href="Text/chapter001.xhtml",
            media_type="application/xhtml+xml",
            html="""
            <html><body>
                <p><span>Will it happen?</span><span>
                </span><span>That is the idea.</span></p>
            </body></html>
            """,
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.text, "Will it happen? That is the idea.")
        self.assertEqual(result.significant_warnings, [])

    def test_extract_text_keeps_pretty_printed_block_spacing_stable(self):
        chapter = EpubSourceChapter(
            title="Chapter 1",
            href="Text/chapter001.xhtml",
            media_type="application/xhtml+xml",
            html="""
            <html>
                <body>
                    <div>
                        <p>Before.</p>
                        <p>After.</p>
                    </div>
                </body>
            </html>
            """,
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.text, "Before.\n\nAfter.")

    def test_extract_text_warns_when_many_inline_whitespace_separators_are_repaired(self):
        separator_count = BeautifulSoupEpubChapterTextExtractor.INLINE_WHITESPACE_REPAIR_WARNING_THRESHOLD
        pieces = []
        for index in range(separator_count + 1):
            if index > 0:
                pieces.append("<span>\n</span>")
            pieces.append(f"<span>Sentence {index + 1}.</span>")
        chapter = EpubSourceChapter(
            title="Chapter 1",
            href="Text/chapter001.xhtml",
            media_type="application/xhtml+xml",
            html=f"<html><body><p>{''.join(pieces)}</p></body></html>",
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertIn("Sentence 1. Sentence 2.", result.text)
        self.assertEqual(len(result.significant_warnings), 1)
        self.assertIn(f"repaired inline spacing {separator_count} times", result.significant_warnings[0])
        self.assertIn("Text/chapter001.xhtml", result.significant_warnings[0])
        self.assertNotIn("whitespace-only inline markup as text separators", result.significant_warnings[0])

    def test_extract_text_does_not_warn_below_inline_whitespace_repair_threshold(self):
        chapter = EpubSourceChapter(
            title="Chapter 1",
            href="Text/chapter001.xhtml",
            media_type="application/xhtml+xml",
            html="<html><body><p><span>One.</span><span>\n</span><span>Two.</span></p></body></html>",
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.text, "One. Two.")
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.significant_warnings, [])

    def test_extract_text_preserves_section_break_spacing_from_empty_block(self):
        chapter = EpubSourceChapter(
            title="Chapter 1",
            href="Text/chapter001.xhtml",
            media_type="application/xhtml+xml",
            html="""
            <html><body>
                <p>Before.</p>
                <div><div><img src="ornament.jpg" alt=""/></div></div>
                <p>After.</p>
            </body></html>
            """,
        )

        result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)

        self.assertEqual(result.text, "Before.\n\n\nAfter.")

    def test_normalize_output_text_caps_section_break_spacing(self):
        text = BeautifulSoupEpubChapterTextExtractor.normalize_output_text("Before.\n\n\n\n\nAfter.")

        self.assertEqual(text, "Before.\n\n\nAfter.")

    def test_format_skipped_section_warning_includes_removed_content_preview(self):
        chapter = EpubSourceChapter(
            title="Copyright",
            href="Text/copyright.xhtml",
            media_type="application/xhtml+xml",
            html="""
            <html><body>
                <h1>Copyright</h1>
                <p>Copyright © 2024 Example Press. All rights reserved. ISBN 978-1-2345-6789-0.</p>
            </body></html>
            """,
        )

        warning = EpubExtractor.format_skipped_section_warning(chapter, "publication metadata text signals (3)")

        self.assertEqual(
            warning,
            "Skipped EPUB section: Text/copyright.xhtml (publication metadata text signals (3)): "
            "Copyright Copyright © 2024 Example Press. All rights reserved. ISBN 978-1-2345-6789-0.",
        )

    def test_format_skipped_section_warning_ellipsizes_removed_content_preview_to_100_chars(self):
        chapter = EpubSourceChapter(
            title="Contents",
            href="Text/toc.xhtml",
            media_type="application/xhtml+xml",
            html="<html><body><h1>Contents</h1><p>" + "word " * 40 + "</p></body></html>",
        )

        warning = EpubExtractor.format_skipped_section_warning(chapter, "table of contents heading signal plus link structure")
        preview = warning.rsplit(": ", 1)[1]

        self.assertEqual(len(preview), 100)
        self.assertTrue(preview.endswith("…"))

    def test_append_skipped_section_warning_makes_skip_visible_to_user_fyi_prompt(self):
        warnings: list[str] = []
        significant_warnings: list[str] = []
        chapter = EpubSourceChapter(
            title="Contents",
            href="Text/toc.xhtml",
            media_type="application/xhtml+xml",
            html="<html><body><h1>Contents</h1><ol><li>Chapter 1</li></ol></body></html>",
        )

        EpubExtractor.append_skipped_section_warning(
            warnings,
            significant_warnings,
            chapter,
            "table of contents heading signal plus link structure",
        )

        self.assertEqual(warnings, significant_warnings)
        self.assertEqual(len(significant_warnings), 1)
        self.assertIn("Contents Chapter 1", significant_warnings[0])


if __name__ == "__main__":
    unittest.main()