import os
import tempfile
import unittest
from unittest.mock import patch

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.text_ops.epub_extractor import BeautifulSoupEpubChapterTextExtractor, EpubExtractor, EpubNavigationTarget, EpubSourceChapter, EpubTextExtractionResult, EpubTextSlice
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


def make_navigation_target(title: str, href: str, order: int, depth: int = 0) -> EpubNavigationTarget:
    target = EpubExtractor.make_navigation_target(href, title, order, depth=depth)
    assert target is not None
    return target


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
        self.items = items
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

    def test_import_epub_preassigns_detected_dialog_to_voice_sample_two(self):
        source_chapters = [
            EpubSourceChapter(
                "Chapter 1",
                "chapter1.xhtml",
                "application/xhtml+xml",
                'He said "Hello." Then left.',
            ),
        ]

        with patch.object(
            EpubExtractor,
            "load_source_chapters",
            return_value=(source_chapters, "", [], []),
        ):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=100,
                segmentation_strategy=SegmentationStrategy.MAX_LEN,
                language_code="en",
                dialog_segmentation=True,
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual(
            [group.text for group in result.phrase_groups],
            ["He said ", '"Hello." ', "Then left."],
        )
        self.assertEqual(
            [group.voice_index for group in result.phrase_groups],
            [-1, 1, -1],
        )

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

    def test_import_epub_merges_leading_ornamental_line_into_first_group(self):
        # A section that opens with an ornamental line (dinkus, asterisk
        # divider, etc) must not produce an ornament-only first PhraseGroup,
        # which would normalize to an empty TTS prompt. The ornament rides
        # with the section's first vocalizable phrase.
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "✦\n\nChapter one prose follows."),
            EpubSourceChapter("Chapter 2", "chapter2.xhtml", "application/xhtml+xml", "* * *\n\nChapter two prose follows."),
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
        self.assertEqual(
            [group.text for group in result.phrase_groups],
            ["✦\n\nChapter one prose follows.", "* * *\n\nChapter two prose follows."],
        )
        for group_index in [0, *result.section_start_indices]:
            self.assertTrue(app_text.is_vocalizable(result.phrase_groups[group_index].text))

    def test_import_epub_merges_leading_ornamental_group_recreated_by_dialog_split(self):
        # Dialog segmentation splits the first phrase at the opening quote of
        # dialog, which re-isolates a leading ornament as its own group. The
        # import pass must fold it back into the first vocalizable group.
        source_chapters = [
            EpubSourceChapter(
                "Chapter 1",
                "chapter1.xhtml",
                "application/xhtml+xml",
                "✦\n\n“Hello,” she said.\n\nMore prose follows here.",
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                dialog_segmentation=True,
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual(
            [group.text for group in result.phrase_groups],
            ["✦\n\n“Hello,” ", "she said.\n\n", "More prose follows here."],
        )
        self.assertEqual(
            [group.voice_index for group in result.phrase_groups],
            [1, -1, -1],
        )
        self.assertTrue(app_text.is_vocalizable(result.phrase_groups[0].text))

    def test_import_epub_skips_section_with_no_vocalizable_text(self):
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter one prose."),
            EpubSourceChapter("Divider", "divider.xhtml", "application/xhtml+xml", "✦\n\n◆ ◆ ◆"),
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

        self.assertEqual(
            [group.text for group in result.phrase_groups],
            ["Chapter one prose.", "Chapter two prose."],
        )
        self.assertEqual(result.section_start_indices, [1])
        self.assertEqual([chapter.title for chapter in result.chapters], ["Chapter 1", "Chapter 2"])
        self.assertIn(
            "EPUB section has no vocalizable text and was skipped: Divider",
            result.warnings,
        )

    def test_import_epub_merges_trailing_ornamental_line_into_last_group(self):
        # A section ending with an ornament line: the trailing ornament has
        # no trailing line breaks (extraction strips them), so only the
        # grouper-level fold can prevent an ornament-only final group.
        source_chapters = [
            EpubSourceChapter("Chapter 1", "chapter1.xhtml", "application/xhtml+xml", "Chapter one prose.\n\n✦"),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
                extractor=StubEpubChapterTextExtractor(),
            )

        self.assertEqual(
            [group.text for group in result.phrase_groups],
            ["Chapter one prose.\n\n✦"],
        )
        self.assertTrue(app_text.is_vocalizable(result.phrase_groups[-1].text))
        self.assertEqual(result.phrase_groups[-1].last_reason, Reason.SECTION_BREAK)

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
        self.assertIn("EPUB navigation is unavailable or unusable", result.warnings[0])
        self.assertEqual(result.warnings, result.significant_warnings)

    def test_import_epub_falls_back_to_spine_boundaries_without_navigation(self):
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

    def test_import_epub_uses_navigation_sections_across_text_and_image_spine_documents(self):
        chapter_one_target = make_navigation_target("Chapter 1", "Text/chapter005.xhtml", 0)
        chapter_two_target = make_navigation_target("Chapter 2", "Text/chapter010.xhtml", 1)
        source_chapters = [
            EpubSourceChapter(
                "Chapter 1",
                "Text/chapter005.xhtml",
                "application/xhtml+xml",
                '<html><body><img src="chapter-one.jpg"/></body></html>',
                [chapter_one_target],
            ),
            EpubSourceChapter(
                "chapter006",
                "Text/chapter006.xhtml",
                "application/xhtml+xml",
                "<html><body><p>“Okay, I gotta ask—”</p></body></html>",
            ),
            EpubSourceChapter(
                "chapter007",
                "Text/chapter007.xhtml",
                "application/xhtml+xml",
                '<html><body><img src="illustration.jpg"/></body></html>',
            ),
            EpubSourceChapter(
                "chapter008",
                "Text/chapter008.xhtml",
                "application/xhtml+xml",
                "<html><body><p>“Can I have this one, Christina?”</p></body></html>",
            ),
            EpubSourceChapter(
                "Chapter 2",
                "Text/chapter010.xhtml",
                "application/xhtml+xml",
                '<html><body><img src="chapter-two.jpg"/></body></html>',
                [chapter_two_target],
            ),
            EpubSourceChapter(
                "chapter011",
                "Text/chapter011.xhtml",
                "application/xhtml+xml",
                "<html><body><h1>Chapter 2</h1><p>The next chapter begins.</p></body></html>",
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.title for chapter in result.chapters], ["Chapter 1", "Chapter 2"])
        self.assertEqual([chapter.href for chapter in result.chapters], ["Text/chapter005.xhtml", "Text/chapter010.xhtml"])
        self.assertIn("Okay, I gotta ask", result.chapters[0].text)
        self.assertIn("Can I have this one, Christina", result.chapters[0].text)
        self.assertNotIn("chapter008", [chapter.title for chapter in result.chapters])
        self.assertEqual(len(result.section_start_indices), 1)
        self.assertEqual(result.phrase_groups[result.section_start_indices[0] - 1].last_reason, Reason.SECTION_BREAK)

    def test_import_epub_splits_navigation_fragments_within_one_document(self):
        source_chapters = [
            EpubSourceChapter(
                "Fallback",
                "Text/story.xhtml",
                "application/xhtml+xml",
                """
                <html><body>
                    <h1 id="one">One</h1><p>First section prose.</p>
                    <a id="two"></a><h1>Two</h1><p>Second section prose.</p>
                </body></html>
                """,
                [
                    make_navigation_target("First", "Text/story.xhtml#one", 0),
                    make_navigation_target("Second", "Text/story.xhtml#two", 1),
                ],
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.title for chapter in result.chapters], ["First", "Second"])
        self.assertIn("One\n\nFirst section prose.", result.chapters[0].text)
        self.assertIn("Two\n\nSecond section prose.", result.chapters[1].text)
        self.assertEqual(result.section_start_indices, [2])

    def test_import_epub_uses_deepest_title_for_duplicate_navigation_location(self):
        source_chapters = [
            EpubSourceChapter(
                "Fallback",
                "Text/story.xhtml",
                "application/xhtml+xml",
                '<html><body><h1 id="start">One</h1><p>Prose.</p></body></html>',
                [
                    make_navigation_target("Part One", "Text/story.xhtml#start", 0),
                    make_navigation_target("Chapter One", "Text/story.xhtml#start", 1, depth=1),
                    make_navigation_target("Duplicate sibling", "Text/story.xhtml#start", 2),
                ],
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.title for chapter in result.chapters], ["Chapter One"])

    def test_import_epub_splits_at_inline_fragment_in_dom_order(self):
        source_chapters = [
            EpubSourceChapter(
                "Fallback",
                "Text/story.xhtml",
                "application/xhtml+xml",
                '<html><body><p id="one">Before <span id="two">after</span> end.</p></body></html>',
                [
                    make_navigation_target("One", "Text/story.xhtml#one", 0),
                    make_navigation_target("Two", "Text/story.xhtml#two", 1),
                ],
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.text for chapter in result.chapters], ["Before", "after end."])

    def test_import_epub_honors_coarse_navigation_across_multiple_text_documents(self):
        source_chapters = [
            EpubSourceChapter(
                "Part One",
                "Text/one.xhtml",
                "application/xhtml+xml",
                "<html><body><p>First document.</p></body></html>",
                [make_navigation_target("Part One", "Text/one.xhtml", 0)],
            ),
            EpubSourceChapter(
                "two",
                "Text/two.xhtml",
                "application/xhtml+xml",
                "<html><body><p>Second document.</p></body></html>",
            ),
            EpubSourceChapter(
                "three",
                "Text/three.xhtml",
                "application/xhtml+xml",
                "<html><body><p>Third document.</p></body></html>",
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual(len(result.chapters), 1)
        self.assertEqual(result.chapters[0].title, "Part One")
        self.assertEqual(result.chapters[0].text, "First document.\n\n\nSecond document.\n\n\nThird document.")
        self.assertEqual(result.section_start_indices, [])

    def test_assemble_navigation_text_chapters_separates_merged_slices_with_two_blank_lines(self):
        source_chapter = EpubSourceChapter(
            "Fallback",
            "Text/story.xhtml",
            "application/xhtml+xml",
            "<html><body><p>Ignored.</p></body></html>",
        )
        result = EpubTextExtractionResult(
            text="First part.\n\nSecond part.",
            slices=[
                EpubTextSlice("First part.", None),
                EpubTextSlice("Second part.", None),
            ],
        )

        text_chapters, used_navigation = EpubExtractor.assemble_navigation_text_chapters(
            [(source_chapter, result)],
            [],
            [],
        )

        self.assertEqual([chapter.text for chapter in text_chapters], ["First part.\n\n\nSecond part."])
        self.assertFalse(used_navigation)

    def test_import_epub_keeps_readable_content_before_first_navigation_target(self):
        source_chapters = [
            EpubSourceChapter(
                "Preface",
                "Text/preface.xhtml",
                "application/xhtml+xml",
                "<html><body><p>Preface prose.</p></body></html>",
            ),
            EpubSourceChapter(
                "Chapter One",
                "Text/chapter.xhtml",
                "application/xhtml+xml",
                "<html><body><h1>Chapter One</h1><p>Chapter prose.</p></body></html>",
                [make_navigation_target("Chapter One", "Text/chapter.xhtml", 0)],
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.title for chapter in result.chapters], ["Preface", "Chapter One"])
        self.assertEqual(result.chapters[0].text, "Preface prose.")
        self.assertEqual(len(result.section_start_indices), 1)

    def test_import_epub_uses_latest_consecutive_empty_navigation_target_for_prose(self):
        source_chapters = [
            EpubSourceChapter(
                "First",
                "Text/first.xhtml",
                "application/xhtml+xml",
                '<html><body><img src="first.jpg"/></body></html>',
                [make_navigation_target("First", "Text/first.xhtml", 0)],
            ),
            EpubSourceChapter(
                "Second",
                "Text/second.xhtml",
                "application/xhtml+xml",
                '<html><body><img src="second.jpg"/></body></html>',
                [make_navigation_target("Second", "Text/second.xhtml", 1)],
            ),
            EpubSourceChapter(
                "prose",
                "Text/prose.xhtml",
                "application/xhtml+xml",
                "<html><body><p>Readable prose.</p></body></html>",
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.title for chapter in result.chapters], ["Second"])
        self.assertEqual(result.chapters[0].text, "Readable prose.")
        self.assertTrue(any("Omitted empty EPUB navigation section" in warning for warning in result.warnings))
        self.assertTrue(any("Omitted empty EPUB navigation section" in warning for warning in result.significant_warnings))

    def test_import_epub_falls_back_to_spine_sections_when_fragment_targets_are_unusable(self):
        source_chapters = [
            EpubSourceChapter(
                "One",
                "Text/one.xhtml",
                "application/xhtml+xml",
                "<html><body><p>First.</p></body></html>",
                [make_navigation_target("Broken", "Text/one.xhtml#missing", 0)],
            ),
            EpubSourceChapter(
                "Two",
                "Text/two.xhtml",
                "application/xhtml+xml",
                "<html><body><p>Second.</p></body></html>",
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.title for chapter in result.chapters], ["One", "Two"])
        self.assertTrue(any("navigation target not found" in warning for warning in result.significant_warnings))
        self.assertTrue(any("derived from spine documents" in warning for warning in result.significant_warnings))

    def test_import_epub_falls_back_globally_when_later_fragment_target_is_unresolved(self):
        source_chapters = [
            EpubSourceChapter(
                "One",
                "Text/one.xhtml",
                "application/xhtml+xml",
                '<html><body><h1 id="start">One</h1><p>First.</p></body></html>',
                [make_navigation_target("One", "Text/one.xhtml#start", 0)],
            ),
            EpubSourceChapter(
                "Two",
                "Text/two.xhtml",
                "application/xhtml+xml",
                "<html><body><h1>Two</h1><p>Second.</p></body></html>",
                [make_navigation_target("Two", "Text/two.xhtml#missing", 1)],
            ),
        ]

        with patch.object(EpubExtractor, "load_source_chapters", return_value=(source_chapters, "", [], [])):
            result = EpubExtractor.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertEqual([chapter.title for chapter in result.chapters], ["One", "Two"])
        self.assertNotIn("Second.", result.chapters[0].text)
        self.assertIn("Second.", result.chapters[1].text)
        self.assertTrue(any("navigation target not found" in warning for warning in result.significant_warnings))
        self.assertTrue(any("derived from spine documents" in warning for warning in result.significant_warnings))

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
        navigation_targets = EpubExtractor.extract_navigation_targets(book)

        self.assertEqual(title_by_href["Text/part0001.xhtml"], "Part One")
        self.assertEqual(title_by_href["Text/chapter 02.xhtml"], "Chapter Two")
        self.assertEqual(
            [(target.path, target.fragment, target.title) for target in navigation_targets],
            [
                ("Text/part0001.xhtml", "chapter", "Part One"),
                ("Text/parent.xhtml", "", "Parent"),
                ("Text/chapter 02.xhtml", "start", "Chapter Two"),
            ],
        )

    def test_extract_navigation_targets_resolves_ncx_links_relative_to_ncx_resource(self):
        book = StubEpubBookWithSpine(
            items=[
                StubEpubSpineItem(
                    "ncx",
                    "Navigation/toc.ncx",
                    "",
                    media_type="application/x-dtbncx+xml",
                ),
            ],
            toc=[StubTocItem("Chapter One", "../Text/chapter1.xhtml#start")],
        )

        targets = EpubExtractor.extract_navigation_targets(book)

        self.assertEqual(targets[0].path, "Text/chapter1.xhtml")
        self.assertEqual(targets[0].fragment, "start")

    def test_make_navigation_target_normalizes_relative_encoded_path_query_and_fragment(self):
        target = EpubExtractor.make_navigation_target(
            "./Text/../Text/chapter%2002.xhtml?edition=1#chapter%20start",
            "Chapter Two",
            3,
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.path, "Text/chapter 02.xhtml")
        self.assertEqual(target.fragment, "chapter start")
        self.assertEqual(target.href, "Text/chapter 02.xhtml#chapter start")
        self.assertIsNone(EpubExtractor.make_navigation_target("https://example.com/chapter", "External", 4))

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
        self.assertEqual(source_chapters[0].navigation_targets[0].fragment, "start")

    def test_load_source_chapters_skips_copyright_pushed_beyond_front_window_by_image_inserts(self):
        # Mirrors a light novel spine: cover, six image-only color inserts, title page, then the
        # copyright page at raw readable index 8. Image-only pages must not consume front matter
        # scan budget, so the labeled copyright page is still recognized as front matter.
        copyright_html = (
            "<html><body>"
            "<h1>Copyright</h1>"
            "<p>The Eminence in Shadow 06</p>"
            "<p>DAISUKE AIZAWA</p>"
            "<p>Copyright © 2025 by Yen Press, LLC</p>"
            "<p>All rights reserved.</p>"
            "<p>First published in Japan in 2023 by KADOKAWA CORPORATION, Tokyo.</p>"
            "<p>Library of Congress Cataloging-in-Publication Data</p>"
            "<p>ISBNs: 979-8-8554-0698-6 (hardcover)</p>"
            "</body></html>"
        )
        items = [StubEpubSpineItem("cover", "Text/cover.xhtml", '<html><body><img src="cover.jpg"/></body></html>')]
        items += [
            StubEpubSpineItem(
                f"insert{i:03d}",
                f"Text/insert{i:03d}.xhtml",
                f'<html><body><img src="insert{i:03d}.jpg"/></body></html>',
            )
            for i in range(1, 7)
        ]
        items += [
            StubEpubSpineItem("titlepage", "Text/titlepage.xhtml", "<html><body><p>The Eminence in Shadow 6</p></body></html>"),
            StubEpubSpineItem("copyright", "Text/copyright.xhtml", copyright_html),
            StubEpubSpineItem("chapter001", "Text/chapter001.xhtml", "<html><body><h1>Prologue</h1><p>Story text begins.</p></body></html>"),
        ]
        book = StubEpubBookWithSpine(
            items=items,
            toc=[
                StubTocItem("Cover", "Text/cover.xhtml"),
                StubTocItem("Title Page", "Text/titlepage.xhtml"),
                StubTocItem("Copyright", "Text/copyright.xhtml"),
                StubTocItem("Prologue", "Text/chapter001.xhtml"),
            ],
        )

        with patch("os.path.exists", return_value=True), \
                patch("importlib.import_module") as import_module, \
                patch.object(EpubExtractor, "extract_title", return_value="Fallback Title"):
            import_module.return_value.read_epub.return_value = book
            source_chapters, _, warnings, significant_warnings = EpubExtractor.load_source_chapters("book.epub")

        kept_hrefs = [chapter.href for chapter in source_chapters]
        self.assertNotIn("Text/copyright.xhtml", kept_hrefs)
        self.assertIn("Text/titlepage.xhtml", kept_hrefs)
        self.assertIn("Text/chapter001.xhtml", kept_hrefs)
        skip_warning = next(warning for warning in warnings if "copyright.xhtml" in warning)
        self.assertIn("publication metadata", skip_warning)
        self.assertIn(skip_warning, significant_warnings)

    def test_load_source_chapters_warns_for_navigation_target_outside_spine(self):
        book = StubEpubBookWithSpine(
            items=[
                StubEpubSpineItem(
                    "chapter1",
                    "Text/chapter1.xhtml",
                    "<html><body><p>Body text.</p></body></html>",
                ),
            ],
            toc=[StubTocItem("Missing", "Text/missing.xhtml")],
        )

        with patch("os.path.exists", return_value=True), \
                patch("importlib.import_module") as import_module, \
                patch.object(EpubExtractor, "extract_title", return_value="Chapter One"):
            import_module.return_value.read_epub.return_value = book
            _, _, warnings, significant_warnings = EpubExtractor.load_source_chapters("book.epub")

        self.assertTrue(any("does not map to a spine document" in warning for warning in warnings))
        self.assertTrue(any("does not map to a spine document" in warning for warning in significant_warnings))

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

    def test_copy_epub_to_project_copies_epub_from_different_location(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as project_dir:
            epub_path = os.path.join(source_dir, "book.epub")
            with open(epub_path, "wb") as f:
                f.write(b"epub bytes")

            err = EpubExtractor.copy_epub_to_project(epub_path, project_dir)

            self.assertEqual(err, "")
            dest_path = os.path.join(project_dir, "project_text.epub")
            with open(dest_path, "rb") as f:
                self.assertEqual(f.read(), b"epub bytes")

    def test_copy_epub_to_project_skips_copy_when_source_is_saved_destination(self):
        with tempfile.TemporaryDirectory() as project_dir:
            dest_path = os.path.join(project_dir, "project_text.epub")
            with open(dest_path, "wb") as f:
                f.write(b"saved epub bytes")

            err = EpubExtractor.copy_epub_to_project(dest_path, project_dir)

            self.assertEqual(err, "")
            with open(dest_path, "rb") as f:
                self.assertEqual(f.read(), b"saved epub bytes")

    def test_copy_epub_to_project_skips_copy_for_relative_path_to_saved_destination(self):
        with tempfile.TemporaryDirectory() as project_dir:
            old_cwd = os.getcwd()
            os.chdir(project_dir)
            try:
                dest_path = os.path.join(project_dir, "project_text.epub")
                with open(dest_path, "wb") as f:
                    f.write(b"saved epub bytes")

                err = EpubExtractor.copy_epub_to_project(os.path.join(".", "project_text.epub"), project_dir)

                self.assertEqual(err, "")
                with open(dest_path, "rb") as f:
                    self.assertEqual(f.read(), b"saved epub bytes")
            finally:
                os.chdir(old_cwd)

    def test_copy_epub_to_project_skips_copy_for_differing_paths_to_same_file(self):
        with tempfile.TemporaryDirectory() as project_dir:
            dest_path = os.path.join(project_dir, "project_text.epub")
            with open(dest_path, "wb") as f:
                f.write(b"saved epub bytes")
            linked_path = os.path.join(project_dir, "linked.epub")
            os.symlink(dest_path, linked_path)

            err = EpubExtractor.copy_epub_to_project(linked_path, project_dir)

            self.assertEqual(err, "")
            self.assertTrue(os.path.islink(linked_path))
            with open(dest_path, "rb") as f:
                self.assertEqual(f.read(), b"saved epub bytes")


if __name__ == "__main__":
    unittest.main()