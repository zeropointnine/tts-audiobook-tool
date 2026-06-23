import unittest
from unittest.mock import patch

from tts_audiobook_tool.app_types import ConcreteWord, SegmentationStrategy, Word
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.enhance import enhance_flow
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult, EpubTextChapter


class TestEnhanceFlow(unittest.TestCase):

    def test_normalize_enhance_source_text_segments_plain_text(self):
        with patch("tts_audiobook_tool.enhance.enhance_flow.printt"):
            source_text = enhance_flow.normalize_enhance_source_text("First sentence. Second sentence.")

        self.assertEqual(source_text.raw_text, "First sentence. Second sentence.")
        self.assertEqual(source_text.source_kind, "text")
        self.assertEqual(source_text.section_ranges, [])
        self.assertGreaterEqual(len(source_text.phrases), 1)

    def test_make_section_ranges_from_group_starts_maps_groups_to_phrase_ranges(self):
        groups = [
            PhraseGroup([Phrase("A. ", Reason.SENTENCE), Phrase("B. ", Reason.SENTENCE)]),
            PhraseGroup([Phrase("C. ", Reason.SENTENCE)]),
            PhraseGroup([Phrase("D. ", Reason.SENTENCE), Phrase("E. ", Reason.SENTENCE)]),
        ]

        ranges = enhance_flow.make_section_ranges_from_group_starts(groups, [1, 2])

        self.assertEqual(ranges, [(0, 2), (2, 3), (3, 5)])

    def test_make_app_metadata_sections_uses_imported_titles_and_ranges(self):
        source_text = enhance_flow.EnhanceSourceText(
            raw_text="",
            phrases=[],
            source_kind="epub",
            section_ranges=[(0, 2), (2, 5)],
            section_titles=["One", "Two"],
        )

        sections = enhance_flow.make_app_metadata_sections(source_text, text_segment_count=4)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "One")
        self.assertEqual(sections[0].start_index, 0)
        self.assertEqual(sections[0].end_index, 2)
        self.assertEqual(sections[1].title, "Two")
        self.assertEqual(sections[1].start_index, 2)
        self.assertEqual(sections[1].end_index, 4)

    def test_load_source_text_for_enhance_uses_epub_importer(self):
        phrase_groups = [
            PhraseGroup([Phrase("First. ", Reason.SENTENCE)]),
            PhraseGroup([Phrase("Second. ", Reason.SENTENCE)]),
        ]
        epub_result = EpubImportResult(
            phrase_groups=phrase_groups,
            raw_text="First.\n\nSecond.",
            section_start_indices=[1],
            chapters=[
                EpubTextChapter("Chapter 1", "one.xhtml", "First."),
                EpubTextChapter("Chapter 2", "two.xhtml", "Second."),
            ],
            book_title="Book Title",
        )
        state = _StubState()

        with patch("tts_audiobook_tool.enhance.enhance_flow.EpubMenuUtil.import_epub", return_value=epub_result) as import_epub, \
                patch("tts_audiobook_tool.enhance.enhance_flow.EpubMenuUtil.print_import_info"):
            source_text = enhance_flow.load_source_text_for_enhance(state, "book.epub")

        self.assertIsNotNone(source_text)
        assert source_text is not None
        import_epub.assert_called_once_with(
            epub_path="book.epub",
            max_words=55,
            segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
            language_code="en",
        )
        self.assertEqual(source_text.source_kind, "epub")
        self.assertEqual(source_text.title, "Book Title")
        self.assertEqual(source_text.section_titles, ["Chapter 1", "Chapter 2"])
        self.assertEqual(source_text.section_ranges, [(0, 1), (1, 2)])
        self.assertEqual([phrase.text for phrase in source_text.phrases], ["First. ", "Second. "])

    def test_make_section_aware_timed_phrases_keeps_global_cursor_across_sections(self):
        source_text = enhance_flow.EnhanceSourceText(
            raw_text="Alpha.\n\nBeta.",
            phrases=[Phrase("Alpha.", Reason.SENTENCE), Phrase("Beta.", Reason.SENTENCE)],
            source_kind="epub",
            section_ranges=[(0, 1), (1, 2)],
            section_titles=["One", "Two"],
        )
        words: list[Word] = [
            ConcreteWord(0.0, 0.5, "alpha", 1.0),
            ConcreteWord(1.0, 1.5, "beta", 1.0),
        ]

        with patch("tts_audiobook_tool.enhance.enhance_flow.printt"):
            timed_phrases, did_interrupt = enhance_flow.make_section_aware_timed_phrases(source_text, words)

        self.assertFalse(did_interrupt)
        self.assertEqual(len(timed_phrases), 2)
        self.assertEqual(timed_phrases[0].time_start, 0.0)
        self.assertEqual(timed_phrases[0].time_end, 1.0)
        self.assertEqual(timed_phrases[1].time_start, 1.0)
        self.assertEqual(timed_phrases[1].time_end, 1.5)


class _StubProject:
    max_words = 55
    segmentation_strategy = SegmentationStrategy.SENTENCE_PLUS
    language_code = "en"


class _StubState:
    project = _StubProject()


if __name__ == "__main__":
    unittest.main()
