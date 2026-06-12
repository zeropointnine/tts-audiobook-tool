import os
import tempfile
import unittest
from unittest.mock import patch

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.epub_menu_util import EpubMenuUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult, EpubTextChapter


class TestEpubMenuUtil(unittest.TestCase):

    def test_ask_epub_path_accepts_epub_and_updates_last_text_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "book.epub")
            with open(epub_path, "w", encoding="utf-8") as file:
                file.write("stub")
            prefs = Prefs()

            with patch("tts_audiobook_tool.ask.ask_file_path", return_value=epub_path):
                path = EpubMenuUtil.ask_epub_path(prefs)

            self.assertEqual(path, epub_path)
            self.assertEqual(prefs.last_text_dir, temp_dir)

    def test_ask_epub_path_rejects_missing_file(self):
        prefs = Prefs()

        with patch("tts_audiobook_tool.ask.ask_file_path", return_value="/missing/book.epub"), \
                patch("tts_audiobook_tool.ask.ask_error") as ask_error:
            path = EpubMenuUtil.ask_epub_path(prefs)

        self.assertEqual(path, "")
        ask_error.assert_called_once_with("No such file")

    def test_ask_epub_path_rejects_non_epub_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = os.path.join(temp_dir, "book.txt")
            with open(text_path, "w", encoding="utf-8") as file:
                file.write("stub")
            prefs = Prefs()

            with patch("tts_audiobook_tool.ask.ask_file_path", return_value=text_path), \
                    patch("tts_audiobook_tool.ask.ask_error") as ask_error:
                path = EpubMenuUtil.ask_epub_path(prefs)

        self.assertEqual(path, "")
        ask_error.assert_called_once_with("Must select an .epub file")

    def test_import_epub_returns_result(self):
        expected = EpubImportResult(
            phrase_groups=[PhraseGroup([Phrase("Text.", Reason.SENTENCE)])],
            raw_text="Text.",
            section_start_indices=[],
            chapters=[EpubTextChapter("Chapter", "chapter.xhtml", "Text.")],
        )

        with patch("tts_audiobook_tool.text_ops.epub_extractor.EpubExtractor.import_epub", return_value=expected):
            result = EpubMenuUtil.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertIs(result, expected)

    def test_import_epub_reports_import_error(self):
        with patch("tts_audiobook_tool.text_ops.epub_extractor.EpubExtractor.import_epub", side_effect=ImportError("missing dep")), \
                patch("tts_audiobook_tool.ask.ask_error") as ask_error:
            result = EpubMenuUtil.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertIsNone(result)
        ask_error.assert_called_once_with("missing dep")

    def test_import_epub_reports_generic_error(self):
        with patch("tts_audiobook_tool.text_ops.epub_extractor.EpubExtractor.import_epub", side_effect=ValueError("bad epub")), \
                patch("tts_audiobook_tool.ask.ask_error") as ask_error:
            result = EpubMenuUtil.import_epub(
                epub_path="book.epub",
                max_words=40,
                segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
                language_code="en",
            )

        self.assertIsNone(result)
        ask_error.assert_called_once_with("Error importing EPUB: bad epub")

    def test_make_text_file_path_uses_unique_incrementing_path_next_to_epub(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "book.epub")
            text_path = os.path.join(temp_dir, "book.txt")
            first_increment_path = os.path.join(temp_dir, "book-1.txt")
            for path in [epub_path, text_path, first_increment_path]:
                with open(path, "w", encoding="utf-8") as file:
                    file.write("stub")

            unique_path = EpubMenuUtil.make_text_file_path(epub_path)

        self.assertEqual(unique_path, os.path.join(temp_dir, "book-2.txt"))

    def test_save_text_file_writes_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "book.txt")

            err = EpubMenuUtil.save_text_file("Text.", file_path)

            self.assertEqual(err, "")
            with open(file_path, "r", encoding="utf-8") as file:
                self.assertEqual(file.read(), "Text.")


if __name__ == "__main__":
    unittest.main()
