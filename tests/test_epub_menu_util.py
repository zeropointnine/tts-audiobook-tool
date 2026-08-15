import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.epub_menu_util import EpubMenuUtil
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult, EpubTextChapter


class TestEpubMenuUtil(unittest.TestCase):

    def test_ask_epub_path_accepts_epub_and_updates_last_text_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "book.epub")
            with open(epub_path, "w", encoding="utf-8") as file:
                file.write("stub")
            prefs = Prefs()

            # Persist prefs to the temp directory, never to the real user directory
            prefs_path = os.path.join(temp_dir, PREFS_FILE_NAME)
            with patch("tts_audiobook_tool.ask.ask_file_path", return_value=epub_path), \
                    patch.object(Prefs, "get_file_path", staticmethod(lambda: prefs_path)):
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

    def test_ask_epub_path_save_never_touches_real_user_prefs_file(self):
        # Regression: ask_epub_path() used to save a default Prefs() to the real user
        # prefs file in the home directory, resetting user preferences. Verify that a
        # save from this flow never creates or modifies the real user prefs file and
        # that the prefs path is isolated to a test temp directory (conftest fixture).
        from tts_audiobook_tool.app_support import app_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "book.epub")
            with open(epub_path, "w", encoding="utf-8") as file:
                file.write("stub")
            prefs = Prefs()

            # The prefs path must be isolated away from the home directory
            prefs_path = Path(Prefs.get_file_path())
            self.assertFalse(
                prefs_path.is_relative_to(Path.home()),
                "prefs path is not isolated from the home directory; "
                "the conftest app-user-dir fixture may not be in effect",
            )

            # Snapshot the real (home-directory) prefs file, if present
            real_prefs_file = Path.home() / app_paths.APP_USER_SUBDIR / PREFS_FILE_NAME
            real_existed = real_prefs_file.exists()
            real_content = real_prefs_file.read_text(encoding="utf-8") if real_existed else None

            with patch("tts_audiobook_tool.ask.ask_file_path", return_value=epub_path):
                EpubMenuUtil.ask_epub_path(prefs)

            # Real user prefs file must be untouched
            self.assertEqual(real_prefs_file.exists(), real_existed)
            if real_existed:
                self.assertEqual(real_prefs_file.read_text(encoding="utf-8"), real_content)

            # The save must have landed in the isolated directory
            self.assertTrue(prefs_path.exists())
            payload = json.loads(prefs_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["last_text_dir"], temp_dir)

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
