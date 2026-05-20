import os
import tempfile
import unittest

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import (
    BOOK_FORMAT,
    book_from_project_text_json_dict,
    book_to_project_text_json_dict,
    load_book_from_project_text_file,
    save_book_to_project_text_file,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason


class TestBookSerialization(unittest.TestCase):
    def make_phrase_group(self, text: str, reason: Reason = Reason.SENTENCE) -> PhraseGroup:
        return PhraseGroup([Phrase(text, reason)])

    def test_book_round_trip_preserves_metadata_sections_and_phrase_groups(self):
        book = Book(
            title="Example Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            segmentation_settings=BookSegmentationSettings(
                language_code="en",
                max_words_per_segment=120,
                strategy=SegmentationStrategy.MULTI_SENTENCE,
            ),
            sections=[
                BookSection(
                    title="Chapter 1",
                    phrase_groups=[self.make_phrase_group("Opening prose.", Reason.SENTENCE)],
                ),
                BookSection(
                    title="Chapter 2",
                    phrase_groups=[self.make_phrase_group("Next chapter.\n\n", Reason.PARAGRAPH)],
                ),
            ],
        )

        payload = book_to_project_text_json_dict(book)
        result = book_from_project_text_json_dict(payload)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(payload["format"], BOOK_FORMAT)
        self.assertEqual(result.title, "Example Book")
        self.assertEqual(result.text_source_kind, "epub")
        self.assertEqual(result.audio_source_kind, "generated")
        self.assertEqual(result.segmentation_settings.language_code, "en")
        self.assertEqual(result.segmentation_settings.max_words_per_segment, 120)
        self.assertEqual(result.segmentation_settings.strategy, SegmentationStrategy.MULTI_SENTENCE)
        self.assertEqual([section.title for section in result.sections], ["Chapter 1", "Chapter 2"])
        self.assertEqual([group.text for group in result.phrase_groups()], ["Opening prose.", "Next chapter.\n\n"])
        self.assertEqual(result.phrase_groups()[1].last_reason, Reason.PARAGRAPH)
        self.assertEqual(result.section_start_indices(), [0, 1])

    def test_phrase_groups_v1_deserializes_to_legacy_flat_book(self):
        phrase_groups = [
            self.make_phrase_group("One.", Reason.SENTENCE),
            self.make_phrase_group("Two.", Reason.SENTENCE),
        ]
        payload = {
            "format": "phrase_groups.v1",
            "phrase_groups": PhraseGroup.phrase_groups_to_json_list(phrase_groups),
        }
        settings = BookSegmentationSettings(
            language_code="en",
            max_words_per_segment=80,
            strategy=SegmentationStrategy.MAX_LEN,
        )

        result = book_from_project_text_json_dict(payload, settings)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.text_source_kind, "legacy_flat")
        self.assertEqual(result.audio_source_kind, "unknown")
        self.assertEqual(result.segmentation_settings, settings)
        self.assertEqual(len(result.sections), 1)
        self.assertEqual([group.text for group in result.sections[0].phrase_groups], ["One.", "Two."])

    def test_bare_phrase_group_list_deserializes_to_legacy_flat_book(self):
        phrase_groups = [self.make_phrase_group("Bare list.", Reason.SENTENCE)]
        payload = PhraseGroup.phrase_groups_to_json_list(phrase_groups)

        result = book_from_project_text_json_dict(payload)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.text_source_kind, "legacy_flat")
        self.assertEqual(len(result.sections), 1)
        self.assertEqual(result.phrase_groups()[0].text, "Bare list.")

    def test_invalid_payload_returns_error_string(self):
        result = book_from_project_text_json_dict({"format": "book.v1"})

        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("missing 'book'", result)

    def test_save_and_load_book_project_text_file(self):
        book = Book(
            title="File Round Trip",
            sections=[BookSection(phrase_groups=[self.make_phrase_group("From disk.")])],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "project_text.json")
            err = save_book_to_project_text_file(path, book)
            self.assertEqual(err, "")

            result = load_book_from_project_text_file(path)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.title, "File Round Trip")
        self.assertEqual(result.phrase_groups()[0].text, "From disk.")


if __name__ == "__main__":
    unittest.main()