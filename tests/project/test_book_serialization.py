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
    def make_phrase_group(
            self,
            text: str,
            reason: Reason = Reason.SENTENCE,
            voice_index: int = 0,
    ) -> PhraseGroup:
        return PhraseGroup([Phrase(text, reason)], voice_index=voice_index)

    def legacy_phrase_groups_json(self, phrase_groups: list[PhraseGroup]) -> list[list[dict]]:
        return [group.to_json_dict_list() for group in phrase_groups]

    def test_book_round_trip_preserves_metadata_sections_and_phrase_groups(self):
        book = Book(
            title="Example Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            segmentation_settings=BookSegmentationSettings(
                language_code="en",
                max_words_per_segment=120,
                strategy=SegmentationStrategy.MULTI_SENTENCE,
                dialog_segmentation=True,
            ),
            sections=[
                BookSection(
                    title="Chapter 1",
                    phrase_groups=[self.make_phrase_group("Opening prose.", Reason.SENTENCE)],
                ),
                BookSection(
                    title="Chapter 2",
                    phrase_groups=[self.make_phrase_group("Next chapter.\n\n", Reason.PARAGRAPH, voice_index=2)],
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
        self.assertEqual(result.segmentation_settings.dialog_segmentation, True)
        self.assertEqual([section.title for section in result.sections], ["Chapter 1", "Chapter 2"])
        self.assertEqual([group.text for group in result.phrase_groups], ["Opening prose.", "Next chapter.\n\n"])
        self.assertEqual([group.voice_index for group in result.phrase_groups], [0, 2])
        self.assertEqual(result.phrase_groups[1].last_reason, Reason.PARAGRAPH)
        self.assertEqual(result.section_start_indices(), [0, 1])

    def test_load_promotes_legacy_ornamental_paragraph_to_space_break(self):
        payload = {
            "format": "book.v2",
            "book": {
                "sections": [{
                    "phrase_groups": [{
                        "voice_index": -1,
                        "phrases": [{
                            "text": "Take good care of Traitre in my absence.\n\n◇\n\n",
                            "reason": "p",
                        }],
                    }],
                }],
            },
        }

        result = book_from_project_text_json_dict(payload)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.phrase_groups[0].last_reason, Reason.SPACE_BREAK)

    def test_load_keeps_paragraph_reason_for_trailing_quoted_beat(self):
        # The legacy ornament repair must not mistake a trailing silent dialog
        # beat (`"..."`) for an ornamental scene break.
        payload = {
            "format": "book.v2",
            "book": {
                "sections": [{
                    "phrase_groups": [{
                        "voice_index": -1,
                        "phrases": [{"text": '"Well?"\n\n"..."\n\n', "reason": "p"}],
                    }],
                }],
            },
        }

        result = book_from_project_text_json_dict(payload)

        assert isinstance(result, Book)
        self.assertEqual(result.phrase_groups[0].last_reason, Reason.PARAGRAPH)

    def test_load_keeps_regular_paragraph_reason(self):
        payload = {
            "format": "book.v2",
            "book": {
                "sections": [{
                    "phrase_groups": [{
                        "voice_index": -1,
                        "phrases": [{"text": "Ordinary paragraph.\n\n", "reason": "p"}],
                    }],
                }],
            },
        }

        result = book_from_project_text_json_dict(payload)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.phrase_groups[0].last_reason, Reason.PARAGRAPH)

    def test_phrase_group_voice_index_defaults_to_first_voice_and_has_setter(self):
        group = PhraseGroup([Phrase("One.", Reason.SENTENCE)])

        self.assertEqual(group.voice_index, 0)
        group.voice_index = 3
        self.assertEqual(group.voice_index, 3)

    def test_book_v3_serializes_phrase_groups_as_objects(self):
        book = Book(sections=[BookSection(phrase_groups=[self.make_phrase_group("One.", voice_index=4)])])

        payload = book_to_project_text_json_dict(book)

        self.assertEqual(payload["format"], "book.v3")
        group_payload = payload["book"]["sections"][0]["phrase_groups"][0]
        self.assertEqual(group_payload["voice_index"], 4)
        self.assertEqual(group_payload["phrases"][0]["text"], "One.")
        # The dialog flag is written only when set
        self.assertNotIn("is_dialog", group_payload)

    def test_dialog_flag_round_trips_independently_of_voice_index(self):
        dialog = PhraseGroup([Phrase('"Hi."', Reason.SENTENCE)], voice_index=0, is_dialog=True)
        narration = PhraseGroup([Phrase("Narration.", Reason.SENTENCE)], voice_index=2)
        book = Book(sections=[BookSection(phrase_groups=[dialog, narration])])

        payload = book_to_project_text_json_dict(book)
        result = book_from_project_text_json_dict(payload)

        self.assertIs(payload["book"]["sections"][0]["phrase_groups"][0]["is_dialog"], True)
        assert isinstance(result, Book)
        self.assertEqual([group.is_dialog for group in result.phrase_groups], [True, False])
        self.assertEqual([group.voice_index for group in result.phrase_groups], [0, 2])

    def test_pre_v3_books_derive_dialog_flags_from_voice_index(self):
        # Before v3, dialog segmentation marked dialog by preassigning voice
        # index 1, and any voice index >= 1 counted as dialog. Loading derives
        # the flags the same way, but only for books segmented with dialog
        # segmentation; a stray is_dialog key in an old book is ignored.
        def payload(dialog_segmentation: bool) -> dict:
            return {
                "format": "book.v2",
                "book": {
                    "segmentation_settings": {"dialog_segmentation": dialog_segmentation},
                    "sections": [{"phrase_groups": [
                        {"voice_index": -1, "is_dialog": True, "phrases": [{"text": "A. ", "reason": "s"}]},
                        {"voice_index": 1, "phrases": [{"text": '"B." ', "reason": "s"}]},
                        {"voice_index": 3, "phrases": [{"text": '"C."', "reason": "s"}]},
                    ]}],
                },
            }

        with_dialog = book_from_project_text_json_dict(payload(True))
        without_dialog = book_from_project_text_json_dict(payload(False))

        assert isinstance(with_dialog, Book) and isinstance(without_dialog, Book)
        self.assertEqual([group.is_dialog for group in with_dialog.phrase_groups], [False, True, True])
        self.assertEqual([group.is_dialog for group in without_dialog.phrase_groups], [False, False, False])

    def test_legacy_phrase_groups_derive_dialog_flags_from_applied_settings(self):
        payload = {
            "format": "phrase_groups.v1",
            "phrase_groups": [
                {"voice_index": -1, "phrases": [{"text": "A. ", "reason": "s"}]},
                {"voice_index": 1, "phrases": [{"text": '"B."', "reason": "s"}]},
            ],
        }

        result = book_from_project_text_json_dict(payload, BookSegmentationSettings(dialog_segmentation=True))

        assert isinstance(result, Book)
        self.assertEqual([group.is_dialog for group in result.phrase_groups], [False, True])

    def test_book_v1_deserializes_legacy_group_arrays_with_default_voice_index(self):
        payload = {
            "format": "book.v1",
            "book": {
                "sections": [{
                    "phrase_groups": self.legacy_phrase_groups_json([self.make_phrase_group("Old.")]),
                }],
            },
        }

        result = book_from_project_text_json_dict(payload)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.phrase_groups[0].text, "Old.")
        self.assertEqual(result.phrase_groups[0].voice_index, 0)

    def test_legacy_unassigned_invalid_or_missing_voice_index_loads_as_zero(self):
        # Earlier versions stored -1 for "unassigned", which always meant the
        # first voice; it loads as 0, as do invalid and missing values.
        values = [-1, None, True, -2, 1.5, "2"]
        group_payloads = [
            {"phrases": [{"text": f"Value {index}.", "reason": "s"}], "voice_index": value}
            for index, value in enumerate(values)
        ]
        group_payloads.append({"phrases": [{"text": "Missing.", "reason": "s"}]})

        result = PhraseGroup.phrase_groups_from_json_list(group_payloads)

        self.assertIsInstance(result, list)
        assert isinstance(result, list)
        self.assertEqual([group.voice_index for group in result], [0] * len(group_payloads))

    def test_phrase_groups_v1_deserializes_to_legacy_flat_book(self):
        phrase_groups = [
            self.make_phrase_group("One.", Reason.SENTENCE),
            self.make_phrase_group("Two.", Reason.SENTENCE),
        ]
        payload = {
            "format": "phrase_groups.v1",
            "phrase_groups": self.legacy_phrase_groups_json(phrase_groups),
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
        payload = self.legacy_phrase_groups_json(phrase_groups)

        result = book_from_project_text_json_dict(payload)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.text_source_kind, "legacy_flat")
        self.assertEqual(len(result.sections), 1)
        self.assertEqual(result.phrase_groups[0].text, "Bare list.")

    def test_invalid_payload_returns_error_string(self):
        result = book_from_project_text_json_dict({"format": "book.v1"})

        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("missing 'book'", result)

    def test_book_v2_rejects_empty_phrase_group_with_specific_error(self):
        payload = {
            "format": "book.v2",
            "book": {
                "sections": [{
                    "phrase_groups": [{"voice_index": -1, "phrases": []}],
                }],
            },
        }

        result = book_from_project_text_json_dict(payload)

        self.assertEqual(
            result,
            "Book section 1: Phrase group 1 contains no phrases",
        )

    def test_phrase_quote_end_reason_round_trips_through_book_json(self):
        book = Book(
            sections=[
                BookSection(
                    phrase_groups=[
                        self.make_phrase_group(
                            'He said "fine" she replied.',
                            Reason.PHRASE_QUOTE_END,
                        ),
                    ],
                ),
            ],
        )

        payload = book_to_project_text_json_dict(book)
        phrase_payload = payload["book"]["sections"][0]["phrase_groups"][0]["phrases"][0]
        self.assertEqual(phrase_payload["reason"], "isqe")

        result = book_from_project_text_json_dict(payload)

        self.assertIsInstance(result, Book)
        assert isinstance(result, Book)
        self.assertEqual(result.phrase_groups[0].phrases[0].reason, Reason.PHRASE_QUOTE_END)
        self.assertEqual(result.phrase_groups[0].last_reason, Reason.PHRASE_QUOTE_END)

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
        self.assertEqual(result.phrase_groups[0].text, "From disk.")


if __name__ == "__main__":
    unittest.main()
