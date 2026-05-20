import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tts_audiobook_tool.app_types.app_metadata import AppMetadata, AppMetadataSection
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.concat_util import make_app_metadata_sections, save_abr_metadata_debug_json
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.app_types import Book, BookSection


class TestAppMetadata(unittest.TestCase):
    def make_phrase_group(self, text: str) -> PhraseGroup:
        return PhraseGroup([Phrase(text, Reason.SENTENCE)])

    def test_app_metadata_round_trip_preserves_sections(self):
        meta = AppMetadata(
            timed_phrases=[TimedPhrase.make_using(Phrase("One.", Reason.SENTENCE), 0.0, 1.0)],
            version=3,
            bookmark_indices=[0],
            raw_text="One.",
            has_section_break_audio=False,
            project_snapshot={},
            sections=[AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)],
        )

        json_string = meta.to_json_string()
        result = AppMetadata.get_from_json_string(json_string)

        self.assertIsInstance(result, AppMetadata)
        assert isinstance(result, AppMetadata)
        self.assertEqual(result.sections, [AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)])

        payload = json.loads(json_string)
        self.assertNotIn("raw_text", payload)
        self.assertEqual(payload["sections"][0]["title"], "Chapter 1")

    def test_app_metadata_parses_legacy_raw_text_and_missing_sections(self):
        payload = {
            "version": 2,
            "raw_text": "eJzzSM3JyVcozy_KSVEEAB0JBF4=",
            "bookmarks": [0],
            "text_segments": [{"text": "Hello.", "time_start": 0.0, "time_end": 1.0}],
            "has_section_break_audio": False,
            "project_snapshot": {},
        }

        result = AppMetadata.get_from_json_string(json.dumps(payload))

        self.assertIsInstance(result, AppMetadata)
        assert isinstance(result, AppMetadata)
        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.sections, [])

    def test_app_metadata_rejects_bad_sections_type(self):
        payload = {
            "version": 3,
            "raw_text": "eJzzSM3JyVcozy_KSVEEAB0JBF4=",
            "bookmarks": [0],
            "text_segments": [{"text": "Hello.", "time_start": 0.0, "time_end": 1.0}],
            "has_section_break_audio": False,
            "project_snapshot": {},
            "sections": {},
        }

        result = AppMetadata.get_from_json_string(json.dumps(payload))

        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("sections", result)

    def test_save_abr_metadata_debug_json_writes_standalone_payload(self):
        meta = AppMetadata(
            timed_phrases=[TimedPhrase.make_using(Phrase("One.", Reason.SENTENCE), 0.0, 1.0)],
            version=3,
            bookmark_indices=[0],
            raw_text="One.",
            has_section_break_audio=False,
            project_snapshot={"voice": "test"},
            sections=[AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.abr.metadata.json"
            err = save_abr_metadata_debug_json(meta, str(path))

            self.assertEqual(err, "")
            self.assertTrue(path.exists())

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 3)
            self.assertNotIn("raw_text", payload)
            self.assertEqual(payload["sections"], [{"title": "Chapter 1", "start_index": 0, "end_index": 1}])
            self.assertEqual(payload["text_segments"], [{"text": "One.", "time_start": 0.0, "time_end": 1.0}])

    def test_make_app_metadata_sections_rebases_and_filters_to_overlapping_sections(self):
        project = Project.model_validate({
            "book": Book(
                sections=[
                    BookSection(title="A", phrase_groups=[self.make_phrase_group("A1")]),
                    BookSection(title="B", phrase_groups=[self.make_phrase_group("B1")]),
                    BookSection(title="C", phrase_groups=[self.make_phrase_group("C1")]),
                ]
            )
        })

        with patch.object(Project, "get_section_ranges", return_value=[(0, 2), (2, 5), (5, 7)]):
            result = make_app_metadata_sections(
                project=project,
                index_start=1,
                index_end=4,
                phrase_to_text_segment_start_indices=[0, 1, 2, 3, 4, 5, 6],
                text_segment_count=7,
            )

        self.assertEqual(result, [
            AppMetadataSection(title="A", start_index=1, end_index=2),
            AppMetadataSection(title="B", start_index=2, end_index=5),
        ])

    def test_make_app_metadata_sections_uses_subdivided_indices(self):
        project = Project.model_validate({
            "book": Book(sections=[BookSection(title="Chapter 1", phrase_groups=[
                self.make_phrase_group("One."),
                self.make_phrase_group("Two."),
            ])]),
        })

        with patch.object(Project, "get_section_ranges", return_value=[(0, 2)]):
            result = make_app_metadata_sections(
                project=project,
                index_start=0,
                index_end=1,
                phrase_to_text_segment_start_indices=[0, 3],
                text_segment_count=5,
            )

        self.assertEqual(result, [
            AppMetadataSection(title="Chapter 1", start_index=0, end_index=5),
        ])


if __name__ == "__main__":
    unittest.main()