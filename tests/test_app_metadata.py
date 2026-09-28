import json
from pathlib import Path
import tempfile
import unittest
from typing import cast
from unittest.mock import patch
from types import SimpleNamespace

from tts_audiobook_tool.app_types.app_metadata import AppMetadata, AppMetadataSection
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.concat_util import ConcatUtil, make_app_metadata_sections, save_abr_metadata_debug_json
from tts_audiobook_tool.constants import ABR_VERSION
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types import ExportType, NormalizationType, SectionMarkerMode
from tts_audiobook_tool.state import State


class TestAppMetadata(unittest.TestCase):
    def make_phrase_group(self, text: str) -> PhraseGroup:
        return PhraseGroup([Phrase(text, Reason.SENTENCE)])

    def test_app_metadata_round_trip_preserves_sections(self):
        meta = AppMetadata(
            timed_phrases=[TimedPhrase.make_using(Phrase("One.", Reason.SENTENCE), 0.0, 1.0)],
            title="Example Book",
            version=4,
            bookmark_indices=[0],
            raw_text="One.",
            has_break_audio=False,
            project_snapshot={},
            sections=[AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)],
            type="generated",
        )

        json_string = meta.to_json_string()
        result = AppMetadata.get_from_json_string(json_string)

        self.assertIsInstance(result, AppMetadata)
        assert isinstance(result, AppMetadata)
        self.assertEqual(result.title, "Example Book")
        self.assertEqual(result.type, "generated")
        self.assertEqual(result.sections, [AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)])

        payload = json.loads(json_string)
        self.assertEqual(payload["title"], "Example Book")
        self.assertEqual(payload["type"], "generated")
        self.assertNotIn("raw_text", payload)
        self.assertEqual(payload["sections"][0]["title"], "Chapter 1")

    def test_app_metadata_round_trip_preserves_mixed_nested_segments(self):
        nested_singleton = [TimedPhrase("Nested singleton.", 1.0, 2.0)]
        nested_pair = [
            TimedPhrase("Nested one, ", 2.0, 2.5),
            TimedPhrase("nested two.", 2.5, 3.0),
        ]
        meta = AppMetadata(
            timed_phrases=[TimedPhrase("Flat.", 0.0, 1.0), nested_singleton, nested_pair],
            title="Nested",
            version=4,
            bookmark_indices=[0, 2],
            raw_text="",
            has_break_audio=False,
            project_snapshot={},
            sections=[AppMetadataSection(title="All", start_index=0, end_index=4)],
        )

        json_string = meta.to_json_string()
        payload = json.loads(json_string)
        self.assertIsInstance(payload["text_segments"][0], dict)
        self.assertEqual(len(payload["text_segments"][1]), 1)
        self.assertEqual(len(payload["text_segments"][2]), 2)

        result = AppMetadata.get_from_json_string(json_string)
        self.assertIsInstance(result, AppMetadata)
        assert isinstance(result, AppMetadata)
        self.assertIsInstance(result.timed_phrases[0], TimedPhrase)
        self.assertIsInstance(result.timed_phrases[1], list)
        self.assertIsInstance(result.timed_phrases[2], list)
        assert isinstance(result.timed_phrases[1], list)
        self.assertEqual(result.timed_phrases[1][0].text, "Nested singleton.")
        self.assertEqual(json.loads(result.to_json_string())["text_segments"], payload["text_segments"])

    def test_app_metadata_rejects_invalid_nested_segments(self):
        segment = {"text": "Hello.", "time_start": 0.0, "time_end": 1.0}
        invalid_values = [
            [[]],
            [[[segment]]],
            [["not an object"]],
            [[{"text": 123, "time_start": 0.0, "time_end": 1.0}]],
        ]

        for text_segments in invalid_values:
            with self.subTest(text_segments=text_segments):
                result = AppMetadata.get_from_json_string(json.dumps({
                    "version": 4,
                    "text_segments": text_segments,
                }))
                self.assertIsInstance(result, str)

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
        self.assertEqual(result.title, "")
        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.sections, [])
        self.assertIsNone(result.type)
        self.assertNotIn("type", json.loads(result.to_json_string()))

    def test_snapshot_written_by_another_os_still_loads_and_repoints_dir(self):
        """
        A version 4 snapshot embeds `dir_path`, an absolute path in the writing
        machine's grammar. It must still parse, its saved voice references must
        be reduced to the app's portable form, and the new project must point at
        its own directory rather than the one recorded on the other machine.
        """
        snapshot = {
            "version": 2,
            "dir_path": "C:\\Users\\lee\\mybook",
            "fish_s2_voice_file_name": [
                "C:\\Users\\lee\\mybook\\voice\\narrator.flac",
                "voice/other.flac",
            ],
            "none_voice_file_name": "C:\\Users\\lee\\mybook\\voice\\none.flac",
        }

        with tempfile.TemporaryDirectory() as tmp:
            project = ProjectTransferUtil.make_project_from_snapshot(tmp, snapshot)

            self.assertEqual(project.dir_path, tmp)
            self.assertEqual(project.fish_s2_voice_file_name, ["narrator.flac", "voice/other.flac"])
            self.assertEqual(project.none_voice_file_name, "none.flac")

    def test_snapshot_dict_omits_machine_local_dir_path(self):
        """
        The snapshot embedded in an ABR file is the portable artifact, so it
        must not carry a machine-local directory. The directory is kept only as
        a display string, and `project.json` still records it for older builds.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project = Project(dir_path=tmp)
            project.fish_s2_voice_file_name = ["voice/narrator.flac"]

            snapshot = ProjectSerializationUtil.to_snapshot_dict(project)
            project_json = ProjectSerializationUtil.to_project_json_dict(project)

            self.assertNotIn("dir_path", snapshot)
            self.assertEqual(snapshot["source_dir_display"], tmp)
            self.assertEqual(project_json["dir_path"], tmp)

            reloaded = ProjectTransferUtil.make_project_from_snapshot(tmp, snapshot)
            self.assertEqual(reloaded.fish_s2_voice_file_name, ["voice/narrator.flac"])
            self.assertEqual(reloaded.dir_path, tmp)

    def test_app_metadata_preserves_conversion_type_and_rejects_invalid_values(self):
        payload = {
            "version": 4,
            "text_segments": [{"text": "Hello.", "time_start": 0.0, "time_end": 1.0}],
            "type": "conversion",
        }
        result = AppMetadata.get_from_json_string(json.dumps(payload))
        self.assertIsInstance(result, AppMetadata)
        assert isinstance(result, AppMetadata)
        self.assertEqual(result.type, "conversion")
        self.assertEqual(json.loads(result.to_json_string())["type"], "conversion")

        for invalid_type in (None, "other", 3, True):
            with self.subTest(invalid_type=invalid_type):
                payload["type"] = invalid_type
                error = AppMetadata.get_from_json_string(json.dumps(payload))
                self.assertIsInstance(error, str)
                self.assertIn("type", error)

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
            title="Example Book",
            version=4,
            bookmark_indices=[0],
            raw_text="One.",
            has_break_audio=False,
            project_snapshot={"voice": "test"},
            sections=[AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.abr.metadata.json"
            err = save_abr_metadata_debug_json(meta, str(path))

            self.assertEqual(err, "")
            self.assertTrue(path.exists())

            # Title/sections/raw_text payload shape is already covered by
            # test_app_metadata_round_trip_preserves_sections; this test pins
            # the fields only this file adds beyond that.
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 4)
            self.assertEqual(payload["text_segments"], [{"text": "One.", "time_start": 0.0, "time_end": 1.0}])
            self.assertEqual(payload["project_snapshot"], {"voice": "test"})

    def test_make_app_metadata_sections_preserves_all_sections_for_split_exports(self):
        project = Project.model_validate({
            "book": Book(
                sections=[
                    BookSection(title="A", phrase_groups=[self.make_phrase_group("A1")]),
                    BookSection(title="B", phrase_groups=[self.make_phrase_group("B1")]),
                    BookSection(title="C", phrase_groups=[self.make_phrase_group("C1")]),
                ]
            )
        })

        with patch("tts_audiobook_tool.concat_util.ProjectBookUtil.get_section_ranges", return_value=[(0, 2), (2, 5), (5, 7)]):
            result = make_app_metadata_sections(
                project=project,
                index_start=1,
                index_end=4,
                phrase_to_text_segment_start_indices=[0, 1, 2, 3, 4, 5, 6],
                text_segment_count=7,
            )

        self.assertEqual(result, [
            AppMetadataSection(title="A", start_index=0, end_index=2),
            AppMetadataSection(title="B", start_index=2, end_index=5),
            AppMetadataSection(title="C", start_index=5, end_index=7),
        ])

    def test_make_app_metadata_sections_uses_subdivided_indices(self):
        project = Project.model_validate({
            "book": Book(sections=[BookSection(title="Chapter 1", phrase_groups=[
                self.make_phrase_group("One."),
                self.make_phrase_group("Two."),
            ])]),
        })
        with patch("tts_audiobook_tool.concat_util.ProjectBookUtil.get_section_ranges", return_value=[(0, 2)]):
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

    def test_concat_make_file_writes_chapter_metadata_before_abr_metadata_for_marker_chapters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Project.model_validate({
                "dir_path": temp_dir,
                "book": Book(sections=[BookSection(title="Only Section", phrase_groups=[
                    self.make_phrase_group("Chapter One"),
                    self.make_phrase_group("Text one."),
                    self.make_phrase_group("Chapter Two"),
                    self.make_phrase_group("Text two."),
                ])]),
                "markers": [2],
                "chapter_mode": SectionMarkerMode.BOOKMARKS.id,
            })
            project.export_type = ExportType.AAC
            project.normalization_type = NormalizationType.DISABLED
            project.use_break_sound_effect = False
            project._sound_segments = SimpleNamespace(get_best_file_for=lambda _: "segment.flac")

            state = SimpleNamespace(
                project=project,
                prefs=SimpleNamespace(aac_bitrate="128k", save_debug_files=False),
            )

            phrases_and_paths = [
                (Phrase("Chapter One", Reason.SENTENCE), "/tmp/one.flac", True),
                (Phrase("Text one.", Reason.SENTENCE), "/tmp/two.flac", False),
                (Phrase("Chapter Two", Reason.SENTENCE), "/tmp/three.flac", True),
                (Phrase("Text two.", Reason.SENTENCE), "/tmp/four.flac", False),
            ]
            sections = [AppMetadataSection(title="Only Section", start_index=0, end_index=4)]
            stem_path = str(Path(temp_dir) / "book")

            with patch("tts_audiobook_tool.concat_util.ProjectTextIOUtil.load_raw_text", return_value="raw"), \
                 patch.object(ConcatUtil, "make_phrases_and_paths", return_value=phrases_and_paths), \
                 patch.object(ConcatUtil, "concatenate_sound_segments", return_value=[1.0, 2.0, 3.0, 4.0]), \
                 patch("tts_audiobook_tool.concat_util.make_app_metadata_sections", return_value=sections), \
                 patch("tts_audiobook_tool.concat_util.m4b_chapter_util.make_metadata", return_value="meta") as make_metadata_mock, \
                 patch("tts_audiobook_tool.concat_util.m4b_chapter_util.make_copy_with_metadata", return_value="") as make_copy_mock, \
                 patch("tts_audiobook_tool.concat_util.AppMetadata.save_to_mp4", return_value="") as save_to_mp4_mock, \
                 patch("tts_audiobook_tool.concat_util.delete_silently"):
                result_path, err = ConcatUtil.make_file(
                    state=cast(State, state),
                    index_start=0,
                    index_end=3,
                    bookmark_indices=[2],
                    stem_path=stem_path,
                )

            self.assertEqual(err, "")
            self.assertEqual(result_path, stem_path + ".abr.m4b")
            make_metadata_mock.assert_called_once()
            make_copy_mock.assert_called_once_with(
                source_path=stem_path + " [concat].m4b",
                dest_path=stem_path + " [chaptermeta].m4b",
                metadata="meta",
            )
            save_to_mp4_mock.assert_called_once()
            self.assertEqual(save_to_mp4_mock.call_args.args[0].version, ABR_VERSION)
            self.assertEqual(save_to_mp4_mock.call_args.args[0].type, "generated")
            self.assertEqual(save_to_mp4_mock.call_args.args[1], stem_path + " [chaptermeta].m4b")
            self.assertEqual(save_to_mp4_mock.call_args.args[2], stem_path + ".abr.m4b")


if __name__ == "__main__":
    unittest.main()
