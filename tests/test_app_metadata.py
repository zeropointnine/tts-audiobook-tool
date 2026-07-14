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
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types import ExportType, NormalizationType, SectionMarkerMode
from tts_audiobook_tool.menus import concat_menu
from tts_audiobook_tool.state import State


class TestAppMetadata(unittest.TestCase):
    def make_phrase_group(self, text: str) -> PhraseGroup:
        return PhraseGroup([Phrase(text, Reason.SENTENCE)])

    def test_app_metadata_round_trip_preserves_sections(self):
        meta = AppMetadata(
            timed_phrases=[TimedPhrase.make_using(Phrase("One.", Reason.SENTENCE), 0.0, 1.0)],
            title="Example Book",
            version=3,
            bookmark_indices=[0],
            raw_text="One.",
            has_break_audio=False,
            project_snapshot={},
            sections=[AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)],
        )

        json_string = meta.to_json_string()
        result = AppMetadata.get_from_json_string(json_string)

        self.assertIsInstance(result, AppMetadata)
        assert isinstance(result, AppMetadata)
        self.assertEqual(result.title, "Example Book")
        self.assertEqual(result.sections, [AppMetadataSection(title="Chapter 1", start_index=0, end_index=1)])

        payload = json.loads(json_string)
        self.assertEqual(payload["title"], "Example Book")
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
        self.assertEqual(result.title, "")
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
            title="Example Book",
            version=3,
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

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["title"], "Example Book")
            self.assertEqual(payload["version"], 3)
            self.assertNotIn("raw_text", payload)
            self.assertEqual(payload["sections"], [{"title": "Chapter 1", "start_index": 0, "end_index": 1}])
            self.assertEqual(payload["text_segments"], [{"text": "One.", "time_start": 0.0, "time_end": 1.0}])

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

    def test_ask_output_indices_and_make_single_file_uses_markers_as_bookmarks_for_single_book_section(self):
        project = Project.model_validate({
            "phrase_groups": [
                self.make_phrase_group("One."),
                self.make_phrase_group("Two."),
                self.make_phrase_group("Three."),
            ],
            "markers": [1, 2],
            "book": Book(sections=[BookSection(title="Chapter 1", phrase_groups=[
                self.make_phrase_group("One."),
                self.make_phrase_group("Two."),
                self.make_phrase_group("Three."),
            ])]),
        })
        state = SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(project_dir="/tmp"),
        )
        state.project.markers = [1, 2]
        state.project.export_type = ExportType.AAC
        state.project.chapter_mode = SectionMarkerMode.BOOKMARKS
        state.project._sound_segments = SimpleNamespace(num_generated=lambda: 1)

        with patch.object(concat_menu.ask, "ask_confirm", return_value=True), \
            patch.object(concat_menu.OutputRangeInfo, "make_single_info", return_value=SimpleNamespace(num_files_exist=1, num_segments=3)), \
             patch.object(concat_menu.ConcatUtil, "make_files") as make_files_mock, \
             patch.object(concat_menu, "printt"):
            concat_menu.ask_output_indices_and_make(cast(State, state))

        make_files_mock.assert_called_once_with(
            state=state,
            file_cut_indices=[],
            bookmark_indices=[1, 2],
        )

    def test_ask_output_indices_and_make_single_file_ignores_markers_as_bookmarks_for_multiple_book_sections(self):
        project = Project.model_validate({
            "phrase_groups": [
                self.make_phrase_group("One."),
                self.make_phrase_group("Two."),
                self.make_phrase_group("Three."),
            ],
            "markers": [1, 2],
            "book": Book(sections=[
                BookSection(title="Chapter 1", phrase_groups=[self.make_phrase_group("One.")]),
                BookSection(title="Chapter 2", phrase_groups=[self.make_phrase_group("Two."), self.make_phrase_group("Three.")]),
            ]),
        })
        state = SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(project_dir="/tmp"),
        )
        state.project.markers = [1, 2]
        state.project.export_type = ExportType.AAC
        state.project.chapter_mode = SectionMarkerMode.BOOKMARKS
        state.project._sound_segments = SimpleNamespace(num_generated=lambda: 1)

        with patch.object(concat_menu.ask, "ask_confirm", return_value=True), \
            patch.object(concat_menu.OutputRangeInfo, "make_single_info", return_value=SimpleNamespace(num_files_exist=1, num_segments=3)), \
             patch.object(concat_menu.ConcatUtil, "make_files") as make_files_mock, \
             patch.object(concat_menu, "printt"):
            concat_menu.ask_output_indices_and_make(cast(State, state))

        make_files_mock.assert_called_once_with(
            state=state,
            file_cut_indices=[],
            bookmark_indices=[],
        )

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
            self.assertEqual(save_to_mp4_mock.call_args.args[1], stem_path + " [chaptermeta].m4b")
            self.assertEqual(save_to_mp4_mock.call_args.args[2], stem_path + ".abr.m4b")


if __name__ == "__main__":
    unittest.main()
