import json
import os
import tempfile
import unittest
from unittest.mock import patch

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SectionMarkerMode, SegmentationStrategy, VoiceSelectMode
from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME, PROJECT_TEXT_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class TestProjectBookIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        L.init("test-project-book-integration")

    def make_phrase_group(self, text: str, reason: Reason = Reason.SENTENCE) -> PhraseGroup:
        return PhraseGroup([Phrase(text, reason)])

    def legacy_phrase_groups_json(self, phrase_groups: list[PhraseGroup]) -> list[list[dict]]:
        return [group.to_json_dict_list() for group in phrase_groups]

    def write_minimal_project_json(self, project_dir: str, extra: dict | None = None) -> None:
        payload = {
            "version": 2,
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
            "applied_dialog_segmentation": True,
        }
        if extra:
            payload.update(extra)
        with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "w", encoding="utf-8") as file:
            json.dump(payload, file)

    def write_complete_project_json(self, project_dir: str, current_model_type: object) -> None:
        payload = ProjectSerializationUtil.to_project_json_dict(Project())
        payload["current_model_type"] = current_model_type
        with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "w", encoding="utf-8") as file:
            json.dump(payload, file)

    def test_project_model_validate_keeps_flat_phrase_groups_as_single_book_section(self):
        phrase_groups = [
            self.make_phrase_group("One."),
            self.make_phrase_group("Two."),
            self.make_phrase_group("Three."),
        ]

        project = Project.model_validate({
            "phrase_groups": phrase_groups,
            "markers": [2],
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
            "applied_dialog_segmentation": True,
        })

        self.assertEqual(project.book.text_source_kind, "legacy_flat")
        self.assertEqual(project.book.audio_source_kind, "unknown")
        self.assertEqual(project.book.segmentation_settings.language_code, "en")
        self.assertEqual(project.book.segmentation_settings.strategy, SegmentationStrategy.MULTI_SENTENCE)
        self.assertEqual(project.book.segmentation_settings.max_words_per_segment, 80)
        self.assertEqual(project.applied_dialog_segmentation, True)
        self.assertEqual(project.book.segmentation_settings.dialog_segmentation, True)
        self.assertEqual(project.phrase_groups, phrase_groups)
        self.assertEqual(project.markers, {2})
        self.assertEqual([len(section.phrase_groups) for section in project.book.sections], [3])

    def test_project_to_dict_excludes_legacy_applied_fields(self):
        project = Project.model_validate({
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
        })

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertNotIn("applied_language_code", payload)
        self.assertNotIn("applied_strategy", payload)
        self.assertNotIn("applied_max_words", payload)
        self.assertNotIn("applied_dialog_segmentation", payload)

    def test_project_to_dict_includes_moss_audio_top_p_and_top_k(self):
        project = Project.model_validate({
            "moss_delay_top_p": 0.72,
            "moss_delay_top_k": 37,
            "moss_local_top_p": 0.82,
            "moss_local_top_k": 47,
        })

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(payload["moss_delay_top_p"], 0.72)
        self.assertEqual(payload["moss_delay_top_k"], 37)
        self.assertEqual(payload["moss_local_top_p"], 0.82)
        self.assertEqual(payload["moss_local_top_k"], 47)

    def test_project_to_dict_includes_qwen3_server_concurrent_requests(self):
        project = Project.model_validate({
            "qwen3_server_concurrent_requests": 3,
        })

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(payload["qwen3_server_concurrent_requests"], 3)

    def test_project_model_validate_normalizes_legacy_voice_strings_to_lists(self):
        project = Project.model_validate({
            "fish_s1_voice_file_name": "sample_s1.flac",
            "fish_s1_voice_text": "sample text",
            "glm_voice_file_name": "sample_glm.flac",
            "glm_voice_text": "glm text",
        })

        self.assertEqual(project.fish_s1_voice_file_name, ["sample_s1.flac"])
        self.assertEqual(project.fish_s1_voice_transcript, ["sample text"])
        self.assertEqual(project.glm_voice_file_name, ["sample_glm.flac"])
        self.assertEqual(project.glm_voice_transcript, ["glm text"])

    def test_project_model_validate_preserves_voice_lists_and_filters_invalid_items(self):
        project = Project.model_validate({
            "moss_voice_file_name": ["one.flac", "", 3, "two.flac"],
            "moss_voice_transcript": ["one", None, "two"],
        })

        self.assertEqual(project.moss_voice_file_name, ["one.flac", "two.flac"])
        self.assertEqual(project.moss_voice_transcript, ["one", "two"])

    def test_project_model_validate_collects_warnings_only_with_explicit_context(self):
        warnings: list[str] = []

        Project.model_validate(
            {"streaming_chat": "invalid"},
            context={"warnings": warnings},
        )

        self.assertTrue(any("streaming_chat" in warning for warning in warnings))
        warning_count = len(warnings)

        Project.model_validate({"streaming_chat": "invalid"})

        self.assertEqual(len(warnings), warning_count)

    def test_project_load_reports_normalization_warnings(self):
        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir, {"streaming_chat": "invalid"})

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.project_support.project_load_util.printt") as print_mock, \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue") as continue_mock:
                result = ProjectUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(result, Project)
        self.assertTrue(any(
            "streaming_chat" in str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        ))
        continue_mock.assert_called_once_with()

    def test_project_current_model_type_defaults_and_normalizes_by_id(self):
        self.assertEqual(Project().current_model_type, TtsModelType.NONE)
        self.assertEqual(
            Project.model_validate({"current_model_type": TtsModelType.CHATTERBOX.value.id}).current_model_type,
            TtsModelType.CHATTERBOX,
        )
        self.assertEqual(
            Project.model_validate({"current_model_type": TtsModelType.CHATTERBOX}).current_model_type,
            TtsModelType.CHATTERBOX,
        )
        self.assertEqual(
            Project.model_validate({"current_model_type": "unknown-model"}).current_model_type,
            TtsModelType.NONE,
        )

    def test_project_current_model_type_serializes_using_id(self):
        project = Project.model_validate({"current_model_type": TtsModelType.CHATTERBOX.value.id})

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(payload["current_model_type"], TtsModelType.CHATTERBOX.value.id)

    def test_project_save_stamps_current_runtime_model_type(self):
        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)

            with patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.MIRA):
                error = project.save()

            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertEqual(error, "")
        self.assertEqual(project.current_model_type, TtsModelType.MIRA)
        self.assertEqual(payload["current_model_type"], TtsModelType.MIRA.value.id)

    def test_project_save_does_not_replace_current_model_type_with_none(self):
        with tempfile.TemporaryDirectory() as project_dir:
            project = Project.model_validate({
                "dir_path": project_dir,
                "current_model_type": TtsModelType.CHATTERBOX.value.id,
            })

            with patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.NONE):
                error = project.save()

            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertEqual(error, "")
        self.assertEqual(project.current_model_type, TtsModelType.CHATTERBOX)
        self.assertEqual(payload["current_model_type"], TtsModelType.CHATTERBOX.value.id)

    def test_project_load_reports_and_acknowledges_different_previous_model(self):
        with tempfile.TemporaryDirectory() as project_dir:
            self.write_complete_project_json(project_dir, TtsModelType.CHATTERBOX.value.id)

            with patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.MIRA), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue") as continue_mock:
                result = ProjectLoadUtil.load_using_dir_path(project_dir)
                with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                    payload = json.load(file)
                reopened_result = ProjectLoadUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(result, Project)
        self.assertEqual(result.current_model_type, TtsModelType.NONE)
        self.assertEqual(payload["current_model_type"], TtsModelType.NONE.value.id)
        self.assertIsInstance(reopened_result, Project)
        continue_mock.assert_called_once_with()

    def test_project_load_skips_previous_model_report_when_not_applicable(self):
        cases = [
            ("matching model", TtsModelType.CHATTERBOX.value.id, TtsModelType.CHATTERBOX, True),
            ("previous model is none", TtsModelType.NONE.value.id, TtsModelType.MIRA, True),
            ("previous model id is invalid", "unknown-model", TtsModelType.MIRA, True),
            ("noninteractive load", TtsModelType.CHATTERBOX.value.id, TtsModelType.MIRA, False),
        ]
        for label, stored_type, runtime_type, prompt_on_warnings in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as project_dir:
                self.write_complete_project_json(project_dir, stored_type)
                with patch("tts_audiobook_tool.tts.Tts.get_type", return_value=runtime_type), \
                        patch("tts_audiobook_tool.ask.ask_enter_to_continue") as continue_mock:
                    result = ProjectLoadUtil.load_using_dir_path(
                        project_dir,
                        prompt_on_warnings=prompt_on_warnings,
                    )

                self.assertIsInstance(result, Project)
                continue_mock.assert_not_called()

    def test_project_load_skips_previous_model_report_without_proper_name(self):
        with tempfile.TemporaryDirectory() as project_dir:
            self.write_complete_project_json(project_dir, TtsModelType.CHATTERBOX.value.id)

            with patch.dict(TtsModelType.CHATTERBOX.value.ui, {}, clear=True), \
                    patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.MIRA), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue") as continue_mock:
                result = ProjectLoadUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(result, Project)
        continue_mock.assert_not_called()

    def test_project_to_dict_serializes_single_voice_item_as_string_and_multiple_as_list(self):
        project = Project.model_validate({
            "qwen3_voice_file_name": ["one.flac"],
            "qwen3_voice_transcript": ["one"],
            "fish_s2_voice_file_name": ["one.flac", "two.flac"],
            "fish_s2_voice_transcript": ["one", "two"],
        })

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(payload["qwen3_voice_file_name"], "one.flac")
        self.assertEqual(payload["qwen3_voice_transcript"], "one")
        self.assertEqual(payload["fish_s2_voice_file_name"], ["one.flac", "two.flac"])
        self.assertEqual(payload["fish_s2_voice_transcript"], ["one", "two"])

    def test_project_to_dict_keeps_oute_voice_json_as_dict(self):
        voice_json = {"speaker": "default"}
        project = Project.model_validate({
            "oute_voice_json": voice_json,
        })

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(project.oute_voice_json, voice_json)
        self.assertNotIsInstance(project.oute_voice_json, list)
        self.assertNotIn("oute_voice_json", payload)

    def test_project_normalizes_qwen3_server_concurrent_requests(self):
        project = Project.model_validate({
            "qwen3_server_concurrent_requests": 0,
        })

        self.assertEqual(project.qwen3_server_concurrent_requests, 1)

    def test_project_to_dict_includes_higgs_v3_voice_generation_settings(self):
        project = Project.model_validate({
            "higgs_v3_temperature": 0.43,
            "higgs_v3_top_p": 0.87,
            "higgs_v3_top_k": 42,
            "higgs_v3_seed": 12345,
        })

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(payload["higgs_v3_temperature"], 0.43)
        self.assertEqual(payload["higgs_v3_top_p"], 0.87)
        self.assertEqual(payload["higgs_v3_top_k"], 42)
        self.assertEqual(payload["higgs_v3_seed"], 12345)

    def test_project_model_validate_normalizes_higgs_v3_seed(self):
        project = Project.model_validate({
            "higgs_v3_seed": 12345.0,
        })

        self.assertEqual(project.higgs_v3_seed, 12345)

    def test_project_model_validate_rejects_invalid_higgs_v3_seed(self):
        project = Project.model_validate({
            "higgs_v3_seed": -2,
        })

        self.assertEqual(project.higgs_v3_seed, -1)

    def test_project_transfer_field_set_matches_serialized_project_settings(self):
        self.assertEqual(ProjectTransferUtil.get_missing_project_settings_transfer_fields(Project), [])

    def test_project_model_validate_migrates_legacy_moss_audio_top_p_and_top_k(self):
        project = Project.model_validate({
            "moss_top_p": 0.72,
            "moss_top_k": 37.0,
        })

        self.assertEqual(project.moss_delay_top_p, 0.72)
        self.assertEqual(project.moss_delay_top_k, 37)

    def test_project_to_dict_includes_moss_target(self):
        target = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
        project = Project.model_validate({
            "moss_target": target,
        })

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(payload["moss_target"], target)

    def test_project_model_validate_normalizes_moss_audio_top_p_and_top_k(self):
        project = Project.model_validate({
            "moss_delay_top_p": 0.72,
            "moss_delay_top_k": 37.0,
            "moss_local_top_p": 0.82,
            "moss_local_top_k": 47.0,
        })

        self.assertEqual(project.moss_delay_top_p, 0.72)
        self.assertEqual(project.moss_delay_top_k, 37)
        self.assertEqual(project.moss_local_top_p, 0.82)
        self.assertEqual(project.moss_local_top_k, 47)

    def test_project_model_validate_accepts_moss_audio_top_k_minimum(self):
        moss_delay_top_k_min = MossConfigs.DELAY.value.audio_top_k_min
        moss_local_top_k_min = MossConfigs.LOCAL.value.audio_top_k_min
        project = Project.model_validate({
            "moss_delay_top_k": moss_delay_top_k_min,
            "moss_local_top_k": moss_local_top_k_min,
        })

        self.assertEqual(project.moss_delay_top_k, moss_delay_top_k_min)
        self.assertEqual(project.moss_local_top_k, moss_local_top_k_min)

    def test_project_model_validate_rejects_moss_audio_top_k_below_minimum(self):
        moss_delay_top_k_min = MossConfigs.DELAY.value.audio_top_k_min
        moss_local_top_k_min = MossConfigs.LOCAL.value.audio_top_k_min
        project = Project.model_validate({
            "moss_delay_top_k": moss_delay_top_k_min - 1,
            "moss_local_top_k": moss_local_top_k_min - 1,
        })

        self.assertEqual(project.moss_delay_top_k, -1)
        self.assertEqual(project.moss_local_top_k, -1)

    def test_project_model_validate_preserves_moss_audio_top_k_default_sentinel(self):
        project = Project.model_validate({
            "moss_delay_top_k": -1,
            "moss_local_top_k": -1,
        })

        self.assertEqual(project.moss_delay_top_k, -1)
        self.assertEqual(project.moss_local_top_k, -1)

    def test_get_book_segmentation_settings_falls_back_to_legacy_fields_without_book_sections(self):
        project = Project.model_validate({
            "applied_language_code": "es",
            "applied_strategy": "max_len",
            "applied_max_words": 42,
            "applied_dialog_segmentation": True,
        })

        settings = ProjectBookUtil.get_book_segmentation_settings(project)

        self.assertEqual(settings.language_code, "es")
        self.assertEqual(settings.strategy, SegmentationStrategy.MAX_LEN)
        self.assertEqual(settings.max_words_per_segment, 42)
        self.assertEqual(settings.dialog_segmentation, True)

    def test_project_loads_legacy_project_text_as_book_and_preserves_flat_compatibility(self):
        phrase_groups = [self.make_phrase_group("One."), self.make_phrase_group("Two.")]
        text_payload = {
            "format": "phrase_groups.v1",
            "phrase_groups": self.legacy_phrase_groups_json(phrase_groups),
        }

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir, {"chapter_indices": [1]})
            with open(os.path.join(project_dir, PROJECT_TEXT_FILE_NAME), "w", encoding="utf-8") as file:
                json.dump(text_payload, file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(result, Project)
        assert isinstance(result, Project)
        self.assertEqual(result.book.text_source_kind, "legacy_flat")
        self.assertEqual([group.text for group in result.phrase_groups], ["One.", "Two."])
        self.assertEqual(result.markers, {1})
        self.assertEqual([len(section.phrase_groups) for section in result.book.sections], [2])

    def test_project_load_migrates_phrase_groups_v1_project_text_to_book_v2(self):
        phrase_groups = [self.make_phrase_group("One."), self.make_phrase_group("Two.")]
        text_payload = {
            "format": "phrase_groups.v1",
            "phrase_groups": self.legacy_phrase_groups_json(phrase_groups),
        }

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir, {"chapter_indices": [1]})
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(text_payload, file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(text_path, "r", encoding="utf-8") as file:
                migrated_payload = json.load(file)
            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                migrated_project_payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(migrated_payload["format"], "book.v2")
        self.assertEqual(migrated_payload["book"]["text_source_kind"], "legacy_flat")
        self.assertEqual(len(migrated_payload["book"]["sections"]), 1)
        self.assertEqual(migrated_project_payload["markers"], [1])
        self.assertNotIn("chapter_indices", migrated_project_payload)
        self.assertNotIn("applied_language_code", migrated_project_payload)
        self.assertNotIn("applied_strategy", migrated_project_payload)
        self.assertNotIn("applied_max_words", migrated_project_payload)
        self.assertNotIn("applied_dialog_segmentation", migrated_project_payload)

    def test_project_load_removes_stale_applied_fields_from_project_json_with_book_v2_text(self):
        book = Book(
            title="Already Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            segmentation_settings=BookSegmentationSettings(
                language_code="en",
                max_words_per_segment=120,
                strategy=SegmentationStrategy.MAX_LEN,
            ),
            sections=[BookSection(phrase_groups=[self.make_phrase_group("One.")])],
        )

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir)
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(book_to_project_text_json_dict(book), file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                migrated_project_payload = json.load(file)
            with open(text_path, "r", encoding="utf-8") as file:
                text_project_payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(text_project_payload["format"], "book.v2")
        self.assertIn("markers", migrated_project_payload)
        self.assertNotIn("chapter_indices", migrated_project_payload)
        self.assertNotIn("applied_language_code", migrated_project_payload)
        self.assertNotIn("applied_strategy", migrated_project_payload)
        self.assertNotIn("applied_max_words", migrated_project_payload)
        self.assertNotIn("applied_dialog_segmentation", migrated_project_payload)

    def test_project_load_migrates_bare_list_project_text_to_book_v2(self):
        phrase_groups = [self.make_phrase_group("Bare list.")]
        text_payload = self.legacy_phrase_groups_json(phrase_groups)

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir)
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(text_payload, file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(text_path, "r", encoding="utf-8") as file:
                migrated_payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(migrated_payload["format"], "book.v2")
        group_payload = migrated_payload["book"]["sections"][0]["phrase_groups"][0]
        self.assertEqual(group_payload["voice_index"], -1)
        self.assertEqual(group_payload["phrases"][0]["text"], "Bare list.")

    def test_project_load_migrates_book_v1_project_text_to_book_v2(self):
        book = Book(
            title="Already Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            sections=[BookSection(phrase_groups=[self.make_phrase_group("One.")])],
        )

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir)
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            payload = book_to_project_text_json_dict(book)
            payload["format"] = "book.v1"
            for section in payload["book"]["sections"]:
                section["phrase_groups"] = [
                    group["phrases"] for group in section["phrase_groups"]
                ]
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(payload, file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(text_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(payload["format"], "book.v2")
        self.assertEqual(payload["book"]["title"], "Already Book")
        self.assertEqual(payload["book"]["sections"][0]["phrase_groups"][0]["voice_index"], -1)

    def test_project_save_writes_book_v2_project_text(self):
        book = Book(
            title="Saved Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            segmentation_settings=BookSegmentationSettings(
                language_code="en",
                max_words_per_segment=120,
                strategy=SegmentationStrategy.MAX_LEN,
                dialog_segmentation=True,
            ),
            sections=[
                BookSection(title="Chapter 1", phrase_groups=[self.make_phrase_group("One.")]),
                BookSection(title="Chapter 2", phrase_groups=[self.make_phrase_group("Two.")]),
            ],
        )

        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir, book=book)
            ProjectBookUtil.sync_flat_text_from_book(project)
            err = ProjectTextIOUtil.save_book(project)
            if not err:
                err = project.save()
            self.assertEqual(err, "")

            with open(os.path.join(project_dir, PROJECT_TEXT_FILE_NAME), "r", encoding="utf-8") as file:
                payload = json.load(file)
            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                project_payload = json.load(file)

        self.assertEqual(payload["format"], "book.v2")
        self.assertEqual(payload["book"]["title"], "Saved Book")
        self.assertEqual(payload["book"]["sections"][1]["title"], "Chapter 2")
        self.assertEqual(payload["book"]["segmentation_settings"]["language_code"], "en")
        self.assertEqual(payload["book"]["segmentation_settings"]["max_words_per_segment"], 120)
        self.assertEqual(payload["book"]["segmentation_settings"]["strategy"], "max_len")
        self.assertEqual(payload["book"]["segmentation_settings"]["dialog_segmentation"], True)
        self.assertEqual(project_payload["markers"], [])
        self.assertNotIn("chapter_indices", project_payload)
        self.assertNotIn("applied_language_code", project_payload)
        self.assertNotIn("applied_strategy", project_payload)
        self.assertNotIn("applied_dialog_segmentation", project_payload)

    def test_project_model_validate_coerces_bookmark_mode_for_multi_section_books(self):
        project = Project.model_validate({
            "chapter_mode": SectionMarkerMode.BOOKMARKS.id,
            "book": Book(sections=[
                BookSection(title="Chapter 1", phrase_groups=[self.make_phrase_group("One.")]),
                BookSection(title="Chapter 2", phrase_groups=[self.make_phrase_group("Two.")]),
            ]),
        })

        self.assertEqual(project.chapter_mode, SectionMarkerMode.FILES)

    def test_project_model_validate_keeps_bookmark_mode_for_single_section_books(self):
        project = Project.model_validate({
            "chapter_mode": SectionMarkerMode.BOOKMARKS.id,
            "book": Book(sections=[
                BookSection(title="Chapter 1", phrase_groups=[self.make_phrase_group("One.")]),
            ]),
        })

        self.assertEqual(project.chapter_mode, SectionMarkerMode.BOOKMARKS)

    def test_project_save_coerces_bookmark_mode_for_multi_section_books(self):
        book = Book(sections=[
            BookSection(title="Chapter 1", phrase_groups=[self.make_phrase_group("One.")]),
            BookSection(title="Chapter 2", phrase_groups=[self.make_phrase_group("Two.")]),
        ])

        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir, book=book, chapter_mode=SectionMarkerMode.BOOKMARKS)
            ProjectBookUtil.sync_flat_text_from_book(project)

            err = project.save()
            self.assertEqual(err, "")

            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertEqual(project.chapter_mode, SectionMarkerMode.FILES)
        self.assertEqual(payload["chapter_mode"], SectionMarkerMode.FILES.id)
        self.assertNotIn("applied_max_words", payload)
        self.assertNotIn("applied_dialog_segmentation", payload)

    def test_project_model_validate_accepts_legacy_chapter_indices_alias(self):
        phrase_groups = [
            self.make_phrase_group("One."),
            self.make_phrase_group("Two."),
            self.make_phrase_group("Three."),
        ]

        project = Project.model_validate({
            "phrase_groups": phrase_groups,
            "chapter_indices": [2],
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
        })

        self.assertEqual(project.markers, {2})

    def test_set_phrase_groups_and_save_creates_plain_text_book(self):
        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)
            ProjectTextIOUtil.set_phrase_groups_and_save(
                project,
                phrase_groups=[self.make_phrase_group("One.")],
                strategy=SegmentationStrategy.SENTENCE_PLUS,
                max_words=50,
                language_code="en",
                dialog_segmentation=True,
                raw_text="One.",
                title="Manual Title",
                text_source_kind="manual",
            )

            with open(os.path.join(project_dir, PROJECT_TEXT_FILE_NAME), "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertEqual(project.book.text_source_kind, "manual")
        self.assertEqual(project.book.title, "Manual Title")
        self.assertEqual(project.book.audio_source_kind, "generated")
        self.assertEqual(project.applied_max_words, 50)
        self.assertEqual(project.applied_dialog_segmentation, True)
        self.assertEqual(project.book.segmentation_settings.dialog_segmentation, True)
        self.assertEqual(payload["book"]["segmentation_settings"]["dialog_segmentation"], True)
        self.assertEqual(project.markers, set())
        self.assertEqual(payload["format"], "book.v2")
        self.assertEqual(payload["book"]["title"], "Manual Title")
        self.assertEqual(payload["book"]["text_source_kind"], "manual")

    def test_dialog_voice_preassignments_survive_project_save_and_reload(self):
        raw_text = 'He said "Hello." Then left.'
        phrase_groups = PhraseGrouper.text_to_groups(
            raw_text,
            max_words=100,
            strategy=SegmentationStrategy.MAX_LEN,
            pysbd_lang="en",
            dialog_segmentation=True,
        )

        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)
            project.voice_select_mode = VoiceSelectMode.DISABLED
            ProjectTextIOUtil.set_phrase_groups_and_save(
                project,
                phrase_groups=phrase_groups,
                strategy=SegmentationStrategy.MAX_LEN,
                max_words=100,
                language_code="en",
                dialog_segmentation=True,
                raw_text=raw_text,
            )
            self.assertEqual(project.voice_select_mode, VoiceSelectMode.DISABLED)

            with patch(
                "tts_audiobook_tool.project_support.project_util.Tts.get_type",
                return_value=TtsModelType.NONE,
            ), patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                reloaded = ProjectUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(reloaded, Project)
        self.assertEqual(
            [group.voice_index for group in reloaded.phrase_groups],
            [-1, 1, -1],
        )
        self.assertTrue(reloaded.applied_dialog_segmentation)

    def test_set_phrase_groups_and_save_clears_markers_for_plain_text_import(self):
        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)
            ProjectTextIOUtil.set_phrase_groups_and_save(
                project,
                phrase_groups=[self.make_phrase_group("One."), self.make_phrase_group("Two.")],
                strategy=SegmentationStrategy.SENTENCE_PLUS,
                max_words=50,
                language_code="en",
                raw_text="One.\nTwo.",
                title="source-file",
                text_source_kind="plain_text",
            )

        self.assertEqual(project.book.text_source_kind, "plain_text")
        self.assertEqual(project.book.title, "source-file")
        self.assertEqual(project.markers, set())

    def test_set_phrase_groups_chapters_and_save_creates_epub_book_sections(self):
        phrase_groups = [
            self.make_phrase_group("One."),
            self.make_phrase_group("Two."),
            self.make_phrase_group("Three."),
        ]

        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)
            ProjectTextIOUtil.set_phrase_groups_chapters_and_save(
                project,
                phrase_groups=phrase_groups,
                section_start_indices=[2],
                strategy=SegmentationStrategy.MULTI_SENTENCE,
                max_words=80,
                language_code="en",
                dialog_segmentation=True,
                raw_text="One. Two. Three.",
                title="Example Book",
                section_titles=["Chapter 1", "Chapter 2"],
            )

        self.assertEqual(project.book.title, "Example Book")
        self.assertEqual(project.book.text_source_kind, "epub")
        self.assertEqual(project.markers, set())
        self.assertEqual(project.applied_dialog_segmentation, True)
        self.assertEqual(project.book.segmentation_settings.dialog_segmentation, True)
        self.assertEqual([section.title for section in project.book.sections], ["Chapter 1", "Chapter 2"])
        self.assertEqual([len(section.phrase_groups) for section in project.book.sections], [2, 1])

    def test_project_load_keeps_epub_book_sections_and_does_not_repopulate_markers(self):
        book = Book(
            title="Example Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            segmentation_settings=BookSegmentationSettings(
                language_code="en",
                max_words_per_segment=80,
                strategy=SegmentationStrategy.MULTI_SENTENCE,
            ),
            sections=[
                BookSection(title="Chapter 1", phrase_groups=[self.make_phrase_group("One."), self.make_phrase_group("Two.")]),
                BookSection(title="Chapter 2", phrase_groups=[self.make_phrase_group("Three.")]),
            ],
        )

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir, {"markers": []})
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(book_to_project_text_json_dict(book), file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(result, Project)
        assert isinstance(result, Project)
        self.assertEqual(result.markers, set())
        self.assertEqual([section.title for section in result.book.sections], ["Chapter 1", "Chapter 2"])
        self.assertEqual([len(section.phrase_groups) for section in result.book.sections], [2, 1])

    def test_project_markers_persist_independently_of_epub_book_sections(self):
        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)
            ProjectTextIOUtil.set_phrase_groups_chapters_and_save(
                project,
                phrase_groups=[
                    self.make_phrase_group("One."),
                    self.make_phrase_group("Two."),
                    self.make_phrase_group("Three."),
                ],
                section_start_indices=[2],
                strategy=SegmentationStrategy.MULTI_SENTENCE,
                max_words=80,
                language_code="en",
                raw_text="One. Two. Three.",
                title="Example Book",
                section_titles=["Chapter 1", "Chapter 2"],
            )
            project.markers = {1}
            project.save()

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelType.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                reloaded = ProjectUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(reloaded, Project)
        assert isinstance(reloaded, Project)
        self.assertEqual(reloaded.markers, {1})
        self.assertEqual([section.title for section in reloaded.book.sections], ["Chapter 1", "Chapter 2"])
        self.assertEqual([len(section.phrase_groups) for section in reloaded.book.sections], [2, 1])

    def test_project_model_validate_drops_markers_when_any_equal_phrase_group_count(self):
        phrase_groups = [
            self.make_phrase_group("One."),
            self.make_phrase_group("Two."),
            self.make_phrase_group("Three."),
        ]

        project = Project.model_validate({
            "phrase_groups": phrase_groups,
            "markers": [2, 3],
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
        })

        self.assertEqual(project.markers, set())

    def test_project_model_validate_deduplicates_and_sorts_markers(self):
        phrase_groups = [
            self.make_phrase_group(f"Line {i}.")
            for i in range(1, 6)
        ]

        project = Project.model_validate({
            "phrase_groups": phrase_groups,
            "markers": [3, 1, 3],
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
        })

        self.assertEqual(project.markers, {1, 3})

    def test_project_model_validate_discards_non_positive_markers(self):
        phrase_groups = [
            self.make_phrase_group("One."),
            self.make_phrase_group("Two."),
        ]

        project = Project.model_validate({
            "phrase_groups": phrase_groups,
            "markers": [-2, -1, 0, 1],
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
        })

        self.assertEqual(project.markers, {1})

    def test_markers_setter_filters_non_positive_values_and_returns_copy(self):
        project = Project()
        project.markers = {-1, 0, 1, 3}

        markers = project.markers
        markers.add(5)

        self.assertEqual(project.markers, {1, 3})
        self.assertEqual(markers, {1, 3, 5})

    def test_markers_serialize_as_a_sorted_json_array(self):
        project = Project()
        project.markers = {3, 1, 2}

        payload = ProjectSerializationUtil.to_project_json_dict(project)

        self.assertEqual(payload["markers"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
