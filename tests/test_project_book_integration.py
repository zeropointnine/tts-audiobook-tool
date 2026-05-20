import json
import os
import tempfile
import unittest
from unittest.mock import patch

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME, PROJECT_TEXT_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos


class TestProjectBookIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        L.init("test-project-book-integration")

    def make_phrase_group(self, text: str, reason: Reason = Reason.SENTENCE) -> PhraseGroup:
        return PhraseGroup([Phrase(text, reason)])

    def write_minimal_project_json(self, project_dir: str, extra: dict | None = None) -> None:
        payload = {
            "version": 2,
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
        }
        if extra:
            payload.update(extra)
        with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "w", encoding="utf-8") as file:
            json.dump(payload, file)

    def test_project_model_validate_creates_book_from_flat_phrase_groups_and_section_dividers(self):
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

        self.assertEqual(project.book.text_source_kind, "legacy_flat")
        self.assertEqual(project.book.audio_source_kind, "unknown")
        self.assertEqual(project.book.segmentation_settings.language_code, "en")
        self.assertEqual(project.book.segmentation_settings.strategy, SegmentationStrategy.MULTI_SENTENCE)
        self.assertEqual(project.book.segmentation_settings.max_words_per_segment, 80)
        self.assertEqual(project.phrase_groups, phrase_groups)
        self.assertEqual(project.section_dividers, [2])
        self.assertEqual([len(section.phrase_groups) for section in project.book.sections], [2, 1])

    def test_project_to_dict_excludes_legacy_applied_fields(self):
        project = Project.model_validate({
            "applied_language_code": "en",
            "applied_strategy": "multi",
            "applied_max_words": 80,
        })

        payload = project.to_dict()

        self.assertNotIn("applied_language_code", payload)
        self.assertNotIn("applied_strategy", payload)
        self.assertNotIn("applied_max_words", payload)

    def test_get_book_segmentation_settings_falls_back_to_legacy_fields_without_book_sections(self):
        project = Project.model_validate({
            "applied_language_code": "es",
            "applied_strategy": "max_len",
            "applied_max_words": 42,
        })

        settings = project.get_book_segmentation_settings()

        self.assertEqual(settings.language_code, "es")
        self.assertEqual(settings.strategy, SegmentationStrategy.MAX_LEN)
        self.assertEqual(settings.max_words_per_segment, 42)

    def test_project_loads_legacy_project_text_as_book_and_preserves_flat_compatibility(self):
        phrase_groups = [self.make_phrase_group("One."), self.make_phrase_group("Two.")]
        text_payload = {
            "format": "phrase_groups.v1",
            "phrase_groups": PhraseGroup.phrase_groups_to_json_list(phrase_groups),
        }

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir, {"chapter_indices": [1]})
            with open(os.path.join(project_dir, PROJECT_TEXT_FILE_NAME), "w", encoding="utf-8") as file:
                json.dump(text_payload, file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelInfos.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

        self.assertIsInstance(result, Project)
        assert isinstance(result, Project)
        self.assertEqual(result.book.text_source_kind, "legacy_flat")
        self.assertEqual([group.text for group in result.phrase_groups], ["One.", "Two."])
        self.assertEqual(result.section_dividers, [1])
        self.assertEqual([len(section.phrase_groups) for section in result.book.sections], [1, 1])

    def test_project_load_migrates_phrase_groups_v1_project_text_to_book_v1(self):
        phrase_groups = [self.make_phrase_group("One."), self.make_phrase_group("Two.")]
        text_payload = {
            "format": "phrase_groups.v1",
            "phrase_groups": PhraseGroup.phrase_groups_to_json_list(phrase_groups),
        }

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir, {"chapter_indices": [1]})
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(text_payload, file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelInfos.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(text_path, "r", encoding="utf-8") as file:
                migrated_payload = json.load(file)
            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                migrated_project_payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(migrated_payload["format"], "book.v1")
        self.assertEqual(migrated_payload["book"]["text_source_kind"], "legacy_flat")
        self.assertEqual(len(migrated_payload["book"]["sections"]), 2)
        self.assertNotIn("applied_language_code", migrated_project_payload)
        self.assertNotIn("applied_strategy", migrated_project_payload)
        self.assertNotIn("applied_max_words", migrated_project_payload)

    def test_project_load_removes_stale_applied_fields_from_project_json_with_book_v1_text(self):
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

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelInfos.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                migrated_project_payload = json.load(file)
            with open(text_path, "r", encoding="utf-8") as file:
                text_project_payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(text_project_payload["format"], "book.v1")
        self.assertNotIn("applied_language_code", migrated_project_payload)
        self.assertNotIn("applied_strategy", migrated_project_payload)
        self.assertNotIn("applied_max_words", migrated_project_payload)

    def test_project_load_migrates_bare_list_project_text_to_book_v1(self):
        phrase_groups = [self.make_phrase_group("Bare list.")]
        text_payload = PhraseGroup.phrase_groups_to_json_list(phrase_groups)

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir)
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(text_payload, file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelInfos.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(text_path, "r", encoding="utf-8") as file:
                migrated_payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(migrated_payload["format"], "book.v1")
        self.assertEqual(migrated_payload["book"]["sections"][0]["phrase_groups"][0][0]["text"], "Bare list.")

    def test_project_load_keeps_book_v1_project_text_as_book_v1(self):
        book = Book(
            title="Already Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            sections=[BookSection(phrase_groups=[self.make_phrase_group("One.")])],
        )

        with tempfile.TemporaryDirectory() as project_dir:
            self.write_minimal_project_json(project_dir)
            text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
            with open(text_path, "w", encoding="utf-8") as file:
                json.dump(book_to_project_text_json_dict(book), file)

            with patch("tts_audiobook_tool.project_support.project_util.Tts.get_type", return_value=TtsModelInfos.NONE), \
                    patch("tts_audiobook_tool.ask.ask_enter_to_continue"):
                result = ProjectUtil.load_using_dir_path(project_dir)

            with open(text_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertIsInstance(result, Project)
        self.assertEqual(payload["format"], "book.v1")
        self.assertEqual(payload["book"]["title"], "Already Book")

    def test_project_save_writes_book_v1_project_text(self):
        book = Book(
            title="Saved Book",
            text_source_kind="epub",
            audio_source_kind="generated",
            segmentation_settings=BookSegmentationSettings(
                language_code="en",
                max_words_per_segment=120,
                strategy=SegmentationStrategy.MAX_LEN,
            ),
            sections=[
                BookSection(title="Chapter 1", phrase_groups=[self.make_phrase_group("One.")]),
                BookSection(title="Chapter 2", phrase_groups=[self.make_phrase_group("Two.")]),
            ],
        )

        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir, book=book)
            project.sync_flat_text_from_book()
            err = project.save(force_phrase_groups=True)
            self.assertEqual(err, "")

            with open(os.path.join(project_dir, PROJECT_TEXT_FILE_NAME), "r", encoding="utf-8") as file:
                payload = json.load(file)
            with open(os.path.join(project_dir, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as file:
                project_payload = json.load(file)

        self.assertEqual(payload["format"], "book.v1")
        self.assertEqual(payload["book"]["title"], "Saved Book")
        self.assertEqual(payload["book"]["sections"][1]["title"], "Chapter 2")
        self.assertEqual(payload["book"]["segmentation_settings"]["language_code"], "en")
        self.assertEqual(payload["book"]["segmentation_settings"]["max_words_per_segment"], 120)
        self.assertEqual(payload["book"]["segmentation_settings"]["strategy"], "max_len")
        self.assertNotIn("applied_language_code", project_payload)
        self.assertNotIn("applied_strategy", project_payload)
        self.assertNotIn("applied_max_words", project_payload)

    def test_set_phrase_groups_and_save_creates_plain_text_book(self):
        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)
            project.set_phrase_groups_and_save(
                phrase_groups=[self.make_phrase_group("One.")],
                strategy=SegmentationStrategy.NORMAL,
                max_words=50,
                language_code="en",
                raw_text="One.",
                text_source_kind="manual",
            )

            with open(os.path.join(project_dir, PROJECT_TEXT_FILE_NAME), "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertEqual(project.book.text_source_kind, "manual")
        self.assertEqual(project.book.audio_source_kind, "generated")
        self.assertEqual(project.applied_max_words, 50)
        self.assertEqual(project.section_dividers, [])
        self.assertEqual(payload["format"], "book.v1")
        self.assertEqual(payload["book"]["text_source_kind"], "manual")

    def test_set_phrase_groups_chapters_and_save_creates_epub_book_sections(self):
        phrase_groups = [
            self.make_phrase_group("One."),
            self.make_phrase_group("Two."),
            self.make_phrase_group("Three."),
        ]

        with tempfile.TemporaryDirectory() as project_dir:
            project = Project(dir_path=project_dir)
            project.set_phrase_groups_chapters_and_save(
                phrase_groups=phrase_groups,
                section_dividers=[2],
                strategy=SegmentationStrategy.MULTI_SENTENCE,
                max_words=80,
                language_code="en",
                raw_text="One. Two. Three.",
                title="Example Book",
                section_titles=["Chapter 1", "Chapter 2"],
            )

        self.assertEqual(project.book.title, "Example Book")
        self.assertEqual(project.book.text_source_kind, "epub")
        self.assertEqual(project.section_dividers, [2])
        self.assertEqual([section.title for section in project.book.sections], ["Chapter 1", "Chapter 2"])
        self.assertEqual([len(section.phrase_groups) for section in project.book.sections], [2, 1])


if __name__ == "__main__":
    unittest.main()