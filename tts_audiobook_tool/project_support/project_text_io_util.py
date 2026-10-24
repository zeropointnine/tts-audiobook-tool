from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.constants import PROJECT_TEXT_FILE_NAME, PROJECT_TEXT_RAW_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.util import COL_ERROR, printt
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types.phrase import PhraseGroup
    from tts_audiobook_tool.project import Project


class ProjectTextIOUtil:
    """
    Project text persistence and import-commit helpers.
    """

    @staticmethod
    def save_book(project: Project) -> str:
        file_path = project.project_text_path
        err = JsonSaveUtil.save(
            JsonArtifactType.PROJECT_TEXT,
            file_path,
            lambda: ProjectBookUtil.phrase_groups_to_dict(project),
        )
        if err:
            printt(f"\n{COL_ERROR}{err}\n")
            return err

        L.d(f"Saved {PROJECT_TEXT_FILE_NAME}: {file_path}")
        return ""

    @staticmethod
    def make_import_book(
            phrase_groups: list[PhraseGroup],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            dialog_segmentation: bool = False,
            title: str = "",
            text_source_kind: str = "plain_text",
            section_start_indices: list[int] | None = None,
            section_titles: list[str] | None = None,
    ) -> Book:
        settings = BookSegmentationSettings(
            language_code=language_code,
            max_words_per_segment=max_words,
            strategy=strategy,
            dialog_segmentation=dialog_segmentation,
        )
        if section_start_indices is not None:
            return ProjectBookUtil.make_book_from_flat_compatibility_fields(
                phrase_groups=phrase_groups,
                section_start_indices=section_start_indices,
                segmentation_settings=settings,
                text_source_kind=text_source_kind,
                audio_source_kind="generated",
                title=title,
                section_titles=section_titles,
            )
        return Book(
            title=title,
            text_source_kind=text_source_kind,
            audio_source_kind="generated",
            segmentation_settings=settings,
            sections=[BookSection(phrase_groups=phrase_groups)],
        )

    @staticmethod
    def set_phrase_groups_and_save(
            project: Project,
            phrase_groups: list[PhraseGroup],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            raw_text: str,
            dialog_segmentation: bool = False,
            title: str = "",
            text_source_kind: str = "plain_text",
    ) -> str:
        book = ProjectTextIOUtil.make_import_book(
            phrase_groups, strategy, max_words, language_code,
            dialog_segmentation, title, text_source_kind,
        )
        return ProjectTextIOUtil._save_import(project, book, raw_text)

    @staticmethod
    def set_phrase_groups_chapters_and_save(
            project: Project,
            phrase_groups: list[PhraseGroup],
            section_start_indices: list[int],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            raw_text: str,
            dialog_segmentation: bool = False,
            title: str = "",
            section_titles: list[str] | None = None,
    ) -> str:
        book = ProjectTextIOUtil.make_import_book(
            phrase_groups, strategy, max_words, language_code,
            dialog_segmentation, title, "epub", section_start_indices, section_titles,
        )
        return ProjectTextIOUtil._save_import(project, book, raw_text)

    @staticmethod
    def _save_import(project: Project, book: Book, raw_text: str) -> str:
        old_state = (project.book, project.markers, project.generate_range_string,
                     project.realtime_line_range)
        project.book = book
        project.markers = set()
        project.generate_range_string = ""
        project.realtime_line_range = None
        error = ProjectTextIOUtil.save_book(project)
        if error:
            (project.book, project.markers, project.generate_range_string,
             project.realtime_line_range) = old_state
            return error
        error = project.save()
        if error:
            return f"Project text was saved, but project metadata could not be saved: {error}"
        error = ProjectTextIOUtil._save_raw_text(project, raw_text)
        if error:
            return f"Project text was saved, but the raw source text could not be saved: {error}"
        # Business rule: importing text discards all saved dialog assignments
        # (unlike text editing, which keeps still-matching ranges). The roster
        # and voice mappings are kept. Only done once the new text is persisted.
        if project.dir_path:
            error = SpeakerAssignFileUtil.delete_attributions_file(project.dir_path)
            if error:
                return f"Project text was saved, but saved dialog assignments could not be cleared: {error}"
        return ""

    @staticmethod
    def load_raw_text(project: Project) -> str:
        file_path = os.path.join(project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            L.e(f"Error saving raw text: {e}")
            return ""

    @staticmethod
    def _save_raw_text(project: Project, raw_text: str) -> str:
        file_path = os.path.join(project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(raw_text)
        except Exception as e:
            error = f"Error saving raw text: {e}"
            L.e(error)
            return error
        return ""
