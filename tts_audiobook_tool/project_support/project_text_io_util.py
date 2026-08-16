from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.constants import PROJECT_TEXT_FILE_NAME, PROJECT_TEXT_RAW_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.util import COL_ERROR, printt

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
    def set_phrase_groups_and_save(
            project: Project,
            phrase_groups: list[PhraseGroup],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            raw_text: str,
            title: str="",
            text_source_kind: str="plain_text",
    ) -> None:

        settings = BookSegmentationSettings(
            language_code=language_code,
            max_words_per_segment=max_words,
            strategy=strategy,
        )
        book = Book(
            title=title,
            text_source_kind=text_source_kind,
            audio_source_kind="generated",
            segmentation_settings=settings,
            sections=[BookSection(phrase_groups=phrase_groups)],
        )

        project.book = book
        ProjectBookUtil.sync_flat_text_from_book(project)
        project.markers = set()
        project.generate_range_string = ""
        project.realtime_line_range = None
        ProjectTextIOUtil.save_book(project)
        project.save()

        ProjectTextIOUtil._save_raw_text(project, raw_text)

    @staticmethod
    def set_phrase_groups_chapters_and_save(
            project: Project,
            phrase_groups: list[PhraseGroup],
            section_start_indices: list[int],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            raw_text: str,
            title: str="",
            section_titles: list[str] | None=None,
    ) -> None:

        settings = BookSegmentationSettings(
            language_code=language_code,
            max_words_per_segment=max_words,
            strategy=strategy,
        )
        book = ProjectBookUtil.make_book_from_flat_compatibility_fields(
            phrase_groups=phrase_groups,
            section_start_indices=section_start_indices,
            segmentation_settings=settings,
            text_source_kind="epub",
            audio_source_kind="generated",
            title=title,
            section_titles=section_titles,
        )

        project.book = book
        ProjectBookUtil.sync_flat_text_from_book(project)
        project.markers = set()
        project.generate_range_string = ""
        project.realtime_line_range = None
        ProjectTextIOUtil.save_book(project)
        project.save()

        ProjectTextIOUtil._save_raw_text(project, raw_text)

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
    def _save_raw_text(project: Project, raw_text: str) -> None:
        file_path = os.path.join(project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(raw_text)
        except Exception as e:
            L.e(f"Error saving raw text: {e}")
