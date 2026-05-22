from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.constants import PROJECT_TEXT_FILE_NAME, PROJECT_TEXT_RAW_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.util import COL_ERROR, make_error_string, printt

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types.phrase import PhraseGroup
    from tts_audiobook_tool.project import Project


class ProjectTextIOUtil:
    """
    Project text persistence and import-commit helpers.
    """

    @staticmethod
    def save_phrase_groups(project: Project) -> str:
        file_path = project.project_text_path
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(ProjectBookUtil.phrase_groups_to_dict(project), file, indent=4)
            project._phrase_groups_dirty = False
            project._phrase_groups_inline_source = ""
            L.d(f"Saved {PROJECT_TEXT_FILE_NAME}: {file_path}")
            return ""
        except Exception as e:
            err = make_error_string(e)
            printt(f"\n{COL_ERROR}{err}\n")
            return err

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

        with project.batch():
            project.book = book
            ProjectBookUtil.sync_flat_text_from_book(project)
            project.markers = []
            project.generate_range_string = ""
            project.realtime_line_range = None

        ProjectTextIOUtil.save_raw_text(project, raw_text)

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

        with project.batch():
            project.book = book
            ProjectBookUtil.sync_flat_text_from_book(project)
            project.markers = []
            project.generate_range_string = ""
            project.realtime_line_range = None

        ProjectTextIOUtil.save_raw_text(project, raw_text)

    @staticmethod
    def save_raw_text(project: Project, raw_text: str) -> None:
        file_path = os.path.join(project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(raw_text)
        except Exception as e:
            L.e(f"Error saving raw text: {e}")

    @staticmethod
    def load_raw_text(project: Project) -> str:
        file_path = os.path.join(project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            L.e(f"Error saving raw text: {e}")
            return ""