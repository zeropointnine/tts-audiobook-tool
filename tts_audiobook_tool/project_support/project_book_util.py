from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings
from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.util import make_file_line_ranges

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class ProjectBookUtil:
    """
    Book/phrase-group compatibility helpers for `Project`.
    """

    @staticmethod
    def make_book_from_flat_compatibility_fields(
            phrase_groups: list[PhraseGroup],
            section_start_indices: list[int] | None,
            segmentation_settings: BookSegmentationSettings,
            text_source_kind: str,
            audio_source_kind: str,
            title: str="",
            section_titles: list[str] | None=None,
    ) -> Book:
        section_start_indices = sorted(section_start_indices or [])
        section_titles = section_titles or []
        valid_section_starts = [index for index in section_start_indices if 0 < index <= len(phrase_groups)]
        starts = [0, *valid_section_starts]
        sections: list[BookSection] = []
        for section_index, start in enumerate(starts):
            end = starts[section_index + 1] if section_index + 1 < len(starts) else len(phrase_groups)
            title_value = section_titles[section_index] if section_index < len(section_titles) else ""
            sections.append(BookSection(
                phrase_groups=phrase_groups[start:end],
                title=title_value,
            ))
        if not sections:
            sections = [BookSection(phrase_groups=[])]
        return Book(
            sections=sections,
            title=title,
            text_source_kind=text_source_kind,
            audio_source_kind=audio_source_kind,
            segmentation_settings=segmentation_settings,
        )

    @staticmethod
    def sync_parse_dict_flat_text_from_book(d: dict) -> None:
        book = d.get('book')
        if not isinstance(book, Book):
            return
        settings = book.segmentation_settings
        d['phrase_groups'] = book.phrase_groups()
        d['applied_language_code'] = settings.language_code
        d['applied_max_words'] = settings.max_words_per_segment
        d['applied_strategy'] = settings.strategy

    @staticmethod
    def sync_flat_text_from_book(project: Project) -> None:
        settings = project.book.segmentation_settings
        super(type(project), project).__setattr__('phrase_groups', project.book.phrase_groups())
        super(type(project), project).__setattr__('applied_language_code', settings.language_code)
        super(type(project), project).__setattr__('applied_max_words', settings.max_words_per_segment)
        super(type(project), project).__setattr__('applied_strategy', settings.strategy)

    @staticmethod
    def ensure_book_from_flat_text(project: Project) -> None:
        if project.book.sections or not project.phrase_groups:
            return
        settings = BookSegmentationSettings(
            language_code=project.applied_language_code,
            max_words_per_segment=project.applied_max_words,
            strategy=project.applied_strategy or BookSegmentationSettings().strategy,
        )
        super(type(project), project).__setattr__('book', Book(
            sections=[BookSection(phrase_groups=project.phrase_groups)],
            segmentation_settings=settings,
            text_source_kind="legacy_flat",
            audio_source_kind="unknown",
        ))

    @staticmethod
    def get_book_segmentation_settings(project: Project) -> BookSegmentationSettings:
        if project.book.sections:
            return project.book.segmentation_settings
        return BookSegmentationSettings(
            language_code=project.applied_language_code,
            max_words_per_segment=project.applied_max_words,
            strategy=project.applied_strategy or BookSegmentationSettings().strategy,
        )

    @staticmethod
    def get_flat_phrase_groups(project: Project) -> list[PhraseGroup]:
        return project.book.phrase_groups() if project.book.sections else project.phrase_groups

    @staticmethod
    def get_section_start_indices(project: Project) -> list[int]:
        if project.book.sections:
            return project.book.section_start_indices()
        return [0, *project.markers]

    @staticmethod
    def get_section_ranges(project: Project) -> list[tuple[int, int]]:
        if project.book.sections:
            return project.book.section_ranges()
        return make_file_line_ranges(project.markers, len(project.phrase_groups))

    @staticmethod
    def phrase_groups_to_dict(project: Project) -> dict:
        ProjectBookUtil.ensure_book_from_flat_text(project)
        return book_to_project_text_json_dict(project.book)