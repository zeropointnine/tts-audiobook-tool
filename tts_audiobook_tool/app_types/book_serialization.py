from __future__ import annotations

import json
from typing import Any

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.util import make_error_string


BOOK_FORMAT = "book.v1"
PHRASE_GROUPS_FORMAT = "phrase_groups.v1"
LEGACY_LIST_FORMAT = "legacy_list"


def get_project_text_format(value: Any) -> str | None:
    if isinstance(value, list):
        return LEGACY_LIST_FORMAT
    if isinstance(value, dict):
        if value.get("format") == BOOK_FORMAT:
            return BOOK_FORMAT
        if value.get("format") == PHRASE_GROUPS_FORMAT or "phrase_groups" in value:
            return PHRASE_GROUPS_FORMAT
    return None


def book_to_json_dict(book: Book) -> dict[str, Any]:
    return {
        "title": book.title,
        "text_source_kind": book.text_source_kind,
        "audio_source_kind": book.audio_source_kind,
        "segmentation_settings": segmentation_settings_to_json_dict(book.segmentation_settings),
        "sections": [section_to_json_dict(section) for section in book.sections],
    }


def book_to_project_text_json_dict(book: Book) -> dict[str, Any]:
    return {
        "format": BOOK_FORMAT,
        "book": book_to_json_dict(book),
    }


def book_from_json_dict(value: Any) -> Book | str:
    if not isinstance(value, dict):
        return f"Expected book dict. Value: {value}"

    settings = segmentation_settings_from_json_dict(value.get("segmentation_settings", {}))
    if isinstance(settings, str):
        return settings

    raw_sections = value.get("sections")
    if not isinstance(raw_sections, list):
        return "Book missing or invalid 'sections'"

    sections: list[BookSection] = []
    for raw_section in raw_sections:
        section = section_from_json_dict(raw_section)
        if isinstance(section, str):
            return section
        sections.append(section)

    return Book(
        sections=sections,
        title=get_string_value(value, "title"),
        text_source_kind=get_string_value(value, "text_source_kind"),
        audio_source_kind=get_string_value(value, "audio_source_kind"),
        segmentation_settings=settings,
    )


def book_from_project_text_json_dict(
        value: Any,
        legacy_segmentation_settings: BookSegmentationSettings | None = None,
) -> Book | str:
    if isinstance(value, list):
        return legacy_phrase_groups_to_book(value, legacy_segmentation_settings)

    if not isinstance(value, dict):
        return f"Project text file bad type: {type(value)}"

    format_value = value.get("format", PHRASE_GROUPS_FORMAT)
    if format_value == BOOK_FORMAT:
        if "book" not in value:
            return "Project text file missing 'book'"
        return book_from_json_dict(value["book"])

    if format_value == PHRASE_GROUPS_FORMAT or "phrase_groups" in value:
        if "phrase_groups" not in value:
            return "Project text file missing 'phrase_groups'"
        return legacy_phrase_groups_to_book(value["phrase_groups"], legacy_segmentation_settings)

    return f"Unsupported project text format: {format_value}"


def load_book_from_project_text_file(
        path: str,
        legacy_segmentation_settings: BookSegmentationSettings | None = None,
) -> Book | str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as e:
        return f"Error loading project text: {e}"

    result = book_from_project_text_json_dict(payload, legacy_segmentation_settings)
    if isinstance(result, str):
        return f"Error parsing project text: {result}"
    return result


def save_book_to_project_text_file(path: str, book: Book) -> str:
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(book_to_project_text_json_dict(book), file, indent=4)
        return ""
    except Exception as e:
        return make_error_string(e)


def segmentation_settings_to_json_dict(settings: BookSegmentationSettings) -> dict[str, Any]:
    return {
        "language_code": settings.language_code,
        "max_words_per_segment": settings.max_words_per_segment,
        "strategy": settings.strategy.id,
    }


def segmentation_settings_from_json_dict(value: Any) -> BookSegmentationSettings | str:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        return f"Expected segmentation_settings dict. Value: {value}"

    language_code = value.get("language_code", "")
    if not isinstance(language_code, str):
        language_code = ""

    max_words = value.get("max_words_per_segment", 0)
    if not isinstance(max_words, int) or max_words < 0:
        max_words = 0

    raw_strategy = value.get("strategy", "")
    strategy = SegmentationStrategy.from_id(raw_strategy) if isinstance(raw_strategy, str) else None
    if strategy is None:
        strategy = BookSegmentationSettings().strategy

    return BookSegmentationSettings(
        language_code=language_code,
        max_words_per_segment=max_words,
        strategy=strategy,
    )


def section_to_json_dict(section: BookSection) -> dict[str, Any]:
    return {
        "title": section.title,
        "phrase_groups": PhraseGroup.phrase_groups_to_json_list(section.phrase_groups),
    }


def section_from_json_dict(value: Any) -> BookSection | str:
    if not isinstance(value, dict):
        return f"Expected section dict. Value: {value}"
    if "phrase_groups" not in value:
        return "Book section missing 'phrase_groups'"

    phrase_groups = PhraseGroup.phrase_groups_from_json_list(value["phrase_groups"])
    if isinstance(phrase_groups, str):
        return phrase_groups

    return BookSection(
        phrase_groups=phrase_groups,
        title=get_string_value(value, "title"),
    )


def legacy_phrase_groups_to_book(
        value: Any,
        legacy_segmentation_settings: BookSegmentationSettings | None = None,
) -> Book | str:
    phrase_groups = PhraseGroup.phrase_groups_from_json_list(value)
    if isinstance(phrase_groups, str):
        return phrase_groups

    return Book(
        text_source_kind="legacy_flat",
        audio_source_kind="unknown",
        segmentation_settings=legacy_segmentation_settings or BookSegmentationSettings(),
        sections=[BookSection(phrase_groups=phrase_groups)],
    )


def get_string_value(value: dict[str, Any], key: str) -> str:
    raw_value = value.get(key, "")
    return raw_value if isinstance(raw_value, str) else ""