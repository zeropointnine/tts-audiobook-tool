"""Persistence of the confirmed audiobook file selection."""

import pytest

from tts_audiobook_tool.app_types import Book, BookSection, SectionMarkerMode
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil


def test_create_files_round_trip() -> None:
    # Loading must preserve the selection rather than treating fields as edits.
    project = Project(create_files="1-2, 5", chapter_mode=SectionMarkerMode.FILES)
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["create_files"] == "1-2, 5"
    assert "concat_files_range_string" not in payload
    ProjectSerializationUtil.normalize_loaded_project_dict(payload)
    restored = Project.model_validate(payload)
    assert restored.create_files == project.create_files
    assert restored.chapter_mode == project.chapter_mode
    assert "concat_files_range_basis" not in payload


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, ""),
        ({"create_files": None}, ""),
        ({"create_files": [1, 2]}, ""),
        ({"concat_files_range_string": "1-2"}, "1-2"),
        ({"create_files": "3", "concat_files_range_string": "1-2"}, "3"),
    ],
)
def test_create_files_load_defaults_and_legacy_selection(fields, expected) -> None:
    # Old projects remain readable; the new field wins if both keys exist.
    ProjectSerializationUtil.normalize_loaded_project_dict(fields)
    assert Project.model_validate(fields).create_files == expected


@pytest.mark.parametrize("field", ["markers", "marker_indices"])
@pytest.mark.parametrize("markers", [{2}, {1, 2, 3}, set()])
def test_marker_count_change_permanently_clears_selection(field, markers) -> None:
    # Both the public setter and direct field assignment invalidate the selection.
    project = Project(marker_indices={2, 4}, create_files="1, 3")
    setattr(project, field, markers)
    assert project.create_files == ""
    setattr(project, field, {2, 4})
    assert project.create_files == ""
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert Project.model_validate(payload).create_files == ""


def test_moving_markers_preserves_selection() -> None:
    project = Project(marker_indices={2, 4}, create_files="1, 3")
    project.markers = {0, 1, 3}
    assert project.markers == {1, 3}
    assert project.create_files == "1, 3"


@pytest.mark.parametrize("mode", [SectionMarkerMode.FILES, SectionMarkerMode.BOOKMARKS])
def test_changing_mode_permanently_clears_selection(mode) -> None:
    # Re-selecting the current mode is harmless; a round trip cannot revive it.
    other = SectionMarkerMode.BOOKMARKS if mode == SectionMarkerMode.FILES else SectionMarkerMode.FILES
    project = Project.model_validate({"chapter_mode": mode.id, "create_files": "2"})
    project.chapter_mode = mode
    assert project.create_files == "2"
    project.chapter_mode = other
    assert project.create_files == ""
    project.chapter_mode = mode
    assert project.create_files == ""


def test_normalizing_mode_also_clears_selection() -> None:
    # Multi-section books force file mode, including when normalized on save.
    project = Project(book=Book(sections=[BookSection(phrase_groups=[]), BookSection(phrase_groups=[])]))
    project.chapter_mode = SectionMarkerMode.BOOKMARKS
    project.create_files = "2"
    assert project.normalize_chapter_mode() is True
    assert project.chapter_mode == SectionMarkerMode.FILES
    assert project.create_files == ""
