from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonSaveUtil
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil


@pytest.fixture(autouse=True)
def disable_debug_logging(monkeypatch) -> None:
    monkeypatch.setattr(L, "d", lambda _message="": None)


def record_json_writes(monkeypatch) -> list[str]:
    """Patches JsonSaveUtil.save to record artifact types (still saving)."""
    writes: list[str] = []
    real_save = JsonSaveUtil.save

    def record_save(artifact_type, path, payload_factory):
        writes.append(artifact_type.name)
        return real_save(artifact_type, path, payload_factory)

    monkeypatch.setattr(JsonSaveUtil, "save", record_save)
    return writes


def test_project_assignment_does_not_save_until_explicitly_requested(
    tmp_path: Path,
) -> None:
    project = Project(dir_path=str(tmp_path))

    project.language_code = "es"

    assert not (tmp_path / "project.json").exists()

    assert project.save() == ""
    payload = json.loads((tmp_path / "project.json").read_text(encoding="utf-8"))
    assert payload["language_code"] == "es"


def test_project_save_writes_only_project_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = Project(dir_path=str(tmp_path))
    project.book = Book(sections=[BookSection(phrase_groups=[])])
    writes = record_json_writes(monkeypatch)

    assert project.save() == ""
    assert writes == ["PROJECT"]
    assert (tmp_path / "project.json").exists()
    assert not (tmp_path / "project_text.json").exists()


def test_save_book_writes_only_project_text_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = Project(dir_path=str(tmp_path))
    project.book = Book(sections=[BookSection(phrase_groups=[])])
    writes = record_json_writes(monkeypatch)

    assert ProjectTextIOUtil.save_book(project) == ""
    assert writes == ["PROJECT_TEXT"]
    assert not (tmp_path / "project.json").exists()
    assert (tmp_path / "project_text.json").exists()


def test_text_import_commits_each_json_artifact_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = Project(dir_path=str(tmp_path))
    writes = record_json_writes(monkeypatch)

    ProjectTextIOUtil.set_phrase_groups_and_save(
        project=project,
        phrase_groups=[],
        strategy=project.segmentation_strategy,
        max_words=project.max_words,
        language_code=project.language_code,
        dialog_segmentation=project.dialog_segmentation,
        raw_text="",
    )

    assert writes == ["PROJECT_TEXT", "PROJECT"]
