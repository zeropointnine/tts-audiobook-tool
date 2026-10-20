from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from tts_audiobook_tool.app_support.JsonSaveUtil import (
    JsonArtifactType,
    JsonSaveUtil,
)


def test_save_atomically_replaces_json_file(tmp_path: Path) -> None:
    destination = tmp_path / "project.json"
    destination.write_text('{"old": true}', encoding="utf-8")

    error = JsonSaveUtil.save(
        JsonArtifactType.PROJECT,
        destination,
        lambda: {"new": "value"},
    )

    assert error == ""
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": "value"}
    assert list(tmp_path.glob(".project.json.*.tmp")) == []


def test_payload_factory_failure_preserves_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "project.json"
    original = '{"old": true}'
    destination.write_text(original, encoding="utf-8")

    def fail() -> dict:
        raise RuntimeError("factory failed")

    error = JsonSaveUtil.save(JsonArtifactType.PROJECT, destination, fail)

    assert "factory failed" in error
    assert destination.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [destination]


def test_non_finite_number_failure_preserves_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "prefs.json"
    original = '{"old": true}'
    destination.write_text(original, encoding="utf-8")

    error = JsonSaveUtil.save(
        JsonArtifactType.PREFS,
        destination,
        lambda: {"invalid": float("nan")},
    )

    assert "Out of range float values" in error
    assert destination.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [destination]


def test_replace_failure_preserves_existing_file_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "project_text.json"
    original = '{"old": true}'
    destination.write_text(original, encoding="utf-8")

    def fail_replace(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("tts_audiobook_tool.app_support.JsonSaveUtil.os.replace", fail_replace)

    error = JsonSaveUtil.save(
        JsonArtifactType.PROJECT_TEXT,
        destination,
        lambda: {"new": True},
    )

    assert "replace failed" in error
    assert destination.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".project_text.json.*.tmp")) == []


def test_same_artifact_payload_factories_do_not_overlap(tmp_path: Path) -> None:
    destination = tmp_path / "project.json"
    first_factory_entered = threading.Event()
    release_first_factory = threading.Event()
    second_factory_entered = threading.Event()

    def first_factory() -> dict:
        first_factory_entered.set()
        assert release_first_factory.wait(timeout=2)
        return {"writer": 1}

    def second_factory() -> dict:
        second_factory_entered.set()
        return {"writer": 2}

    first = threading.Thread(
        target=JsonSaveUtil.save,
        args=(JsonArtifactType.PROJECT, destination, first_factory),
    )
    second = threading.Thread(
        target=JsonSaveUtil.save,
        args=(JsonArtifactType.PROJECT, destination, second_factory),
    )

    first.start()
    assert first_factory_entered.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    assert not second_factory_entered.is_set()

    release_first_factory.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_factory_entered.is_set()
    assert json.loads(destination.read_text(encoding="utf-8")) == {"writer": 2}
