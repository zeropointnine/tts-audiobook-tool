from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs


def test_prefs_assignment_does_not_save_until_explicitly_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(tmp_path / PREFS_FILE_NAME)))
    prefs = Prefs()

    prefs.llm_url = "https://example.com"

    destination = tmp_path / PREFS_FILE_NAME
    assert not destination.exists()

    assert prefs.save() == ""
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["llm_url"] == "https://example.com"


def test_compound_prefs_change_is_persisted_with_one_explicit_save(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(tmp_path / PREFS_FILE_NAME)))
    prefs = Prefs()
    writes: list[JsonArtifactType] = []
    real_save = JsonSaveUtil.save

    def record_save(artifact_type, path, payload_factory):
        writes.append(artifact_type)
        return real_save(artifact_type, path, payload_factory)

    monkeypatch.setattr(JsonSaveUtil, "save", record_save)

    prefs.llm_system_prompt = "Be concise."
    prefs.system_prompt_preset = ""

    assert writes == []
    assert prefs.save() == ""
    assert writes == [JsonArtifactType.PREFS]

    payload = json.loads((tmp_path / PREFS_FILE_NAME).read_text(encoding="utf-8"))
    assert payload["llm_system_prompt"] == "Be concise."
    assert payload["system_prompt_preset"] == ""


def test_save_gen_log_defaults_false_and_round_trips(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(tmp_path / PREFS_FILE_NAME)))

    assert Prefs().save_gen_log is False

    prefs = Prefs()
    prefs.save_gen_log = True
    prefs.save()
    payload = json.loads((tmp_path / PREFS_FILE_NAME).read_text(encoding="utf-8"))
    assert payload["save_gen_log"] is True

    assert Prefs.load().save_gen_log is True


def test_save_gen_log_invalid_value_is_normalized_to_false(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    destination.write_text(json.dumps({"save_gen_log": "yes"}), encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    prefs = Prefs.load()

    assert prefs.save_gen_log is False
    normalized = json.loads(destination.read_text(encoding="utf-8"))
    assert normalized["save_gen_log"] is False


def test_hint_mutations_require_explicit_save(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(tmp_path / PREFS_FILE_NAME)))
    prefs = Prefs()

    prefs.set_hint_true("example")

    destination = tmp_path / PREFS_FILE_NAME
    assert not destination.exists()
    assert prefs.get_hint("example")

    assert prefs.save() == ""
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["hints"] == {"example": True}


def test_malformed_json_is_quarantined_before_defaults_are_created(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    malformed = '{"llm_url": "https://recover-me.example"'
    destination.write_text(malformed, encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    prefs = Prefs.load()

    quarantined = list(tmp_path.glob(f"{PREFS_FILE_NAME}.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == malformed
    assert json.loads(destination.read_text(encoding="utf-8"))["llm_url"] == ""
    assert prefs.llm_url == ""
    output = capsys.readouterr().out
    assert "Preferences recovery warning" in output
    assert str(quarantined[0]) in output


@pytest.mark.parametrize("root_value", [[], "text", 7, True, None])
def test_non_object_json_root_is_quarantined(
    tmp_path: Path,
    monkeypatch,
    root_value,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    original = json.dumps(root_value)
    destination.write_text(original, encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    Prefs.load()

    quarantined = list(tmp_path.glob(f"{PREFS_FILE_NAME}.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == original
    assert isinstance(json.loads(destination.read_text(encoding="utf-8")), dict)


def test_invalid_field_is_salvaged_without_quarantining_parseable_object(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    destination.write_text(
        json.dumps({
            "llm_url": "https://valid.example",
            "hints": ["invalid"],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    prefs = Prefs.load()

    assert prefs.llm_url == "https://valid.example"
    assert not prefs.get_hint("invalid")
    assert list(tmp_path.glob(f"{PREFS_FILE_NAME}.*.corrupt")) == []
    normalized = json.loads(destination.read_text(encoding="utf-8"))
    assert normalized["llm_url"] == "https://valid.example"
    assert normalized["hints"] == {}


def test_read_error_does_not_rename_or_overwrite_preferences(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    original = '{"llm_url": "https://valid.example"}'
    destination.write_text(original, encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    def fail_open(*_args, **_kwargs):
        raise PermissionError("read denied")

    monkeypatch.setattr("builtins.open", fail_open)

    with pytest.raises(RuntimeError, match="Error reading preferences file"):
        Prefs.load()

    assert destination.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(f"{PREFS_FILE_NAME}.*.corrupt")) == []


def test_quarantine_failure_preserves_original_and_stops_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    malformed = "{bad json"
    destination.write_text(malformed, encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    def fail_quarantine(_file_path: str) -> str:
        raise PermissionError("rename denied")

    monkeypatch.setattr(Prefs, "quarantine_file", staticmethod(fail_quarantine))

    with pytest.raises(RuntimeError, match="could not be preserved"):
        Prefs.load()

    assert destination.read_text(encoding="utf-8") == malformed
    assert list(tmp_path.glob(f"{PREFS_FILE_NAME}.*.corrupt")) == []


def test_default_save_failure_keeps_quarantine_and_uses_in_memory_defaults(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    malformed = "{bad json"
    destination.write_text(malformed, encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))
    monkeypatch.setattr(Prefs, "save", lambda _self: "disk full")

    prefs = Prefs.load()

    assert prefs.llm_url == ""
    assert not destination.exists()
    quarantined = list(tmp_path.glob(f"{PREFS_FILE_NAME}.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == malformed
    assert "continuing with in-memory defaults" in capsys.readouterr().out


def test_quarantine_path_is_unique_when_timestamp_collides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME

    class FixedDateTime:
        @classmethod
        def now(cls, _timezone):
            return datetime(2026, 8, 1, 17, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("tts_audiobook_tool.prefs.datetime", FixedDateTime)
    destination.write_text("first", encoding="utf-8")
    first = Prefs.quarantine_file(str(destination))
    destination.write_text("second", encoding="utf-8")
    second = Prefs.quarantine_file(str(destination))

    assert first != second
    assert first.endswith(".corrupt")
    assert second.endswith("-1.corrupt")
    assert Path(first).read_text(encoding="utf-8") == "first"
    assert Path(second).read_text(encoding="utf-8") == "second"


def test_sgl_omni_url_defaults_to_empty() -> None:
    assert Prefs().sgl_omni_url == ""


def test_sgl_omni_url_load_normalizes_unset_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    # Missing file => missing key => empty
    assert Prefs.load().sgl_omni_url == ""

    destination.write_text(json.dumps({"sgl_omni_url": ""}), encoding="utf-8")
    assert Prefs.load().sgl_omni_url == ""

    destination.write_text(json.dumps({"sgl_omni_url": "   "}), encoding="utf-8")
    assert Prefs.load().sgl_omni_url == ""

    destination.write_text(json.dumps({"sgl_omni_url": 42}), encoding="utf-8")
    assert Prefs.load().sgl_omni_url == ""

    destination.write_text(
        json.dumps({"sgl_omni_url": "http://example.test:9009"}),
        encoding="utf-8",
    )
    assert Prefs.load().sgl_omni_url == "http://example.test:9009"


def test_sgl_omni_url_setter_strips_and_allows_empty() -> None:
    prefs = Prefs()
    prefs.sgl_omni_url = "  http://example.test:9009  "
    assert prefs.sgl_omni_url == "http://example.test:9009"

    prefs.sgl_omni_url = "   "
    assert prefs.sgl_omni_url == ""


def test_sgl_omni_url_round_trips_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    prefs = Prefs()
    assert prefs.save() == ""

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["sgl_omni_url"] == ""
    assert Prefs.load().sgl_omni_url == ""
