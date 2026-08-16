from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.start import Start, resolve_project_override


def _valid(path: str) -> str:
    return ""

def _invalid(path: str) -> str:
    return f"Doesn't exist: {path}"

def _report_should_not_be_called() -> None:
    raise AssertionError("report_invalid should not be called")

def _confirm_should_not_be_called(message: str) -> bool:
    raise AssertionError(f"confirm should not be called (got {message!r})")

def _enter_should_not_be_called() -> None:
    raise AssertionError("enter_to_continue should not be called")


# --- Pure helper ---

def test_valid_override_returns_path_and_continues():
    assert resolve_project_override(
        "/p/new", "/p/stored", _valid, _report_should_not_be_called, _confirm_should_not_be_called, _enter_should_not_be_called
    ) == ("/p/new", True)

def test_invalid_with_stored_and_yes_keeps_stored():
    # report_invalid is expected to be called (error is shown before the prompt)
    assert resolve_project_override(
        "/p/bad", "/p/stored", _invalid, lambda: None, lambda m: True, _enter_should_not_be_called
    ) == ("/p/stored", True)

def test_invalid_with_stored_and_no_aborts():
    assert resolve_project_override(
        "/p/bad", "/p/stored", _invalid, lambda: None, lambda m: False, _enter_should_not_be_called
    ) == ("", False)

def test_invalid_without_stored_prompts_enter_and_continues_empty():
    calls: list[str] = []
    assert resolve_project_override(
        "/p/bad", "", _invalid, lambda: calls.append("report"), _confirm_should_not_be_called,
        lambda: calls.append("enter"),
    ) == ("", True)
    assert calls == ["report", "enter"]

def test_invalid_reports_before_any_prompt():
    calls: list[str] = []
    resolve_project_override(
        "/p/bad", "/p/stored", _invalid, lambda: calls.append("report"),
        lambda m: calls.append("confirm") or True, lambda: None,
    )
    assert calls == ["report", "confirm"]


# --- apply_project_override integration (app mode) ---

def _make_start(monkeypatch, args: list[str], stored_project_dir: str, tmp_path: Path) -> Start:
    """Builds a Start instance with the given CLI args, bypassing argparse."""
    monkeypatch.setattr("sys.argv", ["tts_audiobook_tool", *args])

    prefs_file = tmp_path / "prefs" / PREFS_FILE_NAME
    prefs_file.parent.mkdir(parents=True, exist_ok=True)
    prefs_file.write_text(json.dumps({"project_dir": stored_project_dir}), encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(prefs_file)))

    start = Start.__new__(Start)
    start.is_server = "--server" in args
    start.server_host = "127.0.0.1"
    start.server_port = 5001
    start.project_path = args[args.index("--project") + 1] if "--project" in args else ""
    return start

def _stored_project_dir(tmp_path: Path) -> str:
    payload = json.loads((tmp_path / "prefs" / PREFS_FILE_NAME).read_text(encoding="utf-8"))
    return payload["project_dir"]

def test_no_flag_is_noop(tmp_path: Path, monkeypatch, capsys):
    start = _make_start(monkeypatch, [], "/p/stored", tmp_path)

    start.apply_project_override()

    assert _stored_project_dir(tmp_path) == "/p/stored"
    assert capsys.readouterr().out == ""

def test_valid_override_updates_prefs_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ProjectLoadUtil, "is_valid_project_dir", staticmethod(lambda d: ""))
    start = _make_start(monkeypatch, ["--project", "/p/new"], "/p/stored", tmp_path)

    start.apply_project_override()

    assert _stored_project_dir(tmp_path) == "/p/new"

def test_invalid_no_stored_prompts_enter_and_continues_with_empty_path(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(ProjectLoadUtil, "is_valid_project_dir", staticmethod(_invalid))
    enter_calls: list[str] = []
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue", lambda *args, **kwargs: enter_calls.append("enter"))
    start = _make_start(monkeypatch, ["--project", "/p/bad"], "", tmp_path)

    start.apply_project_override()

    assert _stored_project_dir(tmp_path) == ""
    assert "Bad project path." in capsys.readouterr().out
    assert enter_calls == ["enter"]

def test_invalid_stored_confirm_yes_keeps_stored(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(ProjectLoadUtil, "is_valid_project_dir", staticmethod(_invalid))
    confirm_messages: list[str] = []
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_confirm", lambda message: confirm_messages.append(message) or True)
    start = _make_start(monkeypatch, ["--project", "/p/bad"], "/p/stored", tmp_path)

    start.apply_project_override()

    assert _stored_project_dir(tmp_path) == "/p/stored"
    out = capsys.readouterr().out
    assert "Bad project path." in out
    assert confirm_messages == ["Do you want to load the last used project path (/p/stored)?"]

def test_invalid_stored_confirm_no_exits(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ProjectLoadUtil, "is_valid_project_dir", staticmethod(_invalid))
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_confirm", lambda message: False)
    start = _make_start(monkeypatch, ["--project", "/p/bad"], "/p/stored", tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        start.apply_project_override()

    assert excinfo.value.code == 1
    assert _stored_project_dir(tmp_path) == "/p/stored"


# --- Server mode: validate + exit on failure; never touches prefs, no prompt ---

def test_server_invalid_path_exits_without_touching_prefs_or_prompting(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(ProjectLoadUtil, "is_valid_project_dir", staticmethod(_invalid))
    confirm_messages: list[str] = []
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_confirm", lambda message: confirm_messages.append(message) or False)
    start = _make_start(monkeypatch, ["--server", "--project", "/p/bad"], "/p/stored", tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        start.apply_project_override()

    assert excinfo.value.code == 1
    assert _stored_project_dir(tmp_path) == "/p/stored"
    assert confirm_messages == []
    assert "does not appear to be a project directory" in capsys.readouterr().out

def test_server_valid_path_is_noop_for_prefs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ProjectLoadUtil, "is_valid_project_dir", staticmethod(lambda d: ""))
    start = _make_start(monkeypatch, ["--server", "--project", "/p/new"], "/p/stored", tmp_path)

    start.apply_project_override()

    assert _stored_project_dir(tmp_path) == "/p/stored"