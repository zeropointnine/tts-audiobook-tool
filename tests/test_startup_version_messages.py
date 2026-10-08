from __future__ import annotations

import json
from pathlib import Path

from tts_audiobook_tool.app_support import app_hint_util
from tts_audiobook_tool.app_support import hints as hints_module
from tts_audiobook_tool.constants_startup_messaging import (
    STARTUP_VERSION_CODE,
    STARTUP_VERSION_MESSAGES,
)
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs

MESSAGES = [(2, "stuff"), (3, "added ModelX"), (4, "migrated x to y")]


def _patch(monkeypatch, tmp_path: Path, current: int, messages=MESSAGES) -> list:
    """Redirects the prefs file, stubs the hint display, and sets the constants. Returns the shown hints."""
    import tts_audiobook_tool.constants_startup_messaging as c
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(tmp_path / PREFS_FILE_NAME)))
    monkeypatch.setattr(c, "STARTUP_VERSION_CODE", current)
    monkeypatch.setattr(c, "STARTUP_VERSION_MESSAGES", messages)
    shown: list = []
    monkeypatch.setattr(hints_module, "show_hint", lambda h, **kw: (shown.append(h), True)[1])
    return shown


def test_constant_matches_highest_message_code() -> None:
    # Guards against bumping one of the two constants and forgetting the other
    if STARTUP_VERSION_MESSAGES:
        assert STARTUP_VERSION_CODE == max(code for code, _ in STARTUP_VERSION_MESSAGES)
    else:
        assert STARTUP_VERSION_CODE == 1


def test_qualifying_messages_are_only_those_after_stored_code() -> None:
    # Stored 2, current 4: only the messages for 3 and 4 qualify
    assert app_hint_util.get_startup_version_messages(2, 4, MESSAGES) == ["added ModelX", "migrated x to y"]
    # Messages beyond the current code never qualify
    assert app_hint_util.get_startup_version_messages(2, 3, MESSAGES) == ["added ModelX"]
    # No stored value means no history: nothing qualifies
    assert app_hint_util.get_startup_version_messages(None, 4, MESSAGES) == []


def test_upgrade_shows_hint_then_saves_code(monkeypatch, tmp_path: Path) -> None:
    shown = _patch(monkeypatch, tmp_path, current=4)
    prefs = Prefs(startup_version_code=2)

    app_hint_util.show_startup_version_messages(prefs)

    assert len(shown) == 1
    assert shown[0].heading == "New features since the last time you ran tts-audiobook-tool:"
    assert shown[0].text == "- added ModelX\n- migrated x to y"
    assert prefs.startup_version_code == 4
    assert json.loads((tmp_path / PREFS_FILE_NAME).read_text(encoding="utf-8"))["startup_version_code"] == 4


def test_no_stored_code_is_populated_silently(monkeypatch, tmp_path: Path) -> None:
    # New user, or user updating to the first version with this feature
    shown = _patch(monkeypatch, tmp_path, current=4)
    prefs = Prefs()

    app_hint_util.show_startup_version_messages(prefs)

    assert shown == []
    assert prefs.startup_version_code == 4


def test_downgrade_leaves_stored_code_alone(monkeypatch, tmp_path: Path) -> None:
    shown = _patch(monkeypatch, tmp_path, current=3)
    prefs = Prefs(startup_version_code=4)

    app_hint_util.show_startup_version_messages(prefs)

    assert shown == []
    assert prefs.startup_version_code == 4
    assert not (tmp_path / PREFS_FILE_NAME).exists()


def test_load_distinguishes_missing_and_invalid_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(tmp_path / PREFS_FILE_NAME)))
    path = tmp_path / PREFS_FILE_NAME

    # Existing file from before the feature: stays None, with no rewrite forced by it
    path.write_text(json.dumps({}), encoding="utf-8")
    assert Prefs.load().startup_version_code is None

    path.write_text(json.dumps({"startup_version_code": 3}), encoding="utf-8")
    assert Prefs.load().startup_version_code == 3

    path.write_text(json.dumps({"startup_version_code": "x"}), encoding="utf-8")
    assert Prefs.load().startup_version_code is None
