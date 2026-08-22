from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs


@pytest.mark.parametrize(
    ("legacy_value", "expected_hint_keys"),
    [
        (False, ["full_screen_ui"]),
        (True, []),
    ],
)
def test_legacy_menu_preference_is_removed_immediately(
    tmp_path: Path,
    monkeypatch,
    legacy_value: bool,
    expected_hint_keys: list[str],
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    destination.write_text(
        json.dumps({"menu_clears_screen": legacy_value}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))

    shown: list[str] = []
    monkeypatch.setattr(
        hints,
        "show_hint",
        lambda hint, **_kwargs: (shown.append(hint.key), True)[1],
    )

    prefs = Prefs.load(save_if_dirty=False)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert "menu_clears_screen" not in payload
    assert shown == expected_hint_keys
    assert not hasattr(prefs, "menu_clears_screen")


def test_absent_legacy_preference_does_not_trigger_migration_hint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    original_payload = {"llm_url": "https://example.test"}
    destination.write_text(json.dumps(original_payload), encoding="utf-8")
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))
    monkeypatch.setattr(
        hints,
        "show_hint",
        lambda *_args, **_kwargs: pytest.fail("migration hint should not be shown"),
    )

    Prefs.load(save_if_dirty=False)

    assert json.loads(destination.read_text(encoding="utf-8")) == original_payload


def test_new_preferences_do_not_persist_menu_preference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME
    monkeypatch.setattr(Prefs, "get_file_path", staticmethod(lambda: str(destination)))
    monkeypatch.setattr(
        hints,
        "show_hint",
        lambda *_args, **_kwargs: pytest.fail("migration hint should not be shown"),
    )

    prefs = Prefs.new_and_save()

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert "menu_clears_screen" not in payload
    assert not hasattr(prefs, "menu_clears_screen")
