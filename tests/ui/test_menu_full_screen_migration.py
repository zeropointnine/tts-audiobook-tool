from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.prefs import Prefs


@pytest.mark.parametrize(
    ("legacy_value", "expected_hint_keys"),
    [
        (False, ["full_screen_ui"]),
        (True, []),
    ],
)
def test_legacy_menu_preference_is_removed_immediately(
    prefs_path: Path,
    monkeypatch,
    legacy_value: bool,
    expected_hint_keys: list[str],
) -> None:
    prefs_path.write_text(
        json.dumps({"menu_clears_screen": legacy_value}),
        encoding="utf-8",
    )

    shown: list[str] = []
    monkeypatch.setattr(
        hints,
        "show_hint",
        lambda hint, **_kwargs: (shown.append(hint.key), True)[1],
    )

    prefs = Prefs.load(save_if_dirty=False)

    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert "menu_clears_screen" not in payload
    assert shown == expected_hint_keys
    assert not hasattr(prefs, "menu_clears_screen")


def test_new_preferences_round_trip_without_rewriting(prefs_path: Path, monkeypatch) -> None:
    # Retain creation and clean-load coverage without removed-field assertions.
    monkeypatch.setattr(
        hints,
        "show_hint",
        lambda *_args, **_kwargs: pytest.fail("clean preferences should not show a hint"),
    )
    created = Prefs.new_and_save()
    original_payload = prefs_path.read_text(encoding="utf-8")

    loaded = Prefs.load(save_if_dirty=False)

    assert loaded.llm_url == created.llm_url
    assert prefs_path.read_text(encoding="utf-8") == original_payload
