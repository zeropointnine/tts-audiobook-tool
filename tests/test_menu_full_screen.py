from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.menu_status import MenuStatus
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.state import State


def make_state() -> State:
    return cast(State, SimpleNamespace(prefs=Prefs()))


def test_menu_heading_clears_screen_and_prints_status(monkeypatch) -> None:
    state = make_state()
    clears: list[str] = []
    statuses: list[State] = []
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.os.system", clears.append)
    monkeypatch.setattr(MenuStatus, "print_block", statuses.append)

    MenuUtil.print_heading(state, "Heading")

    assert len(clears) == 1
    assert statuses == [state]


def test_non_menu_heading_clears_without_status(monkeypatch) -> None:
    state = make_state()
    clears: list[str] = []
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.os.system", clears.append)
    monkeypatch.setattr(
        MenuStatus,
        "print_block",
        lambda _state: pytest.fail("non-menu heading should not print menu status"),
    )

    MenuUtil.print_heading(state, "Heading", non_menu=True)

    assert len(clears) == 1


def test_menu_calls_on_shown_after_render(monkeypatch) -> None:
    state = make_state()
    shown: list[bool] = []
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.os.system", lambda _: None)
    monkeypatch.setattr(MenuStatus, "print_block", lambda _: None)
    monkeypatch.setattr("tts_audiobook_tool.tts.Tts.update_tts_type", lambda: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.ask.ask_hotkey", lambda: "q")

    MenuUtil.menu(
        state,
        "Heading",
        [MenuItem("Quit", lambda *_: True, hotkey="q")],
        is_submenu=False,
        on_shown=lambda: shown.append(True),
    )

    assert shown == [True]


def test_dont_clear_suppresses_clear_and_status(monkeypatch) -> None:
    state = make_state()
    monkeypatch.setattr(
        "tts_audiobook_tool.menus.menu_util.os.system",
        lambda _command: pytest.fail("screen should not be cleared"),
    )
    monkeypatch.setattr(
        MenuStatus,
        "print_block",
        lambda _state: pytest.fail("status should not be printed"),
    )

    MenuUtil.print_heading(state, "Heading", dont_clear=True)
