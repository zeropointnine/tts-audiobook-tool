from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus import menu_util
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State


def test_options_menu_only_calls_on_select_for_changed_value(monkeypatch) -> None:
    state = cast(State, SimpleNamespace())
    captured_items: list[MenuItem] = []
    selected: list[str] = []

    def capture_menu(**kwargs) -> None:
        captured_items.extend(kwargs["items"])

    monkeypatch.setattr(MenuUtil, "menu", capture_menu)

    MenuUtil.options_menu(
        state=state,
        heading_text="Mode",
        labels=["A", "B"],
        values=["a", "b"],
        current_value="a",
        default_value="a",
        on_select=selected.append,
    )

    captured_items[0].handler(state, captured_items[0])
    captured_items[1].handler(state, captured_items[1])

    assert selected == ["b"]


def test_item_factory_runs_after_status_heading_reconciles_selection(monkeypatch):
    state = cast(State, SimpleNamespace(project=SimpleNamespace(tts_model_type="old-model")))
    rendered = []
    monkeypatch.setattr(menu_util.MenuStatus, "prepare_tts", lambda _: None)
    monkeypatch.setattr(menu_util.MenuStatus, "show_pending_tts_model_hint", lambda *args, **kwargs: None)
    monkeypatch.setattr(MenuUtil, "is_first_submenu", False)
    monkeypatch.setattr(menu_util, "printt", lambda text="": rendered.append(text))
    monkeypatch.setattr(menu_util.ask, "can_hotkey", True)
    monkeypatch.setattr(menu_util.ask, "ask_hotkey", lambda: "\n")

    def print_heading(current, *args, **kwargs):
        current.project.tts_model_type = "new-model"

    monkeypatch.setattr(MenuUtil, "print_heading", print_heading)
    MenuUtil.menu(state, "Voice", lambda current: [
        MenuItem(current.project.tts_model_type, lambda *_: None),
    ])
    assert any("new-model" in text for text in rendered)
    assert not any("old-model" in text for text in rendered)


@pytest.mark.parametrize("is_screen", [False, True])
@pytest.mark.parametrize("subheading, text", [
    ("Model guidance", "Model guidance"),
    (lambda _: "Model guidance", "Model guidance"),
    ("Model guidance\n", "Model guidance"),
    (lambda _: "Model guidance\r\n\r\n", "Model guidance"),
    ("First line\n\nSecond line\n", "First line\n\nSecond line"),
    ("\n\n", ""),
    (None, ""),
    ("", ""),
    (lambda _: "", ""),
])
def test_subheading_has_a_trailing_blank_line(monkeypatch, is_screen, subheading, text):
    state = cast(State, SimpleNamespace())
    printed: list[str] = []
    monkeypatch.setattr(menu_util, "printt", lambda value="": printed.append(value))
    monkeypatch.setattr(MenuUtil, "print_heading", lambda *args, **kwargs: None)

    if is_screen:
        MenuUtil.print_screen_heading(state, "Heading", subheading=subheading)
        assert printed == ([text, ""] if text else [])
    else:
        monkeypatch.setattr(menu_util.MenuStatus, "prepare_tts", lambda _: None)
        monkeypatch.setattr(menu_util.MenuStatus, "show_pending_tts_model_hint", lambda *args, **kwargs: None)
        monkeypatch.setattr(MenuUtil, "is_first_submenu", False)
        monkeypatch.setattr(menu_util.ask, "can_hotkey", True)
        monkeypatch.setattr(menu_util.ask, "ask_hotkey", lambda: "\n")
        MenuUtil.menu(state, "Heading", [MenuItem("Item", lambda *_: None)], subheading=subheading)
        if text:
            assert printed[:2] == [text, ""]
            assert "Item" in printed[2]
        else:
            assert "Item" in printed[0]
