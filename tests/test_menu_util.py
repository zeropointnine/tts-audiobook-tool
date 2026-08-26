from types import SimpleNamespace
from typing import cast

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
