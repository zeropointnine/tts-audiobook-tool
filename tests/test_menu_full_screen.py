from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.menu_status import MenuStatus
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.state import State


def make_state() -> State:
    return cast(State, SimpleNamespace(prefs=Prefs(), project=object()))


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
    monkeypatch.setattr(MenuStatus, "prepare_tts", lambda _: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.ask.ask_hotkey", lambda: "q")

    MenuUtil.menu(
        state,
        "Heading",
        [MenuItem("Quit", lambda *_: True, hotkey="q")],
        is_submenu=False,
        on_shown=lambda: shown.append(True),
    )

    assert shown == [True]


def test_blank_line_before_item_is_rendered_without_changing_superlabel_spacing(monkeypatch) -> None:
    state = make_state()
    output: list[str] = []
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.os.system", lambda _: None)
    monkeypatch.setattr(MenuStatus, "print_block", lambda _: None)
    monkeypatch.setattr(MenuStatus, "prepare_tts", lambda _: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.ask.ask_hotkey", lambda: "q")
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.printt", lambda value="": output.append(value))

    MenuUtil.menu(
        state,
        "Heading",
        [
            MenuItem("First", lambda *_: None, hotkey="1"),
            MenuItem(
                "Optional",
                lambda *_: None,
                hotkey="2",
                superlabel="Tail",
                superlabel_no_blank_line=True,
                blank_line_before=True,
            ),
            MenuItem("Quit", lambda *_: True, hotkey="q"),
        ],
        is_submenu=False,
    )

    optional_index = next(index for index, value in enumerate(output) if "Optional" in value)
    assert output[optional_index - 2] == ""
    assert "Tail" in output[optional_index - 1]
    assert output.count("") == 4  # status gap, heading gap, item separator, footer


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


@pytest.mark.parametrize("is_submenu", [False, True])
def test_deferred_tts_hint_follows_menu_and_on_shown_without_pausing(monkeypatch, capsys, is_submenu):
    model = TtsModelType.require_by_id("echo_tts_audiocpp")
    state = cast(State, SimpleNamespace(
        prefs=Prefs(), project=Project(tts_model_type="chatterbox_local"),
        pending_tts_model_change=None,
    ))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [model])
    monkeypatch.setattr(Tts, "bind_project", lambda _: None)
    monkeypatch.setattr(Tts, "get_active_type", lambda: model)
    monkeypatch.setattr(MenuUtil, "is_first_submenu", True)
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.os.system", lambda _: None)
    monkeypatch.setattr(MenuStatus, "print_block", MenuStatus.prepare_tts)
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("FYI must not request acknowledgment"))
    monkeypatch.setattr("tts_audiobook_tool.app_support.hints.show_hint",
                        lambda *_: pytest.fail("FYI must not animate or save preferences"))
    rendered = []

    def read_input():
        rendered.append(strip_ansi_codes(capsys.readouterr().out))
        return "q"

    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.ask.ask_hotkey", read_input)
    MenuUtil.menu(
        state, "Heading", [MenuItem("Quit", lambda *_: len(rendered) == 2, hotkey="q")],
        is_submenu=is_submenu, on_shown=lambda: print("Existing on_shown output"),
    )

    assert len(rendered) == 2
    first = rendered[0]
    assert first.index("Heading") < first.index("Quit") < first.index("Existing on_shown output")
    assert first.index("Existing on_shown output") < first.index("🔔 FYI")
    if is_submenu:
        assert first.index("to go back one level") < first.index("🔔 FYI")
    assert "previously using TTS model Chatterbox TTS" in first
    assert f"currently active model, {model.value.ui['proper_name']}" in first
    assert "Press enter:" not in first
    assert "FYI" not in rendered[1]
    assert state.pending_tts_model_change is None


def test_local_fyi_skips_submenu_and_shows_once_at_first_main_menu(monkeypatch, capsys):
    state = State.for_worker(Prefs())
    state._project = Project(tts_model_type="echo_tts_audiocpp")
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("chatterbox_local"),))
    monkeypatch.setattr(Tts, "bind_project", lambda _: None)
    monkeypatch.setattr(Tts, "get_active_type", lambda: TtsModelType.require_by_id("chatterbox_local"))
    monkeypatch.setattr(MenuUtil, "is_first_submenu", False)
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.os.system", lambda _: None)
    monkeypatch.setattr(MenuStatus, "print_block", MenuStatus.prepare_tts)
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("Local startup FYI must not pause"))
    rendered = []

    def read_input():
        rendered.append(strip_ansi_codes(capsys.readouterr().out))
        return "q"

    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.ask.ask_hotkey", read_input)
    MenuUtil.menu(state, "Early submenu", [MenuItem("Back", lambda *_: True, hotkey="q")])
    assert "FYI" not in rendered[0]
    assert state.pending_tts_model_change is not None
    assert not state.has_shown_main_menu

    def main_shown():
        state.mark_main_menu_shown()
        print("Main menu on_shown output")

    def action(*_):
        if len(rendered) == 2:
            state.project.tts_model_type = "echo_tts_audiocpp"
            return None
        return True

    MenuUtil.menu(
        state, "Main menu", [MenuItem("Quit", action, hotkey="q")],
        is_submenu=False, on_shown=main_shown,
    )

    assert len(rendered) == 3
    assert rendered[1].index("Quit") < rendered[1].index("Main menu on_shown output")
    assert rendered[1].index("Main menu on_shown output") < rendered[1].index("🔔 FYI")
    assert "previously using TTS model Echo-TTS" in rendered[1]
    assert "currently active model, Chatterbox TTS" in rendered[1]
    assert "Press enter:" not in rendered[1]
    assert "FYI" not in rendered[2]
    assert state.project.tts_model_type == "chatterbox_local"
    assert state.pending_tts_model_change is None


def test_heading_only_screen_leaves_tts_notice_for_next_full_menu(monkeypatch):
    from tts_audiobook_tool.state import PendingTtsModelChange

    model = TtsModelType.require_by_id("echo_tts_audiocpp")
    pending = PendingTtsModelChange("chatterbox_local", model.id)
    state = cast(State, SimpleNamespace(
        prefs=Prefs(), project=Project(tts_model_type=model.id), pending_tts_model_change=pending,
    ))
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_util.os.system", lambda _: None)
    monkeypatch.setattr(MenuStatus, "print_block", lambda _: None)

    MenuUtil.print_screen_heading(state, "Prompt screen")

    assert state.pending_tts_model_change is pending
