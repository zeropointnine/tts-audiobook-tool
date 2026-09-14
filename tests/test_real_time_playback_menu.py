from types import SimpleNamespace
from typing import Any, cast

import tts_audiobook_tool.menus.real_time_playback_menu as menu_module
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.real_time_playback_menu import RealTimePlaybackMenu, do_start
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes


def _make_state(
    *,
    phrase_groups: list[object] | None = None,
    realtime_line_range: tuple[int, int] | None = None,
) -> State:
    project = SimpleNamespace(
        phrase_groups=phrase_groups if phrase_groups is not None else [object(), object(), object()],
        realtime_line_range=realtime_line_range,
        realtime_save=True,
        save=lambda: None,
    )
    return cast(State, SimpleNamespace(project=project, prefs=object()))


def test_menu_omits_text_source_and_keeps_project_options(monkeypatch) -> None:
    state = _make_state()
    captured: dict[str, Any] = {}

    def capture_menu(_state: State, _heading: str, items: list[MenuItem], **_kwargs: Any) -> None:
        captured["items"] = items

    monkeypatch.setattr(MenuUtil, "menu", capture_menu)
    monkeypatch.setattr(
        menu_module.readiness,
        "get_generate_blocker_text",
        lambda _state, verbose: "",
    )

    RealTimePlaybackMenu.menu(state)

    items = captured["items"]
    labels = [
        strip_ansi_codes(item.label(state) if callable(item.label) else item.label)
        for item in items
    ]
    assert labels == [
        "Start",
        "Line range (currently: all)",
        "Save output (currently: True)",
    ]
    assert items[1].superlabel == "Options"


def test_line_range_uses_project_text_and_persists_selection(monkeypatch) -> None:
    save_calls: list[None] = []
    state = _make_state(phrase_groups=[object()] * 5, realtime_line_range=(2, 3))
    state.project.save = lambda: save_calls.append(None)  # type: ignore[method-assign]
    feedback: list[tuple[object, ...]] = []

    monkeypatch.setattr(menu_module, "printt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(menu_module.ask, "ask_input", lambda **_kwargs: "3-5")
    monkeypatch.setattr(
        menu_module,
        "print_feedback",
        lambda *args, **_kwargs: feedback.append(args),
    )

    RealTimePlaybackMenu.ask_line_range(state)

    assert state.project.realtime_line_range == (3, 5)
    assert save_calls == [None]
    assert feedback == [("Line range set:", "3-5 (end)")]


def test_start_always_passes_project_text_and_range(monkeypatch) -> None:
    phrase_groups = [object(), object()]
    state = _make_state(phrase_groups=phrase_groups, realtime_line_range=(2, 2))
    calls: list[tuple[State, list[object], tuple[int, int] | None]] = []

    monkeypatch.setattr(
        menu_module.readiness,
        "get_generate_blocker_text",
        lambda _state, verbose: "",
    )
    monkeypatch.setattr(
        menu_module.app_hint_util,
        "show_pre_inference_hints",
        lambda _prefs, _project: True,
    )
    monkeypatch.setattr(menu_module.ask, "can_hotkey", False)
    monkeypatch.setattr(
        menu_module,
        "run_real_time_playback_modal",
        lambda *, state, phrase_groups, line_range: calls.append(
            (state, phrase_groups, line_range)
        ),
    )

    do_start(state)

    assert calls == [(state, phrase_groups, (2, 2))]


def test_start_rejects_missing_project_text(monkeypatch) -> None:
    state = _make_state(phrase_groups=[])
    feedback: list[tuple[object, ...]] = []
    launches: list[None] = []

    monkeypatch.setattr(
        menu_module,
        "print_feedback",
        lambda *args, **_kwargs: feedback.append(args),
    )
    monkeypatch.setattr(
        menu_module,
        "run_real_time_playback_modal",
        lambda **_kwargs: launches.append(None),
    )

    do_start(state)

    assert feedback == [("No text segments specified",)]
    assert launches == []
