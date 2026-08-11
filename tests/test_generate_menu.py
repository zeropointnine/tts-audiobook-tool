from types import SimpleNamespace
from typing import Any, cast

import tts_audiobook_tool.menus.generate_menu as generate_menu_module
from tts_audiobook_tool.menus.generate_menu import GenerateMenu
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppStylesheetFailed,
    EditorClosed,
    EditorSaveFailed,
)
from tts_audiobook_tool.textual.generate_editor import QuickGenerationRequested
from tts_audiobook_tool.tts import Tts


def capture_generate_menu(monkeypatch) -> tuple[State, list[MenuItem]]:
    project = SimpleNamespace(
        phrase_groups=[],
        generate_range_string="none",
        sound_segments=SimpleNamespace(
            get_best_item_for=lambda index: None,
            num_generated=lambda: 0,
        ),
    )
    state = cast(State, SimpleNamespace(project=project))
    captured: dict[str, Any] = {}

    def capture_menu(
        passed_state: State,
        heading,
        items,
        **kwargs,
    ) -> None:
        captured["items"] = items

    monkeypatch.setattr(MenuUtil, "menu", capture_menu)
    monkeypatch.setattr(
        Tts,
        "get_type",
        lambda: SimpleNamespace(value=SimpleNamespace(can_batch=False)),
    )

    GenerateMenu.menu(state)
    items_maker = captured["items"]
    return state, items_maker(state)


def test_generate_menu_replaces_legacy_generation_entries(monkeypatch):
    monkeypatch.setattr(
        generate_menu_module.readiness,
        "get_generate_blocker_text",
        lambda state, verbose: "",
    )

    state, items = capture_generate_menu(monkeypatch)

    first_label = items[0].label
    assert callable(first_label)
    assert strip_ansi_codes(first_label(state)) == (
        "Generate sound segments (0 lines queued)"
    )
    assert len(items) == 5


def test_generate_menu_shows_blocker_but_delegates_launch_validation(monkeypatch):
    blocker_calls: list[bool] = []
    blocker = "Generation is unavailable"
    editor_calls: list[State] = []

    def get_blocker(state: State, verbose: bool) -> str:
        blocker_calls.append(verbose)
        return blocker if verbose else "Unavailable"

    monkeypatch.setattr(
        generate_menu_module.readiness,
        "get_generate_blocker_text",
        get_blocker,
    )
    monkeypatch.setattr(
        GenerateMenu,
        "edit_generation_queue",
        lambda passed_state: editor_calls.append(passed_state),
    )
    state, items = capture_generate_menu(monkeypatch)
    generate_item = items[0]

    label = generate_item.label
    assert callable(label)
    assert "Unavailable" in label(state)
    generate_item.handler(state, generate_item)

    assert blocker_calls == [False]
    assert editor_calls == [state]


def test_generation_workflow_closed_result_performs_no_generation(monkeypatch) -> None:
    state = cast(State, SimpleNamespace(project=object()))
    generation_calls: list[State] = []
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "persist_range_without_generated_items",
        lambda _: "",
    )
    monkeypatch.setattr(generate_menu_module, "GenerateEditor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        generate_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorClosed()),
    )
    monkeypatch.setattr(
        generate_menu_module.GenerateUtil,
        "do_generate_using_project_and_state",
        lambda passed_state: generation_calls.append(passed_state),
    )

    GenerateMenu.run_editor(state)

    assert generation_calls == []


def test_generation_workflow_quick_generates_and_reopens(monkeypatch) -> None:
    state = cast(State, SimpleNamespace(project=object()))
    restored_indices: list[int | None] = []
    quick_generation_calls: list[tuple[State, int]] = []
    run_results = iter(
        [
            ContentAppCompleted(QuickGenerationRequested(2)),
            ContentAppCompleted(EditorClosed()),
        ]
    )

    def make_editor(_state: State, quick_gen_restore_phrase_index=None):
        restored_indices.append(quick_gen_restore_phrase_index)
        return object()

    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "persist_range_without_generated_items",
        lambda _: "",
    )
    monkeypatch.setattr(generate_menu_module, "GenerateEditor", make_editor)
    monkeypatch.setattr(
        generate_menu_module,
        "run_content_textual_app",
        lambda _: next(run_results),
    )
    monkeypatch.setattr(
        generate_menu_module.GenerateUtil,
        "do_quick_generate",
        lambda passed_state, index: quick_generation_calls.append(
            (passed_state, index)
        ),
    )

    GenerateMenu.run_editor(state)

    assert restored_indices == [None, 2]
    assert quick_generation_calls == [(state, 2)]


def test_generation_workflow_reports_quick_save_error_and_continues(monkeypatch) -> None:
    state = cast(State, SimpleNamespace(project=object()))
    feedback_calls: list[tuple[str, bool]] = []
    run_results = iter(
        [
            ContentAppCompleted(
                QuickGenerationRequested(1, "Save failed: disk full")
            ),
            ContentAppCompleted(EditorClosed()),
        ]
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "persist_range_without_generated_items",
        lambda _: "",
    )
    monkeypatch.setattr(generate_menu_module, "GenerateEditor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        generate_menu_module,
        "run_content_textual_app",
        lambda _: next(run_results),
    )
    monkeypatch.setattr(
        generate_menu_module.GenerateUtil,
        "do_quick_generate",
        lambda *_: None,
    )
    monkeypatch.setattr(
        generate_menu_module,
        "print_feedback",
        lambda message, is_error=False, **_: feedback_calls.append(
            (message, is_error)
        ),
    )

    GenerateMenu.run_editor(state)

    assert feedback_calls == [("Save failed: disk full", True)]


def test_generation_workflow_reports_save_failure_without_generation(monkeypatch) -> None:
    state = cast(State, SimpleNamespace(project=object()))
    feedback_calls: list[tuple[str, bool]] = []
    generation_calls: list[State] = []
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "persist_range_without_generated_items",
        lambda _: "",
    )
    monkeypatch.setattr(generate_menu_module, "GenerateEditor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        generate_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorSaveFailed("Save failed: disk full")),
    )
    monkeypatch.setattr(
        generate_menu_module.GenerateUtil,
        "do_generate_using_project_and_state",
        lambda passed_state: generation_calls.append(passed_state),
    )
    monkeypatch.setattr(
        generate_menu_module,
        "print_feedback",
        lambda message, is_error=False, **_: feedback_calls.append(
            (message, is_error)
        ),
    )

    GenerateMenu.run_editor(state)

    assert feedback_calls == [("Save failed: disk full", True)]
    assert generation_calls == []


def test_generation_workflow_reports_cleanup_and_launch_failures(monkeypatch) -> None:
    state = cast(State, SimpleNamespace(project=object()))
    feedback_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "persist_range_without_generated_items",
        lambda _: "Save failed: cleanup failed",
    )
    monkeypatch.setattr(generate_menu_module, "GenerateEditor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        generate_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppStylesheetFailed("Couldn't load textual css"),
    )
    monkeypatch.setattr(
        generate_menu_module,
        "print_feedback",
        lambda message, is_error=False, **_: feedback_calls.append(
            (message, is_error)
        ),
    )

    GenerateMenu.run_editor(state)

    assert feedback_calls == [
        ("Save failed: cleanup failed", True),
        ("Couldn't load textual css", True),
    ]
