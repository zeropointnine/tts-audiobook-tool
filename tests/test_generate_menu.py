from types import SimpleNamespace
from typing import Any, cast

import tts_audiobook_tool.menus.generate_menu as generate_menu_module
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
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
from tts_audiobook_tool.textual.generation_app import GenerationModalResult
from tts_audiobook_tool.model_worker_protocol import GenerationTerminalStatus
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
    assert strip_ansi_codes(first_label(state)) == "Start (0 lines queued)"
    assert len(items) == 6


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
        generate_menu_module,
        "do_generate",
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

    def make_editor(_state: State, quick_gen_index=None):
        restored_indices.append(quick_gen_index)
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
    feedback_calls: list[str] = []
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
    monkeypatch.setattr(generate_menu_module.ask, "ask_error", feedback_calls.append)

    GenerateMenu.run_editor(state)

    assert feedback_calls == ["Save failed: disk full"]


def test_generation_workflow_reports_save_failure_without_generation(monkeypatch) -> None:
    state = cast(State, SimpleNamespace(project=object()))
    feedback_calls: list[str] = []
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
    monkeypatch.setattr(generate_menu_module.ask, "ask_error", feedback_calls.append)

    GenerateMenu.run_editor(state)

    assert feedback_calls == ["Save failed: disk full"]
    assert generation_calls == []


def test_generation_workflow_reports_cleanup_and_launch_failures(monkeypatch) -> None:
    state = cast(State, SimpleNamespace(project=object()))
    feedback_calls: list[str] = []
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
    monkeypatch.setattr(generate_menu_module.ask, "ask_error", feedback_calls.append)

    GenerateMenu.run_editor(state)

    assert feedback_calls == [
        "Save failed: cleanup failed",
        "Couldn't load textual css",
    ]


def test_validation_confirmation_line_reflects_whisper_setting() -> None:
    disabled_state = cast(
        State,
        SimpleNamespace(
            prefs=SimpleNamespace(is_validation_disabled=True),
            project=SimpleNamespace(strictness=Strictness.LOW),
        ),
    )
    enabled_state = cast(
        State,
        SimpleNamespace(
            prefs=SimpleNamespace(is_validation_disabled=False),
            project=SimpleNamespace(strictness=Strictness.LOW),
        ),
    )

    assert generate_menu_module.make_validation_confirmation_line(disabled_state) == (
        "- Speech to text validation: Disabled"
    )
    assert generate_menu_module.make_validation_confirmation_line(enabled_state) == (
        "- Word error tolerance: Loose"
    )


def test_auto_concat_runs_only_after_successful_generation(monkeypatch) -> None:
    sound_segments = SimpleNamespace(
        num_generated=lambda: 0,
        num_generated_in_current_range=lambda: 0,
    )
    project = SimpleNamespace(
        phrase_groups=[object()],
        sound_segments=sound_segments,
        gen_auto_concat=True,
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(is_validation_disabled=True),
        ),
    )
    concat_calls: list[State] = []
    printed: list[str] = []
    result_status = [GenerationTerminalStatus.COMPLETED]

    monkeypatch.setattr(
        generate_menu_module.readiness,
        "get_generate_blocker_text",
        lambda _state, verbose: "",
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "get_selected_indices_not_generated",
        lambda _project: {0},
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "generate_range_string_display",
        lambda _project: "all",
    )
    monkeypatch.setattr(
        generate_menu_module.app_hint_util,
        "show_pre_inference_hints",
        lambda _prefs, _project: True,
    )
    monkeypatch.setattr(generate_menu_module.MenuUtil, "print_screen_heading", lambda *_: None)
    monkeypatch.setattr(
        generate_menu_module.Tts,
        "get_type",
        lambda: SimpleNamespace(value=SimpleNamespace(can_batch=False)),
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectVoiceUtil,
        "get_voice_values",
        lambda _project, _tts_type: [],
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectVoiceUtil,
        "get_batch_size",
        lambda _project: 1,
    )
    monkeypatch.setattr(generate_menu_module.ask, "ask_confirm", lambda _prompt: True)
    monkeypatch.setattr(
        generate_menu_module,
        "printt",
        lambda value="", *_args, **_kwargs: printed.append(value),
    )
    monkeypatch.setattr(
        generate_menu_module,
        "run_generation_app",
        lambda **_kwargs: GenerationModalResult(result_status[0], "", ""),
    )
    monkeypatch.setattr(
        generate_menu_module.ConcatUtil,
        "auto_concat_after_generation",
        lambda passed_state: concat_calls.append(passed_state),
    )

    cases = [
        (GenerationTerminalStatus.COMPLETED, True),
        (GenerationTerminalStatus.CANCELLED, False),
        (GenerationTerminalStatus.ABORTED, False),
        (GenerationTerminalStatus.FAILED, False),
        (GenerationTerminalStatus.WORKER_RESET, False),
    ]
    for status, should_concat in cases:
        result_status[0] = status
        concat_calls.clear()
        printed.clear()

        generate_menu_module.do_generate(state)

        assert concat_calls == ([state] if should_concat else [])
        assert printed.index("- Speech to text validation: Disabled") < printed.index(
            "- Will concatenate audio file/s when finished"
        )


def make_phrase_group(voice_index: int) -> PhraseGroup:
    return PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=voice_index)


def test_out_of_range_voice_count_ignores_unassigned_default() -> None:
    # voice_index == -1 is the "no explicit assignment" default (first voice sample).
    # Only indices past the end of the voice sample list are stale.
    groups = [make_phrase_group(voice_index) for voice_index in (-1, 0, 1, 2, 5)]

    assert generate_menu_module.count_out_of_range_voice_indices(groups, {0, 1, 2, 3, 4}, 2) == 2
    assert generate_menu_module.count_out_of_range_voice_indices(groups, [0], 2) == 0
    assert generate_menu_module.count_out_of_range_voice_indices(groups, range(5), 3) == 1
    assert generate_menu_module.count_out_of_range_voice_indices(groups, [], 2) == 0
    assert generate_menu_module.count_out_of_range_voice_indices(groups, range(1), 0) == 0
