from types import SimpleNamespace
from typing import Any, cast

import pytest

import tts_audiobook_tool.menus.generate_menu as generate_menu_module
from tts_audiobook_tool.app_types import Hint, VoiceSelectMode
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


def run_do_generate(
    monkeypatch,
    *,
    voice_select_mode: VoiceSelectMode,
    num_voice_values: int,
    batch_size: int,
) -> list:
    """ Runs do_generate() up to the start confirmation.

    Returns the list of hints shown via hints.show_hint_if_necessary.
    """
    project = SimpleNamespace(
        phrase_groups=[
            make_phrase_group(i % max(1, num_voice_values)) for i in range(2)
        ],
        voice_select_mode=voice_select_mode,
        gen_auto_concat=False,
        sound_segments=SimpleNamespace(
            get_best_item_for=lambda index: None,
            num_generated=lambda: 0,
            num_generated_in_current_range=lambda: 0,
        ),
    )
    state = cast(State, SimpleNamespace(project=project, prefs=object()))

    shown_hints: list[Hint] = []

    def capture_hint(prefs, hint, **kwargs) -> bool:
        shown_hints.append(hint)
        return True

    monkeypatch.setattr(
        generate_menu_module.hints, "show_hint_if_necessary", capture_hint
    )
    monkeypatch.setattr(
        generate_menu_module.readiness,
        "get_generate_blocker_text",
        lambda state, verbose: "",
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "get_selected_indices_not_generated",
        lambda project: [0, 1],
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectUtil,
        "generate_range_string_display",
        lambda project: "1-2",
    )
    monkeypatch.setattr(
        generate_menu_module.app_hint_util,
        "show_pre_inference_hints",
        lambda prefs, project: True,
    )
    monkeypatch.setattr(
        generate_menu_module.MenuUtil,
        "print_screen_heading",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(generate_menu_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Tts,
        "get_type",
        lambda: SimpleNamespace(
            value=SimpleNamespace(can_batch=True, batch_size_attr="batch_size")
        ),
    )
    monkeypatch.setattr(Tts, "is_sgl_mode", staticmethod(lambda: False))
    monkeypatch.setattr(
        generate_menu_module.ProjectVoiceUtil,
        "get_batch_size",
        lambda project: batch_size,
    )
    monkeypatch.setattr(
        generate_menu_module.ProjectVoiceUtil,
        "get_voice_values",
        lambda project, tts_type: [f"voice-{i}.flac" for i in range(num_voice_values)],
    )
    monkeypatch.setattr(
        generate_menu_module.Stt, "should_skip", staticmethod(lambda state: True)
    )
    monkeypatch.setattr(
        generate_menu_module.ask, "ask_confirm", lambda *args, **kwargs: False
    )

    generate_menu_module.do_generate(state)
    return shown_hints


@pytest.mark.parametrize(
    ("voice_select_mode", "num_voice_values", "batch_size", "expect_hint"),
    [
        # Multi-voice batch mode (the trigger condition)
        (VoiceSelectMode.USER_DEFINED, 2, 4, True),
        # Not user-defined: auto-advance rotates voices but has no per-line grouping
        (VoiceSelectMode.AUTO_ADVANCE, 2, 4, False),
        # Batching disabled: one line per call, no reordering
        (VoiceSelectMode.USER_DEFINED, 2, 1, False),
        # Single voice sample: grouping is a no-op
        (VoiceSelectMode.USER_DEFINED, 1, 4, False),
    ],
)
def test_do_generate_shows_multi_voice_batch_hint_only_when_triggered(
    monkeypatch,
    voice_select_mode: VoiceSelectMode,
    num_voice_values: int,
    batch_size: int,
    expect_hint: bool,
) -> None:
    shown = run_do_generate(
        monkeypatch,
        voice_select_mode=voice_select_mode,
        num_voice_values=num_voice_values,
        batch_size=batch_size,
    )

    if expect_hint:
        assert [hint.key for hint in shown] == ["batch_multi_voice"]
    else:
        assert shown == []
