import asyncio
from dataclasses import dataclass

import pytest
from textual.css.errors import StylesheetError
from tts_audiobook_tool.textual import voice_line_editor
from tts_audiobook_tool.textual.save_changes_dialog import (
    ExitDecision,
    SaveChangesDialog,
)
from tts_audiobook_tool.textual.voice_line_editor import (
    NonWrappingOptionList,
    VoiceLineEditorTextualApp,
)
from textual.containers import Horizontal
from textual.widgets import Button, Input, Static


@dataclass
class StubPhraseGroup:
    presentable_text: str
    voice_index: int = -1


@dataclass
class StubProject:
    phrase_groups: list[StubPhraseGroup]


def make_app(
    num_lines: int = 12, voice_sample_count: int = 9
) -> tuple[VoiceLineEditorTextualApp, StubProject]:
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(num_lines)]
    )
    return VoiceLineEditorTextualApp(project, voice_sample_count), project


def run(coroutine) -> None:
    asyncio.run(coroutine)


@pytest.fixture(autouse=True)
def stub_voice_values(monkeypatch) -> None:
    """Keep editor rendering independent of application-wide TTS initialization."""
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: [f"voice-{index + 1}" for index in range(9)],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())


def test_shift_navigation_extends_and_reversing_shrinks_selection() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "shift+down", "shift+down")
            assert app.selected_index == 3
            assert app.selection_anchor_index == 0
            assert app.selected_indices == {0, 1, 2, 3}

            await pilot.press("shift+up", "shift+up")
            assert app.selected_index == 1
            assert app.selected_indices == {0, 1}

            prompts = [app.format_line(index) for index in range(4)]
            assert all(str(prompt).startswith("[") for prompt in prompts)
            assert str(prompts[0].style) == "#888888 reverse"
            assert str(prompts[1].style) == ""
            assert str(prompts[2].style) == ""

    run(exercise())


def test_selection_status_describes_only_multi_line_selections() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            status = app.query_one("#selection-status", Static)
            assert str(status.render()) == ""

            await pilot.press("shift+down")
            assert str(status.render()) == "2 lines selected"

            await pilot.press("shift+down")
            assert str(status.render()) == "3 lines selected"

            await pilot.press("down")
            assert str(status.render()) == ""

    run(exercise())


def test_ctrl_a_selects_all_lines_and_retains_current_line_as_anchor() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "down", "down", "ctrl+a")
            assert app.selected_index == 3
            assert app.selection_anchor_index == 3
            assert app.selected_indices == set(range(12))
            status = app.query_one("#selection-status", Static)
            assert str(status.render()) == "12 lines selected"

    run(exercise())


def test_ctrl_f_opens_find_bar_and_focuses_input() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            find_bar = app.query_one("#find-bar", Horizontal)
            find_input = app.query_one("#find-input", Input)
            status = app.query_one("#selection-status", Static)
            assert find_bar.display is False
            assert status.display is True

            await pilot.press("ctrl+f")

            assert app.find_active is True
            assert app.find_search_start_index == 0
            assert find_bar.display is True
            assert status.display is False
            assert find_input.has_focus is True
            find_label = app.query_one("#find-label", Static)
            assert find_label.styles.text_style.italic is True
            assert find_label.styles.color.hex == "#FFAA44"
            assert find_input.styles.text_style.italic is not True
            assert find_input.styles.background.hex != "#888888"

    run(exercise())


def test_find_waits_for_enter_then_searches_from_stable_origin_and_wraps() -> None:
    project = StubProject(
        [
            StubPhraseGroup("Needle before origin"),
            StubPhraseGroup("unrelated"),
            StubPhraseGroup("Needle first after origin"),
            StubPhraseGroup("Needle second after origin"),
        ]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "ctrl+f")
            assert app.find_search_start_index == 1

            await pilot.press(*"needle")
            await pilot.pause()
            assert app.selected_index == 1

            await pilot.press(*" second")
            await pilot.pause()
            assert app.selected_index == 1

            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 3

            find_input = app.query_one("#find-input", Input)
            find_input.value = "before"
            await pilot.pause()
            assert app.selected_index == 3

            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 0

    run(exercise())


def test_empty_and_no_match_find_queries_leave_selection_unchanged() -> None:
    project = StubProject(
        [StubPhraseGroup("alpha"), StubPhraseGroup("beta"), StubPhraseGroup("gamma")]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "ctrl+f")
            find_input = app.query_one("#find-input", Input)

            find_input.value = "missing"
            await pilot.pause()
            assert app.selected_index == 1

            find_input.value = ""
            await pilot.pause()
            assert app.selected_index == 1

    run(exercise())


def test_find_query_is_retained_selected_and_can_include_number_keys() -> None:
    project = StubProject(
        [StubPhraseGroup("start"), StubPhraseGroup("chapter 2"), StubPhraseGroup("end")]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=2)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+f", *"chapter 2")
            await pilot.pause()
            assert app.selected_index == 0
            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 1
            assert app.staged_voice_indices == [-1, -1, -1]

            await pilot.press("escape", "ctrl+f")
            find_input = app.query_one("#find-input", Input)
            assert find_input.value == "chapter 2"
            assert find_input.selection.start == 0
            assert find_input.selection.end == len("chapter 2")

    run(exercise())


def test_reopening_find_preserves_current_match_until_enter() -> None:
    project = StubProject(
        [
            StubPhraseGroup("needle first"),
            StubPhraseGroup("unrelated"),
            StubPhraseGroup("needle second"),
            StubPhraseGroup("needle third"),
        ]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+f", *"needle", "enter", "escape")
            await pilot.pause()
            assert app.selected_index == 2

            await pilot.press("ctrl+f")
            await pilot.pause()
            assert app.selected_index == 2

            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 3

            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 0

    run(exercise())


def test_enter_advances_matches_without_blurring_find() -> None:
    project = StubProject(
        [
            StubPhraseGroup("needle first"),
            StubPhraseGroup("needle second"),
            StubPhraseGroup("needle third"),
        ]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            find_input = app.query_one("#find-input", Input)
            find_result = app.query_one("#find-result", Static)
            await pilot.press("ctrl+f", *"needle")
            await pilot.pause()
            assert app.selected_index == 0
            assert str(find_result.render()) == ""

            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 1
            assert str(find_result.render()) == "2 of 3"
            assert str(app.format_line(1).style) == "#888888 reverse"
            assert app.find_active is True
            assert find_input.has_focus is True

            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 2
            assert str(find_result.render()) == "3 of 3"
            assert app.find_active is True
            assert find_input.has_focus is True

            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_index == 0
            assert str(find_result.render()) == "1 of 3"
            assert app.find_active is True
            assert find_input.has_focus is True

    run(exercise())


def test_find_match_highlight_is_removed_when_find_blurs() -> None:
    project = StubProject(
        [StubPhraseGroup("start"), StubPhraseGroup("needle"), StubPhraseGroup("end")]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+f", *"needle", "enter")
            await pilot.pause()
            assert app.find_match_index == 1
            assert str(app.format_line(1).style) == "#888888 reverse"

            await pilot.press("escape")
            assert app.find_match_index is None
            assert str(app.format_line(1).style) == ""

    run(exercise())


def test_shift_enter_moves_backward_after_query_is_submitted() -> None:
    project = StubProject(
        [
            StubPhraseGroup("needle first"),
            StubPhraseGroup("needle second"),
            StubPhraseGroup("needle third"),
        ]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            find_input = app.query_one("#find-input", Input)
            find_result = app.query_one("#find-result", Static)
            await pilot.press("ctrl+f", *"needle", "shift+enter")
            await pilot.pause()
            assert app.selected_index == 0
            assert str(find_result.render()) == ""

            await pilot.press("enter", "shift+enter")
            await pilot.pause()
            assert app.selected_index == 0
            assert str(find_result.render()) == "1 of 3"
            assert find_input.has_focus is True

            await pilot.press("shift+enter")
            await pilot.pause()
            assert app.selected_index == 2
            assert str(find_result.render()) == "3 of 3"
            assert find_input.has_focus is True

    run(exercise())


def test_find_reports_no_matches_and_clears_feedback_when_query_changes() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            find_result = app.query_one("#find-result", Static)
            await pilot.press("ctrl+f", *"missing", "enter")
            await pilot.pause()
            assert app.selected_index == 0
            assert str(find_result.render()) == "No matches"

            await pilot.press("x")
            await pilot.pause()
            assert app.selected_index == 0
            assert str(find_result.render()) == ""

    run(exercise())


def test_find_result_reserves_twelve_right_aligned_columns() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+f")
            find_result = app.query_one("#find-result", Static)
            assert find_result.styles.width.value == 12
            assert find_result.styles.content_align_horizontal == "right"

    run(exercise())


def test_escape_and_outside_click_dismiss_find_without_exiting() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            find_bar = app.query_one("#find-bar", Horizontal)

            await pilot.press("ctrl+f", "escape")
            assert app.find_active is False
            assert find_bar.display is False
            assert app.is_running is True

            await pilot.press("ctrl+f")
            await pilot.click(app.query_one("#line-list", NonWrappingOptionList))
            assert app.find_active is False
            assert app.is_running is True

            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running is False

    run(exercise())


def test_header_limits_voice_key_range_to_available_samples() -> None:
    app, _ = make_app(voice_sample_count=2)

    assert str(app.header_lines[1]) == (
        "- Use number keys [1] to [2] to set voice sample for selected text line/s"
    )


def test_shared_css_is_loaded() -> None:
    app, _ = make_app(1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            test_widget = Static(id="textual-shared-css-test")
            await app.mount(test_widget)
            await pilot.pause()

            assert test_widget.styles.color.hex == "#123456"

    run(exercise())


def test_unshifted_navigation_collapses_selection_even_at_an_edge() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+end")
            assert app.selected_indices == set(range(12))

            await pilot.press("end")
            assert app.selected_index == 11
            assert app.selection_anchor_index == 11
            assert app.selected_indices == {11}

            await pilot.press("up")
            assert app.selected_index == 10
            assert app.selected_indices == {10}

    run(exercise())


def test_shift_home_end_and_page_navigation_use_contiguous_selection() -> None:
    app, _ = make_app(40)

    async def exercise() -> None:
        async with app.run_test(size=(80, 16)) as pilot:
            option_list = app.query_one("#line-list", NonWrappingOptionList)

            await pilot.press("down", "down", "down", "down", "down")
            assert app.selected_indices == {5}

            await pilot.press("shift+home")
            assert app.selected_indices == set(range(6))

            await pilot.press("shift+end")
            assert app.selected_indices == set(range(5, 40))

            await pilot.press("home", "shift+pagedown")
            assert app.selection_anchor_index == 0
            assert app.selected_index == option_list.highlighted
            assert app.selected_index is not None and app.selected_index > 0
            assert app.selected_indices == set(range(app.selected_index + 1))

            await pilot.press("shift+pageup")
            assert app.selected_index == 0
            assert app.selected_indices == {0}

    run(exercise())


def test_number_hotkey_assigns_voice_to_every_selected_line(monkeypatch) -> None:
    app, project = make_app(6)
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "shift+down", "2")
            assert app.staged_voice_indices == [1, 1, 1, -1, -1, -1]
            assert [group.voice_index for group in project.phrase_groups] == [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ]
            assert app.has_changes is True
            assert app.selected_indices == {2}
            assert app.selection_anchor_index == 2

            await pilot.press("2")
            assert app.has_changes is True

            await pilot.press("3")
            assert [group.voice_index for group in project.phrase_groups] == [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ]
            assert app.has_changes is True

    run(exercise())


def test_shift_click_selects_from_anchor_and_plain_click_collapses() -> None:
    app, _ = make_app(20)

    async def exercise() -> None:
        async with app.run_test(size=(80, 28)) as pilot:
            option_list = app.query_one("#line-list", NonWrappingOptionList)

            await pilot.press("down", "down", "down")
            assert app.selection_anchor_index == 3

            await pilot.click(option_list, offset=(10, 5), shift=True)
            assert app.selected_index == 5
            assert app.selection_anchor_index == 3
            assert app.selected_indices == {3, 4, 5}

            await pilot.click(option_list, offset=(10, 1), shift=True)
            assert app.selected_index == 1
            assert app.selection_anchor_index == 3
            assert app.selected_indices == {1, 2, 3}

            await pilot.click(option_list, offset=(10, 3))
            assert app.selected_index == 3
            assert app.selection_anchor_index == 3
            assert app.selected_indices == {3}

    run(exercise())


def test_reverting_staged_values_makes_editor_clean(monkeypatch) -> None:
    project = StubProject(
        [StubPhraseGroup("Line 1", voice_index=0), StubPhraseGroup("Line 2")]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=2)
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2")
            assert app.has_changes is True
            assert project.phrase_groups[0].voice_index == 0

            await pilot.press("1")
            assert app.has_changes is False
            assert project.phrase_groups[0].voice_index == 0

    run(exercise())


def test_clean_escape_exits_without_confirmation() -> None:
    app, _ = make_app(2)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running is False

    run(exercise())


def test_dirty_escape_cancel_returns_to_editor_without_mutating_project(
    monkeypatch,
) -> None:
    app, project = make_app(2, voice_sample_count=2)
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2", "escape")
            assert isinstance(app.screen, SaveChangesDialog)

            await pilot.press("escape")
            assert not isinstance(app.screen, SaveChangesDialog)
            assert app.is_running is True
            assert project.phrase_groups[0].voice_index == -1
            assert app.has_changes is True

    run(exercise())


def test_dirty_escape_discard_exits_without_mutating_project(monkeypatch) -> None:
    app, project = make_app(2, voice_sample_count=2)
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2", "escape", "d")
            await pilot.pause()
            assert app.is_running is False
            assert project.phrase_groups[0].voice_index == -1
            assert app.did_save_changes is False

    run(exercise())


def test_save_button_commits_staged_values_and_persists_once(monkeypatch) -> None:
    app, project = make_app(3, voice_sample_count=2)
    saves: list[StubProject] = []
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())
    monkeypatch.setattr(
        voice_line_editor.ProjectTextIOUtil,
        "save_phrase_groups",
        lambda saved_project: saves.append(saved_project) or "",
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "2", "escape")
            assert [group.voice_index for group in project.phrase_groups] == [
                -1,
                -1,
                -1,
            ]

            save_button = app.screen.query_one("#save", Button)
            await pilot.click(save_button)
            await pilot.pause()
            assert app.is_running is False

        assert [group.voice_index for group in project.phrase_groups] == [1, 1, -1]
        assert saves == [project]
        assert app.did_save_changes is True
        assert app.save_error == ""

    run(exercise())


def test_save_failure_rolls_back_project_and_records_error(monkeypatch) -> None:
    project = StubProject(
        [StubPhraseGroup("Line 1", voice_index=0), StubPhraseGroup("Line 2")]
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=2)
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())
    monkeypatch.setattr(
        voice_line_editor.ProjectTextIOUtil,
        "save_phrase_groups",
        lambda *_: "disk full",
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2", "escape", "s")
            await pilot.pause()

        assert [group.voice_index for group in project.phrase_groups] == [0, -1]
        assert app.did_save_changes is False
        assert app.save_error == "Save failed: disk full"

    run(exercise())


def test_unexpected_save_exception_rolls_back_project_and_records_error(
    monkeypatch,
) -> None:
    app, project = make_app(1, voice_sample_count=2)
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())

    def fail_save(*_) -> str:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        voice_line_editor.ProjectTextIOUtil,
        "save_phrase_groups",
        fail_save,
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2", "escape", "s")
            await pilot.pause()

        assert project.phrase_groups[0].voice_index == -1
        assert app.did_save_changes is False
        assert app.save_error == "Save failed: RuntimeError: unexpected failure"

    run(exercise())


def test_start_reports_save_failure_as_error_feedback(monkeypatch) -> None:
    project = StubProject([StubPhraseGroup("Line 1")])
    feedback_calls: list[tuple[str, bool]] = []

    def run_with_save_error(app: VoiceLineEditorTextualApp, **_) -> None:
        app.save_error = "Save failed: disk full"

    monkeypatch.setattr(VoiceLineEditorTextualApp, "run", run_with_save_error)
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1"],
    )
    monkeypatch.setattr(
        voice_line_editor,
        "print_feedback",
        lambda message, **kwargs: feedback_calls.append(
            (message, kwargs.get("is_error", False))
        ),
    )

    VoiceLineEditorTextualApp.start(project)  # type: ignore[arg-type]

    assert feedback_calls == [("Save failed: disk full", True)]


def test_start_reports_css_load_failure_as_error_feedback(monkeypatch) -> None:
    project = StubProject([StubPhraseGroup("Line 1")])
    feedback_calls: list[tuple[str, bool]] = []

    def run_with_css_error(app: VoiceLineEditorTextualApp, **_) -> None:
        app._exception = StylesheetError("unable to read CSS file")

    monkeypatch.setattr(VoiceLineEditorTextualApp, "run", run_with_css_error)
    monkeypatch.setattr(voice_line_editor.Tts, "get_type", lambda: object())
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1"],
    )
    monkeypatch.setattr(
        voice_line_editor,
        "print_feedback",
        lambda message, **kwargs: feedback_calls.append(
            (message, kwargs.get("is_error", False))
        ),
    )

    VoiceLineEditorTextualApp.start(project)  # type: ignore[arg-type]

    assert feedback_calls == [("Couldn't load textual css", True)]


def test_dialog_decision_values_are_explicit() -> None:
    assert ExitDecision.SAVE.value == "save"
    assert ExitDecision.DISCARD.value == "discard"
    assert ExitDecision.CANCEL.value == "cancel"


def test_dialog_button_labels_preserve_literal_brackets() -> None:
    app, _ = make_app(1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(SaveChangesDialog())
            await pilot.pause()

            assert str(app.screen.query_one("#save", Button).label) == "[S]ave"
            assert str(app.screen.query_one("#discard", Button).label) == "[D]iscard"

    run(exercise())
