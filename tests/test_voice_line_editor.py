import asyncio
from dataclasses import dataclass

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
from textual.widgets import Button, Static


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
