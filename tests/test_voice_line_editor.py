import asyncio
from dataclasses import dataclass
from typing import cast

import pytest
from rich.console import Console
from rich.text import Text
from textual.css.errors import StylesheetError
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual import voice_line_editor
from tts_audiobook_tool.textual.textual_shared import NonWrappingOptionList
from tts_audiobook_tool.textual.voice_line_editor import VoiceLineEditorTextualApp
from textual.widgets import Button


@dataclass
class StubPhraseGroup:
    presentable_text: str
    voice_index: int = -1


@dataclass
class StubProject:
    phrase_groups: list[StubPhraseGroup]


def make_editor(
    project: StubProject, voice_sample_count: int
) -> VoiceLineEditorTextualApp:
    return VoiceLineEditorTextualApp(cast(Project, project), voice_sample_count)


def make_app(
    num_lines: int = 12, voice_sample_count: int = 9
) -> tuple[VoiceLineEditorTextualApp, StubProject]:
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(num_lines)]
    )
    return make_editor(project, voice_sample_count), project


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


def test_header_limits_voice_key_range_to_available_samples() -> None:
    app, _ = make_app(voice_sample_count=2)

    assert all(isinstance(line, str) for line in app.header_lines)
    assert all(not line.endswith(Ansi.RESET) for line in app.header_lines)
    assert Text.from_ansi(app.header_lines[1]).plain == (
        "- Use number keys [1] to [2] to set voice sample for selected text line/s"
    )


def test_inactive_selected_line_dim_background_extends_to_full_row_width() -> None:
    app, _ = make_app(2)

    async def exercise() -> None:
        async with app.run_test(size=(50, 20)) as pilot:
            await pilot.press("shift+down")
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            line = option_list.render_line(0)
            assert line.cell_length == option_list.scrollable_content_region.width
            assert all(
                segment.style is not None
                and segment.style.reverse
                and segment.style.color is not None
                and tuple(segment.style.color.get_truecolor()) == (136, 136, 136)
                for segment in line
            )

    run(exercise())


def test_long_text_wraps_with_hanging_indent_and_is_limited_to_three_lines() -> None:
    app, project = make_app(1)
    project.phrase_groups[0].presentable_text = (
        "one two three four five six seven eight nine"
    )
    console = Console(width=36, force_terminal=False, color_system=None)

    rendered_lines = console.render_lines(
        app.format_line(0), console.options, pad=False
    )
    rendered = ["".join(segment.text for segment in line) for line in rendered_lines]

    assert rendered == [
        "[00001] [Voice sample 1] one two",
        "                         three four",
        "                         five six…",
    ]


def test_inactive_selected_wrapped_line_dim_background_extends_each_row() -> None:
    app, project = make_app(2)
    project.phrase_groups[0].presentable_text = "one two three four five"

    async def exercise() -> None:
        async with app.run_test(size=(36, 20)) as pilot:
            await pilot.press("shift+down")
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            rendered_lines = [option_list.render_line(y) for y in range(3)]
            assert all(
                line.cell_length == option_list.scrollable_content_region.width
                for line in rendered_lines
            )
            assert all(
                segment.style is not None
                and segment.style.reverse
                and segment.style.color is not None
                and tuple(segment.style.color.get_truecolor()) == (136, 136, 136)
                for line in rendered_lines
                for segment in line
            )

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


def test_voice_assignment_batches_prompt_updates_without_reflow(monkeypatch) -> None:
    app, _ = make_app(6)
    refresh_calls: list[tuple[list[int], bool]] = []
    monkeypatch.setattr(
        app,
        "refresh_lines",
        lambda indices, *, reflow=True: refresh_calls.append(
            (list(indices), reflow)
        ),
    )
    monkeypatch.setattr(app, "collapse_current_selection", lambda: None)

    app.selected_indices = {0, 1, 2}
    app.action_assign_voice(1)

    assert refresh_calls == [([0, 1, 2], False)]


def test_replacing_out_of_range_voice_requests_reflow(monkeypatch) -> None:
    project = StubProject([StubPhraseGroup("Line 1", voice_index=8)])
    app = make_editor(project, voice_sample_count=2)
    refresh_calls: list[tuple[list[int], bool]] = []
    monkeypatch.setattr(
        app,
        "refresh_lines",
        lambda indices, *, reflow=True: refresh_calls.append(
            (list(indices), reflow)
        ),
    )
    monkeypatch.setattr(app, "collapse_current_selection", lambda: None)

    app.action_assign_voice(1)

    assert refresh_calls == [([0], True)]


def test_reverting_staged_values_makes_editor_clean(monkeypatch) -> None:
    project = StubProject(
        [StubPhraseGroup("Line 1", voice_index=0), StubPhraseGroup("Line 2")]
    )
    app = make_editor(project, voice_sample_count=2)
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

            yes_button = app.screen.query_one("#yes", Button)
            await pilot.click(yes_button)
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
    app = make_editor(project, voice_sample_count=2)
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
            await pilot.press("2", "escape", "y")
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
            await pilot.press("2", "escape", "y")
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
    monkeypatch.setattr(
        VoiceLineEditorTextualApp, "check_terminal_support", lambda: True
    )
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
    monkeypatch.setattr(
        VoiceLineEditorTextualApp, "check_terminal_support", lambda: True
    )
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
