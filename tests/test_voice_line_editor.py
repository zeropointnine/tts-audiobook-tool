import asyncio
from dataclasses import dataclass
from typing import cast

import pytest
from textual.css.errors import StylesheetError
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.textual import voice_line_editor
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

    assert str(app.header_lines[1]) == (
        "- Use number keys [1] to [2] to set voice sample for selected text line/s"
    )


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
