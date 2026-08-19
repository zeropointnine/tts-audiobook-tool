import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
from rich.console import Console
from rich.text import Text
from textual.widgets import Button, Input, Static
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.textual import voice_line_editor
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppStylesheetFailed,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.manual_selection_dialog import ManualSelectionDialog
from tts_audiobook_tool.textual.textual_shared import NonWrappingOptionList
from tts_audiobook_tool.textual.voice_line_editor import (
    VoiceLineEditorTextualApp,
    VoiceLinePhraseGroupItem,
    VoiceLineSectionItem,
)


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
    app = VoiceLineEditorTextualApp(cast(Project, project), voice_sample_count)
    app.load_content()
    return app


def make_app(
    num_lines: int = 12, voice_sample_count: int = 9
) -> tuple[VoiceLineEditorTextualApp, StubProject]:
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(num_lines)]
    )
    return make_editor(project, voice_sample_count), project


def make_phrase_group(text: str, voice_index: int = -1) -> PhraseGroup:
    return PhraseGroup(
        phrases=[Phrase(text, Reason.SENTENCE)],
        voice_index=voice_index,
    )


def make_sectioned_editor(
    sections: list[BookSection], voice_sample_count: int = 2
) -> VoiceLineEditorTextualApp:
    project = Project.model_validate({"book": Book(sections=sections)})
    app = VoiceLineEditorTextualApp(project, voice_sample_count)
    app.load_content()
    return app


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


def test_rows_are_deferred_but_voice_header_data_is_loaded_synchronously() -> None:
    project = StubProject([StubPhraseGroup("Line 1")])
    app = VoiceLineEditorTextualApp(cast(Project, project), voice_sample_count=2)

    assert Text.from_ansi(app.header_lines[3]).plain == (
        "- Use number keys [1] to [2] to set voice sample for selected text line/s"
    )
    assert app.content_initialized is False
    assert app.phrase_indices == []
    assert app.original_voice_indices == []
    assert app.staged_voice_indices == []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.content_initialized is True
            assert app.phrase_indices == [0]
            assert app.original_voice_indices == [-1]
            assert app.staged_voice_indices == [-1]

    run(exercise())


def test_voice_actions_are_ignored_before_deferred_content_loads() -> None:
    project = StubProject([StubPhraseGroup("Line 1")])
    app = VoiceLineEditorTextualApp(cast(Project, project), voice_sample_count=2)

    app.action_assign_voice(1)
    app.commit_changes_and_exit()

    assert project.phrase_groups[0].voice_index == -1
    assert app.original_voice_indices == []
    assert app.staged_voice_indices == []
    assert app.has_changes is False


def test_header_limits_voice_key_range_to_available_samples() -> None:
    app, _ = make_app(voice_sample_count=2)

    assert all(isinstance(line, str) for line in app.header_lines)
    assert all(not line.endswith(Ansi.RESET) for line in app.header_lines)
    assert Text.from_ansi(app.header_lines[3]).plain == (
        "- Use number keys [1] to [2] to set voice sample for selected text line/s"
    )


def test_single_section_lists_only_phrase_group_rows() -> None:
    app = make_sectioned_editor(
        [
            BookSection(
                title="Only section",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            )
        ]
    )

    assert all(isinstance(item, VoiceLinePhraseGroupItem) for item in app.list_items)
    assert [str(app.format_line(index)) for index in range(len(app.list_items))] == [
        "[00001] [Voice 1] One.",
        "[00002] [Voice 1] Two.",
    ]
    assert app.find_match_indices("only section") == []


def test_multiple_sections_insert_generated_section_rows() -> None:
    app = make_sectioned_editor(
        [
            BookSection(
                title="Opening",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            ),
            BookSection(
                title="Middle",
                phrase_groups=[make_phrase_group("Three.")],
            ),
        ]
    )

    assert [type(item) for item in app.list_items] == [
        VoiceLineSectionItem,
        VoiceLinePhraseGroupItem,
        VoiceLinePhraseGroupItem,
        VoiceLineSectionItem,
        VoiceLinePhraseGroupItem,
    ]
    assert [str(app.format_line(index)) for index in range(len(app.list_items))] == [
        "\nSection 1/2: Opening (2 lines)\n\n",
        "[00001] [Voice 1] One.",
        "[00002] [Voice 1] Two.",
        "\nSection 2/2: Middle (1 line)\n\n",
        "[00003] [Voice 1] Three.",
    ]
    assert app.format_line(0).spans == []


def test_voice_assignment_uses_selected_phrase_when_section_is_highlighted() -> None:
    app = make_sectioned_editor(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ]
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2")
            assert app.staged_voice_indices == [-1, -1]

            await pilot.press("down", "shift+up")
            assert app.selected_index == 0
            assert app.selected_indices == {0, 1}
            assert app.highlighted_content_line_index() is None

            await pilot.press("2")
            assert app.staged_voice_indices == [1, -1]
            assert app.selected_indices == {0}

    run(exercise())


def test_empty_sections_are_hidden_and_all_empty_sections_show_empty_state() -> None:
    app = make_sectioned_editor(
        [
            BookSection(title="Empty", phrase_groups=[]),
            BookSection(title="Text", phrase_groups=[make_phrase_group("One.")]),
        ]
    )
    empty_app = make_sectioned_editor(
        [
            BookSection(title="Empty one", phrase_groups=[]),
            BookSection(title="Empty two", phrase_groups=[]),
        ]
    )

    assert [str(app.format_line(index)) for index in range(len(app.list_items))] == [
        "\nSection 2/2: Text (1 line)\n\n",
        "[00001] [Voice 1] One.",
    ]
    assert empty_app.list_items == []

    async def exercise() -> None:
        async with empty_app.run_test():
            assert (
                empty_app.query_one("#line-list", NonWrappingOptionList).display
                is False
            )
            assert empty_app.query_one("#empty-state", Static).display is True

    run(exercise())


def test_find_searches_generated_section_text_and_phrase_text() -> None:
    app = make_sectioned_editor(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(
                title="Needle Chapter",
                phrase_groups=[make_phrase_group("Haystack.")],
            ),
            BookSection(title="Ending", phrase_groups=[make_phrase_group("Three.")]),
        ]
    )

    assert app.find_match_indices("section 2/3") == [2]
    assert app.find_match_indices("needle chapter (1 line)") == [2]
    assert app.find_match_indices("haystack") == [3]

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+f")
            find_input = app.query_one("#find-input", Input)
            find_input.value = "Section 2/3"
            await pilot.press("enter")
            assert app.find_match_index == 2
            assert app.selected_index == 2

    run(exercise())


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
    project.phrase_groups[
        0
    ].presentable_text = "one two three four five six seven eight nine"
    console = Console(width=36, force_terminal=False, color_system=None)

    rendered_lines = console.render_lines(
        app.format_line(0), console.options, pad=False
    )
    rendered = ["".join(segment.text for segment in line) for line in rendered_lines]

    assert rendered == [
        "[00001] [Voice 1] one two",
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


def test_multiline_selection_leaves_section_rows_visually_unchanged_and_uncounted() -> (
    None
):
    app = make_sectioned_editor(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ]
    )

    async def exercise() -> None:
        async with app.run_test(size=(60, 24)) as pilot:
            # Move to the first phrase, then extend through the next heading and phrase.
            await pilot.press("down", "shift+down", "shift+down")
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            assert app.selected_indices == {1, 2, 3}
            assert app.selection_status_text == "2 lines selected"
            assert option_list.inactive_selection_indices == {1}

            section_row_y = next(
                y
                for y, (option_index, _line_offset) in enumerate(option_list._lines)
                if option_index == 2
            )
            section_line = option_list.render_line(section_row_y + 1)
            assert not any(
                segment.style is not None and segment.style.reverse
                for segment in section_line
            )

    run(exercise())


def test_manual_selection_uses_project_line_numbers_and_excludes_section_rows() -> None:
    app = make_sectioned_editor(
        [
            BookSection(
                title="Opening",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            ),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Three.")]),
        ]
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            app.screen.query_one("#manual-selection-input", Input).value = "1, 3"
            await pilot.press("enter")

            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == {1, 4}
            assert app.selected_index == 4
            assert app.selection_anchor_index == 4
            assert app.query_one("#line-list", NonWrappingOptionList).highlighted == 4
            assert all(
                isinstance(app.list_items[index], VoiceLinePhraseGroupItem)
                for index in app.selected_indices
            )

    run(exercise())


def test_number_hotkey_ignores_highlighted_section_row() -> None:
    app = make_sectioned_editor(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ]
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            assert app.selected_index == 0
            await pilot.press("2")
            assert app.staged_voice_indices == [-1, -1]
            assert app.selected_indices == {0}

    run(exercise())


def test_number_hotkey_assigns_only_phrase_rows_when_selection_crosses_section() -> (
    None
):
    app = make_sectioned_editor(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ]
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "shift+down", "shift+down", "2")
            assert app.staged_voice_indices == [1, 1]
            assert app.selected_indices == {3}
            assert app.selection_anchor_index == 3

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
        lambda indices, *, reflow=True: refresh_calls.append((list(indices), reflow)),
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
        lambda indices, *, reflow=True: refresh_calls.append((list(indices), reflow)),
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
        "save_book",
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
        assert app.return_value == EditorSaved()

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
        "save_book",
        lambda *_: "disk full",
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2", "escape", "y")
            await pilot.pause()

        assert [group.voice_index for group in project.phrase_groups] == [0, -1]
        assert app.return_value == EditorSaveFailed("Save failed: disk full")

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
        "save_book",
        fail_save,
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2", "escape", "y")
            await pilot.pause()

        assert project.phrase_groups[0].voice_index == -1
        assert app.return_value == EditorSaveFailed(
            "Save failed: RuntimeError: unexpected failure"
        )

    run(exercise())


def test_voice_menu_reports_save_failure_as_error_feedback(monkeypatch) -> None:
    project = StubProject([StubPhraseGroup("Line 1")])
    state = cast(State, SimpleNamespace(project=project))
    feedback_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        voice_menu_shared,
        "VoiceLineEditorTextualApp",
        lambda _: object(),
    )
    monkeypatch.setattr(
        voice_menu_shared,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorSaveFailed("Save failed: disk full")),
    )
    monkeypatch.setattr(
        voice_menu_shared,
        "print_feedback",
        lambda message, **kwargs: feedback_calls.append(
            (message, kwargs.get("is_error", False))
        ),
    )

    VoiceMenuShared.assign_voice_samples_to_text_lines(state)

    assert feedback_calls == [("Save failed: disk full", True)]


def test_voice_menu_reports_css_load_failure_as_error_feedback(monkeypatch) -> None:
    project = StubProject([StubPhraseGroup("Line 1")])
    state = cast(State, SimpleNamespace(project=project))
    feedback_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        voice_menu_shared,
        "VoiceLineEditorTextualApp",
        lambda _: object(),
    )
    monkeypatch.setattr(
        voice_menu_shared,
        "run_content_textual_app",
        lambda _: ContentAppStylesheetFailed("Couldn't load textual css"),
    )
    monkeypatch.setattr(
        voice_menu_shared,
        "print_feedback",
        lambda message, **kwargs: feedback_calls.append(
            (message, kwargs.get("is_error", False))
        ),
    )

    VoiceMenuShared.assign_voice_samples_to_text_lines(state)

    assert feedback_calls == [("Couldn't load textual css", True)]
