from typing import cast

import pytest
from rich.text import Text
from textual.widgets import Button, Input
from tts_audiobook_tool.app_types import BookSection
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual import voice_line_editor
from tts_audiobook_tool.textual.content_textual_app import (
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
from textual_editor_stubs import (
    make_phrase_group,
    make_project,
    run,
    StubPhraseGroup,
    StubProject,
)


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


def make_sectioned_editor(
    sections: list[BookSection], voice_sample_count: int = 2
) -> VoiceLineEditorTextualApp:
    app = VoiceLineEditorTextualApp(make_project(sections), voice_sample_count)
    app.load_content()
    return app


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
    assert app.find_match_indices("00002") == [3]


def test_find_text_strings_expose_voice_label_separately_from_phrase_text() -> None:
    app = make_sectioned_editor(
        [
            BookSection(
                title="Opening",
                phrase_groups=[
                    make_phrase_group("One.", voice_index=0),
                    make_phrase_group("Two.", voice_index=1),
                ],
            ),
            BookSection(
                title="Ending",
                phrase_groups=[make_phrase_group("Three.", voice_index=5)],
            ),
        ],
        voice_sample_count=2,
    )

    # Rows: [section, One.(v0), Two.(v1), section, Three.(v5 with 2 samples)]
    assert app.find_text_strings(1) == ["00001", "Voice 1", "One."]
    assert app.find_text_strings(2) == ["00002", "Voice 2", "Two."]
    assert app.find_text_strings(3) == ["Section 2/2: Ending (1 line)"]
    assert app.find_text_strings(4) == ["00003", "Voice 6 *OUT OF RANGE*", "Three."]

    # Fields are searched separately, so voice-label metadata matches even
    # when it is not part of the phrase text.
    app = make_sectioned_editor(
        [
            BookSection(
                title="Opening",
                phrase_groups=[
                    make_phrase_group("One.", voice_index=0),
                    make_phrase_group("Two.", voice_index=1),
                ],
            ),
            BookSection(title="Ending", phrase_groups=[make_phrase_group("Three.")]),
        ],
        voice_sample_count=2,
    )

    # Rows: One. is "Voice 1", Two. is "Voice 2", Three. defaults to "Voice 1".
    assert app.find_match_indices("voice 2") == [2]
    assert app.find_match_indices("VOICE 1") == [1, 4]
    assert app.find_match_indices("voice 3") == []
    # Fields are tested separately, so no match can span the label/content join.
    assert app.find_match_indices("voice 1o") == []
    # The displayed line number is searchable; section rows carry no number.
    assert app.find_match_indices("00003") == [4]
    assert app.find_match_indices("00004") == []

    # An out-of-range voice label renders with a note that is searchable too.
    app = make_sectioned_editor(
        [
            BookSection(
                title="Opening",
                phrase_groups=[make_phrase_group("One.", voice_index=5)],
            ),
            BookSection(title="Ending", phrase_groups=[make_phrase_group("Two.")]),
        ],
        voice_sample_count=2,
    )

    # Rows: [section, One.(v5 -> "Voice 6 *OUT OF RANGE*"), section, Two.(v-1)]
    assert app.find_match_indices("out of range") == [1]
    assert app.find_match_indices("voice 6") == [1]
    assert app.find_match_indices("voice 2") == []


def test_find_metadata_tracks_reassigned_voice() -> None:
    app = make_sectioned_editor(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ],
        voice_sample_count=2,
    )

    assert app.find_match_indices("voice 2") == []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "2")
            # Row 1 (One.) is now staged with voice index 1.
            assert app.staged_voice_indices == [1, -1]
            assert app.find_match_indices("voice 2") == [1]
            assert app.find_match_indices("voice 1") == [3]

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
            # A highlighted section row is never assigned a voice
            await pilot.press("2")
            assert app.staged_voice_indices == [-1, -1]
            assert app.selected_indices == {0}

            # A selection spanning a section heading assigns the phrase row only
            await pilot.press("down", "shift+up")
            assert app.selected_index == 0
            assert app.selected_indices == {0, 1}
            assert app.highlighted_content_line_index() is None

            await pilot.press("2")
            assert app.staged_voice_indices == [1, -1]
            assert app.selected_indices == {0}

            # A selection spanning phrase rows on both sides of a heading
            await pilot.press("home", "down", "shift+down", "shift+down", "2")
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


