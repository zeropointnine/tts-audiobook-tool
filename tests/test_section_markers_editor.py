import pytest
from typing import cast

from rich.style import Style
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from tts_audiobook_tool.constants import COL_GRAY
from tts_audiobook_tool.app_types import BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.textual.section_markers_dialog import (
    SectionMarkersDialog,
    SectionMarkersStep,
    make_blank_line_marker_indices,
)
from tts_audiobook_tool.textual.content_textual_app import (
    EditorClosed,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.section_markers_editor import (
    SectionMarkersEditor,
    SectionMarkersPhraseGroupItem,
    SectionMarkersSectionItem,
)

from textual_editor_stubs import (
    make_phrase_group,
    make_project,
    make_project_with_markers,
    run,
)

def make_space_break_group(text: str) -> PhraseGroup:
    return PhraseGroup(phrases=[Phrase(text=text, reason=Reason.SPACE_BREAK)])

def style_at(text: Text, offset: int) -> Style:
    for start, end, style in text.spans:
        if start <= offset < end:
            return Style.parse(style) if isinstance(style, str) else style
    return Style()

def make_markers_editor(
    sections: list[BookSection], markers: list[int]
) -> tuple[SectionMarkersEditor, Project]:
    project = make_project_with_markers(sections, markers)
    app = SectionMarkersEditor(project)
    app.load_content()
    return app, project

def make_loaded_editor(project: Project) -> SectionMarkersEditor:
    app = SectionMarkersEditor(project)
    app.load_content()
    return app

def test_editor_projects_multiple_sections_and_phrase_rows_in_book_order() -> None:
    app = make_loaded_editor(
        make_project(
            [
                BookSection(
                    title="Opening",
                    phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
                ),
                BookSection(
                    title="Ending",
                    phrase_groups=[make_phrase_group("Three.")],
                ),
            ]
        )
    )

    assert [type(item) for item in app.list_items] == [
        SectionMarkersSectionItem,
        SectionMarkersPhraseGroupItem,
        SectionMarkersPhraseGroupItem,
        SectionMarkersSectionItem,
        SectionMarkersPhraseGroupItem,
    ]
    assert str(app.format_line(0)) == "\nSection 1/2: Opening (2 lines)\n\n"
    assert str(app.format_line(1)) == "00001  One."
    assert str(app.format_line(4)) == "00003  Three."
    assert app.content_line_index(0) is None
    assert app.content_line_index(1) == 0
    assert app.find_match_indices("ending") == [3]
    assert app.find_match_indices("three") == [4]
    assert app.find_match_indices("00003") == [4]
    assert app.find_match_indices("00004") == []

def test_editor_omits_heading_for_a_single_section() -> None:
    app = make_loaded_editor(
        make_project(
            [BookSection(title="Only", phrase_groups=[make_phrase_group("One.")])]
        )
    )

    assert app.list_items == [SectionMarkersPhraseGroupItem(0)]
    assert str(app.format_line(0)) == "00001  One."

def test_markers_panel_text_enumerates_current_markers() -> None:
    app, _ = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                ]
            )
        ],
        [0, 2],
    )

    text = app.markers_panel_text()
    assert text.plain == (
        "Current section markers (1 item)\n\n00003  Three.\n"
    )
    header = "Current section markers (1 item)"
    # Header and line numbers use the default color; phrase text is dim.
    assert style_at(text, 0).color is None
    assert style_at(text, len(header) + 2).color is None
    dim_color = Style.parse("#888888").color
    assert style_at(text, text.plain.index("Three.")).color == dim_color

    empty_app, _ = make_markers_editor([BookSection(phrase_groups=[])], [])
    empty_text = empty_app.markers_panel_text()
    assert empty_text.plain == "Current section markers (0 items)\n\nNone"
    assert style_at(empty_text, empty_text.plain.index("None")).color == dim_color

def test_markers_panel_reserves_its_last_line_for_overflow_count() -> None:
    app = SectionMarkersEditor(
        make_project_with_markers(
            [
                BookSection(
                    phrase_groups=[make_phrase_group(f"Line {index}.") for index in range(9)]
                )
            ],
            list(range(1, 9)),
        )
    )

    async def exercise() -> None:
        async with app.run_test(size=(100, 12)) as pilot:
            await pilot.pause()
            panel = app.query_one("#markers-panel", Static)
            text = cast(Text, panel.content)
            assert text.plain == (
                "Current section markers (8 items)\n\n"
                "00002  Line 1.\n"
                "                    +7 more items"
            )
            assert style_at(text, text.plain.index("+7 more items")).color == (
                style_at(Text.from_ansi(f"{COL_GRAY}x"), 0).color
            )

    run(exercise())

def test_space_toggles_marker_on_highlighted_line_and_updates_panel() -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                ]
            )
        ],
        [2],
    )

    assert str(app.format_line(0)) == "00001  One."
    assert str(app.format_line(2)) == "00003* Three."

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            side_panel = app.query_one("#side-panel", Vertical)
            assert [child.id for child in side_panel.children] == [
                "markers-panel"
            ]
            assert app.query_one("#side-panel-divider")

            await pilot.press("shift+down")
            await pilot.press("space")
            assert app.staged_markers == {1, 2}
            assert app.has_changes is True
            assert str(app.format_line(1)) == "00002* Two."

            panel = app.query_one("#markers-panel", Static)
            assert cast(Text, panel.content).plain == (
                "Current section markers (2 items)\n\n"
                "00002  Two.\n00003  Three.\n"
            )

            await pilot.press("space")
            assert app.staged_markers == {2}
            assert app.has_changes is False
            assert cast(Text, panel.content).plain == (
                "Current section markers (1 item)\n\n00003  Three.\n"
            )

            # The project is not mutated until the exit is confirmed.
            assert project.markers == {2}

    run(exercise())

def test_space_on_first_line_shows_toast_without_adding_marker() -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("space")

            assert app.staged_markers == set()
            assert app.toast_text == "Adding first line is not allowed"
            assert project.markers == set()

    run(exercise())

def test_header_documents_the_space_toggle_and_escape_finish_keys() -> None:
    app, _ = make_markers_editor([BookSection(phrase_groups=[])], [])

    assert Text.from_ansi(app.header_lines[2]).plain == (
        "- Press [SPACE] to toggle a section marker on the highlighted line"
    )
    assert Text.from_ansi(app.header_lines[3]).plain == (
        "- [M] More options (add manually, by regex, by blank lines, clear)"
    )
    assert Text.from_ansi(app.header_lines[4]).plain == "- Press [ESC] to finish"

def test_marker_row_indices_maps_markers_past_section_headings() -> None:
    app, _ = make_markers_editor(
        [
            BookSection(title="A", phrase_groups=[make_phrase_group("One.")]),
            BookSection(
                title="B",
                phrase_groups=[make_phrase_group("Two."), make_phrase_group("Three.")],
            ),
        ],
        [2],
    )

    assert app.marker_row_indices() == [4]

@pytest.mark.parametrize(
    ("line_count", "markers", "presses", "expected_selected_indices"),
    [
        pytest.param(
            3,
            [1, 2],
            ("]", "]", "[", "["),
            (1, 2, 1, 2),
            id="wraps-around-at-both-ends",
        ),
        pytest.param(
            4,
            [3],
            ("]", "["),
            (3, 3),
            id="skips-rows-without-markers",
        ),
        pytest.param(
            2,
            [],
            ("]", "["),
            (0, 0),
            id="ignored-when-no-markers-exist",
        ),
    ],
)
def test_bracket_keys_jump_to_next_and_previous_markers_wrapping_around(
    line_count: int,
    markers: list[int],
    presses: tuple[str, ...],
    expected_selected_indices: tuple[int, ...],
) -> None:
    app, _ = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group(f"Line {index + 1}.")
                    for index in range(line_count)
                ]
            )
        ],
        markers,
    )

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.selected_index == 0
            for press, expected in zip(presses, expected_selected_indices):
                await pilot.press(press)
                assert app.selected_index == expected

    run(exercise())

@pytest.mark.parametrize(
    ("key", "preloaded"),
    [
        pytest.param("space", True, id="space-ignored-after-load-flag-cleared"),
        pytest.param("m", False, id="m-ignored-before-content-loads"),
    ],
)
def test_editor_actions_are_ignored_before_content_is_initialized(
    key: str, preloaded: bool
) -> None:
    if preloaded:
        app, _ = make_markers_editor(
            [BookSection(phrase_groups=[make_phrase_group("One.")])], []
        )
        app.content_initialized = False
    else:
        app = SectionMarkersEditor(
            make_project([BookSection(phrase_groups=[])])
        )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press(key)
            if key == "space":
                assert app.staged_markers == set()
                assert app.has_changes is False
            else:
                assert not isinstance(app.screen, SectionMarkersDialog)

    run(exercise())

def test_confirmed_exit_applies_staged_markers_and_saves(monkeypatch) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                ]
            )
        ],
        [],
    )
    saves: list[Project] = []
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self) or "")

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "space", "escape")
            assert project.markers == set()

            yes_button = app.screen.query_one("#yes", Button)
            await pilot.click(yes_button)
            await pilot.pause()
            assert app.is_running is False

        assert project.markers == {1}
        assert app.staged_markers == {1}
        assert app.return_value == EditorSaved()

    run(exercise())

    # Declining the exit dialog discards staged markers without saving.
    declined_app, declined_project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                ]
            )
        ],
        [],
    )

    async def exercise_decline() -> None:
        async with declined_app.run_test() as pilot:
            await pilot.press("shift+down", "space", "escape")
            assert declined_app.has_changes is True
            await pilot.press("n")
            await pilot.pause()

        assert declined_project.markers == set()
        assert declined_app.return_value == EditorClosed()

    run(exercise_decline())

    assert saves == [project]

def test_save_failure_rolls_back_markers_and_records_error(monkeypatch) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                ]
            )
        ],
        [],
    )
    monkeypatch.setattr(Project, "save", lambda self: "disk full")

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "space", "escape", "y")
            await pilot.pause()

        assert project.markers == set()
        assert app.return_value == EditorSaveFailed("Save failed: disk full")

    run(exercise())

def test_section_markers_dialog_width_tracks_terminal_with_limits() -> None:
    async def get_dialog_region(terminal_width: int) -> tuple[int, int]:
        app = make_loaded_editor(
            make_project(
                [BookSection(phrase_groups=[make_phrase_group("One.")])]
            )
        )
        async with app.run_test(size=(terminal_width, 30)) as pilot:
            await pilot.press("m")
            await pilot.pause()
            dialog = app.screen.query_one("#section-markers-dialog", Vertical)
            return dialog.region.x, dialog.region.width

    async def exercise() -> None:
        # The dialog fills the terminal inside two-column margins until it
        # reaches its maximum width, and remains centered after that point.
        assert await get_dialog_region(44) == (2, 40)
        assert await get_dialog_region(80) == (2, 76)
        assert await get_dialog_region(120) == (20, 80)

    run(exercise())

def test_blank_lines_step_shows_description_and_no_matches() -> None:
    app = make_loaded_editor(
        make_project(
            [BookSection(phrase_groups=[make_phrase_group("One.")])]
        )
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "3")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            title = str(
                dialog.query_one("#section-markers-title", Static).render()
            )
            assert "Add at blank lines" in title
            body = str(dialog.query_one("#section-markers-body", Static).render())
            assert (
                "Adds section markers wherever 2+ consecutive blank lines" in body
            )
            assert '"\\n\\n\\n"' in body
            assert "No matches found" not in body
            status_widget = dialog.query_one("#section-markers-status", Static)
            assert str(status_widget.render()) == "No matches found"
            assert status_widget.styles.text_align == "center"
            assert status_widget.display
            assert status_widget.visible
            input_widget = dialog.query_one("#section-markers-input", Input)
            assert not input_widget.display
            buttons_row = dialog.query_one(
                "#section-markers-buttons", Horizontal
            )
            assert not buttons_row.display

            # Confirming with no matches is a no-op.
            await pilot.press("y")
            await pilot.pause()
            assert isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == set()

    run(exercise())

def test_blank_lines_step_confirms_and_adds_markers() -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_space_break_group("Break."),
                    make_phrase_group("Three."),
                    make_phrase_group("Four."),
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "3")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            body = str(dialog.query_one("#section-markers-body", Static).render())
            assert "Num matches found: 1" not in body
            assert "No matches found" not in body
            status_widget = dialog.query_one("#section-markers-status", Static)
            assert str(status_widget.render()) == "Num matches found: 1"
            assert not str(status_widget.render()).startswith(" ")
            assert status_widget.styles.text_align == "center"
            buttons_row = dialog.query_one(
                "#section-markers-buttons", Horizontal
            )
            assert buttons_row.display
            assert buttons_row.visible

            await pilot.click(dialog.query_one("#yes", Button))
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == {2}
            assert app.toast_text == "Added 1 section marker"
            panel_text = str(app.query_one("#markers-panel", Static).render())
            assert "Three." in panel_text

    run(exercise())

def test_make_blank_line_marker_indices() -> None:
    # No breaks -> no markers.
    assert make_blank_line_marker_indices([make_phrase_group("One.")]) == []
    # Marker is index+1 of the preceding group with a SPACE_BREAK phrase.
    groups = [
        make_phrase_group("One."),
        make_space_break_group("Break."),
        make_phrase_group("Three."),
    ]
    assert make_blank_line_marker_indices(groups) == [2]
    # A SPACE_BREAK anywhere in the group counts, not just the last phrase.
    mixed = PhraseGroup(
        phrases=[
            Phrase(text="Break. ", reason=Reason.SPACE_BREAK),
            Phrase(text="More.", reason=Reason.SENTENCE),
        ]
    )
    assert (
        make_blank_line_marker_indices(
            [make_phrase_group("One."), mixed, make_phrase_group("Three.")]
        )
        == [2]
    )
    # A SPACE_BREAK in the last group never yields a marker.
    assert make_blank_line_marker_indices([make_space_break_group("Only.")]) == []
    # Multiple breaks -> multiple sorted indices.
    multi = [
        make_phrase_group("One."),
        make_space_break_group("Break."),
        make_phrase_group("Three."),
        make_space_break_group("Break."),
        make_phrase_group("Five."),
    ]
    assert make_blank_line_marker_indices(multi) == [2, 4]

def test_manual_step_shows_line_number_input() -> None:
    app = make_loaded_editor(
        make_project(
            [BookSection(phrase_groups=[make_phrase_group("One.")])]
        )
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "1")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            body = str(dialog.query_one("#section-markers-body", Static).render())
            assert "Enter line number/s" in body
            assert "Eg, \"105, 200\"" in body
            input_widget = dialog.query_one("#section-markers-input", Input)
            assert input_widget.display
            assert dialog.focused is input_widget

    run(exercise())

@pytest.mark.parametrize(
    ("initial_markers", "line_input", "expected_staged"),
    [
        pytest.param([], "2, 3", {1, 2}, id="adds-new-markers"),
        pytest.param(
            [1], "3, 2", {1, 2}, id="merges-with-existing-staged"
        ),
    ],
)
def test_manual_entry_adds_markers_to_staged_set_and_updates_view(
    initial_markers: list[int], line_input: str, expected_staged: set[int]
) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                ]
            )
        ],
        initial_markers,
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "1")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            dialog.query_one("#section-markers-input", Input).value = line_input
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == expected_staged
            panel_text = str(
                app.query_one("#markers-panel", Static).render()
            )
            assert "Two." in panel_text
            assert "Three." in panel_text

    run(exercise())

def test_manual_entry_drops_line_one_and_deduplicates() -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "1")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            dialog.query_one("#section-markers-input", Input).value = "1, 1, 2"
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == {1}

    run(exercise())

@pytest.mark.parametrize(
    ("line_input", "expected_error"),
    [
        pytest.param("2, ten", "Parse error: ten", id="parse-error"),
        pytest.param("5", "Index out of range: 5", id="out-of-range"),
    ],
)
def test_manual_entry_error_shown_below_input(
    line_input: str, expected_error: str
) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "1")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            dialog.query_one("#section-markers-input", Input).value = line_input
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, SectionMarkersDialog)
            error_text = str(
                dialog.query_one("#section-markers-error", Static).render()
            )
            assert expected_error in error_text
            assert app.staged_markers == set()

    run(exercise())

@pytest.mark.parametrize(
    ("step_key", "expected_step"),
    [
        pytest.param("1", SectionMarkersStep.MANUAL, id="manual-step"),
        pytest.param("2", SectionMarkersStep.REGEX, id="regex-step"),
    ],
)
def test_empty_input_closes_dialog_without_changes(
    step_key: str, expected_step: SectionMarkersStep
) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", step_key)
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            assert dialog.step is expected_step
            dialog.query_one("#section-markers-input", Input).value = ""
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == set()

    run(exercise())

def test_escape_closes_section_markers_dialog_at_any_step() -> None:
    app = make_loaded_editor(
        make_project(
            [BookSection(phrase_groups=[make_phrase_group("One.")])]
        )
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, SectionMarkersDialog)
            await pilot.press("escape")
            assert not isinstance(app.screen, SectionMarkersDialog)

            await pilot.press("m", "2")
            await pilot.pause()
            assert isinstance(app.screen, SectionMarkersDialog)
            await pilot.press("escape")
            assert not isinstance(app.screen, SectionMarkersDialog)

            await pilot.press("m", "1")
            await pilot.pause()
            assert isinstance(app.screen, SectionMarkersDialog)
            await pilot.press("escape")
            assert not isinstance(app.screen, SectionMarkersDialog)

    run(exercise())

def test_regex_step_shows_pattern_input() -> None:
    app = make_loaded_editor(
        make_project(
            [
                BookSection(
                    phrase_groups=[
                        make_phrase_group("Intro."),
                        make_phrase_group("Chapter 241: Chapter Name"),
                    ]
                )
            ]
        )
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "2")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            title = str(
                dialog.query_one("#section-markers-title", Static).render()
            )
            assert "Enter a regular expression" in title
            body = str(dialog.query_one("#section-markers-body", Static).render())
            assert "you could enter \"Chapter \\d+\"" in body
            input_widget = dialog.query_one("#section-markers-input", Input)
            assert input_widget.display
            assert dialog.focused is input_widget

    run(exercise())

@pytest.mark.parametrize(
    ("phrase_texts", "pattern", "expected_staged", "expected_toast"),
    [
        pytest.param(
            ["Intro.", "Chapter 1: One.", "Chapter 2: Two.", "Outro."],
            "Chapter \\d+",
            {1, 2},
            "Added 2 section markers",
            id="matches-capitalized-chapters",
        ),
        pytest.param(
            ["Intro.", "Chapter 1: One."],
            "chapter \\d+",
            {1},
            "Added 1 section marker",
            id="case-insensitive-match",
        ),
    ],
)
def test_regex_entry_adds_markers_to_staged_set_and_updates_view(
    phrase_texts: list[str],
    pattern: str,
    expected_staged: set[int],
    expected_toast: str,
) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group(text) for text in phrase_texts
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "2")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            dialog.query_one("#section-markers-input", Input).value = pattern
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == expected_staged
            assert app.toast_text == expected_toast
            panel_text = str(
                app.query_one("#markers-panel", Static).render()
            )
            assert "Chapter 1: One." in panel_text

    run(exercise())

@pytest.mark.parametrize(
    ("pattern", "expected_error"),
    [
        pytest.param("^\\s*$", "No matches", id="no-matches"),
        pytest.param("Chapter (\\d+", "Syntax error", id="syntax-error"),
    ],
)
def test_regex_entry_error_shown_below_input(
    pattern: str, expected_error: str
) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Intro."),
                    make_phrase_group("Chapter 1: One."),
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "2")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            dialog.query_one("#section-markers-input", Input).value = pattern
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, SectionMarkersDialog)
            error_text = str(
                dialog.query_one("#section-markers-error", Static).render()
            )
            assert expected_error in error_text
            assert app.staged_markers == set()

    run(exercise())

def test_regex_entry_drops_line_one() -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Chapter 1: One."),
                    make_phrase_group("Chapter 2: Two."),
                ]
            )
        ],
        [],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "2")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            dialog.query_one("#section-markers-input", Input).value = "Chapter \\d+"
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == {1}

    run(exercise())

def test_menu_shows_clear_option_only_when_markers_exist() -> None:
    with_markers = make_loaded_editor(
        make_project_with_markers(
            [
                BookSection(
                    phrase_groups=[
                        make_phrase_group("One."),
                        make_phrase_group("Two."),
                    ]
                )
            ],
            [1],
        )
    )

    async def exercise_with_markers() -> None:
        async with with_markers.run_test() as pilot:
            await pilot.press("m")
            await pilot.pause()
            body_text = str(
                with_markers.screen.query_one(
                    "#section-markers-body", Static
                ).render()
            )
            assert "[1] Enter line number/s" in body_text
            assert "[2] Add using regular expression" in body_text
            assert "[3] Add at blank lines" in body_text
            assert "[4] Clear section markers" in body_text

    run(exercise_with_markers())

    without_markers = make_loaded_editor(
        make_project(
            [BookSection(phrase_groups=[make_phrase_group("One.")])]
        )
    )

    async def exercise_without_markers() -> None:
        async with without_markers.run_test() as pilot:
            await pilot.press("m")
            await pilot.pause()
            body_text = str(
                without_markers.screen.query_one(
                    "#section-markers-body", Static
                ).render()
            )
            assert "[1] Enter line number/s" in body_text
            assert "[2] Add using regular expression" in body_text
            assert "[3] Add at blank lines" in body_text
            assert "[4] Clear section markers" not in body_text

            dialog = without_markers.screen
            await pilot.press("4")
            await pilot.pause()
            assert isinstance(without_markers.screen, SectionMarkersDialog)
            assert dialog.step is SectionMarkersStep.MENU

    run(exercise_without_markers())

@pytest.mark.parametrize(
    ("phrase_count", "markers", "expected_prompt"),
    [
        pytest.param(
            3, [1, 2], "Clear all 2 section markers?", id="plural-count"
        ),
        pytest.param(
            2, [1], "Clear all 1 section marker?", id="singular-count"
        ),
    ],
)
def test_four_opens_clear_step_with_buttons_when_markers_exist(
    phrase_count: int, markers: list[int], expected_prompt: str
) -> None:
    app = make_loaded_editor(
        make_project_with_markers(
            [
                BookSection(
                    phrase_groups=[
                        make_phrase_group(f"Line {index + 1}.")
                        for index in range(phrase_count)
                    ]
                )
            ],
            markers,
        )
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "4")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            assert dialog.step is SectionMarkersStep.CLEAR
            body_widget = dialog.query_one("#section-markers-body", Static)
            body = str(body_widget.render())
            assert expected_prompt in body
            assert "body-centered" in body_widget.classes
            buttons_row = dialog.query_one(
                "#section-markers-buttons", Horizontal
            )
            assert buttons_row.display
            assert buttons_row.visible
            assert dialog.query_one("#yes", Button) is not None
            assert dialog.query_one("#no", Button) is not None
            input_widget = dialog.query_one("#section-markers-input", Input)
            assert not input_widget.display

    run(exercise())

@pytest.mark.parametrize(
    ("phrase_count", "markers", "confirm_via", "expected_toast"),
    [
        pytest.param(
            3,
            [1, 2],
            "key",
            "Cleared all 2 section markers",
            id="y-key-plural",
        ),
        pytest.param(
            2,
            [1],
            "button",
            "Cleared all 1 section marker",
            id="yes-button-singular",
        ),
    ],
)
def test_confirmed_clear_resets_staged_markers_and_view(
    phrase_count: int,
    markers: list[int],
    confirm_via: str,
    expected_toast: str,
) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group(f"Line {index + 1}.")
                    for index in range(phrase_count)
                ]
            )
        ],
        markers,
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "4")
            await pilot.pause()
            assert isinstance(app.screen, SectionMarkersDialog)
            if confirm_via == "key":
                await pilot.press("y")
            else:
                dialog = app.screen
                assert isinstance(dialog, SectionMarkersDialog)
                await pilot.click(dialog.query_one("#yes", Button))
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == set()
            assert app.toast_text == expected_toast
            panel_text = str(app.query_one("#markers-panel", Static).render())
            assert "None" in panel_text
            assert project.markers == set(markers)

    run(exercise())

@pytest.mark.parametrize(
    "dismissal",
    [
        pytest.param("n", id="n-key"),
        pytest.param("#no", id="no-button"),
        pytest.param("escape", id="escape"),
    ],
)
def test_declined_clear_preserves_staged_markers(dismissal: str) -> None:
    app, project = make_markers_editor(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                ]
            )
        ],
        [1],
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "4")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            if dismissal == "#no":
                await pilot.click(dialog.query_one("#no", Button))
            else:
                await pilot.press(dismissal)
            await pilot.pause()

            assert not isinstance(app.screen, SectionMarkersDialog)
            assert app.staged_markers == {1}
            assert app.toast_text == ""
            assert project.markers == {1}

    run(exercise())

@pytest.mark.parametrize(
    ("key", "step_key", "markers"),
    [
        pytest.param("n", None, [1], id="n-ignored-in-menu-step"),
        pytest.param("n", "1", [1], id="n-ignored-in-manual-step"),
        pytest.param(
            "y", None, [], id="y-ignored-in-menu-step-without-markers"
        ),
        pytest.param(
            "y", "1", [], id="y-ignored-in-manual-step-without-markers"
        ),
        pytest.param(
            "y", "2", [], id="y-ignored-in-regex-step-without-markers"
        ),
        pytest.param("y", None, [1], id="y-ignored-in-menu-step-with-markers"),
    ],
)
def test_confirmation_keys_are_ignored_outside_clear_step(
    key: str, step_key: str | None, markers: list[int]
) -> None:
    app = make_loaded_editor(
        make_project_with_markers(
            [
                BookSection(
                    phrase_groups=[
                        make_phrase_group("One."),
                        make_phrase_group("Two."),
                    ]
                )
            ],
            markers,
        )
    )
    expected_step = {
        None: SectionMarkersStep.MENU,
        "1": SectionMarkersStep.MANUAL,
        "2": SectionMarkersStep.REGEX,
    }[step_key]

    async def exercise() -> None:
        async with app.run_test() as pilot:
            if step_key is None:
                await pilot.press("m")
            else:
                await pilot.press("m", step_key)
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SectionMarkersDialog)
            assert dialog.step is expected_step
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(app.screen, SectionMarkersDialog)
            assert dialog.step is expected_step
            assert app.staged_markers == set(markers)

    run(exercise())

