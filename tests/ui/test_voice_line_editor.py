from typing import cast

import pytest
from rich.text import Text
from textual.widgets import Button, Input, OptionList
from tts_audiobook_tool.app_types import BookSection, BookSegmentationSettings
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual import voice_line_editor
from tts_audiobook_tool.textual.content_textual_app import (
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.manual_selection_dialog import ManualSelectionDialog
from tts_audiobook_tool.textual.multi_select_dialog import MultiSelectDialog
from tts_audiobook_tool.textual.textual_shared import NonWrappingOptionList
from tts_audiobook_tool.textual.voice_select_dialog import VoiceSelectDialog
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
            assert app.original_voice_indices == [0]
            assert app.staged_voice_indices == [0]

    run(exercise())


def test_voice_actions_are_ignored_before_deferred_content_loads() -> None:
    project = StubProject([StubPhraseGroup("Line 1")])
    app = VoiceLineEditorTextualApp(cast(Project, project), voice_sample_count=2)

    app.action_assign_voice(1)
    app.commit_changes_and_exit()

    assert project.phrase_groups[0].voice_index == 0
    assert app.original_voice_indices == []
    assert app.staged_voice_indices == []
    assert app.has_changes is False


def test_header_limits_voice_key_range_to_available_samples() -> None:
    app, _ = make_app(voice_sample_count=2)

    assert all(isinstance(line, str) for line in app.header_lines)
    assert all(not line.endswith(Ansi.RESET) for line in app.header_lines)
    assert Text.from_ansi(app.header_lines[3]).plain == (
        "- Use keys [1] to [2] to set voice sample for selected lines"
        "  - [V] Voice list dialog"
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
        "[00001] [#1] One.",
        "[00002] [#1] Two.",
    ]
    assert app.find_match_indices("only section") == []


def test_voice_column_shows_truncated_name_and_dialog_marker() -> None:
    # Once any voice is named, the voice column holds the first 5 characters of
    # each voice's name (padded blank when unnamed), then a 💬 for dialog lines
    # or matching-width blanks. Unnamed projects omit the name column entirely.
    project = StubProject(
        [
            StubPhraseGroup("Narration.", voice_index=0),
            StubPhraseGroup('"Hi."', voice_index=1, is_dialog=True),
            StubPhraseGroup('"Yo."', voice_index=2, is_dialog=True),
        ],
        voice_references=[Voice(name="Alexander"), Voice(name="Bo"), Voice()],
    )
    app = make_editor(project, voice_sample_count=3)

    assert [str(app.format_line(index)) for index in range(3)] == [
        "[00001] [#1 Alexa]     Narration.",
        '[00002] [#2 Bo   ] 💬  "Hi."',
        '[00003] [#3      ] 💬  "Yo."',
    ]
    # Search uses the displayed label without its padding
    assert app.find_text_strings(1) == ["00002", "#2 Bo", '"Hi."']
    assert app.find_match_indices("alexa") == [0]


@pytest.mark.parametrize(
    ("dialog_segmentation", "expected_row"),
    [(False, "[00001] [#1] One."), (True, "[00001] [#1]     One.")],
)
def test_dialog_marker_column_is_reserved_only_with_dialog_detection(
    dialog_segmentation: bool, expected_row: str
) -> None:
    # Text segmented without dialog detection can have no dialog lines, so the
    # marker column is dropped; with detection it is reserved even for
    # non-dialog lines so rows stay aligned.
    project = make_project(
        [BookSection(title="", phrase_groups=[make_phrase_group("One.")])]
    )
    project.book.segmentation_settings = BookSegmentationSettings(
        dialog_segmentation=dialog_segmentation
    )
    app = VoiceLineEditorTextualApp(project, voice_sample_count=2)
    app.load_content()

    assert str(app.format_line(0)) == expected_row


def test_voice_number_pads_to_two_digits_only_beyond_nine_voices() -> None:
    # With 10+ voices, single-digit numbers are padded so labels stay aligned.
    project = StubProject(
        [StubPhraseGroup("One.", voice_index=0), StubPhraseGroup("Ten.", voice_index=9)]
    )
    app = make_editor(project, voice_sample_count=10)

    assert [str(app.format_line(index)) for index in range(2)] == [
        "[00001] [#1 ] One.",
        "[00002] [#10] Ten.",
    ]


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
        "[00001] [#1] One.",
        "[00002] [#1] Two.",
        "\nSection 2/2: Middle (1 line)\n\n",
        "[00003] [#1] Three.",
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
        "[00001] [#1] One.",
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
    assert app.find_text_strings(1) == ["00001", "#1", "One."]
    assert app.find_text_strings(2) == ["00002", "#2", "Two."]
    assert app.find_text_strings(3) == ["Section 2/2: Ending (1 line)"]
    assert app.find_text_strings(4) == ["00003", "#6 *OUT OF RANGE*", "Three."]

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

    # Rows: One. is "#1", Two. is "#2", Three. defaults to "#1".
    assert app.find_match_indices("#2") == [2]
    assert app.find_match_indices("#1") == [1, 4]
    assert app.find_match_indices("#3") == []
    # Fields are tested separately, so no match can span the label/content join.
    assert app.find_match_indices("#1o") == []
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

    # Rows: [section, One.(v5 -> "#6 *OUT OF RANGE*"), section, Two.(v0)]
    assert app.find_match_indices("out of range") == [1]
    assert app.find_match_indices("#6") == [1]
    assert app.find_match_indices("#2") == []


def test_find_metadata_tracks_reassigned_voice() -> None:
    app = make_sectioned_editor(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ],
        voice_sample_count=2,
    )

    assert app.find_match_indices("#2") == []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "2")
            # Row 1 (One.) is now staged with voice index 1.
            assert app.staged_voice_indices == [1, 0]
            assert app.find_match_indices("#2") == [1]
            assert app.find_match_indices("#1") == [3]

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
            # Manual entry is the last item of the multi-select dialog
            await pilot.press("m")
            assert isinstance(app.screen, MultiSelectDialog)
            await pilot.press("end", "enter")
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
            assert app.staged_voice_indices == [0, 0]
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
            assert app.staged_voice_indices == [0, 0]
            assert app.selected_indices == {0}

            # A selection spanning a section heading assigns the phrase row only
            await pilot.press("down", "shift+up")
            assert app.selected_index == 0
            assert app.selected_indices == {0, 1}
            assert app.highlighted_content_line_index() is None

            await pilot.press("2")
            assert app.staged_voice_indices == [1, 0]
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

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "shift+down", "2")
            assert app.staged_voice_indices == [1, 1, 1, 0, 0, 0]
            assert [group.voice_index for group in project.phrase_groups] == [
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            assert app.has_changes is True
            assert app.selected_indices == {2}
            assert app.selection_anchor_index == 2

            await pilot.press("2")
            assert app.has_changes is True

            await pilot.press("3")
            assert [group.voice_index for group in project.phrase_groups] == [
                0,
                0,
                0,
                0,
                0,
                0,
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

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2")
            assert app.has_changes is True
            assert project.phrase_groups[0].voice_index == 0

            await pilot.press("1")
            assert app.has_changes is False
            assert project.phrase_groups[0].voice_index == 0

    run(exercise())


def make_voiced_app() -> tuple[VoiceLineEditorTextualApp, StubProject]:
    project = StubProject(
        [
            StubPhraseGroup("One.", voice_index=0),
            StubPhraseGroup("Two.", voice_index=1),
            StubPhraseGroup("Three.", voice_index=0),
            StubPhraseGroup("Four."),
        ],
        voice_references=[Voice(name="Alice"), Voice()],
    )
    return make_editor(project, voice_sample_count=2), project


def option_texts(dialog: MultiSelectDialog) -> list[str]:
    option_list = dialog.query_one("#multi-select-list", OptionList)
    return [
        str(option_list.get_option_at_index(index).prompt)
        for index in range(option_list.option_count)
    ]


def test_header_advertises_multi_select_and_m_opens_it() -> None:
    # "m" opens multi-select rather than the base editor's manual-entry
    # dialog (offered from within multi-select instead); "s" does nothing.
    app, _ = make_voiced_app()
    select_line = Text.from_ansi(app.header_lines[2]).plain
    assert select_line.endswith("- [M] Multi-select")
    assert "[S]" not in select_line and "[D]" not in select_line
    assert "Enter manually" not in select_line

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("s", "d")
            assert len(app.screen_stack) == 1
            await pilot.press("m")
            assert isinstance(app.screen, MultiSelectDialog)

    run(exercise())


def test_multi_select_dialog_lists_voices_with_line_counts() -> None:
    # One item per voice sample (name only when set, singular/plural count
    # from staged assignments; a line never explicitly assigned counts as
    # voice 1), then a blank row, then manual entry; the first item starts
    # highlighted, and the cursor skips over the blank row.
    app, _ = make_voiced_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            dialog = app.screen
            assert isinstance(dialog, MultiSelectDialog)
            title = dialog.query_one("#multi-select-title")
            assert str(title.render()).startswith("Multi-select")
            assert option_texts(dialog) == [
                "[1] Voice 1 Alice (3 lines)",
                "[2] Voice 2 (1 line)",
                "",
                "Enter line selections manually",
            ]
            option_list = dialog.query_one("#multi-select-list", OptionList)
            assert option_list.highlighted == 0
            await pilot.press("down", "down")
            assert option_list.highlighted == 3
            await pilot.press("up")
            assert option_list.highlighted == 1
            await pilot.press("home")
            assert option_list.has_focus
            # The highlight must reuse the editor's (visible) cursor style;
            # reverse over default-on-default colors renders as no highlight
            highlight_class = "option-list--option-highlighted"
            dialog_style = option_list.get_component_rich_style(highlight_class)
            editor_style = app.query_one(
                "#line-list", NonWrappingOptionList
            ).get_component_rich_style(highlight_class)
            assert dialog_style == editor_style
            assert dialog_style.bgcolor is not None
            assert not dialog_style.bgcolor.is_default

    run(exercise())


@pytest.mark.parametrize(
    "keys",
    [("down", "enter"), ("down", "space"), ("2",)],
)
def test_choosing_voice_selects_all_its_lines(keys: tuple[str, ...]) -> None:
    # Enter, space or the item's number key choose a voice; its lines become
    # the multi-selection, so a number key then reassigns all of them.
    app, _ = make_voiced_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m", "down")
            await pilot.press("up", *keys)
            assert len(app.screen_stack) == 1
            assert app.selected_indices == {1}

            await pilot.press("m", "enter")
            assert app.selected_indices == {0, 2, 3}
            await pilot.press("2")
            assert app.staged_voice_indices == [1, 1, 1, 1]

    run(exercise())


def test_v_opens_voice_list_which_assigns_to_selected_lines() -> None:
    # "v" titles the dialog by the selection and passes the current voice only
    # for a single line; entering a number assigns it like the number keys do.
    app, _ = make_voiced_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "v")
            dialog = app.screen
            assert isinstance(dialog, VoiceSelectDialog)
            assert dialog.heading == "Line 2"
            assert dialog.voice_names == ["Alice", ""]
            assert dialog.current_voice_index == 1
            await pilot.press("1", "enter")
            assert len(app.screen_stack) == 1
            assert app.staged_voice_indices == [0, 0, 0, 0]

            await pilot.press("shift+down", "shift+down", "v")
            dialog = app.screen
            assert isinstance(dialog, VoiceSelectDialog)
            assert dialog.heading == "3 lines selected"
            assert dialog.current_voice_index is None
            await pilot.press("2", "enter")
            assert app.staged_voice_indices == [0, 1, 1, 1]

    run(exercise())


def test_dialog_lines_item_selects_every_dialog_line() -> None:
    # With dialog-tagged lines present, an item listing their count sits just
    # above manual entry and selects exactly those lines.
    project = StubProject(
        [
            StubPhraseGroup("Narration."),
            StubPhraseGroup('"Hi."', voice_index=1, is_dialog=True),
            StubPhraseGroup("More narration."),
            StubPhraseGroup('"Yo."', is_dialog=True),
        ]
    )
    app = make_editor(project, voice_sample_count=2)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            dialog = app.screen
            assert isinstance(dialog, MultiSelectDialog)
            assert option_texts(dialog)[-3:] == [
                "",
                "All lines tagged as dialog (2 lines)",
                "Enter line selections manually",
            ]
            await pilot.press("end", "up", "enter")
            assert len(app.screen_stack) == 1
            assert app.selected_indices == {1, 3}

    run(exercise())


def test_dialog_lines_item_is_omitted_without_dialog_detection() -> None:
    app, _ = make_voiced_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            dialog = app.screen
            assert isinstance(dialog, MultiSelectDialog)
            assert "All lines tagged as dialog" not in " ".join(option_texts(dialog))

    run(exercise())


def test_multi_select_dialog_closes_on_escape_or_outside_click() -> None:
    app, _ = make_voiced_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.press("escape")
            assert len(app.screen_stack) == 1
            assert app.selected_indices == {0}

            await pilot.press("m")
            dialog = app.screen
            # Clicking inside the dialog keeps it open
            await pilot.click("#multi-select-title")
            assert app.screen is dialog
            await pilot.click(offset=(0, 0))
            assert len(app.screen_stack) == 1
            assert app.selected_indices == {0}

    run(exercise())


def test_multi_select_list_scrolls_when_dialog_reaches_max_height() -> None:
    project = StubProject([StubPhraseGroup("One.", voice_index=0)])
    app = make_editor(project, voice_sample_count=20)

    async def exercise() -> None:
        async with app.run_test(size=(80, 16)) as pilot:
            await pilot.press("m")
            dialog = app.screen
            assert isinstance(dialog, MultiSelectDialog)
            await pilot.pause()
            box = dialog.query_one("#multi-select-dialog")
            assert box.region.height <= 16
            option_list = dialog.query_one("#multi-select-list", OptionList)
            assert option_list.scroll_y == 0
            await pilot.press("end")
            await pilot.pause()
            # 20 voices, the blank row, then manual entry
            assert option_list.highlighted == 21
            assert option_list.scroll_y > 0
            # The list is capped to the dialog's space rather than clipped by it,
            # so the scrolled-to last item is actually visible
            assert option_list.region.bottom <= box.content_region.bottom
            assert (
                option_list.scroll_y + option_list.scrollable_content_region.height
                >= option_list.option_count
            )
            await pilot.press("enter")
            assert isinstance(app.screen, ManualSelectionDialog)

    run(exercise())


def test_save_button_commits_staged_values_and_persists_once(monkeypatch) -> None:
    app, project = make_app(3, voice_sample_count=2)
    saves: list[StubProject] = []
    monkeypatch.setattr(
        voice_line_editor.ProjectVoiceUtil,
        "get_voice_values",
        lambda *_: ["voice-1", "voice-2"],
    )
    monkeypatch.setattr(
        voice_line_editor.ProjectTextIOUtil,
        "save_book",
        lambda saved_project: saves.append(saved_project) or "",
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "2", "escape")
            assert [group.voice_index for group in project.phrase_groups] == [
                0,
                0,
                0,
            ]

            yes_button = app.screen.query_one("#yes", Button)
            await pilot.click(yes_button)
            await pilot.pause()
            assert app.is_running is False

        assert [group.voice_index for group in project.phrase_groups] == [1, 1, 0]
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
    monkeypatch.setattr(
        voice_line_editor.ProjectTextIOUtil,
        "save_book",
        lambda *_: "disk full",
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("2", "escape", "y")
            await pilot.pause()

        assert [group.voice_index for group in project.phrase_groups] == [0, 0]
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

        assert project.phrase_groups[0].voice_index == 0
        assert app.return_value == EditorSaveFailed(
            "Save failed: RuntimeError: unexpected failure"
        )

    run(exercise())


