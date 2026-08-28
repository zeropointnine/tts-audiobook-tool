import pytest
from pathlib import Path

from rich.text import Text
from textual.widgets import Button, Input, OptionList, Static, TextArea

import tts_audiobook_tool.textual.text_editor as text_editor_module
from tts_audiobook_tool.app_types import BookSection, SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.textual.content_textual_app import (
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.manual_selection_dialog import ManualSelectionDialog
from tts_audiobook_tool.textual.text_editor import (
    TextEditor,
    TextEditorPhraseGroupItem,
    TextEditorSectionItem,
)
from tts_audiobook_tool.textual.phrase_group_split_dialog import (
    PhraseGroupSplitDialog,
)
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.segmentation_info_dialog import (
    SegmentationInfoDialog,
)

from textual_editor_stubs import make_phrase_group, make_project, run


def make_loaded_editor(project: Project) -> TextEditor:
    """Construct an editor and explicitly perform its normally deferred load."""
    app = TextEditor(project)
    app.load_content()
    return app


def test_single_section_lists_only_phrase_groups() -> None:
    project = make_project(
        [
            BookSection(
                title="The only section",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            )
        ]
    )

    app = make_loaded_editor(project)

    assert len(app.section_items) == 1
    assert all(
        isinstance(list_item, TextEditorPhraseGroupItem) for list_item in app.list_items
    )
    assert [str(app.format_line(index)) for index in range(len(app.list_items))] == [
        "00001  One.",
        "00002  Two.",
    ]
    assert app.find_match_indices("only section") == []


def test_phrase_rows_show_line_feeds_as_dim_nonbreaking_tokens() -> None:
    phrase_group = PhraseGroup(
        [
            Phrase("  One\t\n", Reason.SENTENCE),
            Phrase("\nTwo\r Three.  ", Reason.SENTENCE),
        ]
    )
    app = make_loaded_editor(make_project([BookSection(phrase_groups=[phrase_group])]))

    formatted_line = app.format_line(0)
    plain_text = str(formatted_line)

    assert plain_text == "00001  One↵\N{NO-BREAK SPACE}↵\N{NO-BREAK SPACE}Two Three."
    assert "\n" not in plain_text
    assert app.find_text_strings(0) == ["00001", "One Two Three."]


def test_phrase_rows_use_original_presentable_text_when_newline_chars_hidden(
    monkeypatch,
) -> None:
    monkeypatch.setattr(text_editor_module, "SHOW_NEWLINE_CHARS", False)
    app = make_loaded_editor(
        make_project([BookSection(phrase_groups=[make_phrase_group("One.\n\nTwo.")])])
    )

    assert str(app.format_line(0)) == "00001  One. Two."


def test_multiple_sections_add_ordered_headers_and_global_phrase_ordinals() -> None:
    project = make_project(
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

    app = make_loaded_editor(project)

    assert [str(app.format_line(index)) for index in range(len(app.list_items))] == [
        "\nSection 1/2: Opening (2 lines)\n\n",
        "00001  One.",
        "00002  Two.",
        "\nSection 2/2: Middle (1 line)\n\n",
        "00003  Three.",
    ]
    assert [type(list_item) for list_item in app.list_items] == [
        TextEditorSectionItem,
        TextEditorPhraseGroupItem,
        TextEditorPhraseGroupItem,
        TextEditorSectionItem,
        TextEditorPhraseGroupItem,
    ]


def test_empty_untitled_section_omits_title_separator_and_shows_zero_lines() -> None:
    project = make_project(
        [
            BookSection(title="Named", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="", phrase_groups=[]),
        ]
    )

    app = make_loaded_editor(project)

    assert [str(app.format_line(index)) for index in range(len(app.list_items))] == [
        "\nSection 1/2: Named (1 line)\n\n",
        "00001  One.",
        "\nSection 2/2 (0 lines)\n\n",
    ]


def test_book_with_only_empty_sections_uses_empty_state_instead_of_headers() -> None:
    project = make_project(
        [
            BookSection(title="One", phrase_groups=[]),
            BookSection(title="Two", phrase_groups=[]),
        ]
    )
    app = make_loaded_editor(project)

    assert app.list_items == []


def test_find_searches_complete_section_headings_and_phrase_group_text() -> None:
    project = make_project(
        [
            BookSection(
                title="Prologue", phrase_groups=[make_phrase_group("Opening.")]
            ),
            BookSection(
                title="Needle Chapter", phrase_groups=[make_phrase_group("Haystack.")]
            ),
        ]
    )

    app = make_loaded_editor(project)

    assert app.find_match_indices("needle") == [2]
    assert app.find_match_indices("haystack") == [3]
    assert app.find_match_indices("section 2/2: needle chapter") == [2]
    assert app.find_match_indices("needle chapter (1 line)") == [2]


def test_staged_hierarchy_is_detached_from_project_book() -> None:
    project = make_project(
        [
            BookSection(
                title="Original title",
                phrase_groups=[make_phrase_group("Original text.")],
            )
        ]
    )

    app = make_loaded_editor(project)
    staged_section = app.section_items[0]
    staged_section.title = "Staged title"
    staged_section.phrase_group_items[0].phrase_group.phrases[0].text = "Staged text."
    staged_section.phrase_group_items.append(
        TextEditorPhraseGroupItem(make_phrase_group("Inserted."), ordinal=2)
    )

    assert project.book.sections[0].title == "Original title"
    assert project.phrase_groups[0].text == "Original text."
    assert len(project.phrase_groups) == 1


def test_delete_ignores_section_rows_rebuilds_ordinals_and_focuses_survivor() -> None:
    project = make_project(
        [
            BookSection(
                title="Opening",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            ),
            BookSection(
                title="Middle",
                phrase_groups=[make_phrase_group("Three."), make_phrase_group("Four.")],
            ),
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.selected_indices = {0, 1, 2}
            app.selected_index = 2
            app.action_delete_phrase_groups()
            await pilot.pause()

            assert [
                item.phrase_group.text for item in app.edit_session.phrase_groups
            ] == [
                "Three.",
                "Four.",
            ]
            assert [section.title for section in app.edit_session.sections] == [
                "Middle"
            ]
            assert [
                item.ordinal
                for item in app.list_items
                if isinstance(item, TextEditorPhraseGroupItem)
            ] == [1, 2]
            assert app.selected_index == 0
            assert app.selected_indices == {0}
            assert str(app.query_one("#status-line", Static).render()) == (
                "2 lines deleted"
            )
            assert project.book.sections[0].title == "Opening"
            assert [item.text for item in project.phrase_groups] == [
                "One.",
                "Two.",
                "Three.",
                "Four.",
            ]
            assert app.has_changes is True

    run(exercise())


def test_delete_reconciles_options_without_formatting_unchanged_rows(
    monkeypatch,
) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                    make_phrase_group("Four."),
                    make_phrase_group("Five."),
                ]
            )
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            option_list = app.query_one("#line-list", OptionList)
            retained_option = option_list.get_option_at_index(0)
            formatted_indices: list[int] = []
            original_format_line = app.format_line

            def record_format_line(index: int):
                formatted_indices.append(index)
                return original_format_line(index)

            monkeypatch.setattr(app, "format_line", record_format_line)
            await pilot.press("down", "down", "down", "x")
            await pilot.pause()

            assert formatted_indices == [3]
            assert option_list.get_option_at_index(0) is retained_option
            assert [str(option.prompt) for option in option_list.options] == [
                "00001  One.",
                "00002  Two.",
                "00003  Three.",
                "00004  Five.",
            ]

    run(exercise())


def test_manual_selection_uses_staged_ordinals_and_excludes_section_rows() -> None:
    project = make_project(
        [
            BookSection(
                title="Opening",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            ),
            BookSection(
                title="Middle",
                phrase_groups=[make_phrase_group("Three."), make_phrase_group("Four.")],
            ),
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "down", "x")
            assert [
                item.ordinal
                for item in app.list_items
                if isinstance(item, TextEditorPhraseGroupItem)
            ] == [1, 2, 3]

            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            app.screen.query_one("#manual-selection-input", Input).value = "2-3"
            await pilot.press("enter")

            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == {3, 4}
            assert app.selected_index == 4
            assert app.selection_anchor_index == 4
            assert app.query_one("#line-list", OptionList).highlighted == 4
            assert all(
                isinstance(app.list_items[index], TextEditorPhraseGroupItem)
                for index in app.selected_indices
            )

    run(exercise())


@pytest.mark.parametrize(
    ("sections", "expected_sections", "expected_phrase_texts", "expected_project_texts"),
    [
        pytest.param(
            [("Opening", ["One.", "Two."]), ("Middle", ["Three."])],
            ["Middle"],
            ["Three."],
            ["One.", "Two.", "Three."],
            id="section-loses-its-phrase-groups",
        ),
        pytest.param(
            [("Only text", ["One.", "Two."]), ("Empty", [])],
            ["Empty"],
            [],
            ["One.", "Two."],
            id="section-can-lose-every-phrase-group",
        ),
    ],
)
def test_delete_single_selected_section_deletes_its_phrase_groups(
    sections: list[tuple[str, list[str]]],
    expected_sections: list[str],
    expected_phrase_texts: list[str],
    expected_project_texts: list[str],
) -> None:
    project = make_project(
        [
            BookSection(
                title=title,
                phrase_groups=[make_phrase_group(text) for text in texts],
            )
            for title, texts in sections
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x")

            assert [section.title for section in app.edit_session.sections] == (
                expected_sections
            )
            assert [
                item.phrase_group.text for item in app.edit_session.phrase_groups
            ] == expected_phrase_texts
            assert str(app.query_one("#status-line", Static).render()) == (
                "2 lines deleted"
            )

    run(exercise())
    assert [item.text for item in project.phrase_groups] == expected_project_texts


def test_split_dialog_partitions_one_group_and_rebuilds_rows() -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    PhraseGroup(
                        [
                            Phrase("First. ", Reason.SENTENCE),
                            Phrase("Second.", Reason.SENTENCE),
                        ],
                        voice_index=2,
                    )
                ]
            )
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("s")
            assert isinstance(app.screen, PhraseGroupSplitDialog)
            await pilot.press("1", "enter")
            await pilot.pause()

            assert [
                item.phrase_group.text for item in app.edit_session.phrase_groups
            ] == [
                "First. ",
                "Second.",
            ]
            assert [
                item.phrase_group.voice_index for item in app.edit_session.phrase_groups
            ] == [
                2,
                2,
            ]
            assert app.selected_index == 1
            assert [item.text for item in project.phrase_groups] == ["First. Second."]

    run(exercise())


def test_split_reconciles_options_without_formatting_unchanged_rows(
    monkeypatch,
) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    PhraseGroup(
                        [
                            Phrase("First. ", Reason.SENTENCE),
                            Phrase("Second.", Reason.SENTENCE),
                        ]
                    ),
                ]
            )
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            option_list = app.query_one("#line-list", OptionList)
            retained_option = option_list.get_option_at_index(0)
            formatted_indices: list[int] = []
            original_format_line = app.format_line

            def record_format_line(index: int):
                formatted_indices.append(index)
                return original_format_line(index)

            monkeypatch.setattr(app, "format_line", record_format_line)
            await pilot.press("down", "s", "1", "enter")
            await pilot.pause()

            assert formatted_indices == [1, 2]
            assert option_list.get_option_at_index(0) is retained_option
            assert [str(option.prompt) for option in option_list.options] == [
                "00001  One.",
                "00002  First.",
                "00003  Second.",
            ]

    run(exercise())


def test_split_is_ignored_for_a_single_phrase_group() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("s")
            assert not isinstance(app.screen, PhraseGroupSplitDialog)
            assert app.has_changes is False

    run(exercise())


@pytest.mark.parametrize(
    ("sections", "markers", "delete_original_index", "snapshot_paths",
     "expected_requested_indices", "expected_copy"),
    [
        pytest.param(
            [("", ["One.", "Two.", "Three."])],
            set(),
            1,
            ["segment-1.flac", "segment-2.flac"],
            [1],
            "Saving these changes requires deleting 2 generated sound segments "
            "from line 2 onward.",
            id="plural-segments-from-first-affected-line",
        ),
        pytest.param(
            [("", ["One.", "Two."])],
            set(),
            0,
            ["segment.flac"],
            [0],
            "Saving these changes requires deleting 1 generated sound segment "
            "from line 1 onward.",
            id="singular-segment",
        ),
        pytest.param(
            [("", ["One.", "Two."])],
            set(),
            0,
            [],
            [0],
            None,
            id="no-segments-plain-dialog",
        ),
        pytest.param(
            [("", ["One.", "Two.", "Three."])],
            {1, 2},
            1,
            [],
            [1],
            "Saving these changes requires deleting 2 section markers "
            "from line 2 onward.",
            id="markers-only",
        ),
        pytest.param(
            [("", ["One.", "Two.", "Three."])],
            {2},
            1,
            ["segment-1.flac", "segment-2.flac"],
            [1],
            "Saving these changes requires deleting 2 generated sound segments "
            "and 1 section marker from line 2 onward.",
            id="segments-and-singular-marker",
        ),
        pytest.param(
            [("Opening", ["One.", "Two."]), ("Middle", ["Three.", "Four."])],
            {2, 3},
            2,
            ["segment.flac"],
            [2],
            "Saving these changes requires deleting 1 generated sound segment "
            "and 2 split points from line 3 onward.",
            id="multi-section-split-points",
        ),
    ],
)
def test_confirmation_warns_about_generated_segments_from_first_affected_line(
    monkeypatch,
    sections: list[tuple[str, list[str]]],
    markers: set[int],
    delete_original_index: int,
    snapshot_paths: list[str],
    expected_requested_indices: list[int],
    expected_copy: str | None,
) -> None:
    project = make_project(
        [
            BookSection(
                title=title,
                phrase_groups=[make_phrase_group(text) for text in texts],
            )
            for title, texts in sections
        ]
    )
    project.markers = markers
    app = make_loaded_editor(project)
    requested_indices: list[int] = []

    def snapshot_paths_from_index(first_index: int) -> list[Path]:
        requested_indices.append(first_index)
        return [Path(name) for name in snapshot_paths]

    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        snapshot_paths_from_index,
    )
    app.edit_session.delete_phrase_groups(
        {app.edit_session.phrase_groups[delete_original_index].item_id}
    )

    dialog = app.make_confirmation_dialog()

    assert requested_indices == expected_requested_indices
    assert isinstance(dialog, SaveChangesDialog)
    if expected_copy is None:
        assert dialog.copy_lines == ["Save changes before exiting?"]
    else:
        assert dialog.copy_lines[2].endswith(expected_copy)


def test_finish_confirm_calls_commit_with_staged_book(monkeypatch) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")]
            )
        ]
    )
    app = make_loaded_editor(project)
    commits: list[tuple[list[str], int | None]] = []

    def commit(**kwargs) -> str:
        commits.append(
            (
                [item.text for item in kwargs["staged_book"].phrase_groups],
                kwargs["earliest_affected_original_index"],
            )
        )
        return ""

    monkeypatch.setattr(
        "tts_audiobook_tool.textual.text_editor.ProjectTextEditUtil.commit",
        commit,
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x", "escape")
            await pilot.click(app.screen.query_one("#yes", Button))
            await pilot.pause()
            assert app.is_running is False

        assert commits == [(["Two."], 0)]
        assert app.return_value == EditorSaved()

    run(exercise())


def test_structural_actions_are_ignored_while_find_is_active() -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    PhraseGroup(
                        [
                            Phrase("One. ", Reason.SENTENCE),
                            Phrase("Two.", Reason.SENTENCE),
                        ]
                    ),
                    make_phrase_group("Three."),
                ]
            )
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+f")
            app.action_delete_phrase_groups()
            app.action_split_phrase_group()
            assert [
                item.phrase_group.text for item in app.edit_session.phrase_groups
            ] == [
                "One. Two.",
                "Three.",
            ]
            assert app.has_changes is False

    run(exercise())


def test_confirm_surfaces_commit_error_without_mutating_project(monkeypatch) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")]
            )
        ]
    )
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        "tts_audiobook_tool.textual.text_editor.ProjectTextEditUtil.commit",
        lambda **_kwargs: "disk full",
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x", "escape")
            await pilot.click(app.screen.query_one("#yes", Button))
            await pilot.pause()
            assert app.is_running is False

        assert [item.text for item in project.phrase_groups] == ["One.", "Two."]
        assert app.return_value == EditorSaveFailed("Save failed: disk full")

    run(exercise())


def test_header_documents_the_info_key() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    app = make_loaded_editor(project)

    assert Text.from_ansi(app.header_lines[3]).plain == (
        "- Press [X] Delete   [S] Split   [E] Edit   [I] Info"
    )


def test_edit_mode_keeps_the_default_header() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    app = make_loaded_editor(project)
    expected_header = list(app.header_lines)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            assert app.is_editing is True
            assert app.header_lines == expected_header

    run(exercise())


def test_edit_panel_shows_header_help_and_four_row_input() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("e")
            await pilot.pause()

            title = app.query_one("#edit-panel-title", Static).content
            word_count = app.query_one("#edit-panel-word-count", Static).content
            help_text = app.query_one("#edit-panel-help", Static).content
            assert title.plain == "Spot edit:"
            assert word_count.plain == "Words: 1/80"
            assert help_text.plain == "Press [ENTER] to confirm  [ESC] to cancel"
            panel = app.query_one("#edit-panel")
            sound_warning = app.query_one("#edit-panel-sound-warning", Static)
            assert sound_warning.display is False
            input_divider = app.query_one("#edit-panel-input-divider")
            text_area = app.query_one("#edit-area", TextArea)
            assert panel.region.height == 9
            assert panel.styles.border_top[0] == "round"
            assert panel.styles.border_bottom[0] == "round"
            assert input_divider.region.y == text_area.region.y - 1
            assert text_area.region.height == 4
            assert text_area.styles.border_top[0] == ""
            assert text_area.styles.border_bottom[0] == ""

    run(exercise())


@pytest.mark.parametrize(
    ("already_edited", "expected_message"),
    [
        (
            False,
            "This line has a sound segment which will be deleted if edited",
        ),
        (
            True,
            "This line has a sound segment which will be deleted because it has "
            "been edited",
        ),
    ],
)
def test_edit_panel_warns_about_existing_sound_segment(
    monkeypatch, already_edited: bool, expected_message: str
) -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "get_filenames_for",
        lambda index: ["00001.flac"] if index == 0 else [],
    )
    if already_edited:
        app.edit_session.update_phrase_group_text(
            app.edit_session.phrase_groups[0].item_id,
            "One changed.",
            max_words=80,
            pysbd_lang="en",
        )

    async def exercise() -> None:
        async with app.run_test(size=(50, 30)) as pilot:
            await pilot.press("e")
            await pilot.pause()

            warning = app.query_one("#edit-panel-sound-warning", Static)
            assert warning.display is True
            assert warning.content.plain == expected_message
            assert any(span.style.italic for span in warning.content.spans)
            assert warning.styles.text_wrap == "nowrap"
            assert warning.styles.text_overflow == "ellipsis"
            assert app.query_one("#edit-panel").region.height == 10

    run(exercise())


def _word_count_span_colors(content: Text) -> dict[str, str]:
    """Map each styled plain-text fragment to its foreground color name."""
    return {
        content.plain[span.start : span.end]: span.style.color.name
        for span in content.spans
        if span.style.color is not None and span.style.color.name
    }


def test_word_count_updates_live_using_canonical_counting() -> None:
    """The counter initializes and updates live with canonical word counting."""
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            counter = app.query_one("#edit-panel-word-count", Static)
            assert counter.content.plain == "Words: 1/80"

            # Multiple whitespace chars collapse into one canonical word count.
            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = "Hello,   world!"
            await pilot.pause()
            assert counter.content.plain == "Words: 2/80"

    run(exercise())


def test_word_count_marks_zero_as_error() -> None:
    """A zero-word edit renders the count in the error color."""
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 3
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = ""
            await pilot.pause()

            counter = app.query_one("#edit-panel-word-count", Static)
            assert counter.content.plain == "Words: 0/3"
            colors = _word_count_span_colors(counter.content)
            assert colors == {"Words: ": "#888888", "0": "#ff0000", "/3": "#888888"}

    run(exercise())


def test_word_count_shows_zero_for_non_vocalizable_text() -> None:
    """Non-vocalizable text reports 0 words and blocks Enter until fixed."""
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = "!!!"
            await pilot.pause()

            counter = app.query_one("#edit-panel-word-count", Static)
            assert counter.content.plain == "Words: 0/80"
            colors = _word_count_span_colors(counter.content)
            assert colors == {"Words: ": "#888888", "0": "#ff0000", "/80": "#888888"}

            await pilot.press("enter")
            await pilot.pause()

            assert app.is_editing is True
            assert app.query_one("#edit-panel").display is True
            assert app.has_changes is False
            assert "One." in str(app.format_line(0))

    run(exercise())


def test_word_count_marks_over_limit_and_ignores_enter() -> None:
    """Over-limit text renders the count in the error color and Enter is ignored."""
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 3
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = "one two three four"
            await pilot.pause()

            counter = app.query_one("#edit-panel-word-count", Static)
            assert counter.content.plain == "Words: 4/3"
            colors = _word_count_span_colors(counter.content)
            assert colors == {"Words: ": "#888888", "4": "#ff0000", "/3": "#888888"}

            await pilot.press("enter")
            await pilot.pause()

            assert app.is_editing is True
            assert app.query_one("#edit-panel").display is True
            assert app.has_changes is False
            assert "One." in str(app.format_line(0))

    run(exercise())


def test_word_count_accepts_exact_limit() -> None:
    """Exactly applied_max_words is valid and commits the edit."""
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 3
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = "one two three"
            await pilot.pause()

            counter = app.query_one("#edit-panel-word-count", Static)
            assert counter.content.plain == "Words: 3/3"

            await pilot.press("enter")
            await pilot.pause()

            assert app.is_editing is False
            assert app.query_one("#edit-panel").display is False
            assert app.has_changes is True

    run(exercise())


def test_i_opens_segmentation_info_dialog() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 80
    project.applied_strategy = SegmentationStrategy.MULTI_SENTENCE
    project.applied_dialog_segmentation = True
    project.applied_language_code = "en"
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("i")
            assert isinstance(app.screen, SegmentationInfoDialog)
            content = app.screen.query_one("#segmentation-info-copy", Static).content
            assert content.plain == (
                "The text was originally imported using the following "
                "segmentation settings:\n"
                "\n"
                "Max words per segment: 80\n"
                "Segmentation strategy: Multiple sentences\n"
                "Dialog segmentation: True\n"
                "Language code: en"
            )
            dim_prefixes = {
                content.plain[span.start:span.end]
                for span in content.spans
                if span.style.color is not None
                and span.style.color.name == "#888888"
            }
            assert dim_prefixes == {
                "Max words per segment:",
                "Segmentation strategy:",
                "Dialog segmentation:",
                "Language code:",
            }

    run(exercise())


def test_info_dialog_shows_none_for_missing_language_code() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 42
    project.applied_strategy = SegmentationStrategy.SENTENCE_PLUS
    project.applied_dialog_segmentation = False
    project.applied_language_code = ""
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("i")
            content = app.screen.query_one("#segmentation-info-copy", Static).content
            assert "Max words per segment: 42" in content.plain
            assert "Segmentation strategy: Sentence+" in content.plain
            assert "Dialog segmentation: False" in content.plain
            assert content.plain.endswith("Language code: (none)")

    run(exercise())


def test_info_dialog_dismisses_on_any_key() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_strategy = SegmentationStrategy.SENTENCE_PLUS
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("i")
            assert isinstance(app.screen, SegmentationInfoDialog)

            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, SegmentationInfoDialog)

            await pilot.press("i")
            assert isinstance(app.screen, SegmentationInfoDialog)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, SegmentationInfoDialog)
            assert app.is_running is True

    run(exercise())


# =============================================================================
# Text Editing Tests (Phase 2)
# =============================================================================


async def edit_first_line_to(pilot, app: "TextEditor", new_text: str) -> None:
    """Open the editor for the currently selected line, replace its content, and confirm.

    Sets the TextArea's text directly instead of simulating keystroke-by-keystroke
    typing: this exercises the edit-confirm flow rather than Textual's own input
    handling, and avoids accidentally typing key *names* (e.g. "enter") as literal
    characters instead of pressing them.

    Assumes the caller has already positioned selected_index where the edit
    should happen (the OptionList selects index 0 by default on load, so most
    callers don't need to press anything first).
    """
    await pilot.press("e")
    text_area = app.query_one("#edit-area", TextArea)
    text_area.text = new_text
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def test_edit_simple_line() -> None:
    """Test 1: Simple edit (without \\n)."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Hello world."),
                    make_phrase_group("Goodbye world."),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            # Content is loaded via call_after_refresh() after on_mount
            await pilot.pause()
            assert app.list_items is not None
            assert app.list_items != []
            # OptionList initializes with selected_index = 0
            assert app.selected_index == 0

            await edit_first_line_to(pilot, app, "Hello world! This is a test.")

            # Verify that the persistent editor pane was hidden
            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is False
            assert app.is_editing is False

            # Verify that has_changes is True
            assert app.has_changes is True

            # Verify that the text appears correctly in the interface
            assert "This is a test" in str(app.format_line(0))

    run(exercise())


def test_edit_multiline_text() -> None:
    """Test 2: Edit containing \\n (multiple lines)."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Line 1.\nLine 2.\nLine 3."),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(
                pilot, app, "Line 1.\nLine 2.\nLine 3.\nLine 4.\nLine 5."
            )

            # Verify that the persistent editor pane was hidden
            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is False
            assert app.is_editing is False

            # Verify that has_changes is True
            assert app.has_changes is True

            # Verify that the text appears correctly with \n
            formatted = str(app.format_line(0))
            assert "Line 4." in formatted
            assert "Line 5." in formatted

    run(exercise())


def test_edit_cancel_with_escape() -> None:
    """Test 3: ESC discards the in-progress edit without changing state."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Hello world."),
                ]
            )
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            # Select the first line
            await pilot.press("down")
            assert app.selected_index == 0

            # Start editing
            await pilot.press("e")
            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is True
            assert app.is_editing is True

            # Change the content, but cancel instead of confirming
            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = "This edit should never be saved."
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            # Verify that the persistent editor pane was hidden
            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is False
            assert app.is_editing is False

            # Verify that nothing was staged, and original text is untouched
            assert app.has_changes is False
            assert "Hello world." in str(app.format_line(0))

    run(exercise())


def test_edit_rejects_empty_text() -> None:
    """Test 4: Empty text is rejected and the editor stays open.

    Zero words are invalid under the applied_max_words requirement, so Enter
    must be ignored: the edit remains active, nothing is staged, and the
    original text is untouched.
    """
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Hello world."),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = ""
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is True
            assert app.is_editing is True
            assert app.has_changes is False
            assert "Hello world." in str(app.format_line(0))

    run(exercise())


def test_edit_blocked_on_section_item() -> None:
    """Test 5: Attempt to edit TextEditorSectionItem (should be blocked)."""
    project = make_project(
        [
            BookSection(
                title="Section 1",
                phrase_groups=[make_phrase_group("Line 1.")],
            ),
            BookSection(
                title="Section 2",
                phrase_groups=[make_phrase_group("Line 2.")],
            ),
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            # Select the section (not a line)
            await pilot.press("down")  # Go to the first line
            await pilot.press("up")    # Go back to the section
            assert app.selected_index == 0
            assert isinstance(app.list_items[0], TextEditorSectionItem)

            # Try to edit - should be blocked
            await pilot.press("e")
            await pilot.pause()

            # The persistent panel should remain hidden
            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is False
            assert app.is_editing is False

    run(exercise())


def test_edit_creates_has_changes() -> None:
    """Test 6: has_changes after confirming an edit."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Original text."),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "Modified")

            assert app.has_changes is True

    run(exercise())


def test_edit_preserves_voice_index() -> None:
    """Test 7: Preservation of voice_index."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    PhraseGroup(
                        phrases=[Phrase("Original.", Reason.SENTENCE)],
                        voice_index=42,
                    ),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "Modified")

            item = app.list_items[0]
            assert isinstance(item, TextEditorPhraseGroupItem)
            assert item.phrase_group.voice_index == 42

    run(exercise())


def test_edit_single_line_without_terminal_punct_uses_canonical_reason() -> None:
    """Test 8: Spot edits retain the canonical segmenter's end Reason."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Original text."),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "Modified")

            item = app.list_items[0]
            assert isinstance(item, TextEditorPhraseGroupItem)
            for phrase in item.phrase_group.phrases:
                assert phrase.reason == Reason.SENTENCE, (
                    f"Expected Reason.SENTENCE, got {phrase.reason}"
                )

    run(exercise())


def test_edit_confirm_without_changes_is_noop() -> None:
    """Test 9: Confirming without touching the text registers no change."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Hello world."),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down")
            await pilot.press("e")
            assert app.is_editing is True

            # Confirm without editing the TextArea content at all
            await pilot.press("enter")
            await pilot.pause()

            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is False
            assert app.is_editing is False
            assert app.has_changes is False
            assert "Hello world." in str(app.format_line(0))

    run(exercise())


def test_edit_confirm_mixed_line_endings_is_noop() -> None:
    """Test 10: Mixed \\r\\n/\\n in stored text still confirms as no-op.

    Regression test for the TextArea \\n -> \\r\\n coalescing bug: a phrase
    group whose raw text already mixes \\r\\n and \\n (e.g. from a
    mixed-origin import) must not be falsely detected as "changed" just
    because the TextArea widget normalizes line terminators on load.
    """
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Line 1.\r\nLine 2.\nLine 3."),
                ]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down")
            await pilot.press("e")
            assert app.is_editing is True

            # Confirm without editing the TextArea content at all
            await pilot.press("enter")
            await pilot.pause()

            assert app.editor_widget is not None
            assert app.query_one("#edit-panel").display is False
            assert app.is_editing is False
            assert app.has_changes is False

    run(exercise())


def test_edit_uses_canonical_segmentation() -> None:
    """Confirming an edit re-creates phrases with the canonical segmenter."""
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    project.applied_max_words = 80
    project.applied_language_code = "en"
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "Hello, world.")

            item = app.list_items[0]
            assert isinstance(item, TextEditorPhraseGroupItem)
            assert [phrase.text for phrase in item.phrase_group.phrases] == [
                "Hello, ",
                "world.",
            ]
            assert [phrase.reason for phrase in item.phrase_group.phrases] == [
                Reason.PHRASE,
                Reason.SENTENCE,
            ]

    run(exercise())


def test_edit_preserves_original_trailing_break() -> None:
    """An edit re-applies the original group's trailing whitespace."""
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One.\n\n")])]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "Two.")

            item = app.list_items[0]
            assert isinstance(item, TextEditorPhraseGroupItem)
            assert item.phrase_group.text == "Two.\n\n"

    run(exercise())


def test_edit_confirmation_warns_only_about_edited_segment(monkeypatch) -> None:
    """Edit-only confirmation deletes only the edited segment, not markers."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")]
            )
        ]
    )
    app = make_loaded_editor(project)
    requested: list[set[int]] = []

    def snapshot_paths_at_indices(indices) -> list[Path]:
        requested.append(set(indices))
        return [Path("segment.flac")]

    monkeypatch.setattr(
        project.sound_segments, "snapshot_paths_at_indices", snapshot_paths_at_indices
    )
    app.edit_session.update_phrase_group_text(
        app.edit_session.phrase_groups[0].item_id,
        "One changed.",
        max_words=80,
        pysbd_lang="en",
    )

    dialog = app.make_confirmation_dialog()

    assert requested == [{0}]
    assert isinstance(dialog, SaveChangesDialog)
    assert dialog.copy_lines[2].endswith(
        "Saving these changes requires deleting 1 generated sound segment."
    )


def test_finish_confirm_commit_edit_uses_narrow_invalidation(monkeypatch) -> None:
    """Committing an edit-only session narrows invalidation to the edited segment."""
    project = make_project(
        [
            BookSection(
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")]
            )
        ]
    )
    project.applied_max_words = 80
    app = make_loaded_editor(project)
    commits: list[tuple[object, object]] = []

    def commit(**kwargs) -> str:
        commits.append(
            (
                kwargs.get("edited_segment_indices"),
                kwargs.get("earliest_affected_original_index"),
            )
        )
        return ""

    monkeypatch.setattr(
        "tts_audiobook_tool.textual.text_editor.ProjectTextEditUtil.commit", commit
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "One changed.")
            await pilot.press("escape")
            await pilot.click(app.screen.query_one("#yes", Button))
            await pilot.pause()
            assert app.is_running is False

        assert commits == [({0}, None)]
        assert app.return_value == EditorSaved()

    run(exercise())
