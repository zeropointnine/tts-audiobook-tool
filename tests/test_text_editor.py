import asyncio
from pathlib import Path

from rich.text import Text
from textual.widgets import Button, Input, OptionList, Static, TextArea

import tts_audiobook_tool.textual.text_editor as text_editor_module
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import COL_DIM
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.textual.content_textual_app import (
    EditorClosed,
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


def make_phrase_group(text: str) -> PhraseGroup:
    return PhraseGroup([Phrase(text, Reason.SENTENCE)])


def make_project(sections: list[BookSection]) -> Project:
    return Project.model_validate({"book": Book(sections=sections)})


def run(coroutine) -> None:
    asyncio.run(coroutine)


def make_loaded_editor(project: Project) -> TextEditor:
    """Construct an editor and explicitly perform its normally deferred load."""
    app = TextEditor(project)
    app.load_content()
    return app


def test_edit_model_loads_only_after_initial_loading_view_draws(monkeypatch) -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    app = TextEditor(project)
    original_initialize = app.initialize_content
    state_seen_by_initializer: list[tuple[str, int]] = []

    def initialize_after_recording_loading_view() -> range:
        empty_state = app.query_one("#empty-state", Static)
        state_seen_by_initializer.append(
            (
                str(empty_state.render()),
                app.query_one("#line-list", OptionList).option_count,
            )
        )
        return original_initialize()

    monkeypatch.setattr(
        app, "initialize_content", initialize_after_recording_loading_view
    )

    assert app.edit_session_or_none is None
    assert app.section_items == []
    assert app.list_items == []
    assert app.phrase_indices == []
    assert app.loading_state_text == "..."
    assert app.empty_state_text == "No text lines"

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert state_seen_by_initializer == [("...", 0)]
            assert app.edit_session_or_none is not None
            assert app.empty_state_text == "No text lines"
            assert app.query_one("#line-list", OptionList).option_count == 1
            assert app.query_one("#line-list", OptionList).display is True

            loaded_session = app.edit_session
            app.load_content()
            assert app.edit_session is loaded_session

    run(exercise())


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
    assert app.find_text(0) == "One Two Three."

    dim_style = Text.from_ansi(f"{COL_DIM}x").spans[0].style
    dim_positions: set[int] = set()
    for span in formatted_line.spans:
        if span.style == dim_style:
            dim_positions.update(range(span.start, span.end))
    newline_token = "↵\N{NO-BREAK SPACE}"
    first_token_start = plain_text.index(newline_token)
    second_token_start = plain_text.index(newline_token, first_token_start + 2)
    assert dim_positions == (
        set(range(len("00001  ")))
        | set(range(first_token_start, first_token_start + 2))
        | set(range(second_token_start, second_token_start + 2))
    )


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
    assert app.format_line(0).spans == []
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

    async def exercise() -> None:
        async with app.run_test():
            option_list = app.query_one("#line-list", OptionList)
            empty_state = app.query_one("#empty-state", Static)
            assert app.list_items == []
            assert option_list.display is False
            assert empty_state.display is True
            assert str(empty_state.render()) == "No text lines"

    run(exercise())


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


def test_selection_status_excludes_selected_section_rows() -> None:
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

    async def exercise() -> None:
        async with app.run_test():
            status_line = app.query_one("#status-line", Static)

            app.selected_indices = {0, 1, 2, 3}
            app.update_selection_status()
            assert str(status_line.render()) == "2 lines selected"

            app.selected_indices = {0, 3}
            app.update_selection_status()
            assert str(status_line.render()) == ""

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


def test_delete_single_selected_section_deletes_its_phrase_groups() -> None:
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

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x")

            assert [section.title for section in app.edit_session.sections] == [
                "Middle"
            ]
            assert [
                item.phrase_group.text for item in app.edit_session.phrase_groups
            ] == ["Three."]
            assert str(app.query_one("#status-line", Static).render()) == (
                "2 lines deleted"
            )
            assert [item.text for item in project.phrase_groups] == [
                "One.",
                "Two.",
                "Three.",
            ]

    run(exercise())


def test_delete_single_section_can_delete_all_phrase_groups() -> None:
    project = make_project(
        [
            BookSection(
                title="Only text",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            ),
            BookSection(title="Empty", phrase_groups=[]),
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x")

            assert [section.title for section in app.edit_session.sections] == [
                "Empty",
            ]
            assert app.edit_session.phrase_groups == []
            assert str(app.query_one("#status-line", Static).render()) == (
                "2 lines deleted"
            )

    run(exercise())


def test_delete_final_phrase_group_shows_non_selectable_empty_state() -> None:
    project = make_project(
        [BookSection(title="Only", phrase_groups=[make_phrase_group("One.")])]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x")

            option_list = app.query_one("#line-list", OptionList)
            empty_state = app.query_one("#empty-state", Static)
            assert option_list.display is False
            assert option_list.option_count == 0
            assert empty_state.display is True
            assert str(empty_state.render()) == "No text lines"
            assert app.selected_index is None
            assert app.selected_indices == set()

    run(exercise())


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


def test_finish_discard_leaves_project_unchanged() -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")]
            )
        ]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x", "escape")
            assert [item.text for item in project.phrase_groups] == ["One.", "Two."]
            await pilot.click(app.screen.query_one("#no", Button))
            await pilot.pause()
            assert app.is_running is False

        assert [item.text for item in project.phrase_groups] == ["One.", "Two."]
        assert app.return_value == EditorClosed()

    run(exercise())


def test_confirmation_warns_about_generated_segments_from_first_affected_line(
    monkeypatch,
) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                ]
            )
        ]
    )
    app = make_loaded_editor(project)
    requested_indices: list[int] = []

    def snapshot_paths(first_index: int) -> list[Path]:
        requested_indices.append(first_index)
        return [Path("segment-1.flac"), Path("segment-2.flac")]

    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        snapshot_paths,
    )
    app.edit_session.delete_phrase_groups({app.edit_session.phrase_groups[1].item_id})

    dialog = app.make_confirmation_dialog()

    assert requested_indices == [1]
    assert dialog.copy_lines[2].endswith(
        "Saving these changes requires deleting 2 generated sound segments "
        "from line 2 onward."
    )


def test_confirmation_uses_singular_segment_copy(monkeypatch) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")]
            )
        ]
    )
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        lambda _first_index: [Path("segment.flac")],
    )
    app.edit_session.delete_phrase_groups({app.edit_session.phrase_groups[0].item_id})

    dialog = app.make_confirmation_dialog()

    assert dialog.copy_lines[2].endswith(
        "Saving these changes requires deleting 1 generated sound segment "
        "from line 1 onward."
    )


def test_confirmation_omits_warning_when_no_generated_segments_exist(
    monkeypatch,
) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")]
            )
        ]
    )
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        lambda _first_index: [],
    )
    app.edit_session.delete_phrase_groups({app.edit_session.phrase_groups[0].item_id})

    dialog = app.make_confirmation_dialog()

    assert isinstance(dialog, SaveChangesDialog)
    assert dialog.copy_lines == ["Save changes before exiting?"]


def test_confirmation_warns_about_markers_only_when_no_segments_exist(
    monkeypatch,
) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                ]
            )
        ]
    )
    project.markers = {1, 2}
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        lambda _first_index: [],
    )
    app.edit_session.delete_phrase_groups({app.edit_session.phrase_groups[1].item_id})

    dialog = app.make_confirmation_dialog()

    assert dialog.copy_lines[2].endswith(
        "Saving these changes requires deleting 2 section markers "
        "from line 2 onward."
    )


def test_confirmation_warns_about_segments_and_singular_marker(monkeypatch) -> None:
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("One."),
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                ]
            )
        ]
    )
    project.markers = {2}
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        lambda _first_index: [Path("segment-1.flac"), Path("segment-2.flac")],
    )
    app.edit_session.delete_phrase_groups({app.edit_session.phrase_groups[1].item_id})

    dialog = app.make_confirmation_dialog()

    assert dialog.copy_lines[2].endswith(
        "Saving these changes requires deleting 2 generated sound segments "
        "and 1 section marker from line 2 onward."
    )


def test_confirmation_uses_split_point_label_for_multi_section_books(
    monkeypatch,
) -> None:
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
    project.markers = {2, 3}
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        lambda _first_index: [Path("segment.flac")],
    )
    app.edit_session.delete_phrase_groups({app.edit_session.phrase_groups[2].item_id})

    dialog = app.make_confirmation_dialog()

    assert dialog.copy_lines[2].endswith(
        "Saving these changes requires deleting 1 generated sound segment "
        "and 2 split points from line 3 onward."
    )


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


def test_finish_without_changes_exits_without_confirmation() -> None:
    project = make_project([BookSection(phrase_groups=[make_phrase_group("One.")])])
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running is False
        assert app.has_changes is False

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
    await pilot.press("ctrl+enter")
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

            # Verify that the widget was removed
            assert app.editor_widget is None
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
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(
                pilot, app, "Line 1.\nLine 2.\nLine 3.\nLine 4.\nLine 5."
            )

            # Verify that the widget was removed
            assert app.editor_widget is None
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
            assert app.is_editing is True

            # Change the content, but cancel instead of confirming
            text_area = app.query_one("#edit-area", TextArea)
            text_area.text = "This edit should never be saved."
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            # Verify that the widget was removed
            assert app.editor_widget is None
            assert app.is_editing is False

            # Verify that nothing was staged, and original text is untouched
            assert app.has_changes is False
            assert "Hello world." in str(app.format_line(0))

    run(exercise())


def test_edit_confirming_empty_text() -> None:
    """Test 4: Confirming an edit with empty text closes the editor cleanly.

    NOTE: this only pins down that the editor doesn't crash or get stuck when
    confirmed with an empty TextArea. Whether an empty phrase group should be
    dropped, kept, or rejected outright is a product decision the app itself
    should encode (e.g. via validation before ctrl+enter is accepted) — verify
    the exact expectation against the current app behavior and tighten this
    assertion (e.g. checking edit_session.phrase_groups) if the app defines one.
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
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "")

            # Verify that the widget was removed and editing state was exited
            # cleanly, regardless of how the app chose to handle empty content.
            assert app.editor_widget is None
            assert app.is_editing is False

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

            # The widget should not have been created
            assert app.editor_widget is None
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
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await edit_first_line_to(pilot, app, "Modified")

            item = app.list_items[0]
            assert isinstance(item, TextEditorPhraseGroupItem)
            assert item.phrase_group.voice_index == 42

    run(exercise())


def test_edit_single_line_edit_receives_Reason_SENTENCE() -> None:
    """Test 8: A single-line edit (no trailing line breaks) receives Reason.SENTENCE.

    Reason is always recomputed from the edited text's own trailing line
    breaks (see update_phrase_group_text docstring) - it is never preserved
    from before the edit. Zero trailing breaks -> Reason.SENTENCE.
    """
    project = make_project(
        [
            BookSection(
                phrase_groups=[
                    make_phrase_group("Original text."),
                ]
            )
        ]
    )
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
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down")
            await pilot.press("e")
            assert app.is_editing is True

            # Confirm without editing the TextArea content at all
            await pilot.press("ctrl+enter")
            await pilot.pause()

            assert app.editor_widget is None
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
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down")
            await pilot.press("e")
            assert app.is_editing is True

            # Confirm without editing the TextArea content at all
            await pilot.press("ctrl+enter")
            await pilot.pause()

            assert app.editor_widget is None
            assert app.is_editing is False
            assert app.has_changes is False

    run(exercise())