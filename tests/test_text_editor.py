import asyncio
from pathlib import Path

from textual.widgets import Button, OptionList, Static

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project
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
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One.")])]
    )
    app = TextEditor(project)
    original_initialize = app.initialize_content
    state_seen_by_initializer: list[tuple[str, int]] = []

    def initialize_after_recording_loading_view() -> range:
        empty_state = app.query_one("#empty-state", Static)
        state_seen_by_initializer.append(
            (str(empty_state.render()), app.query_one("#line-list", OptionList).option_count)
        )
        return original_initialize()

    monkeypatch.setattr(app, "initialize_content", initialize_after_recording_loading_view)

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
        isinstance(list_item, TextEditorPhraseGroupItem)
        for list_item in app.list_items
    )
    assert [str(app.format_line(index)) for index in range(len(app.list_items))] == [
        "00001  One.",
        "00002  Two.",
    ]
    assert app.find_match_indices("only section") == []


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
        "\nSection 1/2: Opening (2 lines)\n",
        "00001  One.",
        "00002  Two.",
        "\nSection 2/2: Middle (1 line)\n",
        "00003  Three.",
    ]
    assert [
        type(list_item)
        for list_item in app.list_items
    ] == [
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
        "\nSection 1/2: Named (1 line)\n",
        "00001  One.",
        "\nSection 2/2 (0 lines)\n",
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


def test_find_searches_section_titles_and_phrase_group_text() -> None:
    project = make_project(
        [
            BookSection(title="Prologue", phrase_groups=[make_phrase_group("Opening.")]),
            BookSection(title="Needle Chapter", phrase_groups=[make_phrase_group("Haystack.")]),
        ]
    )

    app = make_loaded_editor(project)

    assert app.find_match_indices("needle") == [2]
    assert app.find_match_indices("haystack") == [3]
    assert app.find_match_indices("section 2") == []


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

            assert [item.phrase_group.text for item in app.edit_session.phrase_groups] == [
                "Three.",
                "Four.",
            ]
            assert [section.title for section in app.edit_session.sections] == ["Middle"]
            assert [
                item.ordinal
                for item in app.list_items
                if isinstance(item, TextEditorPhraseGroupItem)
            ] == [1, 2]
            assert app.selected_index == 0
            assert app.selected_indices == {0}
            assert str(app.query_one("#selection-status", Static).render()) == (
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
            selection_status = app.query_one("#selection-status", Static)

            app.selected_indices = {0, 1, 2, 3}
            app.update_selection_status()
            assert str(selection_status.render()) == "2 lines selected"

            app.selected_indices = {0, 3}
            app.update_selection_status()
            assert str(selection_status.render()) == ""

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
            assert [item.phrase_group.text for item in app.edit_session.phrase_groups] == [
                "Three."
            ]
            assert str(app.query_one("#selection-status", Static).render()) == (
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
            assert str(app.query_one("#selection-status", Static).render()) == (
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

            assert [item.phrase_group.text for item in app.edit_session.phrase_groups] == [
                "First. ",
                "Second.",
            ]
            assert [item.phrase_group.voice_index for item in app.edit_session.phrase_groups] == [
                2,
                2,
            ]
            assert app.selected_index == 1
            assert [item.text for item in project.phrase_groups] == ["First. Second."]

    run(exercise())


def test_split_is_ignored_for_a_single_phrase_group() -> None:
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One.")])]
    )
    app = make_loaded_editor(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("s")
            assert not isinstance(app.screen, PhraseGroupSplitDialog)
            assert app.has_changes is False

    run(exercise())


def test_finish_discard_leaves_project_unchanged() -> None:
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")])]
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
        assert app.did_save_changes is False

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
    app.edit_session.delete_phrase_groups(
        {app.edit_session.phrase_groups[1].item_id}
    )

    dialog = app.make_confirmation_dialog()

    assert requested_indices == [1]
    assert dialog.warning_text == (
        "Saving these changes requires deleting 2 generated sound segments "
        "from line 2 onward."
    )


def test_confirmation_uses_singular_segment_copy(monkeypatch) -> None:
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")])]
    )
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        lambda _first_index: [Path("segment.flac")],
    )
    app.edit_session.delete_phrase_groups(
        {app.edit_session.phrase_groups[0].item_id}
    )

    dialog = app.make_confirmation_dialog()

    assert dialog.warning_text == (
        "Saving these changes requires deleting 1 generated sound segment "
        "from line 1 onward."
    )


def test_confirmation_omits_warning_when_no_generated_segments_exist(
    monkeypatch,
) -> None:
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")])]
    )
    app = make_loaded_editor(project)
    monkeypatch.setattr(
        project.sound_segments,
        "snapshot_paths_from_index",
        lambda _first_index: [],
    )
    app.edit_session.delete_phrase_groups(
        {app.edit_session.phrase_groups[0].item_id}
    )

    dialog = app.make_confirmation_dialog()

    assert isinstance(dialog, SaveChangesDialog)
    assert dialog.warning_text == ""


def test_finish_confirm_calls_commit_with_staged_book(monkeypatch) -> None:
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")])]
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
        assert app.did_save_changes is True
        assert app.save_error == ""

    run(exercise())


def test_finish_without_changes_exits_without_confirmation() -> None:
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One.")])]
    )
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
            assert [item.phrase_group.text for item in app.edit_session.phrase_groups] == [
                "One. Two.",
                "Three.",
            ]
            assert app.has_changes is False

    run(exercise())


def test_confirm_surfaces_commit_error_without_mutating_project(monkeypatch) -> None:
    project = make_project(
        [BookSection(phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")])]
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
        assert app.did_save_changes is False
        assert app.save_error == "Save failed: disk full"

    run(exercise())
