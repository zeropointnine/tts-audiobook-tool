import pytest
from pathlib import Path

from rich.text import Text
from textual.widgets import Button, Input, OptionList, Static

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
        "- Press [X] to delete selected lines   [S] Split line  - [I] Info"
    )


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
