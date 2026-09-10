from dataclasses import dataclass, field
from typing import cast

from rich.text import Text
from textual.widgets import Input, OptionList, Rule, Static

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.textual.content_textual_app import (
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.textual_shared import (
    STYLE_HIGHLIGHT,
    NonWrappingOptionList,
)
from tts_audiobook_tool.textual.word_substitutions_app import (
    ADD_ITEM_LABEL,
    NO_ITEMS_LABEL,
    WordSubstitutionItem,
    WordSubstitutionSentinel,
    WordSubstitutionsApp,
    fit_cell,
)
from tts_audiobook_tool.textual.word_substitutions_dialog import (
    WordSubstitutionEdit,
    WordSubstitutionsDialog,
)

from textual_editor_stubs import run


@dataclass
class StubWordSubstitutionsProject:
    word_substitutions: dict[str, str] = field(default_factory=dict)
    save_error: str = ""
    save_calls: int = 0

    def save(self) -> str:
        self.save_calls += 1
        return self.save_error


def make_app(
    values: dict[str, str] | None = None,
    save_error: str = "",
) -> tuple[WordSubstitutionsApp, StubWordSubstitutionsProject]:
    project = StubWordSubstitutionsProject(
        dict(values or {}), save_error=save_error
    )
    app = WordSubstitutionsApp(cast(Project, project))
    return app, project


def originals(app: WordSubstitutionsApp) -> list[str]:
    return [
        item.original
        for item in app.list_items
        if isinstance(item, WordSubstitutionItem)
    ]


def test_empty_project_shows_no_items_sentinel_then_add_sentinel() -> None:
    app, _ = make_app()

    assert app.list_items == [
        WordSubstitutionSentinel(NO_ITEMS_LABEL, is_add=False),
        WordSubstitutionSentinel(ADD_ITEM_LABEL, is_add=True),
    ]
    assert app.content_line_index(0) is None
    assert app.content_line_index(1) is None


def test_rows_sort_case_insensitively_with_one_indexed_ordinals() -> None:
    app, _ = make_app({"Zebra": "z", "apple": "a", "Mango": "m"})

    assert originals(app) == ["apple", "Mango", "Zebra"]
    assert app.list_items[0] == WordSubstitutionItem("apple", "a", 1)
    assert app.list_items[1] == WordSubstitutionItem("Mango", "m", 2)
    assert app.list_items[2] == WordSubstitutionItem("Zebra", "z", 3)
    assert app.list_items[-1] == WordSubstitutionSentinel(ADD_ITEM_LABEL, is_add=True)
    assert app.content_line_index(0) == 0
    assert app.content_line_index(3) is None


def test_table_header_and_row_share_column_layout_at_width_60() -> None:
    app, _ = make_app({"Ariekei": "AriaKay"})

    header = app.table_header_row().render_line(60).plain
    row = app.format_line(0).render_line(60).plain

    # ordinal (3) + ordinal gap (2) + original (27) + gap (1) + substitution (27)
    assert len(header) == 60
    assert len(row) == 60
    assert header[:3] == "   "  # no "#" label above the ordinal column
    assert header[5:32] == "Original word:".ljust(27)
    assert header[33:] == "Substitution word:".ljust(27)
    assert row[:3] == "  1"
    assert row[5:32] == "Ariekei".ljust(27)
    assert row[33:] == "AriaKay".ljust(27)


def test_long_values_are_ellipsized_with_a_margin_to_the_exact_row_width() -> None:
    app, _ = make_app({"a" * 40: "b" * 40})

    row = app.format_line(0).render_line(45).plain

    assert len(row) == 45
    assert "a" * 20 not in row  # column width is (45 - 6) // 2 == 19
    # Each cell ends with "…" plus a blank margin cell
    assert row[5:24] == ("a" * 17) + "\u2026 "
    assert row[25:45] == ("b" * 18) + "\u2026 "


def test_fit_cell_ellipsis_reserves_a_trailing_margin_cell() -> None:
    cell = fit_cell("a" * 30, 10, ellipsis=True)

    assert Text(cell).cell_len == 10
    assert cell == ("a" * 8) + "\u2026 "

    # A one-cell column has no room for both text and a margin.
    assert fit_cell("a" * 30, 1, ellipsis=True) == "\u2026"


def test_fit_cell_ellipsis_leaves_values_that_fit_alone() -> None:
    assert fit_cell("a" * 10, 10, ellipsis=True) == "a" * 10
    assert fit_cell("a" * 9, 10, ellipsis=True) == ("a" * 9) + " "
    assert "\u2026" not in fit_cell("a" * 10, 10, ellipsis=True)


def test_values_that_fill_their_column_exactly_are_not_ellipsized() -> None:
    app, _ = make_app({"a" * 19: "b" * 20})

    row = app.format_line(0).render_line(45).plain

    assert row[5:24] == "a" * 19
    assert row[25:45] == "b" * 20
    assert "\u2026" not in row


def test_add_item_sentinel_uses_highlight_color() -> None:
    app, _ = make_app()
    add_index = len(app.list_items) - 1

    row = app.format_line(add_index).render_line(40)

    assert row.plain.strip() == ADD_ITEM_LABEL
    assert row.spans
    assert str(row.spans[0].style) == STYLE_HIGHLIGHT


def test_handle_dialog_result_adds_item_and_selects_it() -> None:
    app, _ = make_app()

    app.handle_dialog_result(WordSubstitutionEdit("Ariekei", "AriaKay"))

    assert app.staged == {"Ariekei": "AriaKay"}
    assert app.list_items[0] == WordSubstitutionItem("Ariekei", "AriaKay", 1)
    assert app.selected_index == 0


def test_handle_dialog_result_rename_resorts_and_follows_key() -> None:
    app, _ = make_app({"apple": "a", "zebra": "z"})

    app.handle_dialog_result(
        WordSubstitutionEdit("aardvark", "zz", previous_original="zebra")
    )

    assert app.staged == {"apple": "a", "aardvark": "zz"}
    assert originals(app) == ["aardvark", "apple"]
    assert app.selected_index == 0


def test_handle_dialog_result_ignores_cancellation() -> None:
    app, _ = make_app({"apple": "a"})

    app.handle_dialog_result(None)

    assert app.staged == {"apple": "a"}
    assert app.has_changes is False


def test_has_changes_tracks_staged_dictionary() -> None:
    app, _ = make_app({"apple": "a"})
    assert app.has_changes is False

    app.handle_dialog_result(WordSubstitutionEdit("apple", "b"))

    assert app.has_changes is True


def test_commit_saves_and_exits_saved() -> None:
    app, project = make_app()
    app.handle_dialog_result(WordSubstitutionEdit("Ariekei", "AriaKay"))
    exits: list[object] = []
    app.exit = exits.append  # type: ignore[method-assign]

    app.commit_changes_and_exit()

    assert project.word_substitutions == {"Ariekei": "AriaKay"}
    assert project.save_calls == 1
    assert exits == [EditorSaved()]


def test_commit_rolls_back_project_on_save_failure() -> None:
    app, project = make_app({"apple": "a"}, save_error="disk full")
    app.handle_dialog_result(WordSubstitutionEdit("apple", "b"))
    exits: list[object] = []
    app.exit = exits.append  # type: ignore[method-assign]

    app.commit_changes_and_exit()

    assert project.word_substitutions == {"apple": "a"}
    assert exits == [EditorSaveFailed("Save failed: disk full")]


def test_mounted_app_shows_header_and_no_items_sentinel_does_nothing() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.query_one("#table-header", Static)
            assert app.query_one("#table-header-divider", Rule)
            assert app.query_one("#status-left", Static).render() == ""
            assert app.query_one("#status-right", Static).render() == ""
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            assert option_list.option_count == 2

            # Row 0 is the "No items currently" sentinel.
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, WordSubstitutionsDialog)

            # Row 1 is "Add item".
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert "Add item" in str(
                app.screen.query_one("#word-substitutions-title", Static).render()
            )
            assert app.screen.query_one("#original-input", Input).value == ""

    run(exercise())


def test_mounted_app_opens_edit_dialog_and_stages_committed_change() -> None:
    app, _ = make_app({"Ariekei": "AriaKay"})

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.query_one("#line-list", OptionList).option_count == 2

            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert "Edit item" in str(
                app.screen.query_one("#word-substitutions-title", Static).render()
            )
            original_input = app.screen.query_one("#original-input", Input)
            assert original_input.value == "Ariekei"

            original_input.value = "Ariekei"
            app.screen.query_one("#replacement-input", Input).value = "Aria-Kay"
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, WordSubstitutionsDialog)
            assert app.staged == {"Ariekei": "Aria-Kay"}
            assert app.has_changes is True

    run(exercise())


def test_pressing_x_deletes_current_row_keeps_position_and_shows_toast() -> None:
    app, _ = make_app({"apple": "a", "Banana": "b", "cherry": "c"})

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            assert originals(app) == ["apple", "Banana", "cherry"]
            await pilot.press("down")  # highlight "Banana"

            await pilot.press("x")
            await pilot.pause()

            assert app.staged == {"apple": "a", "cherry": "c"}
            assert app.has_changes is True
            assert originals(app) == ["apple", "cherry"]
            assert app.list_items[app.selected_index] == WordSubstitutionItem(
                "cherry", "c", 2
            )
            assert str(app.query_one("#status-left", Static).render()) == "Deleted item"

    run(exercise())


def test_pressing_x_on_sentinel_rows_deletes_nothing() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            # Row 0 is "No items currently", row 1 is "Add item".
            await pilot.press("x")
            await pilot.pause()
            assert app.staged == {}
            assert app.has_changes is False
            assert str(app.query_one("#status-left", Static).render()) == ""

            await pilot.press("down")
            await pilot.press("x")
            await pilot.pause()
            assert app.staged == {}
            assert str(app.query_one("#status-left", Static).render()) == ""

    run(exercise())


def test_pressing_x_on_only_item_restores_empty_state() -> None:
    app, _ = make_app({"apple": "a"})

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("x")
            await pilot.pause()

            assert app.staged == {}
            assert app.has_changes is True
            assert app.list_items == [
                WordSubstitutionSentinel(NO_ITEMS_LABEL, is_add=False),
                WordSubstitutionSentinel(ADD_ITEM_LABEL, is_add=True),
            ]
            assert str(app.query_one("#status-left", Static).render()) == "Deleted item"

    run(exercise())
