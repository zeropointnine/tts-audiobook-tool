from dataclasses import dataclass

import pytest
from rich.console import Console
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.css.errors import StylesheetError
from textual.widgets import Button, Input, OptionList, Static

from tts_audiobook_tool.constants import COL_ERROR
from tts_audiobook_tool.textual import content_textual_app
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppFailed,
    ContentAppMissingResult,
    ContentAppStylesheetFailed,
    ContentAppUnavailable,
    ContentTextualApp,
    EditorClosed,
    EditorSaved,
    run_content_textual_app,
)
from tts_audiobook_tool.textual.editor_view_anchor import (
    EditorViewAnchor,
    resolve_anchor_line,
)
from tts_audiobook_tool.textual.manual_selection_dialog import ManualSelectionDialog
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    NonWrappingOptionList,
)
from textual_editor_stubs import run


@dataclass
class StubPhraseGroup:
    presentable_text: str


@dataclass
class StubProject:
    phrase_groups: list[StubPhraseGroup]


class StubContentApp(ContentTextualApp[EditorSaved]):
    def __init__(
        self,
        project: StubProject,
        phrase_indices: list[int] | None = None,
        empty_state_text: str = "No items",
        loading_state_text: str | None = None,
    ) -> None:
        self.changed_phrase_indices: set[int] = set()
        self.refresh_batches: list[list[int]] = []
        self.committed = False
        super().__init__(
            project,  # type: ignore[arg-type]
            ["Content editor"],
            phrase_indices=[2, 0] if phrase_indices is None else phrase_indices,
            empty_state_text=empty_state_text,
            loading_state_text=loading_state_text,
        )

    @property
    def has_changes(self) -> bool:
        return bool(self.changed_phrase_indices)

    def format_line(self, index: int) -> Text:
        phrase_index = self.phrase_indices[index]
        style = "#888888 reverse" if index in self.selected_indices else ""
        return Text(
            self.project.phrase_groups[phrase_index].presentable_text,
            style=style,
        )

    def action_mutate(self) -> None:
        def mutate(_visible_index: int, phrase_index: int) -> bool:
            if phrase_index in self.changed_phrase_indices:
                return False
            self.changed_phrase_indices.add(phrase_index)
            return True

        self.mutate_selected_items(mutate)

    def refresh_lines(self, indices, *, reflow: bool = True) -> None:
        index_list = list(indices)
        self.refresh_batches.append(index_list)
        super().refresh_lines(index_list, reflow=reflow)

    def make_confirmation_dialog(self) -> SaveChangesDialog:
        return SaveChangesDialog(["Apply content changes?"])

    def commit_changes_and_exit(self) -> None:
        self.committed = True
        self.exit(EditorSaved())


class DeferredStubContentApp(StubContentApp):
    def __init__(
        self,
        project: StubProject,
        final_indices: list[int],
        initial_phrase_index: int | None = None,
    ) -> None:
        self.final_indices = final_indices
        self.initial_phrase_index = initial_phrase_index
        self.initialize_calls = 0
        self.state_seen_during_initialize: tuple[str, int] | None = None
        self.lifecycle_events: list[str] = []
        self.state_seen_after_load: tuple[bool, list[int], int | None] | None = None
        super().__init__(
            project,
            empty_state_text="No content",
            loading_state_text="Loading content",
        )

    def initialize_content(self) -> list[int]:
        self.lifecycle_events.append("initialize")
        self.initialize_calls += 1
        self.state_seen_during_initialize = (
            str(self.query_one("#empty-state", Static).render()),
            self.query_one("#line-list", OptionList).option_count,
        )
        return self.final_indices

    def initial_selected_phrase_index(self) -> int | None:
        self.lifecycle_events.append("initial-selection")
        return self.initial_phrase_index

    def on_content_loaded(self) -> None:
        self.lifecycle_events.append("loaded")
        self.state_seen_after_load = (
            self.content_initialized,
            list(self.phrase_indices),
            self.selected_index,
        )


class StructuralRowStubContentApp(StubContentApp):
    """A base-policy fixture whose middle backing item is structural."""

    def content_line_index(self, item_index: int) -> int | None:
        return None if item_index == 1 else item_index


class SingleSelectionPanelStubContentApp(StubContentApp):
    """A fixture exercising the base's opt-in shell capabilities."""

    def __init__(self, project: StubProject) -> None:
        self.changed_phrase_indices = set()
        self.refresh_batches = []
        self.committed = False
        ContentTextualApp.__init__(
            self,
            project,  # type: ignore[arg-type]
            ["Single selection panel editor"],
            phrase_indices=[0, 1, 2],
            multi_select_enabled=False,
            side_panel_enabled=True,
        )

    def compose_side_panel(self):
        yield Static("Panel contents", id="test-panel-content")


def make_app() -> tuple[StubContentApp, StubProject]:
    project = StubProject(
        [
            StubPhraseGroup("first"),
            StubPhraseGroup("hidden"),
            StubPhraseGroup("third needle"),
        ]
    )
    return StubContentApp(project), project


def test_runner_returns_unavailable_without_running_app(monkeypatch) -> None:
    app, _ = make_app()
    run_calls: list[bool] = []
    monkeypatch.setattr(content_textual_app, "can_textual", lambda: False)
    monkeypatch.setattr(app, "run", lambda **_: run_calls.append(True))

    assert run_content_textual_app(app) == ContentAppUnavailable(
        "The current terminal environment does not support full-screen editor",
    )
    assert run_calls == []


def test_runner_wraps_completed_editor_result(monkeypatch) -> None:
    app, _ = make_app()
    monkeypatch.setattr(content_textual_app, "can_textual", lambda: True)
    monkeypatch.setattr(app, "run", lambda **_: EditorSaved())

    assert run_content_textual_app(app) == ContentAppCompleted(EditorSaved())


def test_runner_reports_missing_editor_result(monkeypatch) -> None:
    app, _ = make_app()
    monkeypatch.setattr(content_textual_app, "can_textual", lambda: True)
    monkeypatch.setattr(app, "run", lambda **_: None)

    assert run_content_textual_app(app) == ContentAppMissingResult(
        "Textual editor closed without returning a result"
    )


def test_runner_reports_stylesheet_failure(monkeypatch) -> None:
    app, _ = make_app()
    monkeypatch.setattr(content_textual_app, "can_textual", lambda: True)

    def run_with_css_error(**_) -> None:
        app._exception = StylesheetError("bad css")

    monkeypatch.setattr(app, "run", run_with_css_error)

    assert run_content_textual_app(app) == ContentAppStylesheetFailed(
        "Couldn't load textual css"
    )


def test_runner_reports_unexpected_exception(monkeypatch) -> None:
    app, _ = make_app()
    monkeypatch.setattr(content_textual_app, "can_textual", lambda: True)

    def fail_run(**_) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "run", fail_run)

    assert run_content_textual_app(app) == ContentAppFailed("RuntimeError: boom")


def test_base_retains_project_and_maps_options_to_phrase_groups() -> None:
    app, project = make_app()

    assert app.project is project
    assert app.phrase_indices == [2, 0]
    assert str(app.format_line(0)) == "third needle"
    assert app.find_match_indices("needle") == [0]
    assert app.find_match_indices("hidden") == []


def test_base_formats_and_searches_displayed_line_numbers() -> None:
    app, _ = make_app()

    assert app.format_line_number(1) == "00001"
    assert app.format_line_number(99999) == "99999"
    assert app.format_line_number(100000) == "100000"

    # Rows display their line number via the base default content-line mapping.
    assert app.line_number_text(0) == "00001"
    assert app.line_number_text(1) == "00002"
    assert app.find_text_strings(0) == ["00001", "first"]

    # The stub's [2, 0] mapping means find searches by mapped phrase index:
    # value 2 (row 0) carries number "00003", value 0 (row 1) carries "00001".
    assert app.find_match_indices("00003") == [0]
    assert app.find_match_indices("00001") == [1]
    assert app.find_match_indices("00004") == []


def test_base_formats_section_list_items_with_shared_style() -> None:
    app, _ = make_app()

    section_item = app.format_section_list_item("Section 1/2: Opening", 0)

    assert str(section_item) == "\nSection 1/2: Opening\n\n"
    assert section_item.style == ""

    app.find_match_index = 0

    assert app.format_section_list_item("Section 1/2: Opening", 0).style == (
        "#888888 reverse"
    )


@pytest.mark.parametrize(
    (
        "phrase_indices",
        "line_input",
        "expected_selected_indices",
        "expected_highlighted",
        "expected_toast",
    ),
    [
        # An existing multi-selection is replaced by the manual choice.
        pytest.param(
            [2, 0],
            "1, 3",
            {0, 1},
            0,
            "Selected 2 lines",
            id="mixed-lines",
        ),
        pytest.param([2, 0], "1", {1}, 1, "Selected 1 line", id="single-line"),
        # A later single line after a range becomes the highlight
        pytest.param(
            list(range(10)),
            "2-3, 7",
            {1, 2, 6},
            6,
            "Selected 3 lines",
            id="range-then-later-single-line",
        ),
        # A range end past the last line is clamped without a warning.
        pytest.param(
            [2, 0],
            "1-99",
            {0, 1},
            0,
            "Selected 2 lines",
            id="range-clamps-past-last-line",
        ),
        # Out-of-range single values and negative ranges are dropped.
        pytest.param(
            list(range(8)),
            "6-120, 20, 100-120, -20, 2-5",
            set(range(1, 8)),
            7,
            "Selected 7 lines",
            id="range-clamps-and-drops-out-of-range-values",
        ),
    ],
)
def test_base_manual_selection_replaces_selection_and_highlights_highest_line(
    phrase_indices: list[int],
    line_input: str,
    expected_selected_indices: set[int],
    expected_highlighted: int,
    expected_toast: str,
) -> None:
    phrase_count = max(phrase_indices) + 1
    project = StubProject(
        [StubPhraseGroup(f"line {index + 1}") for index in range(phrase_count)]
    )
    app = StubContentApp(project, phrase_indices=phrase_indices)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+a")
            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            app.screen.query_one("#manual-selection-input", Input).value = line_input
            await pilot.press("enter")

            # The dialog dismissed without surfacing a clamp warning
            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == expected_selected_indices
            assert app.selected_index == expected_highlighted
            assert app.selection_anchor_index == expected_highlighted
            assert app.query_one("#line-list", OptionList).highlighted == (
                expected_highlighted
            )
            assert app.toast_status.left == expected_toast

    run(exercise())


@pytest.mark.parametrize(
    ("start_y", "line_indices", "expected_y"),
    [
        # The final row is above the view and the span doesn't fit: the final
        # row is bottom-aligned (Textual alone would put it at the top).
        pytest.param(150, {0, 50}, lambda height: 51 - height, id="above-bottom-aligns"),
        # The final row is below the view and the span doesn't fit.
        pytest.param(0, {60, 120}, lambda height: 121 - height, id="below-bottom-aligns"),
        # The span fits: the view scrolls the least distance revealing all of it.
        pytest.param(150, {40, 45}, lambda height: 40, id="span-above-fits"),
        pytest.param(0, {100, 105}, lambda height: 106 - height, id="span-below-fits"),
        # A fully visible final row keeps the view, even with rows hidden above.
        pytest.param(50, {0, 55}, lambda height: 50, id="visible-keeps-view"),
    ],
)
def test_base_manual_selection_scrolls_to_reveal_rows_above_final_row(
    start_y: int, line_indices: set[int], expected_y
) -> None:
    app = StubContentApp(make_long_project(200), phrase_indices=list(range(200)))

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            option_list.scroll_to(y=start_y, animate=False, immediate=True)
            await pilot.pause()
            app.handle_manual_selection(line_indices)
            await pilot.pause()

            assert app.selected_index == max(line_indices)
            height = option_list.scrollable_content_region.height
            assert option_list.scroll_offset.y == expected_y(height)

    run(exercise())


def test_selection_end_shows_start_of_final_row_taller_than_viewport() -> None:
    # Rows are 30 lines tall, taller than the viewport, so bottom-aligning the
    # final row would hide its start; its first line is shown instead.
    project = StubProject(
        [StubPhraseGroup("\n".join(f"{index}" for _ in range(30))) for index in range(10)]
    )
    app = StubContentApp(project, phrase_indices=list(range(10)))

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.handle_manual_selection({2, 5})
            await pilot.pause()
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            assert option_list.scroll_offset.y == option_list._index_to_line[5]

    run(exercise())


def test_base_manual_selection_with_no_matching_rows_leaves_selection_unchanged() -> None:
    project = StubProject(
        [
            StubPhraseGroup("first"),
            StubPhraseGroup("section heading"),
            StubPhraseGroup("third"),
        ]
    )
    app = StructuralRowStubContentApp(project, phrase_indices=[0, 1])

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            app.screen.query_one("#manual-selection-input", Input).value = "2"
            await pilot.press("enter")

            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == {0}
            assert app.selected_index == 0
            assert app.selection_anchor_index == 0
            assert app.toast_status.left == ""

    run(exercise())


def test_base_excludes_structural_rows_from_selection_policy_and_mutation() -> None:
    project = StubProject(
        [
            StubPhraseGroup("first"),
            StubPhraseGroup("section heading"),
            StubPhraseGroup("third"),
        ]
    )
    app = StructuralRowStubContentApp(project, phrase_indices=[0, 1, 2])

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "shift+down")
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            assert app.selected_indices == {0, 1, 2}
            assert app.selection_status_text == "2 lines selected"
            assert option_list.inactive_selection_indices == {0}
            assert app.selected_content_line_indices() == {0, 2}

            await pilot.press("up")
            assert app.highlighted_content_line_index() is None
            app.action_mutate()
            assert app.changed_phrase_indices == set()

            await pilot.press("m")
            app.screen.query_one("#manual-selection-input", Input).value = "1, 3"
            await pilot.press("enter")
            assert app.selected_indices == {0, 2}

            # A selection of structural rows alone reports zero content lines
            app.collapse_selection(1)
            assert app.selected_indices == {1}
            assert app.selection_status_text == ""

    run(exercise())


def test_base_css_contains_shared_and_app_specific_rules() -> None:
    assert "#textual-shared-css-test" in ContentTextualApp.CSS
    assert "#line-list" in ContentTextualApp.CSS


def test_shared_css_styles_all_scrollbars_dim() -> None:
    from tts_audiobook_tool.textual.textual_shared import (
        STYLE_DIM,
        TEXTUAL_SHARED_CSS,
        TEXTUAL_SCROLLBAR_CSS,
    )
    from tts_audiobook_tool.textual.worker_app import worker_app_css

    # The single scrollbar-color definition is derived from STYLE_DIM...
    assert f"scrollbar-color: {STYLE_DIM};" in TEXTUAL_SCROLLBAR_CSS
    assert f"scrollbar-color-hover: {STYLE_DIM};" in TEXTUAL_SCROLLBAR_CSS
    assert f"scrollbar-color-active: {STYLE_DIM};" in TEXTUAL_SCROLLBAR_CSS
    # ...and both CSS paths include it: the shared stylesheet and the
    # worker apps, which deliberately do not load TEXTUAL_SHARED_CSS.
    assert TEXTUAL_SCROLLBAR_CSS in TEXTUAL_SHARED_CSS
    assert TEXTUAL_SCROLLBAR_CSS in worker_app_css("generation-divider")


def test_hanging_indent_ignores_non_printing_ansi_prefix_characters() -> None:
    prefix = "\x1b[31mLabel:\x1b[0m "
    text = HangingIndentText.from_ansi(
        prefix + "one two three four",
        content_start=len(prefix),
    )
    console = Console(width=14, force_terminal=False, color_system=None)

    rendered_lines = console.render_lines(text, console.options, pad=False)
    rendered = ["".join(segment.text for segment in line) for line in rendered_lines]

    assert rendered == ["Label: one two", "       three", "       four"]


def test_hanging_indent_caps_lines_and_truncates_hidden_overflow() -> None:
    # At the 18-cell content width the 59-character content wraps to four
    # lines, so the three-line cap and the truncation ellipsis both apply.
    text = HangingIndentText.from_ansi_prefix(
        "[00001] [Line 10] ",
        "one two three four five six seven eight nine ten eleven twelve",
    )
    console = Console(width=36, force_terminal=False, color_system=None)

    rendered_lines = console.render_lines(text, console.options, pad=False)
    rendered = ["".join(segment.text for segment in line) for line in rendered_lines]

    # Continuation lines are indented by the row prefix width.
    indent = " " * 18
    assert rendered == [
        "[00001] [Line 10] one two three four",
        f"{indent}five six seven",
        f"{indent}eight nine ten…",
    ]


def test_base_composes_header_list_status_and_superseding_find_bar() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            assert app.query_one("#header", Vertical)
            assert app.query_one("#line-list")
            status_bar = app.query_one("#status-bar", Horizontal)
            find_bar = app.query_one("#find-bar", Horizontal)
            status_left = app.query_one("#status-left", Static)
            status_right = app.query_one("#status-right", Static)
            assert status_bar.display is True
            assert find_bar.display is False

            await pilot.press("ctrl+f")
            assert status_bar.display is False
            assert status_left.display is False
            assert status_right.display is False
            assert find_bar.display is True

            await pilot.press("escape")
            assert status_bar.display is True
            assert status_left.display is True
            assert status_right.display is True
            assert find_bar.display is False

    run(exercise())


def test_base_side_panel_is_opt_in_and_constrained_to_twenty_through_forty() -> None:
    project = StubProject(
        [StubPhraseGroup("first"), StubPhraseGroup("second"), StubPhraseGroup("third")]
    )
    app = SingleSelectionPanelStubContentApp(project)
    default_app, _ = make_app()

    async def panel_width_at(terminal_width: int) -> int:
        panel_app = SingleSelectionPanelStubContentApp(project)
        async with panel_app.run_test(size=(terminal_width, 30)):
            return panel_app.query_one("#side-panel", Vertical).size.width

    async def exercise() -> None:
        # Editors which do not opt in keep the plain shell
        async with default_app.run_test():
            assert default_app.query("#content-shell")
            assert default_app.query("#content-main")
            assert not default_app.query("#side-panel")
            assert not default_app.query("#side-panel-divider")

        async with app.run_test(size=(100, 30)):
            panel = app.query_one("#side-panel", Vertical)
            assert app.query_one("#side-panel-divider")
            assert str(app.query_one("#test-panel-content", Static).render()) == (
                "Panel contents"
            )
            assert panel.size.width == 35

        assert await panel_width_at(50) == 20
        assert await panel_width_at(140) == 40

    run(exercise())


def test_base_single_selection_disables_every_row_multi_selection_entry_path() -> None:
    project = StubProject(
        [StubPhraseGroup("first"), StubPhraseGroup("second"), StubPhraseGroup("third")]
    )
    app = SingleSelectionPanelStubContentApp(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down")
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            assert app.selected_index == 1
            assert app.selected_indices == {1}
            assert option_list.extend_selection is False

            await pilot.press("ctrl+a")
            assert app.selected_indices == {1}

            await pilot.press("m")
            assert not isinstance(app.screen, ManualSelectionDialog)

            await pilot.press("ctrl+f", "t", "h", "i", "r", "d", "enter")
            assert app.selected_index == 2
            assert app.selected_indices == {2}

    run(exercise())


def test_base_shows_custom_non_selectable_empty_state_and_restores_list() -> None:
    project = StubProject([StubPhraseGroup("first")])
    app = StubContentApp(
        project,
        phrase_indices=[],
        empty_state_text="Nothing available",
    )
    app.update_empty_state_text("Loading items")

    async def exercise() -> None:
        async with app.run_test():
            option_list = app.query_one("#line-list", OptionList)
            empty_state = app.query_one("#empty-state", Static)

            assert option_list.display is False
            assert option_list.option_count == 0
            assert empty_state.display is True
            assert str(empty_state.render()) == "Loading items"
            assert app.selected_index is None
            assert app.selected_indices == set()

            app.update_empty_state_text("Nothing available")
            assert app.empty_state_text == "Nothing available"
            assert str(empty_state.render()) == "Nothing available"
            assert option_list.display is False
            assert empty_state.display is True

            app.replace_phrase_indices([0])

            assert option_list.display is True
            assert empty_state.display is False
            assert option_list.has_focus is True
            assert app.selected_index == 0
            assert option_list.option_count == 1

    run(exercise())


@pytest.mark.parametrize(
    ("final_indices", "initial_phrase_index"),
    [
        pytest.param([1, 0], 0, id="installs-loaded-rows"),
        pytest.param([], None, id="empty-result-keeps-final-empty-copy"),
    ],
)
def test_base_deferred_content_loads_once_after_loading_view_draws(
    final_indices: list[int],
    initial_phrase_index: int | None,
) -> None:
    project = StubProject(
        [
            StubPhraseGroup(f"line {index + 1}")
            for index in range(max(final_indices, default=0) + 1)
        ]
    )
    app = DeferredStubContentApp(
        project,
        final_indices,
        initial_phrase_index=initial_phrase_index,
    )
    expected_selected_index = (
        None
        if not final_indices or initial_phrase_index is None
        else final_indices.index(initial_phrase_index)
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            option_list = app.query_one("#line-list", OptionList)
            empty_state = app.query_one("#empty-state", Static)

            # The loading view is visible while initialize_content runs
            assert app.state_seen_during_initialize == ("Loading content", 0)
            assert app.initialize_calls == 1
            assert app.content_initialized is True
            assert app.phrase_indices == final_indices
            assert app.selected_index == expected_selected_index
            assert option_list.option_count == len(final_indices)
            assert app.lifecycle_events == [
                "initialize",
                "initial-selection",
                "loaded",
            ]
            assert app.state_seen_after_load == (
                True,
                final_indices,
                expected_selected_index,
            )

            if final_indices:
                assert option_list.display is True
                assert empty_state.display is False
            else:
                # An empty result settles on the final empty copy
                assert option_list.display is False
                assert empty_state.display is True
                assert str(empty_state.render()) == "No content"

            app.load_content()
            assert app.initialize_calls == 1
            assert app.lifecycle_events == [
                "initialize",
                "initial-selection",
                "loaded",
            ]

    run(exercise())


def test_base_parses_ansi_header_strings_when_composing_and_updating() -> None:
    app, _ = make_app()
    app.header_lines = ["\x1b[31mInitial header\x1b[0m", "Initial details"]

    async def exercise() -> None:
        async with app.run_test():
            header = app.query_one("#header-line-0", Static)
            initial_renderable = header.content
            assert isinstance(initial_renderable, Text)
            assert initial_renderable.plain == "Initial header"
            assert initial_renderable.spans

            app.update_header(
                [
                    "\x1b[32mUpdated header\x1b[0m",
                    "Updated details",
                    "Truncated overflow",
                ]
            )
            updated_renderable = header.content
            assert isinstance(updated_renderable, Text)
            assert updated_renderable.plain == "Updated header"
            assert updated_renderable.spans
            assert app.header_lines == [
                "\x1b[32mUpdated header\x1b[0m",
                "Updated details",
            ]

            app.update_header(["Short header"])
            assert app.header_lines == ["Short header", ""]
            assert str(app.query_one("#header-line-1", Static).render()) == ""

    run(exercise())


def test_status_modes_render_independent_sides_with_whole_mode_precedence() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test():
            status_bar = app.query_one("#status-bar", Horizontal)
            status_left = app.query_one("#status-left", Static)
            status_right = app.query_one("#status-right", Static)
            assert str(status_left.render()) == ""
            assert str(status_right.render()) == ""
            assert status_bar.has_class("status-pinned")

            app.set_pinned_status(left="Pinned left", right="Pinned right")
            app.set_pinned_status(right="Updated pinned right")
            assert str(status_left.render()) == "Pinned left"
            assert str(status_right.render()) == "Updated pinned right"
            assert status_bar.has_class("status-pinned")

            app.set_selected_status(left="Selected left")
            assert str(status_left.render()) == "Selected left"
            assert str(status_right.render()) == ""
            assert status_bar.has_class("status-selected")

            app.set_selected_status(left="", right="2 lines selected")
            assert str(status_left.render()) == ""
            assert str(status_right.render()) == "2 lines selected"
            assert status_bar.has_class("status-selected")

            app.show_status_toast(left="2 lines deleted", right="Undo unavailable")
            assert str(status_left.render()) == "2 lines deleted"
            assert str(status_right.render()) == "Undo unavailable"
            assert status_bar.has_class("status-toast")

            app.clear_status_toast()
            assert str(status_left.render()) == ""
            assert str(status_right.render()) == "2 lines selected"
            assert status_bar.has_class("status-selected")

            app.set_selected_status(right="")
            assert str(status_left.render()) == "Pinned left"
            assert str(status_right.render()) == "Updated pinned right"
            assert status_bar.has_class("status-pinned")

    run(exercise())


@pytest.mark.parametrize(
    ("width", "expected_overlap"),
    [pytest.param(40, False, id="separate"), pytest.param(15, True, id="overlap")],
)
def test_status_sides_keep_intrinsic_width_and_edge_alignment(
    width: int,
    expected_overlap: bool,
) -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test(size=(width, 24)) as pilot:
            app.set_pinned_status(left="LEFT-12345", right="RIGHT-12345")
            await pilot.pause()
            status_bar = app.query_one("#status-bar", Horizontal)
            status_left = app.query_one("#status-left", Static)
            status_right = app.query_one("#status-right", Static)

            assert status_left.region.width == len("LEFT-12345")
            assert status_right.region.width == len("RIGHT-12345")
            assert status_left.region.x == status_bar.region.x
            assert status_right.region.right == status_bar.region.right
            assert status_left.region.overlaps(status_right.region) is expected_overlap

    run(exercise())


def test_new_toast_restarts_fixed_expiry_window(monkeypatch) -> None:
    app, _ = make_app()
    durations: list[float] = []

    async def exercise() -> None:
        async with app.run_test():
            original_set_timer = app.set_timer

            def record_set_timer(delay, callback, **kwargs):
                durations.append(delay)
                return original_set_timer(delay, callback, **kwargs)

            monkeypatch.setattr(app, "set_timer", record_set_timer)
            status_bar = app.query_one("#status-bar", Horizontal)
            status_left = app.query_one("#status-left", Static)
            status_right = app.query_one("#status-right", Static)
            app.show_status_toast(left="First", right="Old right")
            first_timer = app.toast_timer
            app.show_status_toast(left="Second")

            assert durations == [1.5, 1.5]
            assert first_timer is not None and app.toast_timer is not first_timer
            assert str(status_left.render()) == "Second"
            assert str(status_right.render()) == ""
            assert status_bar.has_class("status-toast")

    run(exercise())


def test_base_mutates_selected_mapped_items_and_confirms_before_commit() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down")
            assert app.selected_indices == {0, 1}

            app.action_mutate()
            assert app.changed_phrase_indices == {0, 2}
            assert app.selected_indices == {1}
            assert app.refresh_batches == [[0, 1]]

            await pilot.press("escape")
            assert isinstance(app.screen, SaveChangesDialog)
            assert app.committed is False

            await pilot.press("escape")
            assert app.is_running is True
            assert app.committed is False

            await pilot.press("escape", "y")
            await pilot.pause()
            assert app.committed is True
            assert app.is_running is False

    run(exercise())


def test_confirmation_dialog_renders_copy_lines_and_grows_to_fit_warnings() -> None:
    app, _ = make_app()
    long_warning = (
        f"{COL_ERROR}Saving these changes requires deleting 12 generated sound "
        "segments and 3 section markers from line 2 onward."
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(
                SaveChangesDialog(
                    [
                        "Apply changes?",
                        "",
                        f"{COL_ERROR}Generated sound segments will be deleted.",
                    ]
                )
            )
            await pilot.pause()
            warning_separator = app.screen.query_one(
                "#save-changes-copy-line-2", Static
            )
            warning = app.screen.query_one("#save-changes-copy-line-3", Static)
            warning_renderable = warning.content

            assert str(warning_separator.render()) == ""
            assert isinstance(warning_renderable, Text)
            assert warning_renderable.plain == (
                "Generated sound segments will be deleted."
            )
            assert warning_renderable.spans
            assert str(warning_renderable.spans[0].style) == "#ff0000"
            assert {button.id for button in app.screen.query(Button)} == {"yes", "no"}

            await pilot.press("escape")

            # The default dialog carries a single copy line
            app.push_screen(SaveChangesDialog())
            await pilot.pause()
            assert (
                str(app.screen.query_one("#save-changes-copy-line-1", Static).render())
                == "Save changes before exiting?"
            )
            assert not app.screen.query("#save-changes-copy-line-2")

            await pilot.press("escape")

            # A long warning wraps and the dialog grows to fit it
            app.push_screen(SaveChangesDialog(["Apply changes?", ""]))
            await pilot.pause()
            baseline_height = app.screen.query_one(
                "#save-changes-dialog", Vertical
            ).size.height
            await pilot.press("escape")
            await pilot.pause()

            app.push_screen(SaveChangesDialog(["Apply changes?", "", long_warning]))
            await pilot.pause()
            warning = app.screen.query_one("#save-changes-copy-line-3", Static)
            dialog_box = app.screen.query_one("#save-changes-dialog", Vertical)

            assert warning.region.height > 1
            assert dialog_box.size.height > baseline_height

    run(exercise())


def test_confirmation_dialog_can_left_align_removal_list() -> None:
    # Consequences align left without changing the shared dialog's prompt layout.
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(SaveChangesDialog(
                [
                    "Save changes before exiting?",
                    "",
                    "The text edits made are structural changes which will result in the removal of:",
                    f"{COL_ERROR}• 2 generated audio segments from line 2 onward.",
                    f"{COL_ERROR}• 1 section marker from line 2 onward.",
                ],
                left_align_from_line=3,
            ))
            await pilot.pause()
            prompt = app.screen.query_one("#save-changes-copy-line-1", Static)
            heading = app.screen.query_one("#save-changes-copy-line-3", Static)
            bullet = app.screen.query_one("#save-changes-copy-line-4", Static)
            assert prompt.styles.content_align_horizontal == "center"
            assert heading.styles.content_align_horizontal == "left"
            assert bullet.styles.content_align_horizontal == "left"
            assert heading.region.height > 1
            assert bullet.content.plain == "• 2 generated audio segments from line 2 onward."

    run(exercise())


def test_confirmation_dialog_can_show_but_disable_yes_button() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(
                SaveChangesDialog(
                    ["Generation is blocked"],
                    confirmation_enabled=False,
                )
            )
            await pilot.pause()

            yes_button = app.screen.query_one("#yes", Button)
            assert yes_button.display is True
            assert yes_button.disabled is True

            await pilot.press("y")
            assert isinstance(app.screen, SaveChangesDialog)

            await pilot.click(yes_button)
            assert isinstance(app.screen, SaveChangesDialog)

            await pilot.press("n")
            assert not isinstance(app.screen, SaveChangesDialog)

            app.push_screen(
                SaveChangesDialog(
                    ["Generation is blocked"],
                    confirmation_enabled=False,
                )
            )
            await pilot.press("escape")
            assert not isinstance(app.screen, SaveChangesDialog)

    run(exercise())


def test_plain_navigation_does_not_replace_prompts() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.refresh_batches.clear()
            await pilot.press("down")
            assert app.selected_indices == {1}
            assert app.refresh_batches == []

            await pilot.press("shift+up")
            assert app.selected_indices == {0, 1}
            assert app.refresh_batches == []

    run(exercise())


def test_base_clean_escape_exits_without_confirmation() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running is False
            assert app.has_changes is False
            assert not isinstance(app.screen, SaveChangesDialog)

        assert app.return_value == EditorClosed()

    run(exercise())


def test_base_confirmed_exit_commits_changes() -> None:
    app, _ = make_app()
    app.changed_phrase_indices.add(0)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            assert isinstance(app.screen, SaveChangesDialog)

            await pilot.press("y")
            await pilot.pause()
            assert app.committed is True
            assert app.is_running is False

        assert app.return_value == EditorSaved()

    run(exercise())


def test_base_discarded_or_cancelled_exit_closes_or_stays_open_without_committing() -> None:
    app, _ = make_app()
    app.changed_phrase_indices.add(0)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            # Cancelling the confirmation dialog keeps the editor open
            await pilot.press("escape")
            assert isinstance(app.screen, SaveChangesDialog)

            await pilot.press("escape")
            assert not isinstance(app.screen, SaveChangesDialog)
            assert app.committed is False
            assert app.is_running is True

            # Declining the dialog closes the editor without committing
            await pilot.press("escape", "n")
            await pilot.pause()
            assert app.committed is False
            assert app.is_running is False

        assert app.return_value == EditorClosed()

    run(exercise())


def test_resolve_anchor_line_follows_text_then_clamps() -> None:
    texts = ["a", "b", "c", "d"]

    def line_text(index: int) -> str:
        return texts[index]

    # Unchanged line resolves to itself
    assert resolve_anchor_line(2, "c", len(texts), line_text) == 2
    # A line which moved (eg, lines inserted above it) is followed by its text
    assert resolve_anchor_line(0, "c", len(texts), line_text) == 2
    # Without a text match the stored index is clamped into range
    assert resolve_anchor_line(9, "gone", len(texts), line_text) == 3
    assert resolve_anchor_line(1, "gone", len(texts), line_text) == 1
    assert resolve_anchor_line(0, "a", 0, line_text) is None


def test_runner_restores_and_stores_view_anchor_per_editor(monkeypatch) -> None:
    app, _ = make_app()
    stored = EditorViewAnchor(top_line=1, top_line_text="hidden")
    captured = EditorViewAnchor(top_line=2, top_line_text="third needle")
    other = EditorViewAnchor(top_line=0, top_line_text="first")
    view_anchors = {"StubContentApp": stored, "OtherEditor": other}
    seen_on_run: list[EditorViewAnchor | None] = []

    def fake_run(**_):
        seen_on_run.append(app.view_anchor)
        app.exit_view_anchor = captured
        return EditorSaved()

    monkeypatch.setattr(content_textual_app, "can_textual", lambda: True)
    monkeypatch.setattr(app, "run", fake_run)

    run_content_textual_app(app, view_anchors)

    assert seen_on_run == [stored]
    # Only this editor's entry is replaced
    assert view_anchors == {"StubContentApp": captured, "OtherEditor": other}


def make_long_project(count: int, prefix: list[str] | None = None) -> StubProject:
    return StubProject(
        [StubPhraseGroup(text) for text in (prefix or [])]
        + [StubPhraseGroup(f"line {index}") for index in range(count)]
    )


def top_row(app: ContentTextualApp) -> int:
    option_list = app.query_one("#line-list", NonWrappingOptionList)
    return option_list._lines[round(option_list.scroll_offset.y)][0]


def test_view_anchor_round_trips_through_exit_and_follows_shifted_lines() -> None:
    # The first session scrolls down and exits; the second session reopens on
    # a project with lines inserted at the start, and lands on the same text.
    first_project = make_long_project(200)
    first_app = DeferredStubContentApp(first_project, list(range(200)))

    async def first_session() -> None:
        async with first_app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            option_list = first_app.query_one("#line-list", OptionList)
            option_list.highlighted = 120
            await pilot.pause()
            first_app.exit(EditorClosed())
            await pilot.pause()

    run(first_session())
    anchor = first_app.exit_view_anchor
    assert anchor is not None
    assert anchor.highlighted_line == 120
    assert anchor.highlighted_line_text == "line 120"
    assert 90 < anchor.top_line < 120
    assert anchor.top_line_text == f"line {anchor.top_line}"

    second_project = make_long_project(200, prefix=["new a", "new b", "new c"])
    second_app = DeferredStubContentApp(second_project, list(range(203)))
    second_app.view_anchor = anchor

    async def second_session() -> None:
        async with second_app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert second_app.selected_index == 123
            assert top_row(second_app) == anchor.top_line + 3

    run(second_session())


def test_view_anchor_highlight_wins_over_partially_scrolled_top_row() -> None:
    # Rows are three lines tall. The top row is scrolled one line out of view
    # and the highlight sits on the bottom row. The anchor only records the
    # top row, so restoring it in full would push the highlight off the bottom;
    # the highlight takes precedence and the view scrolls just enough to show it.
    project = StubProject([StubPhraseGroup(f"line {index}\nb\nc") for index in range(60)])
    first_app = DeferredStubContentApp(project, list(range(60)))

    async def first_session() -> None:
        async with first_app.run_test(size=(80, 25)) as pilot:
            await pilot.pause()
            option_list = first_app.query_one("#line-list", NonWrappingOptionList)
            assert option_list.scrollable_content_region.height == 22
            option_list.highlighted = 17
            await pilot.pause()
            # Row 10 spans y 30-32; rows 11-17 fill y 33-53
            option_list.scroll_to(y=32, animate=False, immediate=True)
            await pilot.pause()
            first_app.exit(EditorClosed())
            await pilot.pause()

    run(first_session())
    anchor = first_app.exit_view_anchor
    assert anchor is not None
    assert (anchor.top_line, anchor.highlighted_line) == (10, 17)

    second_app = DeferredStubContentApp(project, list(range(60)))
    second_app.view_anchor = anchor

    async def second_session() -> None:
        async with second_app.run_test(size=(80, 25)) as pilot:
            await pilot.pause()
            await pilot.pause()
            option_list = second_app.query_one("#line-list", NonWrappingOptionList)
            assert second_app.selected_index == 17
            assert option_list.scroll_offset.y == 32

    run(second_session())


def test_view_anchor_omits_highlight_scrolled_out_of_view() -> None:
    # A highlight scrolled entirely out of view (eg, with the mouse) isn't
    # recorded, so reopening restores the view and highlights its top line.
    project = make_long_project(200)
    first_app = DeferredStubContentApp(project, list(range(200)))

    async def first_session() -> None:
        async with first_app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            option_list = first_app.query_one("#line-list", OptionList)
            option_list.scroll_to(y=150, animate=False, immediate=True)
            await pilot.pause()
            first_app.exit(EditorClosed())
            await pilot.pause()

    run(first_session())
    assert first_app.exit_view_anchor == EditorViewAnchor(
        top_line=150, top_line_text="line 150"
    )

    second_app = DeferredStubContentApp(project, list(range(200)))
    second_app.view_anchor = first_app.exit_view_anchor

    async def second_session() -> None:
        async with second_app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert top_row(second_app) == 150
            assert second_app.selected_index == 150

    run(second_session())


def test_view_anchor_restores_leading_structural_row() -> None:
    # A structural row (eg, a section heading) at the top of the view is
    # restored above the anchor's top content line.
    project = make_long_project(60)
    app = StructuralRowStubContentApp(project, phrase_indices=list(range(60)))

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            option_list = app.query_one("#line-list", OptionList)
            option_list.scroll_to(y=1, animate=False, immediate=True)
            await pilot.pause()
            anchor = app.capture_view_anchor()
            assert anchor == EditorViewAnchor(
                top_line=2,
                top_line_text="line 2",
                top_leading_rows=1,
                # Row 0 (the highlight) was scrolled out of view
            )

    run(exercise())

    class DeferredStructuralRowStubContentApp(DeferredStubContentApp):
        def content_line_index(self, item_index: int) -> int | None:
            return None if item_index == 1 else item_index

    restored_app = DeferredStructuralRowStubContentApp(project, list(range(60)))
    restored_app.view_anchor = EditorViewAnchor(
        top_line=2, top_line_text="line 2", top_leading_rows=1
    )

    async def restore() -> None:
        async with restored_app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert top_row(restored_app) == 1
            assert restored_app.selected_index == 2

    run(restore())
