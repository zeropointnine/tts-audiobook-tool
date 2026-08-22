import asyncio
from dataclasses import dataclass

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
from tts_audiobook_tool.textual.manual_selection_dialog import ManualSelectionDialog
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    NonWrappingOptionList,
)


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
        self.refreshed_indices: list[int] = []
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
        self.refreshed_indices = []
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


def run(coroutine) -> None:
    asyncio.run(coroutine)


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
    assert section_item.spans == []
    assert section_item.style == ""

    app.find_match_index = 0

    assert app.format_section_list_item("Section 1/2: Opening", 0).style == (
        "#888888 reverse"
    )


def test_base_manual_selection_replaces_selection_and_highlights_highest_line() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            app.screen.query_one("#manual-selection-input", Input).value = "1, 3"
            await pilot.press("enter")

            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == {0, 1}
            assert app.selected_index == 0
            assert app.selection_anchor_index == 0
            assert app.query_one("#line-list", OptionList).highlighted == 0
            assert app.toast_text == "Selected 2 lines"

    run(exercise())


def test_base_manual_selection_single_line_shows_singular_toast() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            app.screen.query_one("#manual-selection-input", Input).value = "1"
            await pilot.press("enter")

            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == {1}
            assert app.toast_text == "Selected 1 line"

    run(exercise())


def test_base_manual_selection_silently_clamps_out_of_range_end() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            app.screen.query_one("#manual-selection-input", Input).value = "1-99"
            await pilot.press("enter")

            # The dialog dismissed without surfacing a clamp warning
            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == {0, 1}
            assert app.toast_text == "Selected 2 lines"

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

    run(exercise())


def test_base_css_contains_shared_and_app_specific_rules() -> None:
    assert "#textual-shared-css-test" in ContentTextualApp.CSS
    assert "#line-list" in ContentTextualApp.CSS


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


def test_base_composes_header_list_status_and_superseding_find_bar() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            assert app.query_one("#header", Vertical)
            assert app.query_one("#line-list")
            status_bar = app.query_one("#status-bar", Horizontal)
            find_bar = app.query_one("#find-bar", Horizontal)
            status_line = app.query_one("#status-line", Static)
            assert status_bar.display is True
            assert find_bar.display is False

            await pilot.press("ctrl+f")
            assert status_bar.display is False
            assert status_line.display is False
            assert find_bar.display is True

            await pilot.press("escape")
            assert status_bar.display is True
            assert status_line.display is True
            assert find_bar.display is False

    run(exercise())


def test_base_omits_side_panel_unless_editor_opts_in() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test():
            assert app.query("#content-shell")
            assert app.query("#content-main")
            assert not app.query("#side-panel")
            assert not app.query("#side-panel-divider")

    run(exercise())


def test_base_composes_widget_agnostic_side_panel_when_enabled() -> None:
    project = StubProject(
        [StubPhraseGroup("first"), StubPhraseGroup("second"), StubPhraseGroup("third")]
    )
    app = SingleSelectionPanelStubContentApp(project)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)):
            panel = app.query_one("#side-panel", Vertical)
            assert app.query_one("#side-panel-divider")
            assert str(app.query_one("#test-panel-content", Static).render()) == (
                "Panel contents"
            )
            assert panel.size.width == 35

    run(exercise())


def test_base_side_panel_width_is_constrained_to_twenty_through_forty_columns() -> None:
    project = StubProject(
        [StubPhraseGroup("first"), StubPhraseGroup("second"), StubPhraseGroup("third")]
    )

    async def panel_width_at(terminal_width: int) -> int:
        app = SingleSelectionPanelStubContentApp(project)
        async with app.run_test(size=(terminal_width, 30)):
            return app.query_one("#side-panel", Vertical).size.width

    assert asyncio.run(panel_width_at(50)) == 20
    assert asyncio.run(panel_width_at(140)) == 40


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


def test_base_loads_opt_in_deferred_content_after_loading_view_draws() -> None:
    project = StubProject([StubPhraseGroup("first"), StubPhraseGroup("second")])
    app = DeferredStubContentApp(project, [1, 0], initial_phrase_index=0)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            option_list = app.query_one("#line-list", OptionList)
            empty_state = app.query_one("#empty-state", Static)

            assert app.state_seen_during_initialize == ("Loading content", 0)
            assert app.initialize_calls == 1
            assert app.content_initialized is True
            assert app.phrase_indices == [1, 0]
            assert app.selected_index == 1
            assert option_list.option_count == 2
            assert option_list.display is True
            assert empty_state.display is False
            assert app.lifecycle_events == [
                "initialize",
                "initial-selection",
                "loaded",
            ]
            assert app.state_seen_after_load == (True, [1, 0], 1)

            app.load_content()
            assert app.initialize_calls == 1
            assert app.lifecycle_events == [
                "initialize",
                "initial-selection",
                "loaded",
            ]

    run(exercise())


def test_base_deferred_empty_result_switches_to_final_empty_copy() -> None:
    app = DeferredStubContentApp(StubProject([]), [])

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            option_list = app.query_one("#line-list", OptionList)
            empty_state = app.query_one("#empty-state", Static)

            assert app.state_seen_during_initialize == ("Loading content", 0)
            assert option_list.display is False
            assert empty_state.display is True
            assert str(empty_state.render()) == "No content"

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


def test_status_layers_apply_precedence_and_restore_after_toast() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test():
            status_line = app.query_one("#status-line", Static)
            assert str(status_line.render()) == ""
            assert status_line.has_class("status-pinned")

            app.set_pinned_text("Pinned")
            app.set_selected_text("")
            assert str(status_line.render()) == "Pinned"
            assert status_line.has_class("status-pinned")

            app.set_selected_text("2 lines selected")
            assert str(status_line.render()) == "2 lines selected"
            assert status_line.has_class("status-selected")

            app.set_toast_text("2 lines deleted")
            app.collapse_selection(0)
            assert str(status_line.render()) == "2 lines deleted"
            assert status_line.has_class("status-toast")

            app.clear_toast_text()
            assert str(status_line.render()) == "Pinned"
            assert status_line.has_class("status-pinned")

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
            status_line = app.query_one("#status-line", Static)
            app.set_toast_text("First")
            first_timer = app.toast_timer
            app.set_toast_text("Second")

            assert durations == [1.5, 1.5]
            assert first_timer is not None and app.toast_timer is not first_timer
            assert str(status_line.render()) == "Second"
            assert status_line.has_class("status-toast")

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


def test_confirmation_dialog_renders_ansi_lines_and_only_yes_no_buttons() -> None:
    app, _ = make_app()

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

    run(exercise())


def test_confirmation_dialog_uses_one_default_copy_line() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(SaveChangesDialog())
            await pilot.pause()

            assert (
                str(app.screen.query_one("#save-changes-copy-line-1", Static).render())
                == "Save changes before exiting?"
            )
            assert not app.screen.query("#save-changes-copy-line-2")

    run(exercise())


def test_confirmation_dialog_warning_wraps_and_dialog_grows_to_fit() -> None:
    app, _ = make_app()
    long_warning = (
        f"{COL_ERROR}Saving these changes requires deleting 12 generated sound "
        "segments and 3 section markers from line 2 onward."
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
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
            app.refreshed_indices.clear()
            await pilot.press("down")
            assert app.selected_indices == {1}
            assert app.refreshed_indices == []

            await pilot.press("shift+up")
            assert app.selected_indices == {0, 1}
            assert app.refreshed_indices == []

    run(exercise())


def test_base_clean_escape_exits_without_confirmation() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running is False
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


def test_base_discarded_exit_closes_without_committing() -> None:
    app, _ = make_app()
    app.changed_phrase_indices.add(0)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape", "n")
            await pilot.pause()
            assert app.committed is False
            assert app.is_running is False

        assert app.return_value == EditorClosed()

    run(exercise())


def test_base_cancelled_exit_keeps_editor_open_without_committing() -> None:
    app, _ = make_app()
    app.changed_phrase_indices.add(0)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            assert isinstance(app.screen, SaveChangesDialog)

            await pilot.press("escape")
            assert not isinstance(app.screen, SaveChangesDialog)
            assert app.committed is False
            assert app.is_running is True

    run(exercise())
