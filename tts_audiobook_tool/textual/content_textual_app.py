from collections.abc import Callable, Iterable
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Input, OptionList, Rule, Static
from textual.widgets.option_list import Option

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.textual.save_changes_dialog import (
    ExitDecision,
    SaveChangesDialog,
)
from tts_audiobook_tool.textual.textual_shared import (
    NonWrappingOptionList,
    can_textual,
    load_css,
)
from tts_audiobook_tool.util import print_feedback

HeaderLine = str | Text
SelectedItemMutator = Callable[[int, int], bool]


class ContentTextualApp(App[None]):
    """Base app for selecting and mutating phrase-group-backed content rows."""

    CSS = load_css("textual_shared.css", "content_textual_app.css")
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "quit_editor", "Quit", show=False),
        Binding("ctrl+q", "ignore_ctrl_q", show=False, priority=True),
        Binding("ctrl+a", "select_all", show=False, priority=True),
        Binding("ctrl+f", "open_find", show=False, priority=True),
        # Many terminal emulators report Shift+Enter as plain Enter, so this
        # binding only works where the terminal emits a distinct key sequence.
        Binding("shift+enter", "find_previous", show=False, priority=True),
    ]

    def __init__(
        self,
        project: Project,
        header_lines: Iterable[HeaderLine],
        phrase_indices: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.header_lines = list(header_lines)
        self.phrase_indices = (
            list(range(len(project.phrase_groups)))
            if phrase_indices is None
            else list(phrase_indices)
        )
        self.selected_index = 0 if self.phrase_indices else None
        self.selection_anchor_index = self.selected_index
        self.selected_indices = (
            {self.selected_index} if self.selected_index is not None else set()
        )
        # Typing only changes the query. Enter submits it and enables backward
        # navigation; each submitted search starts after the selected row.
        self.find_active = False
        self.find_search_start_index: int | None = None
        self.find_query_submitted = False
        self.find_match_index: int | None = None
        self.configure()

    def configure(self) -> None:
        """Configure Textual before the full-screen interface starts."""
        self.theme = "ansi-dark"

    @staticmethod
    def check_terminal_support() -> bool:
        """Report when the terminal cannot run a full-screen editor."""
        if not can_textual():
            print_feedback(
                "The current terminal environment does not support full-screen editor",
                is_error=True,
            )
            return False
        else:
            return True

    @property
    def has_changes(self) -> bool:
        """Whether a concrete editor has staged a data mutation."""
        return False

    def format_line(self, index: int) -> Text:
        """Format a visible row; concrete editors must implement this hook."""
        raise NotImplementedError

    def find_text(self, phrase_index: int) -> str:
        """Return the searchable text for a Project phrase group."""
        return self.project.phrase_groups[phrase_index].presentable_text

    def compose_status_widgets(self) -> Iterable[Widget]:
        """Provide widgets for the status row replaced by the find bar."""
        yield Static(self.selection_status_text, id="selection-status", markup=False)

    def make_confirmation_dialog(self) -> SaveChangesDialog:
        """Build the dialog shown before concrete staged changes are committed."""
        return SaveChangesDialog()

    def commit_changes_and_exit(self) -> None:
        """Commit concrete staged changes; concrete editors must implement this."""
        raise NotImplementedError

    def compose(self) -> ComposeResult:
        header = Vertical(
            *(
                Static(
                    line,
                    id=f"header-line-{index}",
                    classes="header-line",
                    markup=False,
                )
                for index, line in enumerate(self.header_lines)
            ),
            id="header",
        )
        header.styles.height = len(self.header_lines)
        yield header
        yield Rule(id="header-divider")
        yield NonWrappingOptionList(
            *(
                Option(self.format_line(index), id=self.option_id(index))
                for index in range(len(self.phrase_indices))
            ),
            id="line-list",
            markup=False,
            compact=True,
            collapse_selection=self.collapse_current_selection,
        )
        yield Horizontal(*self.compose_status_widgets(), id="status-bar")
        yield Horizontal(
            Static(self.find_label_text, id="find-label", markup=False),
            Input(id="find-input", compact=True, select_on_focus=False),
            Static("", id="find-result", markup=False),
            id="find-bar",
        )

    @property
    def find_label_text(self) -> str:
        return "Find: "

    def on_mount(self) -> None:
        self.query_one("#line-list", OptionList).focus()

    def update_header(self, lines: list[HeaderLine]) -> None:
        """Replace the fixed-height header contents."""
        assert len(lines) == len(self.header_lines), (
            f"Expected {len(self.header_lines)} header lines, got {len(lines)}"
        )
        self.header_lines = list(lines)
        if self.is_mounted:
            for index, line in enumerate(self.header_lines):
                self.query_one(f"#header-line-{index}", Static).update(line)

    @property
    def selection_status_text(self) -> str:
        """Describe a multi-line selection, or remain blank for one line."""
        selection_count = len(self.selected_indices)
        return f"{selection_count} lines selected" if selection_count >= 2 else ""

    def update_selection_status(self) -> None:
        """Refresh the selection status within the bottom status row."""
        status_widgets = self.query("#selection-status")
        if status_widgets:
            status_widgets.first(Static).update(self.selection_status_text)

    @staticmethod
    def option_id(index: int) -> str:
        return f"line-{index}"

    def refresh_line(self, index: int) -> None:
        """Refresh one visible option from its backing data."""
        self.query_one("#line-list", OptionList).replace_option_prompt(
            self.option_id(index), self.format_line(index)
        )

    def replace_selection(self, anchor_index: int, target_index: int) -> None:
        """Select the contiguous inclusive range between an anchor and target."""
        first_index = min(anchor_index, target_index)
        last_index = max(anchor_index, target_index)
        new_selected_indices = set(range(first_index, last_index + 1))
        changed_indices = self.selected_indices ^ new_selected_indices
        self.selection_anchor_index = anchor_index
        self.selected_indices = new_selected_indices
        for index in changed_indices:
            self.refresh_line(index)
        self.update_selection_status()

    def collapse_selection(self, index: int) -> None:
        """Select only one index and make it the anchor for future extension."""
        self.replace_selection(index, index)

    def collapse_current_selection(self) -> None:
        """Collapse selection to the current highlight, including at list edges."""
        if self.selected_index is not None:
            self.collapse_selection(self.selected_index)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """Move or extend selection to the newly highlighted row."""
        previous_selected_index = self.selected_index
        self.selected_index = event.option_index
        option_list = self.query_one("#line-list", NonWrappingOptionList)
        if option_list.extend_selection and self.selection_anchor_index is not None:
            self.replace_selection(self.selection_anchor_index, self.selected_index)
        else:
            self.collapse_selection(self.selected_index)
        if (
            previous_selected_index is not None
            and previous_selected_index in self.selected_indices
        ):
            self.refresh_line(previous_selected_index)
        self.refresh_line(self.selected_index)

    def action_quit_editor(self) -> None:
        if self.find_active:
            self.close_find()
            return
        self.request_exit()

    def action_open_find(self) -> None:
        """Open find at the current row, retaining and selecting its query."""
        find_input = self.query_one("#find-input", Input)
        if not self.find_active:
            self.find_active = True
            self.find_search_start_index = self.selected_index
            self.find_query_submitted = False
            status_bar = self.query_one("#status-bar", Horizontal)
            status_bar.display = False
            for child in status_bar.children:
                child.display = False
            self.query_one("#find-bar", Horizontal).display = True
            self.query_one("#find-result", Static).update("")
        find_input.focus()
        find_input.select_all()

    def close_find(self) -> None:
        """Hide the find bar and return keyboard control to the content list."""
        if not self.find_active:
            return
        previous_match_index = self.find_match_index
        self.find_match_index = None
        self.find_active = False
        self.query_one("#find-bar", Horizontal).display = False
        status_bar = self.query_one("#status-bar", Horizontal)
        status_bar.display = True
        for child in status_bar.children:
            child.display = True
        self.query_one("#line-list", OptionList).focus()
        if previous_match_index is not None:
            self.refresh_line(previous_match_index)

    def find_relative_match(
        self, match_indices: list[int], direction: int
    ) -> int | None:
        """Find a match in one direction from the current search start."""
        if not match_indices:
            return None
        phrase_count = len(self.phrase_indices)
        search_start = self.find_search_start_index
        if search_start is None or not 0 <= search_start < phrase_count:
            return match_indices[0]
        match_index_set = set(match_indices)
        indices = (
            (search_start + (direction * offset)) % phrase_count
            for offset in range(1, phrase_count + 1)
        )
        return next((index for index in indices if index in match_index_set), None)

    def find_match_indices(self, query: str) -> list[int]:
        """Return visible rows containing a case-insensitive literal query."""
        if not query:
            return []
        folded_query = query.casefold()
        return [
            index
            for index, phrase_index in enumerate(self.phrase_indices)
            if folded_query in self.find_text(phrase_index).casefold()
        ]

    def on_input_changed(self, event: Input.Changed) -> None:
        """Clear stale match feedback without moving the line selection."""
        if event.input.id == "find-input" and self.find_active:
            self.find_query_submitted = False
            previous_match_index = self.find_match_index
            self.find_match_index = None
            self.query_one("#find-result", Static).update("")
            if previous_match_index is not None:
                self.refresh_line(previous_match_index)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Advance to the next match while retaining find focus."""
        if event.input.id != "find-input" or not self.find_active:
            return
        self.find_query_submitted = True
        self.advance_find(event.value, 1)

    def action_find_previous(self) -> None:
        """Move backward after the current query has been submitted."""
        if not self.find_active or not self.find_query_submitted:
            return
        self.advance_find(self.query_one("#find-input", Input).value, -1)

    def show_find_match(
        self, match_index: int, match_number: int, match_count: int
    ) -> None:
        """Select and present one find result while retaining input focus."""
        previous_match_index = self.find_match_index
        self.find_match_index = match_index
        self.query_one("#line-list", OptionList).highlighted = match_index
        if previous_match_index is not None and previous_match_index != match_index:
            self.refresh_line(previous_match_index)
        self.refresh_line(match_index)
        self.query_one("#find-result", Static).update(
            f"{match_number} of {match_count}"
        )

    def advance_find(self, query: str, direction: int) -> None:
        """Advance through matches and update right-aligned feedback."""
        match_indices = self.find_match_indices(query)
        if not match_indices:
            self.query_one("#find-result", Static).update("No matches")
            return
        self.find_search_start_index = self.selected_index
        match_index = self.find_relative_match(match_indices, direction)
        if match_index is not None:
            match_number = match_indices.index(match_index) + 1
            self.show_find_match(match_index, match_number, len(match_indices))

    def on_input_blurred(self, event: Input.Blurred) -> None:
        if event.input.id == "find-input":
            self.close_find()

    def on_click(self, event: events.Click) -> None:
        """Dismiss find mode for clicks anywhere outside its text input."""
        if self.find_active and event.widget is not self.query_one(
            "#find-input", Input
        ):
            self.close_find()

    def action_select_all(self) -> None:
        """Select every visible row while retaining the current row as anchor."""
        if self.find_active:
            self.query_one("#find-input", Input).select_all()
            return
        if self.selected_index is None:
            return
        new_selected_indices = set(range(len(self.phrase_indices)))
        changed_indices = self.selected_indices ^ new_selected_indices
        self.selection_anchor_index = self.selected_index
        self.selected_indices = new_selected_indices
        for index in changed_indices:
            self.refresh_line(index)
        self.update_selection_status()

    def mutate_selected_items(self, mutator: SelectedItemMutator) -> list[int]:
        """Mutate selected visible rows through their Project phrase indices."""
        if self.find_active or not self.selected_indices:
            return []
        changed_indices = [
            index
            for index in sorted(self.selected_indices)
            if mutator(index, self.phrase_indices[index])
        ]
        for index in changed_indices:
            self.refresh_line(index)
        self.collapse_current_selection()
        return changed_indices

    def replace_phrase_indices(
        self, phrase_indices: Iterable[int], selected_phrase_index: int | None = None
    ) -> None:
        """Replace the visible Project mapping and rebuild the option list."""
        self.phrase_indices = list(phrase_indices)
        if selected_phrase_index in self.phrase_indices:
            self.selected_index = self.phrase_indices.index(selected_phrase_index)
        else:
            self.selected_index = 0 if self.phrase_indices else None
        self.selection_anchor_index = self.selected_index
        self.selected_indices = (
            {self.selected_index} if self.selected_index is not None else set()
        )
        self.find_match_index = None

        option_list = self.query_one("#line-list", OptionList)
        option_list.clear_options()
        option_list.add_options(
            Option(self.format_line(index), id=self.option_id(index))
            for index in range(len(self.phrase_indices))
        )
        option_list.highlighted = self.selected_index
        self.update_selection_status()

    def request_exit(self) -> None:
        """Exit immediately when clean, or confirm concrete staged changes."""
        if not self.has_changes:
            self.exit()
            return
        self.push_screen(self.make_confirmation_dialog(), self.handle_exit_decision)

    def handle_exit_decision(self, decision: ExitDecision | None) -> None:
        """Commit, discard, or retain staged changes based on dialog result."""
        if decision == ExitDecision.CONFIRM:
            self.commit_changes_and_exit()
        elif decision == ExitDecision.DISCARD:
            self.exit()

    def action_ignore_ctrl_q(self) -> None:
        """Override Textual's built-in Ctrl+Q quit binding."""
