from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.css.errors import StylesheetError
from textual.visual import VisualType
from textual.widget import Widget
from textual.widgets import Input, OptionList, Rule, Static
from textual.widgets.option_list import Option
from textual.timer import Timer

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual.manual_selection_dialog import ManualSelectionDialog
from tts_audiobook_tool.textual.save_changes_dialog import (
    ExitDecision,
    SaveChangesDialog,
)
from tts_audiobook_tool.textual.textual_shared import (
    CONTENT_TEXTUAL_APP_CSS,
    HangingIndentText,
    NonWrappingOptionList,
    OptionReconcileItem,
    STYLE_DIM,
    TEXTUAL_SHARED_CSS,
    can_textual,
)

SelectedItemMutator = Callable[[int, int], bool]
TOAST_DURATION_SECONDS = 1.5
EditorResultT = TypeVar("EditorResultT")


@dataclass(frozen=True)
class EditorClosed:
    """The editor closed without committing a change."""


@dataclass(frozen=True)
class EditorSaved:
    """The editor committed and persisted its staged changes."""


@dataclass(frozen=True)
class EditorSaveFailed:
    """The editor could not persist its staged changes."""

    error: str


@dataclass(frozen=True)
class ContentAppCompleted(Generic[EditorResultT]):
    """A Textual editor completed with its domain result."""

    result: EditorResultT


@dataclass(frozen=True)
class ContentAppUnavailable:
    """The terminal environment cannot host the Textual editor."""

    message: str


@dataclass(frozen=True)
class ContentAppStylesheetFailed:
    """Textual could not load the editor stylesheet."""

    message: str


@dataclass(frozen=True)
class ContentAppFailed:
    """The Textual app stopped because of an unexpected exception."""

    message: str


@dataclass(frozen=True)
class ContentAppMissingResult:
    """The Textual app stopped without returning its required domain result."""

    message: str


ContentAppRunResult = (
    ContentAppCompleted[EditorResultT]
    | ContentAppUnavailable
    | ContentAppStylesheetFailed
    | ContentAppFailed
    | ContentAppMissingResult
)


class ContentTextualApp(App[EditorClosed | EditorResultT], Generic[EditorResultT]):
    """Base app for phrase-group-backed, full-screen content editors.

    The app owns the shared Textual shell: header and status presentation,
    visible-row composition, single/range selection, find navigation, and the
    save/discard exit workflow. Visible list positions are intentionally kept
    separate from ``Project.phrase_groups`` indices through ``phrase_indices``;
    this lets concrete editors filter or reorder rows without changing the
    project's domain ordering.

    Subclasses provide domain behavior through hooks such as ``format_line``,
    ``initialize_content``, ``has_changes``, and ``commit_changes_and_exit``.
    Content may be installed immediately or deferred until after mounting, so
    expensive editor-specific snapshots do not delay construction of the UI.
    Mutations remain staged by the subclass while this base coordinates row
    refreshes, selection state, confirmation dialogs, and typed exit results.
    """

    CSS = "\n".join((TEXTUAL_SHARED_CSS, CONTENT_TEXTUAL_APP_CSS))
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "quit_editor", "Quit", show=False),
        Binding("ctrl+q", "ignore_ctrl_q", show=False, priority=True),
        Binding("ctrl+a", "select_all", show=False, priority=True),
        Binding("ctrl+f", "open_find", show=False, priority=True),
        Binding("m", "show_manual_selection", show=False),
        # Many terminal emulators report Shift+Enter as plain Enter, so this
        # binding only works where the terminal emits a distinct key sequence.
        Binding("shift+enter", "find_previous", show=False, priority=True),
    ]

    def __init__(
        self,
        project: Project,
        header_lines: Iterable[str],
        phrase_indices: Iterable[int] | None = None,
        empty_state_text: str = "No items",
        loading_state_text: str | None = None,
        multi_select_enabled: bool = True,
        side_panel_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.project = project
        self.header_lines = list(header_lines)
        self.empty_state_text = empty_state_text
        self.loading_state_text = loading_state_text
        self.multi_select_enabled = multi_select_enabled
        self.side_panel_enabled = side_panel_enabled
        self.content_initialized = loading_state_text is None
        self.phrase_indices = (
            []
            if loading_state_text is not None
            else (
                list(range(len(project.phrase_groups)))
                if phrase_indices is None
                else list(phrase_indices)
            )
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
        self.pinned_text = ""
        self.selected_text = self.selection_status_text
        self.toast_text = ""
        self.toast_timer: Timer | None = None
        self.configure()

    def configure(self) -> None:
        """Configure Textual before the full-screen interface starts."""
        self.theme = "ansi-dark"

    @property
    def has_changes(self) -> bool:
        """Whether a concrete editor has staged a data mutation."""
        return False

    def format_line(self, index: int) -> VisualType:
        """Format a visible row; concrete editors must implement this hook."""
        raise NotImplementedError

    def format_section_list_item(self, text: str, index: int) -> HangingIndentText:
        """Format a structural section row using the shared list presentation."""
        style = f"{STYLE_DIM} reverse" if index == self.find_match_index else ""
        # Rich's line-height measurement doesn't count a final empty line, so
        # two trailing newlines are needed to render one.
        return HangingIndentText.from_ansi(
            f"\n{Ansi.RESET}{text}\n\n",
            content_start=0,
            max_lines=3,
            style=style,
        )

    def initialize_content(self) -> Iterable[int]:
        """Prepare deferred backing state and return the final visible indices."""
        raise NotImplementedError

    def initial_selected_phrase_index(self) -> int | None:
        """Return the phrase to select when deferred content is first installed."""
        return None

    def on_content_loaded(self) -> None:
        """Run subclass work after deferred content has been installed."""

    def load_content(self) -> None:
        """Initialize deferred content once and install every visible row."""
        if self.content_initialized:
            return
        phrase_indices = list(self.initialize_content())
        self.content_initialized = True
        self.update_empty_state_text(self.empty_state_text)
        self.replace_phrase_indices(
            phrase_indices,
            self.initial_selected_phrase_index(),
        )
        self.on_content_loaded()

    def find_text(self, phrase_index: int) -> str:
        """Return the searchable text for a Project phrase group."""
        return self.project.phrase_groups[phrase_index].presentable_text

    def content_line_index(self, item_index: int) -> int | None:
        """Map a backing item to its actionable Project line, if it has one.

        Structural rows such as section headings override this hook to return
        ``None``. The base then excludes those rows from line counts, inactive
        selection styling, manual selection, and content mutations.
        """
        return item_index

    def highlighted_content_line_index(self) -> int | None:
        """Return the highlighted actionable Project line, if any."""
        if self.selected_index is None:
            return None
        return self.content_line_index(self.phrase_indices[self.selected_index])

    def selected_content_line_indices(self) -> set[int]:
        """Return actionable Project lines represented by the current selection."""
        return {
            content_line_index
            for visible_index in self.selected_indices
            if (
                content_line_index := self.content_line_index(
                    self.phrase_indices[visible_index]
                )
            )
            is not None
        }

    def make_confirmation_dialog(self) -> SaveChangesDialog:
        """Build the dialog shown before concrete staged changes are committed."""
        return SaveChangesDialog()

    def commit_changes_and_exit(self) -> None:
        """Commit concrete staged changes; concrete editors must implement this."""
        raise NotImplementedError

    def compose_side_panel(self) -> ComposeResult:
        """Yield optional panel widgets without coupling the shell to their types."""
        yield from ()

    def compose_content_main(self) -> Vertical:
        """Build the shared list and empty-state pane."""
        return Vertical(
            NonWrappingOptionList(
                *(
                    Option(self.format_line(index), id=self.option_id(index))
                    for index in range(len(self.phrase_indices))
                ),
                id="line-list",
                markup=False,
                compact=True,
                collapse_selection=self.collapse_current_selection,
                multi_select_enabled=self.multi_select_enabled,
            ),
            Static(
                (
                    self.empty_state_text
                    if self.content_initialized
                    else self.loading_state_text or self.empty_state_text
                ),
                id="empty-state",
                markup=False,
            ),
            id="content-main",
        )

    def compose(self) -> ComposeResult:
        header = Vertical(
            *(
                Static(
                    Text.from_ansi(f"{Ansi.RESET}{line}"),
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
        content_children: list[Widget] = [self.compose_content_main()]
        if self.side_panel_enabled:
            content_children.extend(
                (
                    Rule(orientation="vertical", id="side-panel-divider"),
                    Vertical(*self.compose_side_panel(), id="side-panel"),
                )
            )
        yield Horizontal(*content_children, id="content-shell")
        yield Horizontal(
            Static(
                self.status_text,
                id="status-line",
                classes=f"status-{self.status_mode}",
                markup=False,
            ),
            id="status-bar",
        )
        yield Horizontal(
            Static(self.find_label_text, id="find-label", markup=False),
            Input(id="find-input", compact=True, select_on_focus=False),
            Static("", id="find-result", markup=False),
            id="find-bar",
        )

    @property
    def find_label_text(self) -> str:
        return "Search text: "

    def on_mount(self) -> None:
        self.sync_empty_state()
        self.update_inactive_selection_style()
        if not self.content_initialized:
            self.call_after_refresh(self.load_content)

    def sync_empty_state(self) -> None:
        """Show either the selectable list or its non-selectable empty state."""
        option_list = self.query_one("#line-list", OptionList)
        empty_state = self.query_one("#empty-state", Static)
        has_items = bool(self.phrase_indices)
        option_list.display = has_items
        empty_state.display = not has_items
        if has_items:
            option_list.focus()
        elif option_list.has_focus:
            self.set_focus(None)

    def update_empty_state_text(self, text: str) -> None:
        """Replace the fallback copy, including in an already-mounted view."""
        self.empty_state_text = text
        if self.is_running:
            self.query_one("#empty-state", Static).update(text)

    def update_header(self, lines: list[str]) -> None:
        """Replace the fixed-height header, truncating overflow and clearing gaps."""
        header_height = len(self.header_lines)
        self.header_lines = [
            *lines[:header_height],
            *("" for _ in range(header_height - len(lines))),
        ]
        if self.is_mounted:
            for index, line in enumerate(self.header_lines):
                self.query_one(f"#header-line-{index}", Static).update(
                    Text.from_ansi(f"{Ansi.RESET}{line}")
                )

    @property
    def selection_status_text(self) -> str:
        """Describe selected content lines, excluding structural rows."""
        selection_count = sum(
            self.content_line_index(self.phrase_indices[index]) is not None
            for index in self.selected_indices
        )
        if selection_count < 2:
            return ""
        return f"{selection_count} lines selected"

    @property
    def status_mode(self) -> str:
        """Return the highest-priority active status layer."""
        if self.toast_timer is not None:
            return "toast"
        if self.selected_text:
            return "selected"
        return "pinned"

    @property
    def status_text(self) -> str:
        """Return the text from the highest-priority active status layer."""
        if self.toast_timer is not None:
            return self.toast_text
        return self.selected_text or self.pinned_text

    def update_status_line(self) -> None:
        """Render the current status layer when the status widget is mounted."""
        status_widgets = self.query("#status-line")
        if not status_widgets:
            return
        status_line = status_widgets.first(Static)
        status_line.update(self.status_text)
        status_line.set_classes(f"status-{self.status_mode}")

    def set_pinned_text(self, text: str) -> None:
        """Set the dim, left-aligned pinned (lowest-priority) status text."""
        self.pinned_text = text
        self.update_status_line()

    def set_selected_text(self, text: str) -> None:
        """Set right-aligned status text which overrides pinned text when non-empty."""
        self.selected_text = text
        self.update_status_line()

    def update_selection_status(self) -> None:
        """Format the current selection into the selected-text status layer."""
        self.set_selected_text(self.selection_status_text)

    def set_toast_text(self, text: str) -> None:
        """Show left-aligned status text for 1.5 seconds, restarting on each call."""
        if self.toast_timer is not None:
            self.toast_timer.stop()
        self.toast_text = text
        self.toast_timer = self.set_timer(
            TOAST_DURATION_SECONDS,
            self.clear_toast_text,
            name="clear-toast-text",
        )
        self.update_status_line()

    def clear_toast_text(self) -> None:
        """Clear toast feedback and reveal the selected or pinned status layer."""
        self.toast_text = ""
        self.toast_timer = None
        self.update_status_line()

    def option_id(self, index: int) -> str:
        return f"line-{index}"

    def refresh_line(self, index: int, *, reflow: bool = True) -> None:
        """Refresh one visible option from its backing data.

        Pass reflow=False when the change is style-only and cannot alter row
        heights, to avoid rebuilding geometry for the full option list.
        """
        self.refresh_lines([index], reflow=reflow)

    def refresh_lines(self, indices: Iterable[int], *, reflow: bool = True) -> None:
        """Refresh several options while invalidating list caches only once."""
        option_list = self.query_one("#line-list", NonWrappingOptionList)
        option_list.replace_option_prompts(
            ((index, self.format_line(index)) for index in indices),
            reflow=reflow,
        )

    def update_inactive_selection_style(self) -> None:
        """Style inactive content lines while leaving structural rows unchanged."""
        inactive_indices = self.selected_indices - (
            {self.selected_index} if self.selected_index is not None else set()
        )
        selectable_inactive_indices = {
            index
            for index in inactive_indices
            if self.content_line_index(self.phrase_indices[index]) is not None
        }
        option_lists = self.query("#line-list")
        if option_lists:
            option_lists.first(NonWrappingOptionList).set_inactive_selection_indices(
                selectable_inactive_indices
            )

    def replace_selection(
        self,
        anchor_index: int,
        target_index: int,
    ) -> None:
        """Select the contiguous inclusive range between an anchor and target."""
        first_index = min(anchor_index, target_index)
        last_index = max(anchor_index, target_index)
        new_selected_indices = set(range(first_index, last_index + 1))
        self.selection_anchor_index = anchor_index
        self.selected_indices = new_selected_indices
        self.update_inactive_selection_style()
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
        self.selected_index = event.option_index
        option_list = self.query_one("#line-list", NonWrappingOptionList)
        if (
            self.multi_select_enabled
            and option_list.extend_selection
            and self.selection_anchor_index is not None
        ):
            self.replace_selection(
                self.selection_anchor_index,
                self.selected_index,
            )
        else:
            self.replace_selection(
                self.selected_index,
                self.selected_index,
            )

    def action_quit_editor(self) -> None:
        if self.find_active:
            self.close_find()
            return
        self.request_exit()

    def action_open_find(self) -> None:
        """Open find at the current row, retaining and selecting its query."""
        # Guard against opening find while a concrete editor is mid-edit (e.g.
        # TextEditor's in-place line editing). This binding is registered with
        # priority=True, so it is resolved by the App before a subclass's
        # on_key handler ever sees the key event - a local prevent_default()
        # there is not enough to stop it.
        if getattr(self, "is_editing", False):
            return
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
            self.refresh_line(previous_match_index, reflow=False)

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
                self.refresh_line(previous_match_index, reflow=False)

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
            self.refresh_lines(
                [previous_match_index, match_index], reflow=False
            )
        else:
            self.refresh_line(match_index, reflow=False)
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
        if getattr(self, "is_editing", False):
            return  # prevents ctrl+a from selecting all segments outside the TextArea
        if self.find_active:
            self.query_one("#find-input", Input).select_all()
            return
        if not self.multi_select_enabled:
            return
        if self.selected_index is None:
            return
        new_selected_indices = set(range(len(self.phrase_indices)))
        self.selection_anchor_index = self.selected_index
        self.selected_indices = new_selected_indices
        self.update_inactive_selection_style()
        self.update_selection_status()

    def manual_selection_line_count(self) -> int:
        """Return the number of one-based lines accepted by manual selection."""
        return len(self.project.phrase_groups)

    def manual_selection_line_index(self, item_index: int) -> int | None:
        """Map one backing item to its zero-based manual line, if selectable."""
        return self.content_line_index(item_index)

    def action_show_manual_selection(self) -> None:
        """Show the manual line-selection dialog for the concrete editor."""
        if (
            not self.multi_select_enabled
            or not self.content_initialized
            or self.find_active
        ):
            return
        line_count = self.manual_selection_line_count()
        if line_count <= 0:
            return
        self.push_screen(
            ManualSelectionDialog(line_count),
            self.handle_manual_selection,
        )

    def handle_manual_selection(self, line_indices: set[int] | None) -> None:
        """Select visible rows mapped to the entered editor line numbers."""
        if line_indices is None:
            return
        matching_rows = [
            (visible_index, manual_line_index)
            for visible_index, item_index in enumerate(self.phrase_indices)
            if (manual_line_index := self.manual_selection_line_index(item_index))
            is not None
            and manual_line_index in line_indices
        ]
        if not matching_rows:
            return

        highest_visible_index = max(matching_rows, key=lambda row: row[1])[0]
        option_list = self.query_one("#line-list", OptionList)
        with option_list.prevent(OptionList.OptionHighlighted):
            option_list.highlighted = highest_visible_index
        self.selected_index = highest_visible_index
        self.selection_anchor_index = highest_visible_index
        self.selected_indices = {visible_index for visible_index, _ in matching_rows}
        self.update_inactive_selection_style()
        self.update_selection_status()
        count = len(matching_rows)
        line_noun = "line" if count == 1 else "lines"
        self.set_toast_text(f"Selected {count} {line_noun}")

    def mutate_selected_items(
        self,
        mutator: SelectedItemMutator,
        *,
        reflow: bool = True,
    ) -> list[int]:
        """Mutate selected content lines, excluding all structural rows."""
        if self.find_active or not self.selected_indices:
            return []
        selected_content_items = [
            (index, content_line_index)
            for index in sorted(self.selected_indices)
            if (
                content_line_index := self.content_line_index(
                    self.phrase_indices[index]
                )
            )
            is not None
        ]
        if not selected_content_items:
            return []
        changed_indices: list[int] = []
        for index, content_line_index in selected_content_items:
            if mutator(index, content_line_index):
                changed_indices.append(index)
        self.refresh_lines(changed_indices, reflow=reflow)
        self.collapse_current_selection()
        return changed_indices

    def replace_phrase_indices(
        self,
        phrase_indices: Iterable[int],
        selected_phrase_index: int | None = None,
        reconcile_items: Iterable[OptionReconcileItem] | None = None,
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

        if not self.is_running:
            return

        option_list = self.query_one("#line-list", NonWrappingOptionList)
        if reconcile_items is None:
            option_list.set_options(
                Option(self.format_line(index), id=self.option_id(index))
                for index in range(len(self.phrase_indices))
            )
        else:
            option_list.reconcile_options(reconcile_items)
        option_list.highlighted = self.selected_index
        self.sync_empty_state()
        self.update_inactive_selection_style()
        self.update_selection_status()

    def should_confirm_exit(self) -> bool:
        """Whether exiting requires a decision from the confirmation dialog."""
        return self.has_changes

    def exit_without_confirmation(self) -> None:
        """Complete an exit which does not require a confirmation dialog."""
        self.exit(EditorClosed())

    def confirm_exit(self) -> None:
        """Complete an exit explicitly confirmed by the user."""
        self.commit_changes_and_exit()

    def discard_exit(self) -> None:
        """Complete an exit after the user declines to commit changes."""
        self.exit(EditorClosed())

    def cancel_exit(self) -> None:
        """Handle cancellation of the exit dialog while keeping the editor open."""

    def request_exit(self) -> None:
        """Apply concrete exit policy, prompting for a decision when required."""
        if not self.should_confirm_exit():
            self.exit_without_confirmation()
            return
        self.push_screen(self.make_confirmation_dialog(), self.handle_exit_decision)

    def handle_exit_decision(self, decision: ExitDecision | None) -> None:
        """Dispatch a dialog result through the concrete editor's exit policy."""
        if decision == ExitDecision.CONFIRM:
            self.confirm_exit()
        elif decision == ExitDecision.DISCARD:
            self.discard_exit()
        elif decision == ExitDecision.CANCEL:
            self.cancel_exit()

    def action_ignore_ctrl_q(self) -> None:
        """Override Textual's built-in Ctrl+Q quit binding."""


def run_content_textual_app(
    app: ContentTextualApp[EditorResultT],
) -> ContentAppRunResult[EditorClosed | EditorResultT]:
    """Run one editor and translate Textual infrastructure into a typed result."""
    if not can_textual():
        return ContentAppUnavailable(
            "The current terminal environment does not support full-screen editor"
        )

    try:
        result = app.run(inline=False)
    except Exception as exception:
        return ContentAppFailed(f"{type(exception).__name__}: {exception}")

    exception = app._exception
    if isinstance(exception, StylesheetError):
        return ContentAppStylesheetFailed("Couldn't load textual css")
    if exception is not None:
        return ContentAppFailed(f"{type(exception).__name__}: {exception}")
    if result is None:
        return ContentAppMissingResult(
            "Textual editor closed without returning a result"
        )
    return ContentAppCompleted(result)