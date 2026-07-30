from collections.abc import Callable
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.css.errors import StylesheetError
from textual.widgets import Input, OptionList, Rule, Static
from textual.widgets.option_list import Option

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.textual.save_changes_dialog import (
    ExitDecision,
    SaveChangesDialog,
)
from tts_audiobook_tool.textual.textual_shared import (
    STYLE_ACCENT,
    STYLE_DIM,
    load_css,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import make_error_string, print_feedback

HeaderLine = str | Text


class NonWrappingOptionList(OptionList):
    BINDINGS: ClassVar[list[BindingType]] = [
        *OptionList.BINDINGS,
        Binding("shift+up", "extend_cursor_up", show=False),
        Binding("shift+down", "extend_cursor_down", show=False),
        Binding("shift+pageup", "extend_page_up", show=False),
        Binding("shift+pagedown", "extend_page_down", show=False),
        Binding("shift+home", "extend_first", show=False),
        Binding("shift+end", "extend_last", show=False),
    ]

    def __init__(
        self,
        *content: Option,
        collapse_selection: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        self.extend_selection = False
        self.collapse_selection = collapse_selection
        super().__init__(*content, **kwargs)

    def prepare_navigation(self, extend_selection: bool) -> None:
        """Record navigation mode and collapse selection for unshifted movement."""
        self.extend_selection = extend_selection
        if not extend_selection and self.collapse_selection is not None:
            self.collapse_selection()

    def move_cursor(self, direction: int) -> None:
        """Move to the next enabled option in one direction without wrapping."""
        if self.highlighted is None:
            return
        stop = -1 if direction < 0 else len(self.options)
        for index in range(self.highlighted + direction, stop, direction):
            if not self.options[index].disabled:
                self.highlighted = index
                return

    def action_cursor_up(self) -> None:
        """Move to the previous enabled option without wrapping at the top."""
        self.prepare_navigation(False)
        self.move_cursor(-1)

    def action_cursor_down(self) -> None:
        """Move to the next enabled option without wrapping at the bottom."""
        self.prepare_navigation(False)
        self.move_cursor(1)

    def action_page_up(self) -> None:
        self.prepare_navigation(False)
        super().action_page_up()

    def action_page_down(self) -> None:
        self.prepare_navigation(False)
        super().action_page_down()

    def action_first(self) -> None:
        self.prepare_navigation(False)
        super().action_first()

    def action_last(self) -> None:
        self.prepare_navigation(False)
        super().action_last()

    def action_extend_cursor_up(self) -> None:
        self.prepare_navigation(True)
        self.move_cursor(-1)

    def action_extend_cursor_down(self) -> None:
        self.prepare_navigation(True)
        self.move_cursor(1)

    def action_extend_page_up(self) -> None:
        self.prepare_navigation(True)
        super().action_page_up()

    def action_extend_page_down(self) -> None:
        self.prepare_navigation(True)
        super().action_page_down()

    def action_extend_first(self) -> None:
        self.prepare_navigation(True)
        super().action_first()

    def action_extend_last(self) -> None:
        self.prepare_navigation(True)
        super().action_last()

    async def _on_click(self, event: events.Click) -> None:
        """Extend from the selection anchor on Shift+click; otherwise collapse."""
        self.prepare_navigation(event.shift)
        await super()._on_click(event)


class VoiceLineEditorTextualApp(App[None]):
    CSS = load_css("textual_shared.css", "voice_line_editor.css")
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "quit_editor", "Quit", show=False),
        Binding("ctrl+q", "ignore_ctrl_q", show=False, priority=True),
        Binding("ctrl+a", "select_all", show=False, priority=True),
        Binding("ctrl+f", "open_find", show=False, priority=True),
        # Many terminal emulators report Shift+Enter as plain Enter, so this
        # binding only works where the terminal emits a distinct key sequence.
        Binding("shift+enter", "find_previous", show=False, priority=True),
        *(
            Binding(str(number), f"assign_voice({number - 1})", show=False)
            for number in range(1, 10)
        ),
    ]

    def __init__(
        self,
        project: Project,
        voice_sample_count: int | None = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.original_voice_indices = [
            phrase_group.voice_index for phrase_group in self.project.phrase_groups
        ]
        self.staged_voice_indices = list(self.original_voice_indices)
        self.did_save_changes = False
        self.save_error = ""
        if voice_sample_count is None:
            voice_sample_count = len(
                ProjectVoiceUtil.get_voice_values(self.project, Tts.get_type())
            )
        highest_voice_key = min(max(voice_sample_count, 1), 9)
        self.header_lines = [
            Text("Edit voice selections", style=STYLE_ACCENT),
            Text(
                f"- Use number keys [1] to [{highest_voice_key}] to set voice sample for selected text line/s",
                style=STYLE_DIM,
            ),
            Text(
                "- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]",
                style=STYLE_DIM,
            ),
            Text(
                "- Select multiple lines by holding [SHIFT] + navigation keys",
                style=STYLE_DIM,
            ),
            Text("- Press [ESC] to finish   - Press [CTRL-F] to find text", style=STYLE_DIM),
        ]
        self.selected_index = 0 if self.project.phrase_groups else None
        self.selection_anchor_index = self.selected_index
        self.selected_indices = (
            {self.selected_index} if self.selected_index is not None else set()
        )
        # Typing only changes the query. Enter submits it and enables backward
        # navigation; each submitted search starts from the currently selected row.
        self.find_active = False
        self.find_search_start_index: int | None = None
        self.find_query_submitted = False
        self.find_match_index: int | None = None
        self.configure()

    def configure(self) -> None:
        """Configure Textual before the full-screen interface starts."""
        self.theme = "ansi-dark"

    @property
    def has_changes(self) -> bool:
        return self.staged_voice_indices != self.original_voice_indices

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
                for index in range(len(self.project.phrase_groups))
            ),
            id="line-list",
            markup=False,
            compact=True,
            collapse_selection=self.collapse_current_selection,
        )
        yield Static(self.selection_status_text, id="selection-status", markup=False)
        yield Horizontal(
            Static("Find: ", id="find-label", markup=False),
            Input(id="find-input", compact=True, select_on_focus=False),
            Static("", id="find-result", markup=False),
            id="find-bar",
        )

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
        """Refresh the status pinned to the terminal's bottom row."""
        status_widgets = self.query("#selection-status")
        if status_widgets:
            status_widgets.first(Static).update(self.selection_status_text)

    def format_line(self, index: int) -> Text:
        """Format one row, styling selected rows except for the active row."""
        phrase_group = self.project.phrase_groups[index]
        voice_index = self.staged_voice_indices[index]
        voice_number = max(voice_index + 1, 1)
        voice_values = ProjectVoiceUtil.get_voice_values(self.project, Tts.get_type())
        # Keep showing the stored voice number, but flag stale selections after voices are removed.
        voice_status = " *OUT OF RANGE*" if voice_index >= len(voice_values) else ""
        content = (
            f"[{index + 1:05d}] [Voice sample {voice_number}{voice_status}] "
            f"{phrase_group.presentable_text}"
        )
        is_find_match = index == self.find_match_index
        is_inactive_selection = (
            index in self.selected_indices and index != self.selected_index
        )
        style = f"{STYLE_DIM} reverse" if is_find_match or is_inactive_selection else ""
        return Text(content, style=style, no_wrap=True, overflow="ellipsis")

    @staticmethod
    def option_id(index: int) -> str:
        return f"line-{index}"

    def refresh_line(self, index: int) -> None:
        """Refresh one option from its backing data."""
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
            self.query_one("#selection-status", Static).display = False
            self.query_one("#find-bar", Horizontal).display = True
            self.query_one("#find-result", Static).update("")
        find_input.focus()
        find_input.select_all()

    def close_find(self) -> None:
        """Hide the find bar and return keyboard control to the line list."""
        if not self.find_active:
            return
        previous_match_index = self.find_match_index
        self.find_match_index = None
        self.find_active = False
        self.query_one("#find-bar", Horizontal).display = False
        self.query_one("#selection-status", Static).display = True
        self.query_one("#line-list", OptionList).focus()
        if previous_match_index is not None:
            self.refresh_line(previous_match_index)

    def find_relative_match(
        self, match_indices: list[int], direction: int
    ) -> int | None:
        """Find a match in one direction from the current search start."""
        if not match_indices:
            return None
        phrase_count = len(self.project.phrase_groups)
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
        """Return all phrase indices containing a case-insensitive literal query."""
        if not query:
            return []
        folded_query = query.casefold()
        return [
            index
            for index, phrase_group in enumerate(self.project.phrase_groups)
            if folded_query in phrase_group.presentable_text.casefold()
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

    def request_exit(self) -> None:
        """Exit immediately when clean, or ask what to do with staged edits."""
        if not self.has_changes:
            self.exit()
            return
        self.push_screen(SaveChangesDialog(), self.handle_exit_decision)

    def handle_exit_decision(self, decision: ExitDecision | None) -> None:
        """Handle the result of the staged-change confirmation dialog."""
        if decision == ExitDecision.SAVE:
            self.save_changes_and_exit()
        elif decision == ExitDecision.DISCARD:
            self.exit()

    def save_changes_and_exit(self) -> None:
        """Apply staged values and persist them, rolling memory back on failure."""
        phrase_groups = self.project.phrase_groups
        if len(phrase_groups) != len(self.staged_voice_indices):
            self.save_error = "Save failed: project text changed while editing"
            self.exit()
            return

        for phrase_group, voice_index in zip(
            phrase_groups, self.staged_voice_indices, strict=True
        ):
            phrase_group.voice_index = voice_index

        try:
            error = ProjectTextIOUtil.save_phrase_groups(self.project)
        except Exception as exception:
            error = make_error_string(exception)
        if error:
            for phrase_group, voice_index in zip(
                phrase_groups, self.original_voice_indices, strict=True
            ):
                phrase_group.voice_index = voice_index
            self.save_error = f"Save failed: {error}"
        else:
            self.did_save_changes = True
        self.exit()

    def action_select_all(self) -> None:
        """Select every line while retaining the current line as the anchor."""
        if self.find_active:
            self.query_one("#find-input", Input).select_all()
            return
        if self.selected_index is None:
            return
        new_selected_indices = set(range(len(self.project.phrase_groups)))
        changed_indices = self.selected_indices ^ new_selected_indices
        self.selection_anchor_index = self.selected_index
        self.selected_indices = new_selected_indices
        for index in changed_indices:
            self.refresh_line(index)
        self.update_selection_status()

    def action_assign_voice(self, voice_index: int) -> None:
        """Assign an available voice sample to all selected phrase groups."""
        if self.find_active:
            return
        if not self.selected_indices:
            return
        voice_values = ProjectVoiceUtil.get_voice_values(self.project, Tts.get_type())
        if voice_index >= len(voice_values):
            return
        changed_indices = [
            index
            for index in self.selected_indices
            if self.staged_voice_indices[index] != voice_index
        ]
        for index in changed_indices:
            self.staged_voice_indices[index] = voice_index
            self.refresh_line(index)
        self.collapse_current_selection()

    def action_ignore_ctrl_q(self) -> None:
        """Override Textual's built-in Ctrl+Q quit binding."""

    @classmethod
    def start(cls, project: Project) -> None:
        """Run an editor for a project and report its save result."""
        app = cls(project)
        app.run(inline=False)
        if isinstance(app._exception, StylesheetError):
            print_feedback("Couldn't load textual css", is_error=True)
        elif app.save_error:
            print_feedback(app.save_error, is_error=True)
        elif app.did_save_changes:
            print_feedback("Saved changes", long_pause=True)
