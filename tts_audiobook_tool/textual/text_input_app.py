import platform
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.css.errors import StylesheetError
from textual.widgets import Rule, Static, TextArea

from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual.save_changes_dialog import (
    ExitDecision,
    SaveChangesDialog,
)
from tts_audiobook_tool.textual.textual_shared import TEXTUAL_SHARED_CSS, can_textual


class TextInputTextualApp(App[str]):
    """Full-screen multiline text input which submits or discards on exit."""

    CSS = TEXTUAL_SHARED_CSS + """
    #text-input {
        height: 1fr;
        border: none;
    }

    #text-input:focus {
        border: none;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "finish", "Finish", show=False),
        Binding("ctrl+a", "select_all", "Select all", show=False, priority=True),
        Binding("ctrl+q", "ignore_ctrl_q", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.theme = "ansi-dark"

    def compose(self) -> ComposeResult:
        paste_shortcut = (
            "[COMMAND+V]" if platform.system() == "Darwin" else "[CTRL+SHIFT+V]"
        )
        header_lines = (
            f"{COL_ACCENT}Enter/paste text of any length",
            f"{COL_DIM}Press {paste_shortcut} to paste from the system clipboard",
            f"{COL_DIM}Press [{COL_ACCENT}ESC{COL_DIM}] to finish",
        )
        yield Vertical(
            *(
                Static(
                    Text.from_ansi(f"{Ansi.RESET}{line}"),
                    id=f"header-line-{index}",
                    classes="header-line",
                    markup=False,
                )
                for index, line in enumerate(header_lines)
            ),
            id="header",
        )
        yield Rule(id="header-divider")
        yield TextArea(id="text-input", theme="vscode_dark")

    def on_mount(self) -> None:
        self.query_one("#header", Vertical).styles.height = 3
        self.query_one("#text-input", TextArea).focus()

    def action_finish(self) -> None:
        text = self.query_one("#text-input", TextArea).text.strip()
        if not text:
            self.exit("")
            return
        self.push_screen(
            SaveChangesDialog(["Save changes?"]),
            lambda decision: self._handle_exit_decision(decision, text),
        )

    def _handle_exit_decision(
        self, decision: ExitDecision | None, text: str
    ) -> None:
        if decision == ExitDecision.CONFIRM:
            self.exit(text)
        elif decision == ExitDecision.DISCARD:
            self.exit("")
        else:
            self.query_one("#text-input", TextArea).focus()

    def action_select_all(self) -> None:
        """Give Ctrl+A the conventional GUI-editor select-all behavior."""
        self.query_one("#text-input", TextArea).select_all()

    def action_ignore_ctrl_q(self) -> None:
        """Override Textual's built-in Ctrl+Q quit binding."""


def run_text_input_app() -> str:
    """Run multiline input, retaining stdin fallback outside full-screen terminals."""
    if not can_textual():
        return ask.ask_multiline()

    app = TextInputTextualApp()
    try:
        result = app.run(inline=False)
    except Exception as exception:
        ask.ask_error(f"{type(exception).__name__}: {exception}")
        return ""

    exception = app._exception
    if isinstance(exception, StylesheetError):
        ask.ask_error("Couldn't load textual css")
        return ""
    if exception is not None:
        ask.ask_error(f"{type(exception).__name__}: {exception}")
        return ""
    if result is None:
        ask.ask_error("Text input closed without returning a result")
        return ""
    return result
