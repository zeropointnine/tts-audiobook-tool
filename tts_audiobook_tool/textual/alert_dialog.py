from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tts_audiobook_tool.constants import COL_ERROR


class AlertDialog(ModalScreen[None]):
    """Show an error alert that can be dismissed with OK or Escape."""

    AUTO_FOCUS = "#ok"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
    ]

    CSS = """
    AlertDialog {
        align: center middle;
        background: transparent;
    }

    #alert-dialog {
        width: 72;
        height: auto;
        max-height: 100%;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #alert-title {
        height: auto;
        text-align: center;
        text-style: bold;
    }

    #alert-copy {
        height: auto;
        margin-top: 1;
        text-align: center;
    }

    #alert-ok {
        height: 3;
        margin-top: 1;
        align-horizontal: center;
    }

    #alert-ok Button {
        min-width: 16;
        color: #888888;
        background: ansi_default;
        border: round #888888;
        text-style: none;
    }

    #alert-ok Button:hover,
    #alert-ok Button:focus,
    #alert-ok Button.-active {
        color: ansi_default;
        background: ansi_default;
        border: round #888888;
        text-style: none;
        background-tint: transparent;
        tint: transparent;
    }
    """

    def __init__(self, title: str | None = None, copy: str | None = None) -> None:
        super().__init__()
        self.title = title
        self.copy = copy

    def compose(self) -> ComposeResult:
        content: list[Static | Vertical] = []
        if self.title:
            content.append(
                Static(
                    Text.from_ansi(f"{COL_ERROR}{self.title}"),
                    id="alert-title",
                    markup=False,
                )
            )
        if self.copy:
            content.append(
                Static(
                    Text.from_ansi(f"{COL_ERROR}{self.copy}"),
                    id="alert-copy",
                    markup=False,
                )
            )
        content.append(
            Vertical(
                Button(Content.from_text("[OK]", markup=False), id="ok"),
                id="alert-ok",
            )
        )
        yield Vertical(*content, id="alert-dialog")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.dismiss()

    def action_close(self) -> None:
        self.dismiss()
