from enum import Enum
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

class ExitDecision(Enum):
    CONFIRM = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


class SaveChangesDialog(ModalScreen[ExitDecision]):

    STYLE_DIM = "#888888"

    AUTO_FOCUS = "#yes"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "confirm", "Yes", show=False),
        Binding("n", "discard", "No", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = f"""
    $dialog-dim: {STYLE_DIM};
    """ + """
    SaveChangesDialog {
        align: center middle;
        background: transparent;
    }

    #save-changes-dialog {
        width: 62;
        height: auto;
        padding: 1 2;
        border: round $dialog-dim;
        background: ansi_default;
    }

    .save-changes-copy {
        height: 1;
        content-align: center middle;
    }

    .save-changes-warning {
        height: auto;
        text-align: center;
    }

    #save-changes-buttons {
        height: 3;
        align-horizontal: center;
    }

    #save-changes-buttons Button {
        min-width: 16;
        margin: 0 1;
        color: $dialog-dim;
        background: ansi_default;
        border: round $dialog-dim;
        text-style: none;
    }

    #save-changes-buttons Button:hover,
    #save-changes-buttons Button:focus,
    #save-changes-buttons Button.-active {
        color: ansi_default;
        background: ansi_default;
        border: round $dialog-dim;
        text-style: none;
        background-tint: transparent;
        tint: transparent;
    }
    """

    def __init__(
        self,
        copy_lines: list[str] | None = None,
        confirmation_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.copy_lines = (
            ["Save changes before exiting?"] if copy_lines is None else copy_lines
        )
        self.confirmation_enabled = confirmation_enabled

    def compose(self) -> ComposeResult:
        yield Vertical(
            *(
                Static(
                    Text.from_ansi(copy_line),
                    id=f"save-changes-copy-line-{index + 1}",
                    classes="save-changes-copy",
                    markup=False,
                )
                for index, copy_line in enumerate(self.copy_lines)
            ),
            Horizontal(
                Button(
                    Content.from_text("[Y]es", markup=False),
                    id="yes",
                    disabled=not self.confirmation_enabled,
                ),
                Button(Content.from_text("[N]o", markup=False), id="no"),
                id="save-changes-buttons",
            ),
            id="save-changes-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decisions = {
            "yes": ExitDecision.CONFIRM,
            "no": ExitDecision.DISCARD,
        }
        if event.button.id == "yes" and not self.confirmation_enabled:
            return
        if event.button.id in decisions:
            self.dismiss(decisions[event.button.id])

    def action_confirm(self) -> None:
        if self.confirmation_enabled:
            self.dismiss(ExitDecision.CONFIRM)

    def action_discard(self) -> None:
        self.dismiss(ExitDecision.DISCARD)

    def action_cancel(self) -> None:
        self.dismiss(ExitDecision.CANCEL)
