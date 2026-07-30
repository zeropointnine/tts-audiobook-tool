from enum import Enum
from typing import ClassVar

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
        copy_line_1: str = "Save changes before exiting?",
        copy_line_2: str = "",
    ) -> None:
        super().__init__()
        self.copy_line_1 = copy_line_1
        self.copy_line_2 = copy_line_2

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                self.copy_line_1,
                id="save-changes-copy-line-1",
                classes="save-changes-copy",
                markup=False,
            ),
            *(
                [
                    Static(
                        self.copy_line_2,
                        id="save-changes-copy-line-2",
                        classes="save-changes-copy",
                        markup=False,
                    )
                ]
                if self.copy_line_2
                else []
            ),
            Horizontal(
                Button(Content.from_text("[Y]es", markup=False), id="yes"),
                Button(Content.from_text("[N]o", markup=False), id="no"),
                Button("Cancel", id="cancel"),
                id="save-changes-buttons",
            ),
            id="save-changes-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decisions = {
            "yes": ExitDecision.CONFIRM,
            "no": ExitDecision.DISCARD,
            "cancel": ExitDecision.CANCEL,
        }
        if event.button.id in decisions:
            self.dismiss(decisions[event.button.id])

    def action_confirm(self) -> None:
        self.dismiss(ExitDecision.CONFIRM)

    def action_discard(self) -> None:
        self.dismiss(ExitDecision.DISCARD)

    def action_cancel(self) -> None:
        self.dismiss(ExitDecision.CANCEL)
