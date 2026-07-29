from enum import Enum
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ExitDecision(Enum):
    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


class SaveChangesDialog(ModalScreen[ExitDecision]):

    STYLE_DIM = "#888888"

    AUTO_FOCUS = "#save"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "save", "Save", show=False),
        Binding("d", "discard", "Discard", show=False),
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
        height: 9;
        padding: 1 2;
        border: round $dialog-dim;
        background: ansi_default;
    }

    #save-changes-question {
        height: 2;
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

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                "Save changes before exiting?",
                id="save-changes-question",
                markup=False,
            ),
            Horizontal(
                Button(Content.from_text("[S]ave", markup=False), id="save"),
                Button(
                    Content.from_text("[D]iscard", markup=False),
                    id="discard",
                ),
                Button("Cancel", id="cancel"),
                id="save-changes-buttons",
            ),
            id="save-changes-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decisions = {
            "save": ExitDecision.SAVE,
            "discard": ExitDecision.DISCARD,
            "cancel": ExitDecision.CANCEL,
        }
        if event.button.id in decisions:
            self.dismiss(decisions[event.button.id])

    def action_save(self) -> None:
        self.dismiss(ExitDecision.SAVE)

    def action_discard(self) -> None:
        self.dismiss(ExitDecision.DISCARD)

    def action_cancel(self) -> None:
        self.dismiss(ExitDecision.CANCEL)
