from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


class SegmentInfoDialog(ModalScreen[None]):
    """ Displays formatted validation info of a generated segment. """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
    ]

    CSS = """
    SegmentInfoDialog {
        align: center middle;
        background: transparent;
    }

    #segment-info-dialog {
        width: 100%;
        height: auto;
        max-height: 100%;
        margin: 2 0;
        padding: 1 2;
        border-top: round #888888;
        border-bottom: round #888888;
        background: ansi_default;
    }

    #segment-info-scroll {
        height: auto;
        max-height: 1fr;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
    }

    #segment-info-content {
        height: auto;
    }
    """

    def __init__(self, info_text: str) -> None:
        super().__init__()
        self.info_text = Text.from_ansi(info_text)

    def compose(self) -> ComposeResult:
        yield Vertical(
            VerticalScroll(
                Static(self.info_text, id="segment-info-content", markup=False),
                id="segment-info-scroll",
            ),
            id="segment-info-dialog",
        )

    def action_close(self) -> None:
        self.dismiss()

    def on_click(self, event: events.Click) -> None:
        """Close only when the modal backdrop, rather than the dialog, is clicked."""
        dialog = self.query_one("#segment-info-dialog", Vertical)
        clicked_widget = event.widget
        if clicked_widget is None or (
            clicked_widget is not dialog and dialog not in clicked_widget.ancestors
        ):
            self.dismiss()
