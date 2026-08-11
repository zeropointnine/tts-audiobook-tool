from collections.abc import Callable
from enum import Enum
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


class SegmentInfoAction(Enum):
    DELETE_GENERATED = "delete_generated"
    QUICK_GENERATE = "quick_generate"


class SegmentInfoDialog(ModalScreen[SegmentInfoAction | None]):
    """ Displays formatted validation info of a generated segment. """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding("p", "play_sound", show=False),
        Binding("x", "delete_generated", show=False),
        Binding("q", "quick_generate", show=False),
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
        margin: 2 2;
        padding: 1 2;
        border: round #888888;
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

    def __init__(self, info_text: str, play_sound: Callable[[], None]) -> None:
        super().__init__()
        self.info_text = Text.from_ansi(info_text)
        self.play_sound = play_sound

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

    def action_play_sound(self) -> None:
        """Toggle playback without closing the segment-info dialog."""
        self.play_sound()

    def action_delete_generated(self) -> None:
        """Close the dialog and request deletion of its displayed segment."""
        self.dismiss(SegmentInfoAction.DELETE_GENERATED)

    def action_quick_generate(self) -> None:
        """Close the dialog and request regeneration of its displayed segment."""
        self.dismiss(SegmentInfoAction.QUICK_GENERATE)

    def on_click(self, event: events.Click) -> None:
        """Close only when the modal backdrop, rather than the dialog, is clicked."""
        dialog = self.query_one("#segment-info-dialog", Vertical)
        clicked_widget = event.widget
        if clicked_widget is None or (
            clicked_widget is not dialog and dialog not in clicked_widget.ancestors
        ):
            self.dismiss()
