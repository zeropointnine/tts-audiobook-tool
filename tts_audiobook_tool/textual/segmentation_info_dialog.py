from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.project import Project


class SegmentationInfoDialog(ModalScreen[None]):
    """Describe the segmentation settings captured when the text was imported.

    The dialog is informational and dismisses on any key press.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False),
    ]

    CSS = """
    SegmentationInfoDialog {
        align: center middle;
        background: transparent;
    }

    #segmentation-info-dialog {
        width: 76;
        height: auto;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #segmentation-info-copy {
        height: auto;
    }
    """

    def __init__(self, project: Project) -> None:
        super().__init__()
        self.project = project

    def compose(self) -> ComposeResult:
        settings = self.project.book.segmentation_settings
        info_lines = [
            "The text was originally imported using the following segmentation settings:",
            "",
            f"{COL_DIM}Max words per segment:{COL_DEFAULT} {settings.max_words_per_segment}",
            (
                f"{COL_DIM}Segmentation strategy:{COL_DEFAULT} "
                f"{settings.strategy.label}"
            ),
            (
                f"{COL_DIM}Dialog segmentation:{COL_DEFAULT} "
                f"{settings.dialog_segmentation}"
            ),
            (
                f"{COL_DIM}Language code:{COL_DEFAULT} "
                f"{settings.language_code or '(none)'}"
            ),
        ]
        yield Vertical(
            Static(
                Text.from_ansi("\n".join(info_lines)),
                id="segmentation-info-copy",
                markup=False,
            ),
            id="segmentation-info-dialog",
        )

    def on_key(self, event: events.Key) -> None:
        self.dismiss()
        event.stop()

    def action_cancel(self) -> None:
        self.dismiss(None)
