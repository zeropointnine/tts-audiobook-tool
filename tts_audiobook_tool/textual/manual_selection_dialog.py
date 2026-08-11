from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM, COL_ERROR
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil


class ManualSelectionDialog(ModalScreen[set[int] | None]):
    """Select editor lines by entering one-based line numbers and ranges."""

    AUTO_FOCUS = "#manual-selection-input"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    ManualSelectionDialog {
        align: center middle;
        background: transparent;
    }

    #manual-selection-dialog {
        width: 62;
        height: auto;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #manual-selection-title,
    #manual-selection-example,
    #manual-selection-error {
        height: 1;
    }
    """

    def __init__(self, line_count: int) -> None:
        super().__init__()
        self.line_count = line_count

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                Text.from_ansi(f"{COL_ACCENT}Enter line selection manually"),
                id="manual-selection-title",
                markup=False,
            ),
            Static(
                Text.from_ansi(f'{COL_DIM}Eg, "5-100, 105"'),
                id="manual-selection-example",
                markup=False,
            ),
            Input(id="manual-selection-input", compact=True),
            Static("", id="manual-selection-error", markup=False),
            id="manual-selection-dialog",
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "manual-selection-input":
            return
        line_indices, errors = RangeStringUtil.parse_ranges_string(
            event.value, self.line_count
        )
        if errors:
            self.query_one("#manual-selection-error", Static).update(
                Text.from_ansi(f"{COL_ERROR}{'; '.join(errors)}")
            )
            return
        self.dismiss(line_indices)

    def action_cancel(self) -> None:
        self.dismiss(None)
