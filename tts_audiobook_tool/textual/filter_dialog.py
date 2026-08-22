from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM

if TYPE_CHECKING:
    from tts_audiobook_tool.textual.generate_editor import FilterType


class FilterDialog(ModalScreen["FilterType | None"]):
    """Choose which generated-editor lines are visible."""

    BINDINGS: ClassVar[list[BindingType]] = [
        *(
            Binding(str(number), f"select_filter({number - 1})", show=False)
            for number in range(1, 6)
        ),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    FilterDialog {
        align: center middle;
        background: transparent;
    }

    #filter-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #filter-title {
        height: 2;
    }

    .filter-option {
        height: 1;
    }
    """

    def __init__(
        self,
        current_filter: FilterType,
        line_counts: Mapping[FilterType, int],
    ) -> None:
        super().__init__()
        self.current_filter = current_filter
        self.line_counts = line_counts

    def compose(self) -> ComposeResult:
        filter_types = list(type(self.current_filter))
        title = f"{COL_ACCENT}Filter lines"
        yield Vertical(
            Static(Text.from_ansi(title), id="filter-title", markup=False),
            *(
                Static(
                    Text.from_ansi(
                        f"[{number}] {filter_type.menu_label}"
                        f" {COL_DIM}({self.line_counts[filter_type]})"
                        + (
                            " (selected)"
                            if filter_type is self.current_filter
                            else ""
                        )
                    ),
                    id=f"filter-option-{number}",
                    classes="filter-option",
                    markup=False,
                )
                for number, filter_type in enumerate(filter_types, start=1)
            ),
            id="filter-dialog",
        )

    def action_select_filter(self, index: int) -> None:
        filter_types = list(type(self.current_filter))
        if 0 <= index < len(filter_types):
            self.dismiss(filter_types[index])

    def action_cancel(self) -> None:
        self.dismiss(None)
