from collections.abc import Iterable
from typing import ClassVar

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from tts_audiobook_tool.constants import COL_ACCENT
from tts_audiobook_tool.textual.multi_select_dialog import (
    DIALOG_CHROME_HEIGHT,
    MultiSelectDialog,
)


class CutPointSelectionDialog(ModalScreen[set[int] | None]):
    """Select one contiguous range bounded by the project's section markers."""

    AUTO_FOCUS = "#multi-select-list"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", show=False),
        Binding("space", "select_highlighted", show=False),
    ]
    # Keep the parent chooser's dimensions, spacing and list styling.
    CSS = MultiSelectDialog.CSS.replace("MultiSelectDialog", "CutPointSelectionDialog")

    def __init__(
        self,
        line_count: int,
        cut_points: Iterable[int],
        *,
        marker_label: str = "section marker",
    ) -> None:
        super().__init__()
        self.heading = f"Select lines using {marker_label}s"
        points = sorted({point for point in cut_points if 0 < point < line_count})
        boundaries = [0, *points, line_count]
        self.ranges = [
            range(start, end)
            for start, end in zip(boundaries, boundaries[1:])
            if start < end
        ]
        self.labels: list[str] = []
        for index, lines in enumerate(self.ranges):
            if not points:
                label = "All lines"
            elif index == 0:
                label = f"Before {marker_label} 1"
            elif index == len(points):
                label = f"{marker_label.capitalize()} {index} to end"
            else:
                label = f"{marker_label.capitalize()} {index} up to {marker_label} {index + 1}"
            self.labels.append(f"{label} (lines {lines.start + 1}-{lines.stop})")

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                Text.from_ansi(f"{COL_ACCENT}{self.heading}\n"),
                id="multi-select-title",
                markup=False,
            ),
            OptionList(
                *(Option(Text(label)) for label in self.labels),
                id="multi-select-list",
                compact=True,
            ),
            id="multi-select-dialog",
        )

    def on_mount(self) -> None:
        # Border and horizontal padding consume six cells; allow a scrollbar.
        self.query_one("#multi-select-dialog", Vertical).styles.width = max(
            62, *(cell_len(label) + 7 for label in [self.heading, *self.labels])
        )
        self.query_one("#multi-select-dialog", Vertical).styles.max_width = "100%"
        self.fit_list_height()
        self.query_one("#multi-select-list", OptionList).highlighted = 0

    def on_resize(self, _event: events.Resize) -> None:
        self.fit_list_height()

    def fit_list_height(self) -> None:
        self.query_one("#multi-select-list", OptionList).styles.max_height = max(
            1, self.size.height - DIALOG_CHROME_HEIGHT
        )

    def choose(self, option_index: int | None) -> None:
        if option_index is not None and 0 <= option_index < len(self.ranges):
            self.dismiss(set(self.ranges[option_index]))

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.choose(event.option_index)

    def action_select_highlighted(self) -> None:
        self.choose(self.query_one("#multi-select-list", OptionList).highlighted)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        dialog = self.query_one("#multi-select-dialog", Vertical)
        if event.widget is None or (
            event.widget is not dialog and dialog not in event.widget.ancestors
        ):
            self.dismiss(None)
