import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.project_support.project_voice_util import VOICE_NAME_MAX_LENGTH

LABEL_UNNAMED = "unnamed"
INPUT_PROMPT = "Enter voice number: "

# Spaces between voice columns
COLUMN_GAP = 4
# The dialog's border (1 per side) plus padding (1 vertical, 2 horizontal per side)
FRAME_WIDTH = 6
FRAME_HEIGHT = 4
# Gap between the dialog and the screen edge (in total, per axis)
SCREEN_MARGIN = 4
# Non-list content rows: title, blank, blank, input
CHROME_ROWS = 4


@dataclass(frozen=True)
class VoiceGridLayout:
    """Placement of the voice list and the resulting dialog size."""

    rows: int
    columns: int  # columns the full list needs
    visible_columns: int  # whole columns that fit the width
    width: int  # dialog outer width
    height: int  # dialog outer height


def make_voice_select_title(selected_line_numbers: Sequence[int]) -> str:
    """'Line 17' for a single one-based line, else 'n lines selected'."""
    if len(selected_line_numbers) == 1:
        return f"Line {selected_line_numbers[0]}"
    return f"{len(selected_line_numbers)} lines selected"


def compute_voice_grid_layout(
    voice_count: int,
    entry_width: int,
    min_content_width: int,
    chrome_rows: int,
    screen_width: int,
    screen_height: int,
) -> VoiceGridLayout:
    """Size the dialog to its content, capped at the screen size minus a margin.

    The list first grows downward; once it reaches the max height it wraps into
    more columns, whose heights are then balanced (eg 14 voices with room for
    13 rows become two columns of 7). Columns which don't entirely fit the max
    width are left off.
    """
    max_content_width = max(1, screen_width - SCREEN_MARGIN - FRAME_WIDTH)
    max_content_height = max(1, screen_height - SCREEN_MARGIN - FRAME_HEIGHT)
    max_rows = max(1, max_content_height - chrome_rows)
    count = max(1, voice_count)
    columns = math.ceil(count / max_rows)
    rows = math.ceil(count / columns)
    visible_columns = min(
        columns, max(1, (max_content_width + COLUMN_GAP) // (entry_width + COLUMN_GAP))
    )
    grid_width = visible_columns * entry_width + (visible_columns - 1) * COLUMN_GAP
    content_width = min(max_content_width, max(grid_width, min_content_width))
    content_height = min(max_content_height, chrome_rows + rows)
    return VoiceGridLayout(
        rows=rows,
        columns=columns,
        visible_columns=visible_columns,
        width=content_width + FRAME_WIDTH,
        height=content_height + FRAME_HEIGHT,
    )


class VoiceSelectDialog(ModalScreen[int | None]):
    """Choose a voice for the selected line/s by typing its number.

    Lists every voice as "[number] name" in columns, highlighting the line's
    current voice (given only when a single line is selected). Dismisses with
    the chosen zero-based voice index, or None when cancelled or left blank.
    """

    AUTO_FOCUS = "#voice-select-input"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    VoiceSelectDialog {
        align: center middle;
        background: transparent;
    }

    #voice-select-dialog {
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #voice-select-title,
    #voice-select-error {
        height: 1;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    #voice-select-list {
        height: auto;
        margin: 1 0;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    #voice-select-input-row {
        height: 1;
    }

    #voice-select-prompt {
        width: auto;
    }

    #voice-select-input {
        width: 1fr;
    }

    #voice-select-error {
        display: none;
    }
    """

    def __init__(
        self,
        title: str,
        voice_names: Sequence[str],
        current_voice_index: int | None = None,
    ) -> None:
        """voice_names has one (possibly blank) entry per selectable voice."""
        super().__init__()
        self.heading = title
        self.voice_names = [name.strip() for name in voice_names]
        self.current_voice_index = current_voice_index
        voice_count = len(self.voice_names)
        self.number_width = len(f"[{voice_count}]")
        name_width = max(
            (len(self.display_name(index)) for index in range(voice_count)),
            default=0,
        )
        # "[1]" padded to the widest number, a space, then the name
        self.entry_width = self.number_width + 1 + name_width
        self.input_width = len(str(voice_count)) + 1
        self.error_text = f"Enter a number from 1 to {voice_count}"
        self.grid_layout: VoiceGridLayout | None = None

    def display_name(self, voice_index: int) -> str:
        name = self.voice_names[voice_index][:VOICE_NAME_MAX_LENGTH]
        return name or LABEL_UNNAMED

    def entry_ansi(self, voice_index: int) -> str:
        """One "[n] name" entry (the number padded to the widest one), padded to
        the column width. The current voice's entry is entirely accented."""
        number_text = f"[{voice_index + 1}]".ljust(self.number_width)
        name = self.display_name(voice_index)
        padding = " " * (self.entry_width - self.number_width - 1 - len(name))
        if voice_index == self.current_voice_index:
            return f"{COL_ACCENT}{number_text} {name}{COL_DEFAULT}{padding}"
        name_color = COL_DEFAULT if self.voice_names[voice_index] else COL_DIM
        return f"{COL_DEFAULT}{number_text} {name_color}{name}{COL_DEFAULT}{padding}"

    def list_text(self, layout: VoiceGridLayout) -> Text:
        """The voices in column-major order, limited to the visible columns."""
        lines: list[str] = []
        for row in range(layout.rows):
            entries = [
                self.entry_ansi(voice_index)
                for column in range(layout.visible_columns)
                if (voice_index := column * layout.rows + row) < len(self.voice_names)
            ]
            lines.append((" " * COLUMN_GAP).join(entries).rstrip())
        return Text.from_ansi("\n".join(lines))

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                Text.from_ansi(f"{COL_ACCENT}{self.heading}"),
                id="voice-select-title",
                markup=False,
            ),
            Static("", id="voice-select-list", markup=False),
            Horizontal(
                Static(INPUT_PROMPT, id="voice-select-prompt", markup=False),
                Input(
                    id="voice-select-input",
                    compact=True,
                    restrict=r"[0-9]*",
                    max_length=len(str(len(self.voice_names))),
                ),
                id="voice-select-input-row",
            ),
            Static("", id="voice-select-error", markup=False),
            id="voice-select-dialog",
        )

    def on_mount(self) -> None:
        self.fit_layout()

    def on_resize(self, _event: events.Resize) -> None:
        self.fit_layout()

    def fit_layout(self) -> None:
        """Size the dialog to the voice grid (see compute_voice_grid_layout)."""
        error = self.query_one("#voice-select-error", Static)
        min_content_width = max(
            len(self.heading),
            len(INPUT_PROMPT) + self.input_width,
            len(self.error_text) if error.display else 0,
        )
        layout = compute_voice_grid_layout(
            voice_count=len(self.voice_names),
            entry_width=self.entry_width,
            min_content_width=min_content_width,
            chrome_rows=CHROME_ROWS + (1 if error.display else 0),
            screen_width=self.size.width,
            screen_height=self.size.height,
        )
        self.grid_layout = layout
        dialog = self.query_one("#voice-select-dialog", Vertical)
        dialog.styles.width = layout.width
        dialog.styles.height = layout.height
        self.query_one("#voice-select-list", Static).update(self.list_text(layout))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        value = event.value.strip()
        if not value:
            self.dismiss(None)
            return
        number = int(value)
        if 1 <= number <= len(self.voice_names):
            self.dismiss(number - 1)
            return
        error = self.query_one("#voice-select-error", Static)
        error.update(Text.from_ansi(f"{COL_ERROR}{self.error_text}"))
        error.display = True
        self.fit_layout()

    def action_cancel(self) -> None:
        self.dismiss(None)
