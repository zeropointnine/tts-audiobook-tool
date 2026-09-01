from __future__ import annotations

from typing import Literal

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from tts_audiobook_tool import app_support
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import duration_string

# The bottom header line shows a prompt driven by the app. Quick generation
# uses auto_return to leave the line blank while returning to the editor.
PromptMode = Literal[
    "default", "cancel_pending", "finished", "auto_continue", "auto_return"
]

_PROMPT_BY_MODE: dict[PromptMode, str] = {
    "default": f"Press [{COL_ACCENT}CTRL-C{COL_DEFAULT}] to interrupt",
    "cancel_pending": (
        f"Press [{COL_ERROR}CTRL-C{COL_DEFAULT}] to kill process and stop immediately"
    ),
    "finished": f"Press [{COL_ACCENT}ENTER{COL_DEFAULT}] to continue",
    "auto_continue": "Proceeding to concatenation...",
    "auto_return": "",
}


class GenerationHeader(Vertical):
    """Top header for the generation screen: a three-row layout of five sub-widgets.

    - row 1: title (left), memory usage (right)
    - row 2: phase status (left), processed/elapsed stats (right)
    - row 3: full-width hotkey prompt

    Each ``update_*`` method renders exactly one sub-widget so the app can
    refresh them independently. ``update_memory_text`` builds its string from
    ``app_support``; status receives the phase label, while stats and hotkey
    receive the underlying values (processed/total/elapsed and prompt mode)
    and format their own strings.

    The title on row 1 is fixed for the session and supplied at construction:
    the editor's quick-generate flow passes its own.
    """

    # Widgets (unlike Screen) must use `DEFAULT_CSS` in this Textual build;
    # the `CSS` class attribute is ignored for non-Screen widgets.
    DEFAULT_CSS = """
    GenerationHeader {
        height: auto;
        padding: 0 1;
    }
    .generation-header-row {
        height: 1;
        width: 100%;
    }
    .generation-header-left {
        width: 1fr;
    }
    .generation-header-right {
        width: auto;
    }
    .generation-header-hotkey {
        height: 1;
        width: 100%;
    }
    """

    def __init__(
        self,
        *,
        title: str = "Generating audio...",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self._title_text = title

    def compose(self) -> ComposeResult:
        with Horizontal(classes="generation-header-row"):
            yield Static(
                Text.from_ansi(f"{COL_ACCENT}{self._title_text}"),
                classes="generation-header-left",
                id="generation-title",
            )
            yield Static(
                "",
                classes="generation-header-right",
                id="generation-memory",
            )
        with Horizontal(classes="generation-header-row"):
            yield Static(
                "",
                classes="generation-header-left",
                id="generation-status",
            )
            yield Static(
                "",
                classes="generation-header-right",
                id="generation-stats",
            )
        yield Static(
            Text.from_ansi(_PROMPT_BY_MODE["default"]),
            classes="generation-header-hotkey",
            id="generation-hotkey",
        )

    def update_memory_text(self) -> None:
        """Render memory info text on the right side of row 1.

        No-op when no memory info is available, so the label never dangles
        on systems without GPU/RAM telemetry.
        """
        memory_string = app_support.make_memory_string(
            base_color=COL_DIM, accent_color=COL_DIM, always_one_decimal=True
        )
        if memory_string:
            self.query_one("#generation-memory", Static).update(
                Text.from_ansi(memory_string)
            )

    def update_status(self, status: str) -> None:
        """Render the current phase on the left of row 2, e.g. "Status: Loading model"."""
        self.query_one("#generation-status", Static).update(
            Text.from_ansi(f"Status: {COL_DIM_ITALICS}{status}{COL_DEFAULT}")
        )

    def update_stats(
        self, processed: int, total: int, elapsed: float, eta_seconds: float | None = None
    ) -> None:
        """Render progress/elapsed stats on the right of row 2.

        The ETA (once enough batch durations have been recorded) is a
        per-batch snapshot, rendered before Elapsed in the same
        right-justified column; it is not ticked down between batches, and
        is rounded to the nearest 10 seconds.
        """
        stats = f"Processed: {processed}/{total}"
        if eta_seconds is not None:
            rounded_eta = round(eta_seconds / 10) * 10
            stats += f"  ETA: {duration_string(rounded_eta, pad_seconds=True)}"
        stats += f"  Elapsed: {duration_string(elapsed, pad_seconds=True)}"
        self.query_one("#generation-stats", Static).update(stats)

    def update_hotkey(self, mode: PromptMode) -> None:
        """Render the full-width hotkey prompt line for the given mode (ANSI colors allowed)."""
        self.query_one(
            "#generation-hotkey", Static
        ).update(Text.from_ansi(_PROMPT_BY_MODE[mode]))
