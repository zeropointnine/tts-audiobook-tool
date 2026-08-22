from __future__ import annotations

from typing import Literal

from rich.console import Console
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Rule, Static

from tts_audiobook_tool import app_support
from tts_audiobook_tool.constants import (
    COL_ACCENT,
    COL_DEFAULT,
    COL_DIM,
    COL_DIM_ITALICS,
    COL_ERROR,
)
from tts_audiobook_tool.util import duration_string


PromptMode = Literal["default", "cancel_pending", "awaiting_continue", "finished"]

# Console used purely for cell-width measurement when word-wrapping the
# source text to the band's width; nothing is ever written to it.
_MEASURE_CONSOLE = Console()


def _fit_two_lines(text: str, width: int) -> str:
    """Word-wrap text to width cells, returning at most two lines.

    Content that would wrap past the second line is replaced by an
    ellipsis at the end of the second line.
    """
    lines = Text(text).wrap(_MEASURE_CONSOLE, width, overflow="fold")
    if len(lines) <= 2:
        return "\n".join(line.plain.rstrip() for line in lines)
    first = lines[0].plain.rstrip()
    # Text.truncate mutates in place and returns None, so keep the Text.
    second = Text(lines[1].plain)
    second.truncate(max(0, width - 3))
    return f"{first}\n{second.plain.rstrip()}..."


_PROMPT_BY_MODE: dict[PromptMode, str] = {
    "default": f"Press [{COL_ACCENT}CTRL-C{COL_DEFAULT}] to interrupt",
    "cancel_pending": (
        f"Press [{COL_ERROR}CTRL-C{COL_DEFAULT}] to kill process and stop immediately"
    ),
    "awaiting_continue": f"Press [{COL_ACCENT}ENTER{COL_DEFAULT}] to finish",
    "finished": f"Press [{COL_ACCENT}ENTER{COL_DEFAULT}] to finish",
}


class RealTimePlaybackHeader(Vertical):
    """Rudimentary realtime status header driven by structured worker events."""

    DEFAULT_CSS = """
    RealTimePlaybackHeader {
        height: auto;
        padding: 0 1;
    }
    .realtime-header-row {
        height: 1;
        width: 100%;
    }
    .realtime-header-left {
        width: 1fr;
    }
    .realtime-header-right {
        width: auto;
    }
    .realtime-header-hotkey {
        height: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="realtime-header-row"):
            yield Static(
                Text.from_ansi(f"{COL_ACCENT}Realtime audiobook playback..."),
                classes="realtime-header-left",
                id="realtime-title",
            )
            yield Static("", classes="realtime-header-right", id="realtime-memory")
        with Horizontal(classes="realtime-header-row"):
            yield Static("", classes="realtime-header-left", id="realtime-status")
            yield Static("", classes="realtime-header-right", id="realtime-stats")
        yield Static(
            Text.from_ansi(_PROMPT_BY_MODE["default"]),
            classes="realtime-header-hotkey",
            id="realtime-hotkey",
        )

    def update_memory_text(self) -> None:
        memory_string = app_support.make_memory_string(
            base_color=COL_DIM,
            accent_color=COL_DIM,
            always_one_decimal=True,
        )
        if memory_string:
            self.query_one("#realtime-memory", Static).update(
                Text.from_ansi(memory_string)
            )

    def update_status(self, status: str) -> None:
        self.query_one("#realtime-status", Static).update(
            Text.from_ansi(f"Status: {COL_DIM_ITALICS}{status}{COL_DEFAULT}")
        )

    def update_stats(
        self,
        processed: int,
        total: int,
        buffer_seconds: float,
        zero_buffer_is_error: bool = True,
    ) -> None:
        buffer_value = duration_string(buffer_seconds, include_tenth=True)
        if buffer_seconds <= 0.0 and zero_buffer_is_error:
            buffer_value = f"{COL_ERROR}{buffer_value}{COL_DEFAULT}"
        stats = f"Processed: {processed}/{total}  Buffer: {buffer_value}"
        self.query_one("#realtime-stats", Static).update(Text.from_ansi(stats))

    def update_hotkey(self, mode: PromptMode) -> None:
        self.query_one("#realtime-hotkey", Static).update(
            Text.from_ansi(_PROMPT_BY_MODE[mode])
        )


class RealTimePlaybackSourceText(Vertical):
    """The two-line source-text band directly below the realtime header.

    Shows the source text of the segment at the current playhead, dimmed
    (the band is quiet context for the source, not a live highlight). The
    app composes the band below the base's shared divider, and the band
    closes itself with its own full-width rule (styled like the shared
    divider: same color and margin), so the two strokes frame the band
    between the header chrome and the worker log. The id selector wins the
    specificity race against Rule's own horizontal defaults, exactly as
    the base's shared-divider rule does.
    """

    DEFAULT_CSS = """
    RealTimePlaybackSourceText {
        height: auto;
    }
    .realtime-source-text {
        height: 2;
        width: 100%;
        padding: 0 1;
    }
    #realtime-source-divider {
        color: #888888;
        margin: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", classes="realtime-source-text", id="realtime-source-text")
        yield Rule(id="realtime-source-divider")

    def update_playing_text(self, text: str) -> None:
        """Shows the source text of the segment at the current playhead.

        The band is two lines tall: the text word-wraps onto the second
        line, and anything that would overflow past it ends in an ellipsis
        on that line. The app refreshes this on its header interval, so the
        wrap re-adapts if the terminal is resized.
        """
        widget = self.query_one("#realtime-source-text", Static)
        if not text:
            widget.update("")
            return
        preview = " ".join(text.split())
        width = widget.content_size.width
        if width > 0:
            preview = _fit_two_lines(preview, width)
        widget.update(Text.from_ansi(f"{COL_DIM}{preview}{COL_DEFAULT}"))
