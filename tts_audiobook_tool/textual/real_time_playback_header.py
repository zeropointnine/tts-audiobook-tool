from __future__ import annotations

from typing import Literal

from rich.console import Console
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Rule, Static

from tts_audiobook_tool import app_support
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound.waveform_util import (
    WAVEFORM_LEVELS_PER_ROW,
    color_played_columns,
    compute_waveform_heights,
    decode_peak_envelope,
    render_centered_waveform_lines,
)
from tts_audiobook_tool.tts import Tts
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
    "default": f"Press [{COL_ACCENT}ESC{COL_DEFAULT}] to interrupt",
    "awaiting_continue": f"Press [{COL_ACCENT}ENTER{COL_DEFAULT}] to finish",
    "finished": f"Press [{COL_ACCENT}ENTER{COL_DEFAULT}] to finish",
}


def _pause_prompt(paused: bool) -> str:
    action = "unpause" if paused else "pause"
    return f"  - [{COL_ACCENT}P{COL_DEFAULT}] to {action}"


def _cancel_pending_prompt() -> str:
    """Prompt shown while a requested cancellation waits for a safe boundary.

    The second-ESC hard reset is gated to the local backend mode: it dumps
    the worker to clear resident *local* model memory, which does not apply
    to SGL-Omni's remote inference. The prompt is therefore resolved when
    rendered, not baked into the dict above (the backend mode is a process
    invariant, but tests patch it per test).
    """
    if Tts.is_remote_mode():
        return f"{COL_DIM}Waiting for the current generation to stop...{COL_DEFAULT}"
    return f"Press [{COL_ERROR}ESC{COL_DEFAULT}] to kill process and stop immediately"


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
                Text.from_ansi(f"{COL_ACCENT}Realtime audiobook playback"),
                classes="realtime-header-left",
                id="realtime-title",
            )
            yield Static("", classes="realtime-header-right", id="realtime-memory")
        with Horizontal(classes="realtime-header-row"):
            yield Static("", classes="realtime-header-left", id="realtime-status")
            yield Static("", classes="realtime-header-right", id="realtime-stats")
        yield Static(
            Text.from_ansi(_PROMPT_BY_MODE["default"] + _pause_prompt(False)),
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
        stats = f"Processed: {processed}/{total}  Buffer: {COL_OK}{buffer_value}{COL_DEFAULT}"
        self.query_one("#realtime-stats", Static).update(Text.from_ansi(stats))

    def update_hotkey(self, mode: PromptMode, paused: bool = False) -> None:
        text = _cancel_pending_prompt() if mode == "cancel_pending" else _PROMPT_BY_MODE[mode]
        if mode in ("default", "awaiting_continue"):
            text += _pause_prompt(paused)
        self.query_one("#realtime-hotkey", Static).update(
            Text.from_ansi(text)
        )


# Waveform band: rows above (positive peaks) and below (negative peaks) the
# zero axis; the band is twice this tall.
WAVEFORM_HALF_ROWS = 3
# Gentler than the trim editor's 1.5: the shorter half-height would
# otherwise erase quiet speech.
WAVEFORM_AMPLITUDE_EXPONENT = 1.25
# Amplitudes are measured against full scale (no per-segment normalization),
# so relative loudness between segments stays visible.
WAVEFORM_REFERENCE_PEAK = 1.0


class RealTimePlaybackWaveform(Static):
    """Conventional waveform of the segment at the current playhead.

    Positive peaks extend up from the zero axis and negative peaks down,
    each measured independently. Bars are COL_DIM; the already-played prefix
    is redrawn in COL_DEFAULT. Silent stretches show the flat zero axis.
    With no envelope (before the first segment) the band is just the axis.
    The app drives `show` from its
    fast waveform tick; the plain rendered rows are cached per (envelope,
    width), and the widget is only updated when the played column changes.
    """

    DEFAULT_CSS = f"""
    RealTimePlaybackWaveform {{
        height: {2 * WAVEFORM_HALF_ROWS};
        width: 100%;
        padding: 0 1;
    }}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", markup=False, **kwargs)
        self._lines_cache: tuple[tuple[bytes, int], list[str]] | None = None
        self._rendered_key: tuple[bytes, int, int] | None = None

    def show(self, envelope: bytes, progress: float) -> None:
        columns = self.content_size.width
        if columns <= 0:
            return
        cache_key = (envelope, columns)
        cached = self._lines_cache
        if cached is not None and cached[0] == cache_key:
            lines = cached[1]
        else:
            # Both halves share the fixed reference peak, so their heights
            # are directly comparable.
            upper_heights, lower_heights = (
                compute_waveform_heights(
                    peaks,
                    2 * columns,
                    max_level=WAVEFORM_HALF_ROWS * WAVEFORM_LEVELS_PER_ROW,
                    exponent=WAVEFORM_AMPLITUDE_EXPONENT,
                    peak=WAVEFORM_REFERENCE_PEAK,
                    # Short segments can have fewer envelope windows than bars.
                    fill_empty_bars=True,
                )
                for peaks in decode_peak_envelope(envelope)
            )
            lines = render_centered_waveform_lines(
                upper_heights, lower_heights, columns, WAVEFORM_HALF_ROWS
            )
            self._lines_cache = (cache_key, lines)
        played_columns = round(max(0.0, min(1.0, progress)) * columns)
        render_key = (envelope, columns, played_columns)
        if render_key == self._rendered_key:
            return
        self._rendered_key = render_key
        self.update(Text.from_ansi(color_played_columns(lines, played_columns)))


class RealTimePlaybackSourceText(Vertical):
    """The two-line source-text band directly below the realtime header.

    Shows the source text of the segment at the current playhead, in
    COL_ACCENT. The app composes the band below the base's shared divider, and the band
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
        /* One-line text sits on the lower row, against the closing rule. */
        content-align-vertical: bottom;
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
        widget.update(Text.from_ansi(f"{COL_ACCENT}{preview}{COL_DEFAULT}"))
