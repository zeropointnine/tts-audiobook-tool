"""Full-screen interactive voice sample crop editor.

Runs over one voice sample's audio: hotkeys adjust the crop span, preview
plays exactly what `ProjectVoiceUtil.apply_voice_crop_and_save` will write
(including the silence-trim post-processing), and ENTER commits through the
caller. See docs-dev/voice-sample-crop.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from rich.text import Text
from textual.widgets import Rule, Static

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.sound.silence_util import SilenceUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.textual.textual_shared import can_textual

__all__ = [
    "PREVIEW_TICK_SECONDS",
    "WAVEFORM_BLOCK_CHARS",
    "WAVEFORM_LEVELS_PER_ROW",
    "WAVEFORM_ROWS",
    "VoiceCropApp",
    "VoiceCropResult",
    "clamp_crop_range",
    "render_crop_bar",
    "run_voice_crop_app",
]

# Hotkey step sizes in seconds: unshifted fine, shifted coarse.
FINE_STEP_S = 0.05
COARSE_STEP_S = 1.0
# Preview playhead animation cadence: 20 Hz.
PREVIEW_TICK_SECONDS = FINE_STEP_S
# Undocumented Shift-P hotkey: preview the last TAIL seconds of the range.
TAIL_PREVIEW_S = 1.0
# Waveform visualization: rows at the waveform's relative max amplitude,
# and the blank row between the visualization and the bar. Quadrant characters
# give each row 2 vertical levels and each character 2 independent bars.
WAVEFORM_ROWS = 6
WAVEFORM_GAP_ROWS = 1
WAVEFORM_LEVELS_PER_ROW = 2
# Halfway between linear amplitude and the previous squared curve.
WAVEFORM_AMPLITUDE_EXPONENT = 1.5
# Indexed by quadrant mask: upper-left=8, upper-right=4, lower-left=2,
# lower-right=1. Index 0 is blank; index 15 fills all four quadrants.
WAVEFORM_BLOCK_CHARS = (
    " ", "▗", "▖", "▄", "▝", "▐", "▞", "▟",
    "▘", "▚", "▌", "▙", "▀", "▜", "▛", "█",
)
# Horizontal margin around the centered bar, and its maximum width.
BAR_MARGIN_COLUMNS = 5
BAR_MAX_WIDTH = 200


@dataclass(frozen=True)
class VoiceCropResult:
    saved: bool
    cleared: bool = False
    start_s: float = 0.0
    end_s: float = 0.0


def clamp_crop_range(
    start_s: float,
    end_s: float,
    duration: float,
    min_span: float,
    *,
    moving: str = "end",
) -> tuple[float, float]:
    """Clamp a candidate range into [0, duration] keeping the min span.

    `moving` names the boundary being adjusted ("start" or "end"); the other
    boundary acts as the wall the min span is measured from. When even the
    full sample cannot satisfy the min span, the full range is returned.
    Pure function; the app calls it after every adjustment.
    """
    start_s = max(0.0, min(start_s, duration))
    end_s = max(0.0, min(end_s, duration))
    if duration < min_span:
        return 0.0, duration
    if moving == "start":
        start_s = max(0.0, min(end_s - min_span, start_s))
    else:
        end_s = min(duration, max(start_s + min_span, end_s))
        if end_s - start_s < min_span:
            # Defensive: a caller-supplied start too close to the end of the
            # sample cannot keep the span; pull it back.
            start_s = max(0.0, end_s - min_span)
    return start_s, end_s


def render_crop_bar(
    duration: float,
    start_s: float,
    end_s: float,
    width: int,
    progress: float | None = None,
) -> str:
    """Render the proportional bar, boundary markers, and labels.

    Pure function of the range and available width. Each inner character
    covers two time slots, matching the waveform's virtual bars. Unselected
    cells remain COL_DIM dashes, enclosed by COL_DIM angle brackets; selected
    halves use ▌/▐ and full cells use █.
    Selected halves use COL_ACCENT. `progress` in [0, 1] changes the played
    prefix to COL_DEFAULT at half-cell precision; None keeps all selected
    halves accented.
    """
    width = max(20, width)  # total rendered width, brackets included
    inner = width - 2
    slots = 2 * inner
    scale = slots / duration if duration > 0 else 0
    start_slot = max(0, min(slots - 1, round(start_s * scale)))
    end_slot = max(start_slot + 1, min(slots, round(end_s * scale)))

    filled_count = end_slot - start_slot
    played_count = 0
    if progress is not None:
        played_count = max(0, min(filled_count, round(progress * filled_count)))
    played_end = start_slot + played_count

    chars = []
    for column in range(inner):
        left_slot = 2 * column
        right_slot = left_slot + 1
        left_selected = start_slot <= left_slot < end_slot
        right_selected = start_slot <= right_slot < end_slot
        mask = (10 if left_selected else 0) | (5 if right_selected else 0)
        if not mask:
            chars.append(f"{COL_DIM}-{COL_DEFAULT}")
            continue
        left_played = left_selected and left_slot < played_end
        right_played = right_selected and right_slot < played_end
        if left_selected and right_selected and left_played and not right_played:
            # One cell contains two colors. ▌ keeps the played left half
            # in the terminal's native foreground; an accent background fills
            # the unplayed right half. Convert foreground SGR 38 to background
            # 48 for either the 256-color or true-color COL_ACCENT encoding.
            accent_background = COL_ACCENT.replace("[38;", "[48;")
            chars.append(f"{COL_DEFAULT}{accent_background}▌{COL_DEFAULT}")
        elif left_played or right_played:
            chars.append(f"{COL_DEFAULT}{WAVEFORM_BLOCK_CHARS[mask]}{COL_DEFAULT}")
        else:
            chars.append(f"{COL_ACCENT}{WAVEFORM_BLOCK_CHARS[mask]}{COL_DEFAULT}")
    bar = f"{COL_DIM}<{COL_DEFAULT}" + "".join(chars) + f"{COL_DIM}>{COL_DEFAULT}"

    # Carets remain terminal-column markers under the first/last selected
    # half-cell. A selection within one character collapses to one caret.
    start_col = start_slot // 2 + 1
    end_col = (end_slot - 1) // 2 + 1
    markers = " " * start_col + "^"
    if end_col > start_col:
        markers += " " * (end_col - start_col - 1) + "^"
    markers = markers.ljust(width)

    start_label = f"Start: {start_s:.2f}s"
    end_label = f"End: {end_s:.2f}s"
    labels = (
        f"{COL_DIM}Start:{COL_DEFAULT} {start_s:.2f}s"
        + " " * max(1, width - len(start_label) - len(end_label))
        + f"{COL_DIM}End:{COL_DEFAULT} {end_s:.2f}s"
    )
    return "\n".join((bar, markers, labels))


def compute_waveform_heights(data, columns: int) -> list[int]:
    """Per-virtual-bar waveform heights, normalized to the global peak.

    `columns` counts sampled bars, not terminal characters (two bars per
    character). `data` is the mono sample array. Each bar takes the max absolute
    amplitude over its slice of the timeline, then raises the peak-normalized
    amplitude to 1.5 to gently suppress quieter portions and emphasize peaks.
    Heights are rounded to the nearest sub-row level; the loudest bar is
    exactly WAVEFORM_ROWS * WAVEFORM_LEVELS_PER_ROW (a full block to the top).
    Silence and amplitudes that round below half a level remain blank.
    """
    if columns <= 0 or data.size == 0:
        return [0] * max(0, columns)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return [0] * columns
    max_level = WAVEFORM_ROWS * WAVEFORM_LEVELS_PER_ROW
    edges = np.linspace(0, data.size, columns + 1, dtype=np.int64)
    heights: list[int] = []
    for i in range(columns):
        chunk = data[edges[i]:edges[i + 1]]
        amplitude = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        # Nonlinear scaling favors aesthetics; blank regions may still contain quiet speech.
        level = round(max_level * (amplitude / peak) ** WAVEFORM_AMPLITUDE_EXPONENT)
        heights.append(min(max_level, max(0, level)))
    return heights


def render_waveform(heights: list[int], columns: int) -> str:
    """Render the top-half waveform block (COL_DIM), top row first.

    `columns` counts terminal characters; `heights` contains twice that many
    virtual bars, ordered left-to-right. Each character pairs two independent
    bottom-aligned bars, with two vertical levels per row. Missing heights are
    zero, extra heights are ignored, and heights are clamped to the block.
    The block is WAVEFORM_ROWS lines tall; each line is one leading space
    (the bar's "<" column) plus `columns` characters. The caller adds the gap
    row(s) and the bar below.
    """
    columns = max(0, columns)
    left_masks = (0, 2, 10)
    right_masks = (0, 1, 5)
    lines = []
    for row_index in range(WAVEFORM_ROWS):
        # Levels covered by the rows strictly below this one.
        below = (WAVEFORM_ROWS - 1 - row_index) * WAVEFORM_LEVELS_PER_ROW
        chars = []
        for column in range(columns):
            left_index = 2 * column
            right_index = left_index + 1
            left_height = heights[left_index] if left_index < len(heights) else 0
            right_height = heights[right_index] if right_index < len(heights) else 0
            left_level = max(0, min(WAVEFORM_LEVELS_PER_ROW, left_height - below))
            right_level = max(0, min(WAVEFORM_LEVELS_PER_ROW, right_height - below))
            mask = left_masks[left_level] | right_masks[right_level]
            chars.append(WAVEFORM_BLOCK_CHARS[mask])
        line = " " + "".join(chars)
        lines.append(f"{COL_DIM}{line}{COL_DEFAULT}")
    return "\n".join(lines)


class CropCenterContainer(Container):
    """Centering container that re-renders the bar on terminal resize.

    Textual delivers Resize events to widgets, not to the App, so the App
    itself cannot observe size changes.
    """

    def on_resize(self, event) -> None:
        app = self.app
        if isinstance(app, VoiceCropApp) and app.is_mounted:
            # Defer like the mount path does: reading container geometry
            # synchronously here can see pre-resize content_size values.
            self.call_after_refresh(app.refresh_bar)


class VoiceCropApp(App[VoiceCropResult]):
    """Interactive crop editor for one voice sample."""

    CSS = """
    Screen {
        background: ansi_default;
        color: ansi_default;
    }
    #crop-header {
        padding: 0 1;
        height: auto;
    }
    #crop-divider {
        color: #888888;
        margin: 0;
    }
    #crop-content {
        height: 1fr;
    }
    #crop-info {
        padding: 1 1 0 1;
        height: auto;
    }
    #crop-center {
        align: center middle;
        height: 1fr;
    }
    #crop-bar {
        width: auto;
        height: auto;
    }
    #crop-durations {
        align: left bottom;
        padding: 0 1 1 1;
        height: auto;
    }
    """

    def __init__(self, sound: Sound, title: str, initial: tuple[float, float] | None) -> None:
        super().__init__()
        self.sound = sound
        self.title_text = f"Trim voice sample: {title}"
        self.duration = sound.duration
        min_span = ProjectVoiceUtil.MIN_VOICE_CROP_DURATION_S
        if initial is not None:
            self.initial = clamp_crop_range(initial[0], initial[1], self.duration, min_span)
        else:
            # Default crop: as much tail audio as the sample allows.
            self.initial = clamp_crop_range(0.0, self.duration, self.duration, min_span)
        self.start_s, self.end_s = self.initial

    def compose(self) -> ComposeResult:
        yield Static("", id="crop-header", markup=False)
        yield Rule(id="crop-divider")
        with Container(id="crop-content"):
            yield Static("", id="crop-info", markup=False)
            with CropCenterContainer(id="crop-center"):
                yield Static("", id="crop-bar", markup=False)
            yield Static("", id="crop-durations", markup=False)

    def on_mount(self) -> None:
        # Match the worker/generation apps' terminal-native palette.
        self.theme = "ansi-dark"
        self.query_one("#crop-header", Static).update(self._header_text())
        self.query_one("#crop-info", Static).update(self._info_text())
        # Container widths are not final until the first layout pass.
        self.call_after_refresh(self.refresh_bar)

    def _header_text(self) -> Text:
        return Text.from_ansi(f"{COL_ACCENT}{self.title_text}{COL_DEFAULT}")

    def _info_text(self) -> Text:
        lines = [
            f"Press [{COL_ACCENT}A/S{COL_DEFAULT}] to adjust start point",
            f"Press [{COL_ACCENT}D/F{COL_DEFAULT}] to adjust end point",
            f"Press [{COL_ACCENT}P{COL_DEFAULT}] to preview",
            f"Press [{COL_ACCENT}X{COL_DEFAULT}] to reset trim",
            f"Press [{COL_ACCENT}ENTER{COL_DEFAULT}] to save, [{COL_ACCENT}ESC{COL_DEFAULT}] to cancel",
            "",
            f"{COL_DIM}Avoid trimming in the middle of words{COL_DEFAULT}",
            f"{COL_DIM}This is a non-destructive edit{COL_DEFAULT}",
        ]
        return Text.from_ansi("\n".join(lines))

    def _bar_width(self) -> int:
        content_width = self.query_one("#crop-content").content_size.width
        return max(20, min(content_width - 2 * BAR_MARGIN_COLUMNS, BAR_MAX_WIDTH))

    def _new_duration(self) -> float:
        """Duration of what will actually be saved (after silence trim).

        Cached per (start_s, end_s): the trim + silence-trim RMS scan is too
        costly to repeat on every refresh_bar, which fires at 20 Hz while the
        preview playhead animates. Invalidated by adjusting the range.
        """
        cached = getattr(self, "_new_duration_cache", None)
        key = (self.start_s, self.end_s)
        if cached is not None and cached[0] == key:
            return cached[1]
        span = SoundUtil.trim(self.sound, self.start_s, self.end_s)
        trimmed, _, _ = SilenceUtil.trim_silence_ends(span)
        self._new_duration_cache = (key, trimmed.duration)
        return trimmed.duration

    def _waveform_heights(self, width: int) -> list[int]:
        """Two waveform heights per inner character column, cached by width.

        Recomputing reduces the sample array into twice the inner character
        count; that is avoided on ordinary keypresses and preview ticks.
        """
        columns = 2 * (max(20, width) - 2)
        cached = getattr(self, "_waveform_cache", None)
        if cached is not None and cached[0] == columns:
            return cached[1]
        heights = compute_waveform_heights(self.sound.data, columns)
        self._waveform_cache = (columns, heights)
        return heights

    def _waveform_text(self, width: int) -> str:
        """Rendered waveform block, cached by column count.

        The waveform depends only on the audio and the width — never on the
        range or playhead — so the string is rebuilt only on resize, not on
        every refresh_bar (which fires at 20 Hz during preview animation).
        """
        columns = max(20, width) - 2
        cached = getattr(self, "_waveform_text_cache", None)
        if cached is not None and cached[0] == columns:
            return cached[1]
        text = render_waveform(self._waveform_heights(width), columns)
        self._waveform_text_cache = (columns, text)
        return text

    def refresh_bar(self, progress: float | None = None) -> None:
        """Re-render the bar and durations for the current range and width."""
        # Exposed as _last_progress for the playhead tests; no other reader.
        self._last_progress = progress
        width = self._bar_width()
        waveform = self._waveform_text(width)
        bar_text = render_crop_bar(self.duration, self.start_s, self.end_s, width, progress)
        gap = " " * (max(20, width) - 1)  # leading "<" column + inner width
        display = "\n".join([waveform] + [gap] * WAVEFORM_GAP_ROWS + [bar_text])
        self.query_one("#crop-bar", Static).update(Text.from_ansi(display))
        durations = (
            f"{COL_DIM}Original duration: {COL_DEFAULT}{self.duration:.2f}s\n"
            f"{COL_DIM} Trimmed duration: {COL_DEFAULT}{self._new_duration():.2f}s"
        )
        self.query_one("#crop-durations", Static).update(Text.from_ansi(durations))

    # --- Hotkeys (bindings, not key_* methods: Textual only dispatches
    # lowercase key methods, so shifted keys must be bindings) ---

    BINDINGS = [
        Binding("ctrl+q", "ignore_ctrl_q", show=False, priority=True),
        Binding("a", f"adjust('start', -{FINE_STEP_S})", show=False),
        Binding("s", f"adjust('start', {FINE_STEP_S})", show=False),
        Binding("d", f"adjust('end', -{FINE_STEP_S})", show=False),
        Binding("f", f"adjust('end', {FINE_STEP_S})", show=False),
        Binding("A", f"adjust('start', -{COARSE_STEP_S})", show=False),
        Binding("S", f"adjust('start', {COARSE_STEP_S})", show=False),
        Binding("D", f"adjust('end', -{COARSE_STEP_S})", show=False),
        Binding("F", f"adjust('end', {COARSE_STEP_S})", show=False),
        Binding("p", "preview", show=False),
        Binding("P", "preview_tail", show=False),
        Binding("x", "revert", show=False),
        Binding("enter", "save", show=False),
        Binding("escape", "cancel", show=False),
    ]

    def action_adjust(self, boundary: str, delta: float) -> None:
        self._cancel_preview_animation()
        PlaySoundUtil.stop_sound_async()
        min_span = ProjectVoiceUtil.MIN_VOICE_CROP_DURATION_S
        start = self.start_s + delta if boundary == "start" else self.start_s
        end = self.end_s + delta if boundary == "end" else self.end_s
        candidate = clamp_crop_range(start, end, self.duration, min_span, moving=boundary)
        if candidate != (self.start_s, self.end_s):
            self.start_s, self.end_s = candidate
            self.refresh_bar()

    def action_preview(self) -> None:
        # Play exactly what will be saved: the span plus the same
        # silence-trim post-processing apply_voice_crop_and_save performs.
        self._cancel_preview_animation()
        PlaySoundUtil.stop_sound_async()
        span = SoundUtil.trim(self.sound, self.start_s, self.end_s)
        preview, leading_trim_s, _ = SilenceUtil.trim_silence_ends(span)
        if not preview.data.size:
            return
        PlaySoundUtil.play_sound_async(preview)

        # Animate the selected segment's played prefix in COL_DEFAULT using
        # simple time interpolation (no direct play-progress access). The
        # silence trim offsets the mapping: the played audio starts
        # `leading_trim_s` into the span and is shorter than the span, so
        # playback fraction f maps to bar progress
        # (leading_trim_s + f * preview.duration) / span.duration.
        span_duration = self.end_s - self.start_s
        self._start_preview_animation(
            play_duration=preview.duration,
            progress_at_start=leading_trim_s / span_duration if span_duration > 0 else 0.0,
            progress_rate=preview.duration / span_duration if span_duration > 0 else 1.0,
        )

    def _start_preview_animation(
            self, play_duration: float, progress_at_start: float, progress_rate: float,
    ) -> None:
        """Animate the playhead over `play_duration` seconds of audio.

        `progress_at_start` / `progress_rate` map playback fraction to bar
        progress, absorbing offsets (silence-trimmed lead-in, tail-only
        playback).
        """
        self._preview_started_at = time.monotonic()
        self._preview_duration = play_duration
        self._preview_progress_at_start = progress_at_start
        self._preview_progress_rate = progress_rate
        self._preview_timer = self.set_interval(PREVIEW_TICK_SECONDS, self._on_preview_tick)

    def _on_preview_tick(self) -> None:
        elapsed = time.monotonic() - self._preview_started_at
        fraction = elapsed / self._preview_duration if self._preview_duration > 0 else 1.0
        if fraction >= 1.0:
            self._cancel_preview_animation()
            self.refresh_bar()
            return
        progress = self._preview_progress_at_start + fraction * self._preview_progress_rate
        self.refresh_bar(max(0.0, min(1.0, progress)))

    def _cancel_preview_animation(self) -> None:
        timer = getattr(self, "_preview_timer", None)
        if timer is not None:
            timer.stop()
            self._preview_timer = None

    def action_preview_tail(self) -> None:
        # Undocumented helper: play the last TAIL_PREVIEW_S seconds of the
        # range raw (no silence trim), for boundary checking. The playhead
        # animates too: the tail starts (span - tail) into the segment, so
        # fraction f maps to progress ((span - tail) + f * tail) / span.
        self._cancel_preview_animation()
        PlaySoundUtil.stop_sound_async()
        span_duration = self.end_s - self.start_s
        if span_duration <= 0:
            return
        tail = min(TAIL_PREVIEW_S, span_duration)
        tail_sound = SoundUtil.trim(self.sound, self.end_s - tail, self.end_s)
        if not tail_sound.data.size:
            return
        PlaySoundUtil.play_sound_async(tail_sound)
        self._start_preview_animation(
            play_duration=tail_sound.duration,
            progress_at_start=(span_duration - tail) / span_duration,
            progress_rate=tail_sound.duration / span_duration,
        )

    def action_revert(self) -> None:
        # Revert an existing crop (the menu flow decides whether a discard
        # is actually warranted when none exists).
        PlaySoundUtil.stop_sound_async()
        self.exit(VoiceCropResult(saved=False, cleared=True))

    def action_save(self) -> None:
        self._cancel_preview_animation()
        PlaySoundUtil.stop_sound_async()
        # Compare the actual slice boundaries before silence trimming. Float
        # drift from reversing an adjustment must not trigger retranscription.
        current_samples = tuple(
            SoundUtil.seconds_to_sample_index(seconds, self.sound.sr)
            for seconds in (self.start_s, self.end_s)
        )
        initial_samples = tuple(
            SoundUtil.seconds_to_sample_index(seconds, self.sound.sr)
            for seconds in self.initial
        )
        if current_samples == initial_samples:
            self.exit(VoiceCropResult(saved=False))
            return
        self.exit(VoiceCropResult(saved=True, start_s=self.start_s, end_s=self.end_s))

    def action_cancel(self) -> None:
        self._cancel_preview_animation()
        PlaySoundUtil.stop_sound_async()
        self.exit(VoiceCropResult(saved=False))

    def action_ignore_ctrl_q(self) -> None:
        """Override Textual's built-in Ctrl+Q quit binding."""


def run_voice_crop_app(
    sound: Sound,
    title: str,
    initial: tuple[float, float] | None,
) -> VoiceCropResult | str:
    """Run the crop editor; returns its result or an error string."""
    if not can_textual():
        return "The current terminal environment does not support the trim editor"
    app = VoiceCropApp(sound, title, initial)
    result = app.run(inline=False)
    if result is None:
        return "Trim editor closed without a result"
    return result
