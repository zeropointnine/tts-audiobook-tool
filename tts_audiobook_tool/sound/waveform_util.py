"""Terminal waveform visualization, shared by the trim editor and realtime playback.

Pipeline: sample data (or a compact peak envelope of it) is reduced to
per-bar integer heights by `compute_waveform_heights`, then drawn with
quadrant block characters, which give each terminal character two
independent bars and each row two vertical levels. Rendering returns plain
(uncolored) lines; callers color them, e.g. with `color_played_columns`.

Two layouts:
- `render_waveform_lines`: bars grow up from a baseline (the trim editor,
  which plots max |amplitude|).
- `render_centered_waveform_lines`: a conventional (editor-style) waveform
  about a zero axis. Each slice's positive peak extends up and its
  negative peak extends down, measured independently, so asymmetric
  signals look asymmetric.

The realtime app never holds the audio itself (it lives in the model
worker), so the worker sends `compute_peak_envelope` bytes with each segment
and the app re-buckets them to its width: the max of per-window maxima
equals the max over the underlying samples, so heights computed from the
envelope match those computed from the samples, up to window granularity at
the bar edges.
"""
from __future__ import annotations

import numpy as np

from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM

__all__ = [
    "ENVELOPE_WINDOW_SECONDS",
    "WAVEFORM_BLOCK_CHARS",
    "WAVEFORM_LEVELS_PER_ROW",
    "WAVEFORM_SILENCE_CHAR",
    "color_played_columns",
    "compute_peak_envelope",
    "compute_waveform_heights",
    "decode_peak_envelope",
    "render_centered_waveform_lines",
    "render_waveform_lines",
]

# Quadrant characters give each row 2 vertical levels.
WAVEFORM_LEVELS_PER_ROW = 2
# Indexed by quadrant mask: upper-left=8, upper-right=4, lower-left=2,
# lower-right=1. Index 0 is blank; index 15 fills all four quadrants.
WAVEFORM_BLOCK_CHARS = (
    " ", "▗", "▖", "▄", "▝", "▐", "▞", "▟",
    "▘", "▚", "▌", "▙", "▀", "▜", "▛", "█",
)
# Centered waveforms draw the zero axis where the row just above the center
# is otherwise empty: a lower 1/8 block on that row. Block
# elements are drawn flush to the cell edges (as the quadrant bars are), so
# the line rests on the center, offset up by only half its thin thickness;
# "_" sits at the font's underline position, visibly above center. (Adding
# an upper 1/8 block below would center it exactly but double its weight.)
WAVEFORM_SILENCE_CHAR = "▁"

# Masks by level (0, 1, 2) for bars that grow upward (bottom-aligned)...
_UP_LEFT_MASKS = (0, 2, 10)
_UP_RIGHT_MASKS = (0, 1, 5)
# ...and for bars that grow downward (top-aligned).
_DOWN_LEFT_MASKS = (0, 8, 10)
_DOWN_RIGHT_MASKS = (0, 4, 5)

# Peak envelope resolution: per window, the positive and the negative peak
# magnitudes, each quantized to one byte (0..255 maps to 0.0..1.0 full scale).
ENVELOPE_WINDOW_SECONDS = 0.01
_ENVELOPE_SCALE = 255


def _window_peaks(samples: np.ndarray, window: int) -> np.ndarray:
    """Max of non-negative `samples` per window (the last may be partial)."""
    pad = (-samples.size) % window
    if pad:
        samples = np.concatenate([samples, np.zeros(pad, dtype=samples.dtype)])
    return samples.reshape(-1, window).max(axis=1)


def compute_peak_envelope(
    data: np.ndarray, sr: int, window_seconds: float = ENVELOPE_WINDOW_SECONDS
) -> bytes:
    """Positive and negative peak per fixed-length window, two bytes each.

    Bytes are interleaved per window: (positive peak, negative peak
    magnitude), each clipped to full scale and quantized to 0..255. A window
    with no positive (or no negative) samples has 0 for that side. `data`
    is mono (n,) or multichannel (n, channels); across channels the highest
    and lowest sample win. Integer PCM is scaled to [-1, 1] by its dtype's
    max. Compact enough to send per segment across the worker queue.
    Concatenating envelopes of consecutive chunks yields the envelope of the
    whole (up to a partial window at each seam).
    """
    if data.size == 0:
        return b""
    samples = data.astype(np.float32, copy=False)
    if np.issubdtype(data.dtype, np.integer):
        samples = samples / float(np.iinfo(data.dtype).max)
    if samples.ndim > 1:
        highest, lowest = samples.max(axis=1), samples.min(axis=1)
    else:
        highest = lowest = samples
    window = max(1, round(sr * window_seconds))
    positive = _window_peaks(np.maximum(highest, 0.0), window)
    negative = _window_peaks(np.maximum(-lowest, 0.0), window)
    peaks = np.stack([positive, negative], axis=1).ravel()
    quantized = np.round(np.clip(peaks, 0.0, 1.0) * _ENVELOPE_SCALE).astype(np.uint8)
    return quantized.tobytes()


def decode_peak_envelope(envelope: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `compute_peak_envelope`: (positive, negative) peak
    magnitudes per window, float32 in [0, 1]."""
    values = np.frombuffer(envelope, dtype=np.uint8, count=len(envelope) // 2 * 2)
    values = values.astype(np.float32) / _ENVELOPE_SCALE
    return values[0::2], values[1::2]


def compute_waveform_heights(
    data: np.ndarray,
    bars: int,
    max_level: int,
    exponent: float,
    peak: float | None = None,
    fill_empty_bars: bool = False,
) -> list[int]:
    """Per-bar integer heights in [0, max_level].

    `bars` counts sampled bars, not terminal characters (two bars per
    character). Each bar takes the max absolute amplitude over its slice of
    the timeline, divides it by `peak` (None: the data's own global peak, so
    the loudest bar reaches max_level), and raises it to `exponent`; values
    above 1 suppress quieter portions and emphasize peaks. Heights are rounded
    to the nearest level, so silence and amplitudes that round below half a
    level are 0. `data` may be raw mono samples or a decoded envelope.

    When there are more bars than data points, some slices are empty. By
    default those bars stay blank. With `fill_empty_bars`, each takes the
    data point under its center instead, stretching coarse data (a short
    segment's envelope on a wide terminal) without a comb of blank gaps.
    """
    if bars <= 0:
        return []
    if data.size == 0:
        return [0] * bars
    if peak is None:
        peak = float(np.max(np.abs(data)))
    if peak <= 0.0:
        return [0] * bars
    edges = np.linspace(0, data.size, bars + 1, dtype=np.int64)
    heights: list[int] = []
    for i in range(bars):
        chunk = data[edges[i]:edges[i + 1]]
        if chunk.size:
            amplitude = float(np.max(np.abs(chunk)))
        elif fill_empty_bars:
            center = min(data.size - 1, int((i + 0.5) * data.size / bars))
            amplitude = float(abs(data[center]))
        else:
            amplitude = 0.0
        # Nonlinear scaling favors aesthetics; blank regions may still contain quiet speech.
        level = round(max_level * min(1.0, amplitude / peak) ** exponent)
        heights.append(min(max_level, max(0, level)))
    return heights


def _bar_pair(heights: list[int], column: int) -> tuple[int, int]:
    left_index = 2 * column
    right_index = left_index + 1
    left = heights[left_index] if left_index < len(heights) else 0
    right = heights[right_index] if right_index < len(heights) else 0
    return left, right


def _render_rows(
    heights: list[int], columns: int, rows: int, *, grow_down: bool
) -> list[str]:
    """Rows of bars anchored at one edge: the bottom edge (bars grow up,
    rows returned top first) or the top edge (bars grow down, rows returned
    from the anchor outward, i.e. also top first)."""
    left_masks = _DOWN_LEFT_MASKS if grow_down else _UP_LEFT_MASKS
    right_masks = _DOWN_RIGHT_MASKS if grow_down else _UP_RIGHT_MASKS
    lines = []
    for row_index in range(rows):
        # Levels covered by the rows between this one and the anchor edge.
        offset = (row_index if grow_down else rows - 1 - row_index) * WAVEFORM_LEVELS_PER_ROW
        chars = []
        for column in range(columns):
            left, right = _bar_pair(heights, column)
            left_level = max(0, min(WAVEFORM_LEVELS_PER_ROW, left - offset))
            right_level = max(0, min(WAVEFORM_LEVELS_PER_ROW, right - offset))
            chars.append(WAVEFORM_BLOCK_CHARS[left_masks[left_level] | right_masks[right_level]])
        lines.append("".join(chars))
    return lines


def render_waveform_lines(heights: list[int], columns: int, rows: int) -> list[str]:
    """Bottom-aligned waveform, `rows` plain lines of `columns` characters,
    top row first.

    `heights` holds two bars per character, left to right, in levels
    (WAVEFORM_LEVELS_PER_ROW per row). Missing heights are zero, extra
    heights are ignored, and heights are clamped to the block.
    """
    return _render_rows(heights, max(0, columns), rows, grow_down=False)


def render_centered_waveform_lines(
    upper_heights: list[int],
    lower_heights: list[int],
    columns: int,
    half_rows: int,
) -> list[str]:
    """Conventional waveform about a horizontal zero axis: `half_rows` lines
    of positive peaks growing up from the center, then `half_rows` lines of
    negative peaks growing down from it (2 * half_rows plain lines, top
    first).

    `upper_heights` / `lower_heights` are the positive / negative peak bars
    (two per character, as in `render_waveform_lines`), so the two halves
    need not match. Characters whose two upper bars are both 0 show
    WAVEFORM_SILENCE_CHAR on the row just above the center, so the zero axis
    stays continuous through silence and through slices with only a
    negative peak. Height handling matches `render_waveform_lines`.
    """
    columns = max(0, columns)
    top = _render_rows(upper_heights, columns, half_rows, grow_down=False)
    bottom = _render_rows(lower_heights, columns, half_rows, grow_down=True)
    if top:
        top[-1] = "".join(
            WAVEFORM_SILENCE_CHAR if max(_bar_pair(upper_heights, column)) <= 0 else char
            for column, char in enumerate(top[-1])
        )
    return top + bottom


def color_played_columns(
    lines: list[str],
    played_columns: int,
    played_color: str = COL_DEFAULT,
    unplayed_color: str = COL_DIM,
) -> str:
    """Join plain waveform lines, coloring the first `played_columns`
    characters of each with `played_color` and the rest `unplayed_color`.

    Precision is one character (two bars): a quadrant glyph cannot carry two
    foreground colors.
    """
    parts = []
    for line in lines:
        split = max(0, min(len(line), played_columns))
        played, unplayed = line[:split], line[split:]
        text = ""
        if played:
            text += f"{played_color}{played}"
        if unplayed:
            text += f"{unplayed_color}{unplayed}"
        parts.append(f"{text}{COL_DEFAULT}")
    return "\n".join(parts)
