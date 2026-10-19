import re

import numpy as np
import pytest

from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.sound.waveform_util import (
    color_played_columns,
    compute_peak_envelope,
    compute_waveform_heights,
    decode_peak_envelope,
    render_centered_waveform_lines,
    render_waveform_lines,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestEnvelope:

    def test_positive_and_negative_peaks_per_window_with_partial_last_window(self):
        data = np.zeros(250, dtype=np.float32)
        data[5] = 0.5
        data[7] = -0.2
        data[249] = -1.0
        envelope = compute_peak_envelope(data, sr=10000)  # 100-sample windows
        # Interleaved (positive, negative) per window.
        assert envelope == bytes([128, 51, 0, 0, 0, 255])

    def test_one_sided_window_has_zero_for_the_other_side(self):
        assert compute_peak_envelope(np.array([0.3, 0.4]), sr=200) == bytes([102, 0])
        assert compute_peak_envelope(np.array([-0.3, -0.4]), sr=200) == bytes([0, 102])

    def test_stereo_takes_highest_and_lowest_across_channels_and_int_pcm_is_scaled(self):
        stereo = np.array([[0.1, -0.6], [0.2, 0.0]], dtype=np.float32)
        assert compute_peak_envelope(stereo, sr=200) == bytes([51, 153])
        pcm = np.array([16384, -32767], dtype=np.int16)
        assert compute_peak_envelope(pcm, sr=200) == bytes([128, 255])

    def test_empty_and_round_trip(self):
        assert compute_peak_envelope(np.zeros(0, dtype=np.float32), sr=100) == b""
        upper, lower = decode_peak_envelope(bytes([0, 255, 51, 102]))
        np.testing.assert_allclose(upper, [0.0, 0.2])
        np.testing.assert_allclose(lower, [1.0, 0.4])
        # A stray odd byte is ignored rather than misaligning the halves.
        upper, lower = decode_peak_envelope(bytes([10, 20, 30]))
        assert upper.size == lower.size == 1

    def test_heights_from_envelope_match_heights_from_samples(self):
        rng = np.random.default_rng(0)
        gains = np.repeat(np.linspace(0.05, 1.0, 40), 100)
        data = (rng.uniform(-1, 1, 4000) * gains).astype(np.float32)
        upper, lower = decode_peak_envelope(compute_peak_envelope(data, sr=1000))
        kwargs = dict(max_level=8, exponent=1.25, peak=1.0)
        # One-byte quantization can tip a bar sitting on a level boundary by
        # one level; otherwise envelope and samples agree.
        for peaks, samples in ((upper, np.maximum(data, 0)), (lower, np.maximum(-data, 0))):
            from_envelope = compute_waveform_heights(peaks, 40, **kwargs)
            from_samples = compute_waveform_heights(samples, 40, **kwargs)
            assert all(abs(a - b) <= 1 for a, b in zip(from_envelope, from_samples))
            assert sum(a != b for a, b in zip(from_envelope, from_samples)) <= 2
            assert len(set(from_samples)) > 4  # the gain ramp spans many levels


class TestHeights:

    def test_fixed_reference_peak_does_not_normalize(self):
        data = np.array([0.5, 0.25, 2.0, 0.0])
        assert compute_waveform_heights(data, 4, max_level=8, exponent=1.0, peak=1.0) == [4, 2, 8, 0]
        # Default (None) normalizes to the data's own peak.
        assert compute_waveform_heights(data, 4, max_level=8, exponent=1.0) == [2, 1, 8, 0]

    def test_more_bars_than_data_blank_or_fill_empty_slices(self):
        data = np.array([1.0, 0.5])
        kwargs = dict(max_level=8, exponent=1.0, peak=1.0)
        # Default: empty slices stay blank (comb pattern).
        assert compute_waveform_heights(data, 4, **kwargs) == [0, 8, 0, 4]
        # Filled: each empty bar takes the data point under its center.
        assert compute_waveform_heights(data, 4, fill_empty_bars=True, **kwargs) == [8, 8, 4, 4]
        stretched = compute_waveform_heights(
            np.full(100, 0.5), 156, fill_empty_bars=True, **kwargs
        )
        assert stretched == [4] * 156


class TestCenteredRender:

    def test_halves_are_drawn_independently(self):
        # Upper: char 0 left 3 (one full row plus half the next), right 0;
        # char 1 full. Lower: char 0 right 2 (one row); char 1 left full.
        lines = render_centered_waveform_lines([3, 0, 8, 8], [0, 2, 8, 0], 2, half_rows=4)
        assert lines == [
            " █",
            " █",
            "▖█",
            "▌█",
            "▐▌",
            " ▌",
            " ▌",
            " ▌",
        ]

    def test_zero_axis_wherever_upper_bars_are_empty(self):
        lines = render_centered_waveform_lines([0, 0, 0, 1], [0, 0, 2, 0], 2, half_rows=2)
        assert lines == ["  ", "▁▗", " ▌", "  "]
        # A slice with only a negative peak keeps the axis continuous.
        assert render_centered_waveform_lines([0, 0], [4, 4], 1, half_rows=2) == [" ", "▁", "█", "█"]
        assert render_centered_waveform_lines([], [], 3, half_rows=2) == ["   ", "▁▁▁", "   ", "   "]

    def test_top_only_render_matches_upper_half(self):
        heights = [3, 0, 8, 8]
        assert render_waveform_lines(heights, 2, 4) == [" █", " █", "▖█", "▌█"]


class TestColorPlayedColumns:

    @pytest.mark.parametrize("played", [0, 2, 5, 9])
    def test_split_keeps_text_and_colors_prefix(self, played):
        text = color_played_columns(["abcde", "fghij"], played)
        lines = text.split("\n")
        assert [ANSI_RE.sub("", line) for line in lines] == ["abcde", "fghij"]
        split = min(played, 5)
        expected = ""
        if split:
            expected += f"{COL_DEFAULT}{'abcde'[:split]}"
        if split < 5:
            expected += f"{COL_DIM}{'abcde'[split:]}"
        assert lines[0] == expected + COL_DEFAULT
