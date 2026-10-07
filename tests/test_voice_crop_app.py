"""Voice crop Textual app: bar rendering, range clamping, and key flow."""
import asyncio
import time

import numpy as np
import pytest
from rich.console import Console
from rich.text import Text

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.textual.voice_crop_app import (
    VoiceCropApp,
    clamp_crop_range,
    render_crop_bar,
)


def make_sound(seconds: float = 10.0, sr: int = 24000) -> Sound:
    rng = np.random.default_rng(0)
    return Sound(rng.uniform(0.01, 0.02, int(seconds * sr)).astype(np.float32), sr)


class TestClampCropRange:

    def test_plain_clamps_into_bounds(self):
        assert clamp_crop_range(-1.0, 5.0, 10.0, 3.0) == (0.0, 5.0)
        # End clamps to duration; the stationary start then violates the
        # min span and is defensively pulled back.
        assert clamp_crop_range(8.0, 11.0, 10.0, 3.0) == (7.0, 10.0)

    def test_min_span_enforced_moving_end(self):
        assert clamp_crop_range(5.0, 6.0, 10.0, 3.0) == (5.0, 8.0)

    def test_min_span_enforced_moving_start(self):
        assert clamp_crop_range(9.0, 9.5, 10.0, 3.0) == (7.0, 10.0)

    def test_full_range_on_short_sample(self):
        assert clamp_crop_range(0.0, 2.0, 2.0, 3.0) == (0.0, 2.0)


class TestRenderCropBar:

    def test_structure_and_alignment(self):
        text = render_crop_bar(20.0, 6.0, 11.5, 42)
        bar, markers, labels = ANSI_RE.sub("", text).split("\n")
        assert bar.startswith("<") and bar.endswith(">")
        assert "█" in bar and "-" in bar
        # Each caret sits under its boundary cell, including half-filled cells.
        selected_columns = [i for i, char in enumerate(bar) if char in "▌▐█"]
        assert markers.index("^") == selected_columns[0]
        assert markers.rindex("^") == selected_columns[-1]
        assert labels.startswith("Start: 6.00s")
        assert labels.endswith("End: 11.50s")
        assert len(labels) == len(bar) == len(markers)

    def test_bar_width_follows_argument(self):
        assert len(ANSI_RE.sub("", render_crop_bar(10.0, 2.0, 5.0, 50)).split("\n")[0]) == 50

    def test_proportional_segments(self):
        bar = ANSI_RE.sub("", render_crop_bar(10.0, 2.5, 7.5, 41)).split("\n")[0]
        inner = bar[1:-1]
        selected_halves = 2 * inner.count("█") + inner.count("▌") + inner.count("▐")
        assert selected_halves / (2 * len(inner)) == pytest.approx(0.5, abs=0.025)

    def test_minimum_width(self):
        assert len(ANSI_RE.sub("", render_crop_bar(10.0, 1.0, 2.0, 5)).split("\n")[0]) >= 20

    @pytest.mark.parametrize("start,end,expected", [
        (0, 1, "▌" + "-" * 17),
        (1, 2, "▐" + "-" * 17),
        (0, 2, "█" + "-" * 17),
        (3, 7, "-▐█▌" + "-" * 14),
        (35, 36, "-" * 17 + "▐"),
        (0, 36, "█" * 18),
    ])
    def test_half_character_boundaries(self, start, end, expected):
        # 36 seconds over 18 cells = one second per half-cell.
        bar, markers, _ = ANSI_RE.sub("", render_crop_bar(36.0, start, end, 20)).split("\n")
        assert bar == "<" + expected + ">"
        assert len(bar) == len(markers) == 20
        assert markers.index("^") == start // 2 + 1
        assert markers.rindex("^") == (end - 1) // 2 + 1
        assert markers.count("^") == (1 if start // 2 == (end - 1) // 2 else 2)

    def test_labels_show_fine_steps(self):
        labels = render_crop_bar(10.0, 0.05, 9.95, 42).split("\n")[-1]
        labels = ANSI_RE.sub("", labels)
        assert labels.startswith("Start: 0.05s")
        assert labels.endswith("End: 9.95s")

    def test_only_label_names_are_dim(self):
        from tts_audiobook_tool.constants import COL_DIM
        console = Console()
        labels = Text.from_ansi(render_crop_bar(10.0, 0.0, 9.95, 42).split("\n")[-1])
        dim = Text.from_ansi(COL_DIM + "x").get_style_at_offset(console, 0).color
        end_offset = labels.plain.index("End:")
        for offset in list(range(6)) + list(range(end_offset, end_offset + 4)):
            assert labels.get_style_at_offset(console, offset).color == dim
        for offset in (6, 7, end_offset + 4, end_offset + 5):
            assert labels.get_style_at_offset(console, offset).color is None
        assert len(labels.plain) == 42

    @pytest.mark.parametrize("duration,start,end", [(0, 0, 0), (36, -1, 100), (36, 36, 36)])
    def test_boundaries_stay_within_bar(self, duration, start, end):
        bar, markers, _ = ANSI_RE.sub("", render_crop_bar(duration, start, end, 20)).split("\n")
        assert len(bar) == len(markers) == 20
        assert bar.startswith("<") and bar.endswith(">")
        assert markers.index("^") >= 1
        assert markers.rindex("^") <= 18


def run_app_exercise(app, keys):
    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            for key in keys:
                await pilot.press(key)
    asyncio.run(exercise())


class TestVoiceCropApp:

    def test_adjust_and_save(self):
        app = VoiceCropApp(make_sound(10.0), "a", None)
        run_app_exercise(app, ["s", "s", "d", "enter"])
        assert app.return_value is not None
        assert app.return_value.saved
        assert app.return_value.start_s == pytest.approx(0.1)
        assert app.return_value.end_s == pytest.approx(9.95)

    def test_shifted_adjustments_use_one_second_steps(self):
        app = VoiceCropApp(make_sound(10.0), "a", None)
        run_app_exercise(app, ["S", "D", "enter"])
        assert app.return_value is not None
        assert app.return_value.start_s == pytest.approx(1.0)
        assert app.return_value.end_s == pytest.approx(9.0)

    def test_fine_adjustments_visible_in_labels(self):
        app = VoiceCropApp(make_sound(10.0), "a", None)

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("s", "d")
                labels = str(app.query_one("#crop-bar").content).split("\n")[-1]
                assert "Start: 0.05s" in labels
                assert "End: 9.95s" in labels
        asyncio.run(exercise())

    def test_unchanged_is_noop(self):
        app = VoiceCropApp(make_sound(10.0), "a", None)
        run_app_exercise(app, ["enter"])
        assert app.return_value is not None
        assert not app.return_value.saved
        assert not app.return_value.cleared

    @pytest.mark.parametrize("keys", [("s", "a"), ("S", "A"), ("f", "d"), ("F", "D")])
    @pytest.mark.parametrize("with_silence", [False, True])
    def test_reversing_adjustments_is_noop_at_sample_precision(self, keys, with_silence):
        from tts_audiobook_tool.sound.silence_util import SilenceUtil
        from tts_audiobook_tool.sound.sound_util import SoundUtil

        sound = make_sound(10.0)
        if with_silence:
            sound.data[:sound.sr] = 0
            sound.data[7 * sound.sr:] = 0
        initial = (0.1, 7.8)
        before = SoundUtil.trim(sound, *initial)
        before_trimmed, _, _ = SilenceUtil.trim_silence_ends(before)
        app = VoiceCropApp(sound, "a", initial)
        run_app_exercise(app, [*keys, "enter"])

        after = SoundUtil.trim(sound, app.start_s, app.end_s)
        after_trimmed, _, _ = SilenceUtil.trim_silence_ends(after)
        np.testing.assert_array_equal(after.data, before.data)
        np.testing.assert_array_equal(after_trimmed.data, before_trimmed.data)
        if with_silence:
            assert before_trimmed.duration < before.duration
        assert app.return_value is not None
        assert not app.return_value.saved
        assert not app.return_value.cleared

    def test_real_one_sample_adjustment_is_saved(self):
        sound = make_sound(10.0)
        app = VoiceCropApp(sound, "a", (0.1, 7.8))
        app.start_s += 1 / sound.sr
        run_app_exercise(app, ["enter"])
        assert app.return_value is not None
        assert app.return_value.saved

    def test_escape_cancels(self):
        app = VoiceCropApp(make_sound(10.0), "a", None)
        run_app_exercise(app, ["escape"])
        assert app.return_value is not None
        assert not app.return_value.saved

    @pytest.mark.parametrize("exit_key", ["escape", "enter", "x"])
    def test_ctrl_q_is_ignored_during_preview(self, monkeypatch, exit_key):
        from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil

        app = VoiceCropApp(make_sound(10.0), "a", (1.0, 5.0))
        playback_calls = []
        monkeypatch.setattr(PlaySoundUtil, "play_sound_async", lambda sound: playback_calls.append("play"))
        monkeypatch.setattr(PlaySoundUtil, "stop_sound_async", lambda: playback_calls.append("stop"))

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("s", "p")
                before = list(playback_calls)
                await pilot.press("ctrl+q")
                assert app.is_running
                assert app.return_value is None
                assert playback_calls == before
                assert (app.start_s, app.end_s) == (1.05, 5.0)
                await pilot.press(exit_key)
        asyncio.run(exercise())

        assert "play" in playback_calls
        assert playback_calls[-1] == "stop"
        assert app.return_value is not None
        assert app.return_value.saved == (exit_key == "enter")
        assert app.return_value.cleared == (exit_key == "x")

    def test_cleared(self):
        app = VoiceCropApp(make_sound(10.0), "a", (1.0, 5.0))
        run_app_exercise(app, ["x"])
        assert app.return_value is not None
        assert app.return_value.cleared

    def test_min_span_wall(self):
        app = VoiceCropApp(make_sound(10.0), "a", None)
        run_app_exercise(app, ["D"] * 80 + ["enter"])
        assert app.return_value is not None
        assert app.return_value.saved
        assert app.return_value.end_s - app.return_value.start_s == pytest.approx(2.0)

    @pytest.mark.parametrize("seconds", [0.5, 1.95, 2.0])
    def test_short_sample_can_play_but_cut_points_cannot_move(self, monkeypatch, seconds):
        from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil

        sound = make_sound(seconds)
        app = VoiceCropApp(sound, "a", None)
        played = []
        monkeypatch.setattr(PlaySoundUtil, "play_sound_async", played.append)
        monkeypatch.setattr(PlaySoundUtil, "stop_sound_async", lambda: False)

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                for key in ("a", "s", "d", "f", "A", "S", "D", "F"):
                    await pilot.press(key)
                    assert (app.start_s, app.end_s) == (0.0, sound.duration)
                await pilot.press("p", "enter")
        asyncio.run(exercise())

        assert len(played) == 1
        np.testing.assert_array_equal(played[0].data, sound.data)
        assert app.return_value is not None
        assert not app.return_value.saved
        assert not app.return_value.cleared

    def test_rendered_dom(self):
        app = VoiceCropApp(make_sound(10.0), "a", None)

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                header = app.query_one("#crop-header").content
                info = app.query_one("#crop-info").content
                bar = app.query_one("#crop-bar").content
                durations = app.query_one("#crop-durations").content
                return header, info, bar, durations

        header, info, bar, durations = asyncio.run(exercise())
        assert "Trim voice sample: a" in str(header)
        assert "A/S" in str(info) and "D/F" in str(info) and "ENTER" in str(info)
        assert "to reset trim" in str(info)
        assert "Avoid trimming in the middle of words" in str(info)
        assert "crop" not in (str(header) + str(info) + str(durations)).lower()
        assert "█" in str(bar) and "^" in str(bar)
        # Waveform block above, gap row, then the bar as the last lines.
        bar_lines = str(bar).split("\n")
        assert len(bar_lines) == 6 + 1 + 3
        assert set(bar_lines[0]) <= {"█", " "}  # waveform top row
        assert bar_lines[6] == " " + " " * (80 - 10 - 2)  # gap row
        # Bar spans the full container width minus the 5-column margins.
        assert len(bar_lines[-1]) == 80 - 10
        assert "Original duration: 10.00s" in str(durations)
        assert "trimmed, 10.00s" in str(durations)


ANSI_RE = __import__("re").compile(r"\x1b\[[0-9;]*m")


class TestRunVoiceCropApp:

    @pytest.mark.parametrize("supported,expected", [
        (False, "The current terminal environment does not support the trim editor"),
        (True, "Trim editor closed without a result"),
    ])
    def test_editor_errors_use_trim_wording(self, monkeypatch, supported, expected):
        import tts_audiobook_tool.textual.voice_crop_app as mod
        monkeypatch.setattr(mod, "can_textual", lambda: supported)
        monkeypatch.setattr(mod.VoiceCropApp, "run", lambda self, **kwargs: None)
        assert mod.run_voice_crop_app(make_sound(), "a", None) == expected


class TestRenderCropBarProgress:

    def test_progress_keeps_layout_and_labels(self):
        plain = render_crop_bar(10.0, 2.0, 8.0, 42)
        playing = render_crop_bar(10.0, 2.0, 8.0, 42, progress=0.5)
        # This progress ends on a whole-cell boundary, so only styles differ.
        assert ANSI_RE.sub("", playing) == ANSI_RE.sub("", plain)
        assert playing.split("\n")[1:] == plain.split("\n")[1:]

    def test_progress_restores_default_from_the_left_edge(self):
        from tts_audiobook_tool.constants import COL_ACCENT
        console = Console()
        accent = Text.from_ansi(COL_ACCENT + "█").get_style_at_offset(console, 0).color
        bar = Text.from_ansi(render_crop_bar(10.0, 2.0, 8.0, 42, progress=0.25).split("\n")[0])
        first_selected = bar.plain.index("█")
        last_selected = bar.plain.rindex("█")
        assert bar.get_style_at_offset(console, first_selected).color is None
        assert bar.get_style_at_offset(console, last_selected).color == accent

    def test_progress_zero_and_full(self):
        from tts_audiobook_tool.constants import COL_ACCENT
        idle = render_crop_bar(10.0, 2.0, 8.0, 42)
        zero = render_crop_bar(10.0, 2.0, 8.0, 42, progress=0.0)
        full = render_crop_bar(10.0, 2.0, 8.0, 42, progress=1.0)
        assert zero == idle
        assert COL_ACCENT in idle.split("\n")[0]
        assert COL_ACCENT not in full.split("\n")[0]
        assert ANSI_RE.sub("", full) == ANSI_RE.sub("", idle)

    def test_progress_changes_color_one_half_at_a_time(self):
        from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM
        console = Console()
        accent = Text.from_ansi(COL_ACCENT + "█").get_style_at_offset(console, 0).color
        dim = Text.from_ansi(COL_DIM + "x").get_style_at_offset(console, 0).color
        # A full 36-slot selection makes one slot exactly 1/36 progress.
        first = Text.from_ansi(render_crop_bar(36, 0, 36, 20, 1 / 36).split("\n")[0])
        second = Text.from_ansi(render_crop_bar(36, 0, 36, 20, 2 / 36).split("\n")[0])
        assert first.plain == "<▌" + "█" * 17 + ">"
        # Native foreground fills the played left half; background accent fills right.
        split_style = first.get_style_at_offset(console, 1)
        assert split_style.bgcolor == accent
        assert split_style.color is None
        assert second.plain == "<" + "█" * 18 + ">"
        assert second.get_style_at_offset(console, 1).color is None
        assert second.get_style_at_offset(console, 1).bgcolor is None
        for text in (first, second):
            # The following unplayed cell stays accented without background spill.
            style = text.get_style_at_offset(console, 2)
            assert style.color == accent and style.bgcolor is None
            # Angle brackets stay dim without accent background spill.
            for offset in (0, 19):
                style = text.get_style_at_offset(console, offset)
                assert style.color == dim and style.bgcolor is None

    @pytest.mark.parametrize("start,end", [(0, 36), (1, 35), (3, 8), (1, 2)])
    @pytest.mark.parametrize("progress", [None, 0, 0.25, 0.5, 0.75, 1])
    def test_progress_colors_exact_prefix_including_half_boundaries(self, start, end, progress):
        from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM
        console = Console()
        accent = Text.from_ansi(COL_ACCENT + "█").get_style_at_offset(console, 0).color
        dim = Text.from_ansi(COL_DIM + "x").get_style_at_offset(console, 0).color
        idle = render_crop_bar(36, start, end, 20).split("\n")
        rendered = render_crop_bar(36, start, end, 20, progress).split("\n")
        text = Text.from_ansi(rendered[0])
        assert len(text.plain) == 20
        assert text.plain.startswith("<") and text.plain.endswith(">")
        for offset in (0, 19):
            style = text.get_style_at_offset(console, offset)
            assert style.color == dim and style.bgcolor is None
        assert rendered[1:] == idle[1:]
        actual_selection = []
        actual_played = []
        for column, glyph in enumerate(text.plain[1:-1], start=1):
            style = text.get_style_at_offset(console, column)
            if style.bgcolor == accent:
                assert glyph == "▌" and style.color is None
                actual_selection.extend([True, True])
                actual_played.extend([True, False])
            else:
                halves = [glyph in "▌█", glyph in "▐█"]
                actual_selection.extend(halves)
                actual_played.extend([selected and style.color is None for selected in halves])
                if any(halves):
                    assert style.color in (None, accent) and style.bgcolor is None
                else:
                    assert glyph == "-"
                    assert style.color == dim and style.bgcolor is None
        played_end = start + (round(progress * (end - start)) if progress is not None else 0)
        assert actual_selection == [start <= slot < end for slot in range(36)]
        assert actual_played == [start <= slot < played_end for slot in range(36)]


class TestPreviewPlayhead:

    def test_preview_ticks_every_fifty_milliseconds(self, monkeypatch):
        import tts_audiobook_tool.textual.voice_crop_app as mod
        app = VoiceCropApp(make_sound(10.0), "a", None)
        timer_calls = []
        bars = []
        now = [0.0]
        monkeypatch.setattr(mod.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(app, "set_interval", lambda interval, callback:
                            timer_calls.append((interval, callback)))
        monkeypatch.setattr(app, "refresh_bar", lambda progress=None:
                            bars.append(render_crop_bar(1.8, 0, 1.8, 20, progress)))
        app._start_preview_animation(1.8, 0.0, 1.0)
        assert timer_calls == [(0.05, app._on_preview_tick)]
        now[0] = 0.05
        app._on_preview_tick()
        now[0] = 0.1
        app._on_preview_tick()
        assert ANSI_RE.sub("", bars[0]).split("\n")[0] == "<▌" + "█" * 17 + ">"
        assert ANSI_RE.sub("", bars[1]).split("\n")[0] == "<" + "█" * 18 + ">"

    def test_preview_animates_with_silence_trim_offset(self):
        from tts_audiobook_tool.constants import COL_ACCENT
        from tts_audiobook_tool.sound import play_sound_util
        import tts_audiobook_tool.textual.voice_crop_app as mod

        # Sound with 1s leading silence and 1s trailing silence inside a 6s
        # span: preview is 4s, playhead starts 1/6 into the segment.
        sr = 24000
        rng = np.random.default_rng(1)
        data = np.concatenate([
            np.zeros(sr, dtype=np.float32),
            rng.uniform(0.05, 0.1, sr * 4).astype(np.float32),
            np.zeros(sr, dtype=np.float32),
        ])
        app = VoiceCropApp(Sound(data, sr), "a", None)
        app.start_s, app.end_s = 0.0, 6.0

        played = []
        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                monkey_target = play_sound_util.PlaySoundUtil
                original = monkey_target.play_sound_async
                monkey_target.play_sound_async = lambda sound: played.append(sound)
                try:
                    await pilot.press("p")
                    await pilot.pause()
                    assert app._preview_timer is not None
                    assert app._preview_duration == pytest.approx(4.0, abs=0.1)
                    assert app._preview_progress_at_start == pytest.approx(1 / 6, abs=0.01)
                    # Halfway through playback.
                    app._preview_started_at = time.monotonic() - app._preview_duration / 2
                    app._on_preview_tick()
                    content = app.query_one("#crop-bar").content
                    assert content.spans, "expected colored spans in the bar"
                    expected = 1 / 6 + 0.5 * (4.0 / 6.0)
                    assert app._last_progress == pytest.approx(expected, abs=0.05)
                    # A split-color cell may use a different glyph, but width,
                    # boundary markers, and labels must stay unchanged.
                    actual = str(content).split("\n")[-3:]
                    plain = ANSI_RE.sub("", render_crop_bar(6.0, 0.0, 6.0, 70)).split("\n")
                    assert len(actual[0]) == len(plain[0])
                    assert actual[1:] == plain[1:]
                    # Finish: tick past the end restores the plain bar.
                    app._preview_started_at -= app._preview_duration
                    app._on_preview_tick()
                    assert app._preview_timer is None
                    assert app._last_progress is None
                finally:
                    monkey_target.play_sound_async = original
        asyncio.run(exercise())


class TestWaveform:

    def test_peak_normalizes_to_max_level(self):
        from tts_audiobook_tool.textual.voice_crop_app import (
            WAVEFORM_LEVELS_PER_ROW,
            WAVEFORM_ROWS,
            compute_waveform_heights,
        )
        # One loud column in the middle, quiet elsewhere.
        data = np.concatenate([
            np.full(100, 0.1, dtype=np.float32),
            np.full(100, 0.8, dtype=np.float32),
            np.full(100, 0.1, dtype=np.float32),
        ])
        heights = compute_waveform_heights(data, 3)
        # (0.1/0.8) ** 1.5 * 12 is about 0.53 levels, rounded to 1.
        assert heights == [1, WAVEFORM_ROWS * WAVEFORM_LEVELS_PER_ROW, 1]

    @pytest.mark.parametrize("gain", [1.0, 0.03])
    def test_gentler_scaling_preserves_peak_normalization(self, gain):
        from tts_audiobook_tool.textual.voice_crop_app import compute_waveform_heights
        data = np.array([1.0, -0.5, 0.25, 0.0]) * gain
        assert compute_waveform_heights(data, 4) == [12, 4, 2, 0]

    def test_nearest_level_rounding_does_not_exaggerate_noise(self):
        from tts_audiobook_tool.textual.voice_crop_app import compute_waveform_heights
        data = np.array([1.0, 0.4, 0.5, 1e-6])
        assert compute_waveform_heights(data, 4) == [12, 3, 4, 0]

    def test_silence_is_blank(self):
        from tts_audiobook_tool.textual.voice_crop_app import compute_waveform_heights
        assert compute_waveform_heights(np.zeros(100, dtype=np.float32), 4) == [0, 0, 0, 0]
        assert compute_waveform_heights(np.zeros(0, dtype=np.float32), 4) == [0, 0, 0, 0]

    @pytest.mark.parametrize("columns", [0, -1])
    def test_nonpositive_sample_count(self, columns):
        from tts_audiobook_tool.textual.voice_crop_app import compute_waveform_heights
        assert compute_waveform_heights(np.ones(4, dtype=np.float32), columns) == []

    def test_audio_shorter_than_bar_count(self):
        from tts_audiobook_tool.textual.voice_crop_app import compute_waveform_heights
        # Empty timeline slices stay silent instead of duplicating samples.
        assert compute_waveform_heights(np.array([1.0, -0.5]), 4) == [0, 12, 0, 4]

    def test_quadrant_mask_lookup(self):
        from tts_audiobook_tool.textual.voice_crop_app import WAVEFORM_BLOCK_CHARS
        # Binary mask bits are upper-left, upper-right, lower-left, lower-right.
        assert WAVEFORM_BLOCK_CHARS == (
            " ", "▗", "▖", "▄", "▝", "▐", "▞", "▟",
            "▘", "▚", "▌", "▙", "▀", "▜", "▛", "█",
        )

    def test_render_shape_and_block_characters(self):
        from tts_audiobook_tool.constants import COL_DIM
        from tts_audiobook_tool.textual.voice_crop_app import WAVEFORM_ROWS, render_waveform
        # Each character pairs two heights. The third has left height 3
        # (one full row plus half the row above), and right height 2.
        text = render_waveform([12, 12, 0, 0, 3, 2], 3)
        lines = text.split("\n")
        assert len(lines) == WAVEFORM_ROWS
        # One leading space aligns character 0 over the bar's first inner char.
        assert all(len(ANSI_RE.sub("", line)) == 4 for line in lines)
        assert ANSI_RE.sub("", lines[0]) == " █  "
        assert ANSI_RE.sub("", lines[-1]) == " █ █"
        assert ANSI_RE.sub("", lines[-2]) == " █ ▖"
        assert COL_DIM in text

    @pytest.mark.parametrize("left,right,glyph", [
        (0, 0, " "), (0, 1, "▗"), (0, 2, "▐"),
        (1, 0, "▖"), (1, 1, "▄"), (1, 2, "▟"),
        (2, 0, "▌"), (2, 1, "▙"), (2, 2, "█"),
    ])
    def test_render_partial_row_uses_quadrants(self, left, right, glyph):
        from tts_audiobook_tool.textual.voice_crop_app import render_waveform
        lines = render_waveform([left, right], 1).split("\n")
        assert ANSI_RE.sub("", lines[-1]) == " " + glyph
        assert all(ANSI_RE.sub("", line) == "  " for line in lines[:-1])

    def test_render_clamps_and_pads_heights(self):
        from tts_audiobook_tool.textual.voice_crop_app import render_waveform
        lines = render_waveform([99, -1, 1], 2).split("\n")
        assert ANSI_RE.sub("", lines[-1]) == " ▌▖"
        assert all(ANSI_RE.sub("", line) == " ▌ " for line in lines[:-1])
        assert all(ANSI_RE.sub("", line) == "   " for line in render_waveform([], 2).split("\n"))
        assert all(ANSI_RE.sub("", line) == "  " for line in render_waveform([0, 0, 12], 1).split("\n"))

    @pytest.mark.parametrize("columns", [0, -2])
    def test_render_nonpositive_width(self, columns):
        from tts_audiobook_tool.textual.voice_crop_app import WAVEFORM_ROWS, render_waveform
        lines = render_waveform([12, 12], columns).split("\n")
        assert len(lines) == WAVEFORM_ROWS
        assert all(ANSI_RE.sub("", line) == " " for line in lines)

    def test_app_samples_two_independent_bars_per_character(self):
        # At the minimum width, 18 characters contain 36 independent samples.
        data = np.tile(np.array([1.0, 0.0], dtype=np.float32), 18)
        app = VoiceCropApp(Sound(data, 24000), "a", None)
        assert app._waveform_heights(20) == [12, 0] * 18
        lines = app._waveform_text(20).split("\n")
        assert all(ANSI_RE.sub("", line) == " " + "▌" * 18 for line in lines)

    def test_waveform_caches_until_width_changes(self, monkeypatch):
        import tts_audiobook_tool.textual.voice_crop_app as mod
        app = VoiceCropApp(make_sound(10.0), "a", None)
        height_calls = []
        render_calls = []
        compute = mod.compute_waveform_heights
        render = mod.render_waveform
        monkeypatch.setattr(mod, "compute_waveform_heights", lambda data, columns:
                            height_calls.append(columns) or compute(data, columns))
        monkeypatch.setattr(mod, "render_waveform", lambda heights, columns:
                            render_calls.append(columns) or render(heights, columns))

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                heights = app._waveform_heights(70)
                text = app._waveform_text(70)
                app.refresh_bar()
                app.refresh_bar()  # same width: both caches reused
                assert app._waveform_heights(70) is heights
                assert app._waveform_text(70) is text
                monkeypatch.setattr(app, "_bar_width", lambda: 90)
                app.refresh_bar()  # width change recomputes both
            assert height_calls == [2 * 68, 2 * 88]
            assert render_calls == [68, 88]
        asyncio.run(exercise())

    def test_shift_p_plays_tail_of_range(self):
        from tts_audiobook_tool.sound import play_sound_util

        app = VoiceCropApp(make_sound(10.0), "a", None)
        app.start_s, app.end_s = 1.0, 6.0
        played = []
        original = play_sound_util.PlaySoundUtil.play_sound_async
        play_sound_util.PlaySoundUtil.play_sound_async = lambda sound: played.append(sound)

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("P")
        try:
            asyncio.run(exercise())
        finally:
            play_sound_util.PlaySoundUtil.play_sound_async = original
        assert len(played) == 1
        # Raw tail: exactly the last 1s of the 1.0-6.0 range, untrimmed.
        assert played[0].duration == pytest.approx(1.0, abs=0.01)

    def test_shift_p_short_range_plays_whole_range(self):
        from tts_audiobook_tool.sound import play_sound_util

        # Samples shorter than the 1s tail preview play their full range.
        app = VoiceCropApp(make_sound(0.5), "a", None)
        played = []
        original = play_sound_util.PlaySoundUtil.play_sound_async
        play_sound_util.PlaySoundUtil.play_sound_async = lambda sound: played.append(sound)

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("P")
        try:
            asyncio.run(exercise())
        finally:
            play_sound_util.PlaySoundUtil.play_sound_async = original
        assert len(played) == 1
        assert played[0].duration == pytest.approx(0.5, abs=0.01)

    def test_shift_p_animates_playhead_from_tail_offset(self):
        from tts_audiobook_tool.sound import play_sound_util

        app = VoiceCropApp(make_sound(10.0), "a", None)
        app.start_s, app.end_s = 1.0, 7.0  # 6s span; tail = last 1s
        played = []
        original = play_sound_util.PlaySoundUtil.play_sound_async
        play_sound_util.PlaySoundUtil.play_sound_async = lambda sound: played.append(sound)

        async def exercise() -> None:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("P")
                await pilot.pause()
                assert app._preview_timer is not None
                # Playhead begins (span - tail)/span into the segment.
                assert app._preview_progress_at_start == pytest.approx(5 / 6, abs=0.01)
                # Halfway through the 1s tail.
                app._preview_started_at = time.monotonic() - app._preview_duration / 2
                app._on_preview_tick()
                assert app._last_progress == pytest.approx(5 / 6 + 0.5 * (1 / 6), abs=0.05)
                # Past the end: plain bar restored.
                app._preview_started_at -= app._preview_duration
                app._on_preview_tick()
                assert app._preview_timer is None
                assert app._last_progress is None
        try:
            asyncio.run(exercise())
        finally:
            play_sound_util.PlaySoundUtil.play_sound_async = original
        assert played and played[0].duration == pytest.approx(1.0, abs=0.01)
