import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.silence_util import GapTrimMap, SilenceGapTrim, SilenceUtil


SR = 16_000


def make_tone(duration: float, amplitude: float = 0.5) -> np.ndarray:
    samples = round(duration * SR)
    t = np.arange(samples, dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def make_gap_sound() -> Sound:
    """ Tone, 1.0s gap, tone, 0.4s gap, tone """
    data = np.concatenate([
        make_tone(0.5),
        np.zeros(round(1.0 * SR), dtype=np.float32),
        make_tone(0.5),
        np.zeros(round(0.4 * SR), dtype=np.float32),
        make_tone(0.5),
    ])
    return Sound(data, SR)


def make_trim(
    original_start: float,
    original_end: float,
    new_start: float,
    new_end: float,
) -> SilenceGapTrim:
    return SilenceGapTrim(
        original_start=original_start,
        original_end=original_end,
        new_start=new_start,
        new_end=new_end,
    )


class TestLimitSilenceGapsRecordsPositions:

    def test_kept_slice_is_centered_on_the_gap(self) -> None:
        sound = make_gap_sound()
        limited, trims = SilenceUtil.limit_silence_gaps(sound, 0.2)

        assert len(trims) == 2
        assert trims[0].original_duration == pytest.approx(1.0, abs=0.05)
        assert trims[1].original_duration == pytest.approx(0.4, abs=0.05)

        for trim in trims:
            mid = (trim.original_start + trim.original_end) / 2.0
            assert trim.new_start == pytest.approx(mid - 0.1, abs=1e-9)
            assert trim.new_end == pytest.approx(mid + 0.1, abs=1e-9)
            assert trim.new_duration == pytest.approx(0.2, abs=1e-9)

        removed = sum(trim.removed_duration for trim in trims)
        assert limited.duration == pytest.approx(sound.duration - removed, abs=0.01)

    def test_zero_threshold_removes_the_whole_gap(self) -> None:
        sound = make_gap_sound()
        _, trims = SilenceUtil.limit_silence_gaps(sound, 0.0)

        assert len(trims) == 2
        for trim in trims:
            mid = (trim.original_start + trim.original_end) / 2.0
            assert trim.new_start == mid
            assert trim.new_end == mid
            assert trim.removed_duration == pytest.approx(trim.original_duration)

    def test_gaps_below_the_threshold_are_not_recorded(self) -> None:
        sound = make_gap_sound()
        limited, trims = SilenceUtil.limit_silence_gaps(sound, 2.0)

        assert trims == []
        assert limited.duration == pytest.approx(sound.duration, abs=1e-9)


class TestGapTrimMap:

    def test_identity_map_passes_times_through(self) -> None:
        gap_map = GapTrimMap.identity(10.0)

        assert gap_map.is_identity
        assert gap_map.new_duration == 10.0
        assert gap_map.map_time(0.0) == 0.0
        assert gap_map.map_time(4.25) == 4.25
        assert gap_map.map_time(10.0) == 10.0

    def test_shifts_are_the_cumulative_removals_before_the_time(self) -> None:
        gap_map = GapTrimMap(10.0, [
            make_trim(2.0, 4.0, 2.75, 3.25),
            make_trim(6.0, 7.0, 6.4, 6.6),
        ])

        assert gap_map.new_duration == pytest.approx(7.7)
        assert gap_map.map_time(0.0) == 0.0
        assert gap_map.map_time(2.0) == pytest.approx(2.0)
        assert gap_map.map_time(3.0) == pytest.approx(2.25)
        assert gap_map.map_time(4.0) == pytest.approx(2.5)
        assert gap_map.map_time(5.0) == pytest.approx(3.5)
        assert gap_map.map_time(6.0) == pytest.approx(4.5)
        assert gap_map.map_time(6.5) == pytest.approx(4.6)
        assert gap_map.map_time(7.0) == pytest.approx(4.7)
        assert gap_map.map_time(10.0) == pytest.approx(7.7)

    def test_times_inside_a_removed_region_clamp_to_the_preceding_piece(self) -> None:
        gap_map = GapTrimMap(10.0, [
            make_trim(2.0, 4.0, 2.75, 3.25),
        ])

        # Removed head and removed tail both clamp to the end of the kept
        # piece that precedes them.
        assert gap_map.map_time(2.1) == pytest.approx(2.0)
        assert gap_map.map_time(2.75) == pytest.approx(2.0)
        assert gap_map.map_time(3.25) == pytest.approx(2.5)
        assert gap_map.map_time(3.9) == pytest.approx(2.5)

    def test_fully_removed_gap_clamps_to_the_gap_start(self) -> None:
        gap_map = GapTrimMap(5.0, [make_trim(1.0, 3.0, 2.0, 2.0)])

        assert gap_map.new_duration == pytest.approx(3.0)
        assert gap_map.map_time(1.5) == pytest.approx(1.0)
        assert gap_map.map_time(2.5) == pytest.approx(1.0)
        assert gap_map.map_time(3.0) == pytest.approx(1.0)
        assert gap_map.map_time(4.0) == pytest.approx(2.0)

    def test_map_is_monotonic_and_endpoints_are_preserved(self) -> None:
        gap_map = GapTrimMap(10.0, [
            make_trim(2.0, 4.0, 2.75, 3.25),
            make_trim(6.0, 7.0, 6.4, 6.6),
        ])

        previous = gap_map.map_time(0.0)
        assert previous == 0.0
        for step in range(1, 1001):
            mapped = gap_map.map_time(step * 0.01)
            assert mapped >= previous - 1e-9
            assert 0.0 <= mapped <= gap_map.new_duration
            previous = mapped
        assert previous == pytest.approx(gap_map.new_duration)

    def test_map_built_from_limit_silence_gaps_matches_the_cut_audio(self) -> None:
        sound = make_gap_sound()
        limited, trims = SilenceUtil.limit_silence_gaps(sound, 0.2)
        gap_map = GapTrimMap(sound.duration, trims)

        assert gap_map.map_time(0.0) == 0.0
        assert gap_map.map_time(sound.duration) == pytest.approx(limited.duration, abs=0.01)

        # The second tone starts at 1.5s, the third at 2.4s in the original.
        # Detection is frame-quantized, hence the wider tolerance.
        assert gap_map.map_time(1.5) == pytest.approx(0.7, abs=0.06)
        assert gap_map.map_time(2.4) == pytest.approx(1.4, abs=0.08)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
