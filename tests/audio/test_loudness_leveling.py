from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_audiobook_tool.app_types import NormalizationType, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.sound.loudness_leveling import LoudnessLeveling
from tts_audiobook_tool.sound.voice_spatial import center_unity_pan_gains
from tests.concat_test_support import concatenate, concatenate_streamed

SR = 48_000
ACX = NormalizationType.DEFAULT.value
STRONGER = NormalizationType.STRONGER.value


def noise(seconds: float, rms: float, seed: int = 0) -> np.ndarray:
    """ Speech stand-in: steady noise at a given RMS """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(round(seconds * SR)) * rms).astype(np.float32)


def db(x: float) -> float:
    return 20 * np.log10(x)


class TestLimiter:

    def test_spike_is_limited_without_touching_the_rest(self) -> None:
        # A single spike well above the ceiling in otherwise quiet audio: only
        # the region around the spike is turned down, and the gain is already
        # reduced shortly before the spike arrives (lookahead).
        data = (np.sin(np.arange(SR) * 2 * np.pi * 200 / SR) * 0.2).astype(np.float64)
        spike_at = SR // 2
        data[spike_at] = 0.95
        out, reduction_db = LoudnessLeveling.limit(data, SR, -6.0)

        assert np.max(np.abs(out)) <= 10 ** (-6 / 20) + 1e-9
        assert reduction_db == pytest.approx(db(0.95) - -6.0, abs=0.01)
        far = round(0.2 * SR)
        np.testing.assert_array_equal(out[:spike_at - far], data[:spike_at - far])
        np.testing.assert_array_equal(out[spike_at + far:], data[spike_at + far:])
        lead = round(LoudnessLeveling.LIMITER_LOOKAHEAD_S * SR) // 2
        assert abs(out[spike_at - lead]) < abs(data[spike_at - lead])

    def test_audio_under_the_ceiling_is_returned_unchanged(self) -> None:
        data = noise(1.0, 0.05).astype(np.float64)
        out, reduction_db = LoudnessLeveling.limit(data, SR, -1.0)
        assert out is data
        assert reduction_db == 0.0

    def test_stereo_channels_share_one_gain(self) -> None:
        # Channel-linked: limiting the loud channel must not shift the image
        left = noise(1.0, 0.3, seed=1).astype(np.float64)
        stereo = np.stack([left, left * 0.25], axis=1)
        out, _ = LoudnessLeveling.limit(stereo, SR, -6.0)
        assert np.max(np.abs(out)) <= 10 ** (-6 / 20) + 1e-9
        np.testing.assert_allclose(out[:, 1], out[:, 0] * 0.25, atol=1e-9)


class TestLevel:

    def test_segments_of_different_loudness_end_at_the_target(self) -> None:
        for rms in (0.08, 0.12, 0.25):
            result = LoudnessLeveling.level(Sound(noise(3.0, rms), SR), ACX)
            after = LoudnessLeveling.measure_lufs(result.sound)
            assert after is not None
            assert after == pytest.approx(ACX.i, abs=0.3)
            assert np.max(np.abs(result.sound.data)) <= 10 ** (LoudnessLeveling.ceiling_db_for(ACX) / 20) + 1e-6

    def test_gain_is_clamped_for_very_quiet_segments(self) -> None:
        # A whisper-level segment is raised, but only by MAX_BOOST_DB
        result = LoudnessLeveling.level(Sound(noise(3.0, 0.004), SR), ACX)
        assert result.gain_db == LoudnessLeveling.MAX_BOOST_DB
        assert result.clamped

    def test_short_segment_gets_the_typical_loudness_gain(self) -> None:
        # Too short to measure: assume it was at the typical un-leveled level
        result = LoudnessLeveling.level(Sound(noise(0.3, 0.1), SR), ACX)
        assert result.measured_lufs is None
        assert result.gain_db == pytest.approx(ACX.i - LoudnessLeveling.REFERENCE_SPEECH_LUFS)

    def test_silent_and_empty_segments_are_safe(self) -> None:
        silent = LoudnessLeveling.level(Sound(np.zeros(SR, dtype=np.float32), SR), ACX)
        assert silent.measured_lufs is None
        assert not np.any(silent.sound.data)
        empty = Sound(np.zeros(0, dtype=np.float32), SR)
        assert LoudnessLeveling.level(empty, ACX).sound is empty

    def test_stereo_output_measures_on_target(self) -> None:
        # BS.1770 sums channel energy: the mono segment is aimed 3 dB lower so
        # the (panned) stereo output measures on target, like a whole-file pass
        import pyloudnorm as pyln
        result = LoudnessLeveling.level(Sound(noise(3.0, 0.1), SR), ACX, pan=0.4, channels=2)
        left, right = center_unity_pan_gains(0.4)
        stereo = np.stack([result.sound.data * left, result.sound.data * right], axis=1)
        assert pyln.Meter(SR).integrated_loudness(stereo) == pytest.approx(ACX.i, abs=0.3)
        assert LoudnessLeveling.reference_offset_db(ACX, 2) == pytest.approx(
            LoudnessLeveling.reference_offset_db(ACX, 1) - 10 * np.log10(2)
        )

    @pytest.mark.parametrize("pan", [-1.0, -0.3, 0.6])
    def test_panned_output_stays_under_the_peak_limit(self, pan: float) -> None:
        # Panning raises the louder channel (center-unity pan law), so the
        # ceiling applied before panning is lowered by that boost
        result = LoudnessLeveling.level(Sound(noise(3.0, 0.3), SR), STRONGER, pan=pan)
        panned_peak = np.max(np.abs(result.sound.data)) * max(center_unity_pan_gains(pan))
        assert panned_peak <= 10 ** ((STRONGER.tp - LoudnessLeveling.TRUE_PEAK_MARGIN_DB) / 20) + 1e-6


def test_mix_limiter_catches_peaks_added_after_leveling() -> None:
    # Eg room reverb adding to already-limited narration: the mixed chunk is
    # brought back under the ceiling; a chunk already under it is untouched
    ceiling = 10 ** (LoudnessLeveling.ceiling_db_for(ACX) / 20)
    loud = Sound(np.stack([noise(1.0, 0.3), noise(1.0, 0.1, seed=3)], axis=1), SR)
    limited, reduction_db = LoudnessLeveling.limit_mix(loud, ACX)
    assert reduction_db > 0
    assert np.max(np.abs(limited.data)) <= ceiling + 1e-6
    quiet = Sound(noise(1.0, 0.01), SR)
    assert LoudnessLeveling.limit_mix(quiet, ACX) == (quiet, 0.0)


class TestConcatWiring:

    def test_concat_levels_each_segment_to_the_target(self) -> None:
        # Adjacent segments that differ by ~10 dB come out at the same loudness
        phrases = [Phrase("One.", Reason.SENTENCE), Phrase("Two.", Reason.SENTENCE)]
        streamed = concatenate_streamed(
            phrases, [noise(3.0, 0.05), noise(3.0, 0.16, seed=2)], SR, loudness=ACX,
        )
        levels = [LoudnessLeveling.measure_lufs(Sound(chunk, SR)) for chunk in streamed]
        assert levels[0] == pytest.approx(ACX.i, abs=0.3)
        assert levels[1] == pytest.approx(ACX.i, abs=0.3)

    def test_room_tone_and_break_effects_move_with_the_target(self) -> None:
        # Both are authored against un-leveled speech, so they are shifted by
        # the same offset as typical speech
        offset = ACX.i - LoudnessLeveling.REFERENCE_SPEECH_LUFS
        appended: list[dict] = []

        def append(sound_in: Sound, **kwargs) -> Sound:
            appended.append(kwargs)
            return sound_in

        bed = MagicMock()
        bed.return_value.apply.side_effect = lambda sound, is_last=False: sound
        with patch("tts_audiobook_tool.concat_util.NoiseBed", bed):
            concatenate(
                [(Phrase("One.", Reason.SENTENCE), "a.flac")],
                render=Sound(noise(1.0, 0.1), SR),
                append=append,
                room_tone_db=-75.0,
                loudness=ACX,
            )
        bed.assert_called_once_with(SR, pytest.approx(-75.0 + offset))
        assert appended[0]["effect_level_offset_db"] == pytest.approx(offset)


def test_ffmpeg_stream_clips_instead_of_wrapping() -> None:
    # Out-of-range floats must saturate; a bare int16 cast would wrap around
    process = MagicMock()
    ConcatUtil.add_audio_to_ffmpeg_stream(process, np.array([1.5, -1.5, 0.5], dtype=np.float32))
    written = np.frombuffer(process.stdin.write.call_args.args[0], dtype=np.int16)
    np.testing.assert_array_equal(written, [32767, -32767, 16383])
