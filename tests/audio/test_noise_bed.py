
import numpy as np
import pytest
from scipy import signal

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.sound.noise_bed import ContinuousNoise, NoiseBed
from concat_test_support import concatenate_streamed

SR = 48_000
LEVEL = -65.0


def lin_to_db(x: float) -> float:
    return 20 * np.log10(x)


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def tone(duration: float) -> np.ndarray:
    t = np.arange(round(duration * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


class TestContinuousNoise:

    def test_chunked_output_equals_one_long_call(self) -> None:
        # The stream continues across calls, so cutting it into segment-sized
        # pieces leaves no seams
        a = ContinuousNoise(SR, seed=1, lowpass_hz=8000, highpass_hz=80)
        b = ContinuousNoise(SR, seed=1, lowpass_hz=8000, highpass_hz=80)
        chunked = np.concatenate([a.next(n) for n in (1000, 1, 4799, 20000)])
        np.testing.assert_allclose(chunked, b.next(len(chunked)), atol=1e-6)

    @pytest.mark.parametrize("lowpass_hz,highpass_hz", [(None, None), (8000, 80)])
    def test_unit_rms(self, lowpass_hz: float | None, highpass_hz: float | None) -> None:
        noise = ContinuousNoise(SR, seed=2, lowpass_hz=lowpass_hz, highpass_hz=highpass_hz).next(SR * 10)
        assert rms(noise) == pytest.approx(1.0, rel=0.05)

    def test_band_limit_removes_air_and_rumble(self) -> None:
        # Relative to pink noise's flat octave bands, the 16 kHz band (11-22 kHz)
        # is cut by over 20 dB and the 31 Hz band clearly reduced; the midrange
        # stays flat
        noise = ContinuousNoise(SR, seed=3, lowpass_hz=NoiseBed.LOWPASS_HZ,
                                highpass_hz=NoiseBed.HIGHPASS_HZ).next(SR * 20)
        f, p = signal.welch(noise, fs=SR, nperseg=8192)

        def band_db(fc: float) -> float:
            m = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
            return 10 * np.log10(np.sum(p[m]))

        ref = band_db(1000)
        assert abs(band_db(2000) - ref) < 1.5
        assert band_db(16000) - ref < -20
        assert band_db(31.25) - ref < -8


class TestNoiseBed:

    def test_constant_level_over_speech_and_silence(self) -> None:
        # The bed is the same everywhere; it doesn't adapt to the content
        bed = NoiseBed(SR, LEVEL)
        silence = np.zeros(SR, dtype=np.float32)
        bed.apply(Sound(np.zeros(round(0.1 * SR), dtype=np.float32), SR))  # past the fade-in
        out_silence = bed.apply(Sound(silence, SR)).data
        speech = tone(1.0)
        out_speech = bed.apply(Sound(speech, SR)).data
        assert lin_to_db(rms(out_silence)) == pytest.approx(LEVEL, abs=0.5)
        assert lin_to_db(rms(out_speech - speech)) == pytest.approx(LEVEL, abs=0.5)

    def test_same_length_and_original_audio_preserved_underneath(self) -> None:
        speech = tone(0.5)
        out = NoiseBed(SR, LEVEL).apply(Sound(speech, SR)).data
        assert len(out) == len(speech)
        assert rms(out - speech) < 10 ** (-60 / 20)

    def test_fades_in_once_at_start_even_across_short_calls(self) -> None:
        # The fade-in spans the first EDGE_FADE_S of the output, however it is
        # chunked; later calls start at full level
        bed = NoiseBed(SR, -20.0)
        fade_n = round(NoiseBed.EDGE_FADE_S * SR)
        quarter = fade_n // 4
        first = bed.apply(Sound(np.zeros(quarter, dtype=np.float32), SR)).data
        rest = bed.apply(Sound(np.zeros(SR, dtype=np.float32), SR)).data
        assert abs(first[0]) < 1e-6
        assert rms(first) < 0.25 * 10 ** (-20 / 20)
        assert lin_to_db(rms(rest[fade_n:])) == pytest.approx(-20.0, abs=0.5)

    def test_is_last_fades_out_to_zero(self) -> None:
        bed = NoiseBed(SR, -20.0)
        out = bed.apply(Sound(np.zeros(SR, dtype=np.float32), SR), is_last=True).data
        assert abs(out[-1]) < 1e-6
        assert lin_to_db(rms(out[round(0.2 * SR):round(0.8 * SR)])) == pytest.approx(-20.0, abs=0.5)

    def test_mismatched_sample_rate_is_left_alone(self) -> None:
        sound = Sound(np.zeros(1000, dtype=np.float32), 24_000)
        assert NoiseBed(SR, LEVEL).apply(sound) is sound


EFFECT_LEVEL = 0.05


def run_concat(phrases: list[Phrase], sounds: list[np.ndarray], room_tone_db: float | None) -> list[np.ndarray]:
    """Concat with real 0.4 s pauses and a constant 0.2 s break "effect"; returns the streamed arrays."""
    return concatenate_streamed(phrases, sounds, SR, effect_level=EFFECT_LEVEL, room_tone_db=room_tone_db)


class TestConcatFlow:

    def test_bed_runs_under_segments_pauses_and_break_effects(self) -> None:
        # One constant bed under everything, break effects included; durations
        # don't change
        sounds = [tone(0.6) for _ in range(3)]
        phrases = [Phrase("One.", Reason.SENTENCE), Phrase("Two.", Reason.SECTION_BREAK),
                   Phrase("Three.", Reason.SENTENCE)]
        streamed = run_concat(phrases, sounds, LEVEL)

        assert [len(x) for x in streamed] == [len(sounds[0]) + round(0.4 * SR),
                                              len(sounds[1]) + round(0.2 * SR),
                                              len(sounds[2]) + round(0.4 * SR)]
        pause = streamed[0][len(sounds[0]):]
        assert lin_to_db(rms(pause)) == pytest.approx(LEVEL, abs=1.0)
        under_speech = streamed[1][:len(sounds[1])] - sounds[1]
        assert lin_to_db(rms(under_speech)) == pytest.approx(LEVEL, abs=1.0)
        under_effect = streamed[1][len(sounds[1]):].astype(np.float64) - EFFECT_LEVEL
        assert lin_to_db(rms(under_effect)) == pytest.approx(LEVEL, abs=1.5)

    def test_output_fades_out_at_the_end(self) -> None:
        streamed = run_concat([Phrase("One.", Reason.SENTENCE)], [tone(0.5)], LEVEL)
        assert abs(streamed[-1][-1]) < 1e-6

    def test_off_adds_nothing(self) -> None:
        sounds = [tone(0.5), tone(0.5)]
        phrases = [Phrase(t, Reason.SENTENCE) for t in ("One.", "Two.")]
        streamed = run_concat(phrases, sounds, None)
        for sound, out in zip(sounds, streamed):
            assert np.array_equal(out[:len(sound)], sound)
            assert rms(out[len(sound):]) == 0.0
