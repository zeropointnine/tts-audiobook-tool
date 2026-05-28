import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil


def make_tone(sr: int, seconds: float, amplitude: float = 0.3, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (np.sin(2 * np.pi * hz * t) * amplitude).astype(np.float32)


def make_noise(sr: int, seconds: float, amplitude: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, amplitude, int(sr * seconds)).astype(np.float32)


def test_trim_trailing_token_noise_trims_loud_isolated_terminal_artifact() -> None:
    sr = 48000
    speech = make_tone(sr, 0.8, amplitude=0.35)
    quiet = np.zeros(int(sr * 0.09), dtype=np.float32)
    artifact = make_noise(sr, 0.035, amplitude=0.14)
    sound = Sound(np.concatenate([speech, quiet, artifact]), sr)

    result = SoundExtraUtil.trim_trailing_token_noise(sound)

    assert result.duration == (len(speech) + len(quiet)) / sr


def test_trim_trailing_token_noise_trims_subtle_ch_like_artifact() -> None:
    sr = 48000
    speech = make_tone(sr, 0.8, amplitude=0.35)
    quiet = make_noise(sr, 0.09, amplitude=0.002, seed=1)
    artifact = make_noise(sr, 0.04, amplitude=0.02, seed=2)
    sound = Sound(np.concatenate([speech, quiet, artifact]), sr)

    result = SoundExtraUtil.trim_trailing_token_noise(sound)

    assert result.duration == (len(speech) + len(quiet)) / sr


def test_trim_trailing_token_noise_allows_tiny_silence_after_artifact() -> None:
    sr = 48000
    speech = make_tone(sr, 0.8, amplitude=0.35)
    quiet = np.zeros(int(sr * 0.04), dtype=np.float32)
    artifact = make_noise(sr, 0.035, amplitude=0.10, seed=4)
    trailing_silence = np.zeros(int(sr * 0.005), dtype=np.float32)
    sound = Sound(np.concatenate([speech, quiet, artifact, trailing_silence]), sr)

    result = SoundExtraUtil.trim_trailing_token_noise(sound)

    assert result.duration == (len(speech) + len(quiet)) / sr


def test_trim_trailing_token_noise_does_not_trim_valid_final_consonant_without_quiet_lead_in() -> None:
    sr = 48000
    speech = make_tone(sr, 0.8, amplitude=0.35)
    final_consonant = make_noise(sr, 0.035, amplitude=0.10, seed=3)
    sound = Sound(np.concatenate([speech, final_consonant]), sr)

    result = SoundExtraUtil.trim_trailing_token_noise(sound)

    assert result.duration == sound.duration


def test_trim_trailing_token_noise_does_not_trim_clean_ending() -> None:
    sr = 48000
    speech = make_tone(sr, 0.8, amplitude=0.35)
    quiet = np.zeros(int(sr * 0.12), dtype=np.float32)
    sound = Sound(np.concatenate([speech, quiet]), sr)

    result = SoundExtraUtil.trim_trailing_token_noise(sound)

    assert result.duration == sound.duration
