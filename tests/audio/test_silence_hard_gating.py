import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.hard_gate_util import (
    HARD_GATING_NEAR_ZERO_AMPLITUDE,
    HardGatingInfo,
    HardGateUtil,
)


SR = 16_000


def speech(seconds: float) -> np.ndarray:
    """ DC-ish stand-in for speech; avoids sine zero-crossing artifacts. """
    return np.full(round(seconds * SR), 0.5, dtype=np.float32)


def room_tone(seconds: float, db: float = -70.0) -> np.ndarray:
    rng = np.random.default_rng(1234)
    amplitude = 10.0 ** (db / 20.0)
    return (rng.standard_normal(round(seconds * SR)) * amplitude).astype(np.float32)


def zeros(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SR), dtype=np.float32)


def test_natural_pause_with_room_tone_is_not_gated():
    sound = Sound(np.concatenate([speech(0.5), room_tone(1.0), speech(0.5)]), SR)
    info = HardGateUtil.inspect(sound)
    assert isinstance(info, HardGatingInfo)
    assert not info.is_gated
    assert info.gap_count == 0
    assert info.longest_gap_ms == 0.0
    assert info.gated_duration == 0.0
    assert info.absolute_zero_pct == 0.0


def test_digital_silence_gap_is_gated():
    sound = Sound(np.concatenate([speech(0.5), zeros(1.0), speech(0.5)]), SR)
    info = HardGateUtil.inspect(sound)
    assert info.is_gated
    assert info.is_severe
    assert info.gap_count == 1
    assert info.longest_gap_ms == pytest.approx(1000.0)
    assert info.gated_duration == pytest.approx(1.0)
    assert info.gated_sample_pct == pytest.approx(50.0)
    assert info.absolute_zero_pct == pytest.approx(50.0)


def test_gap_shorter_than_minimum_is_ignored_by_default():
    sound = Sound(np.concatenate([speech(0.5), zeros(0.02), speech(0.5)]), SR)
    assert not HardGateUtil.inspect(sound).is_gated
    # ... but is reported once the threshold is lowered below its length
    shorter = HardGateUtil.inspect(sound, min_gap_ms=10.0)
    assert shorter.is_gated
    assert not shorter.is_severe
    assert shorter.longest_gap_ms == pytest.approx(20.0)


def test_near_zero_but_not_exact_zeros_are_detected():
    """ Lossy decoders write tiny non-zero values where the source was silent. """
    rng = np.random.default_rng(7)
    quiet = (rng.standard_normal(round(1.0 * SR)) * 1e-7).astype(np.float32)
    sound = Sound(np.concatenate([speech(0.5), quiet, speech(0.5)]), SR)
    info = HardGateUtil.inspect(sound)
    assert info.is_gated
    assert info.longest_gap_ms == pytest.approx(1000.0, abs=2.0)
    assert info.absolute_zero_pct == 0.0


def test_multichannel_input_is_reduced_before_measuring():
    mono = np.concatenate([speech(0.5), zeros(0.5), speech(0.5)])
    stereo = np.stack([mono, mono], axis=1)
    info = HardGateUtil.inspect(Sound(stereo, SR))
    assert info.is_gated
    assert info.longest_gap_ms == pytest.approx(500.0, abs=1.0)


def test_empty_and_degenerate_sounds_do_not_raise():
    for sound in (Sound(np.zeros(0, dtype=np.float32), SR), Sound(np.zeros(100, dtype=np.float32), 0)):
        info = HardGateUtil.inspect(sound)
        assert not info.is_gated
        assert info.duration == 0.0


def test_describe_mentions_the_measurement():
    gated = HardGateUtil.inspect(Sound(np.concatenate([speech(0.5), zeros(1.0), speech(0.5)]), SR))
    text = gated.describe()
    assert "50%" in text and "1000 ms" in text
    clean = HardGateUtil.inspect(Sound(np.concatenate([speech(0.5), room_tone(0.5)]), SR))
    assert clean.describe().startswith("No hard-gated silence")


def test_amplitude_threshold_is_configurable():
    sound = Sound(np.concatenate([speech(0.2), room_tone(0.5, db=-60.0), speech(0.2)]), SR)
    assert not HardGateUtil.inspect(sound).is_gated
    # A threshold above the natural room tone classifies that pause as gated
    assert HardGateUtil.inspect(
        sound, near_zero_amplitude=HARD_GATING_NEAR_ZERO_AMPLITUDE * 1000.0
    ).is_gated


# ---
# Repair


def gated_sound() -> Sound:
    """ Speech, a digital-zero gap with decaying tails on both sides, speech. """
    tail = np.geomspace(3e-4, 2e-5, round(0.05 * SR)).astype(np.float32)
    return Sound(
        np.concatenate([speech(0.5), tail, zeros(0.4), tail[::-1], speech(0.5)]), SR
    )


def test_fill_preserves_length_and_speech_and_clears_gating():
    sound = gated_sound()
    repaired, report = HardGateUtil.fill_core_runs(sound)
    assert len(repaired.data) == len(sound.data)
    assert report.regions_filled == 1
    assert report.filled_ms >= 400.0 - 1.0
    loud = np.abs(sound.data) > 0.01
    assert np.array_equal(repaired.data[loud], sound.data[loud])
    assert not HardGateUtil.inspect(repaired).is_gated
    assert HardGateUtil.inspect(repaired).floor_db == pytest.approx(-65.0, abs=6.0)


def test_fill_leaves_clean_sound_untouched():
    sound = Sound(np.concatenate([speech(0.5), room_tone(0.5), speech(0.5)]), SR)
    repaired, report = HardGateUtil.fill_core_runs(sound)
    assert report.regions_filled == 0
    assert report.spectrum_source == "none"
    assert np.array_equal(repaired.data, sound.data)


def test_fill_leaves_gate_tails_outside_the_zero_run_untouched():
    sound = gated_sound()
    repaired, _ = HardGateUtil.fill_core_runs(sound)
    tail_index = round(0.5 * SR) + 400  # inside the decaying tail, outside the near-zero core
    assert repaired.data[tail_index] == sound.data[tail_index]


def test_fill_levels_and_spectrum_options():
    sound = gated_sound()
    _, pink = HardGateUtil.fill_core_runs(sound, room_tone="pink", floor_db=-60.0)
    _, white = HardGateUtil.fill_core_runs(sound, room_tone="white")
    assert (pink.spectrum_source, pink.target_db, pink.level_source) == ("pink", -60.0, "configured")
    assert white.spectrum_source == "white"
