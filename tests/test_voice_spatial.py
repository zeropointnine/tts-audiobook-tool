"""Parallel narration panning and wet room DSP, independent of channel policy."""

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.room_reverb import RoomReverbProcessor, StreamingReverb
from tts_audiobook_tool.sound.voice_spatial import VoiceSpatialProcessor, center_unity_pan_gains

SR = 24_000


def sound(data) -> Sound:
    return Sound(np.asarray(data, dtype=np.float32), SR)


def noise(n=4000) -> Sound:
    return sound(np.random.default_rng(4).normal(0, 0.03, n))


def room(width=0.0) -> RoomReverbProcessor:
    return RoomReverbProcessor(SR, "small", 17.0, width=width)


@pytest.mark.parametrize("pan", [-1, -0.75, -0.25, 0, 0.25, 0.75, 1])
def test_pan_constant_power_and_direction(pan):
    left, right = center_unity_pan_gains(pan)
    assert left * left + right * right == pytest.approx(2)
    assert (right > left) == (pan > 0)
    if pan == 0:
        assert (left, right) == (1, 1)
    if pan == 0.25:
        assert left == pytest.approx(0.7856949584)
        assert right == pytest.approx(1.1758756024)


@pytest.mark.parametrize("pan", [-2, 2, np.nan, np.inf])
def test_invalid_pan_rejected(pan):
    with pytest.raises(ValueError):
        center_unity_pan_gains(pan)


def test_pan_without_reverb_and_boundary_preserve_shape_dtype_and_input():
    source = sound([0.2, -0.3, 0.4, -0.1])
    before = source.data.copy()
    output = VoiceSpatialProcessor(2).apply(source, pan=0.25, send_end=2)
    left, right = center_unity_pan_gains(0.25)
    np.testing.assert_allclose(output.data[:2], before[:2, None] * [left, right])
    np.testing.assert_array_equal(output.data[2:], np.repeat(before[2:, None], 2, axis=1))
    np.testing.assert_array_equal(source.data, before)
    assert output.data.dtype == source.data.dtype
    assert output.data.shape == (4, 2)
    assert output.sr == source.sr


@pytest.mark.parametrize("with_reverb", [False, True])
def test_mono_does_not_fold_pan_gains(with_reverb):
    source = noise()
    a = VoiceSpatialProcessor(1, room() if with_reverb else None).apply(source, pan=0.25, is_last=True)
    b = room().apply(source, is_last=True) if with_reverb else source
    np.testing.assert_allclose(a.data, b.data, atol=1e-7)


@pytest.mark.parametrize("width", [0.0, 1.0])
@pytest.mark.parametrize("is_last", [False, True])
def test_center_matches_legacy_apply_including_end_fade(width, is_last):
    source = noise()
    legacy = room(width).apply(source, send_end=2000, is_last=is_last).data
    output = VoiceSpatialProcessor(2, room(width)).apply(source, send_end=2000, is_last=is_last).data
    if width == 0:
        legacy = np.repeat(legacy[:, None], 2, axis=1)
    np.testing.assert_allclose(output, legacy, atol=1e-7)


def test_stereo_output_does_not_enable_room_spread():
    source = noise()
    output = VoiceSpatialProcessor(2, room()).apply(source, pan=0.25).data
    gains = center_unity_pan_gains(0.25)
    wet = output - source.data[:, None] * gains
    np.testing.assert_allclose(wet[:, 0], wet[:, 1], atol=1e-7)


def test_voice_change_does_not_reposition_tail():
    source = noise(2000)
    silence = sound(np.zeros(4000))
    outputs = []
    for first_pan, next_pan in [(0.25, -0.75), (-0.25, 0.75)]:
        processor = VoiceSpatialProcessor(2, room(1))
        processor.apply(source, pan=first_pan)
        outputs.append(processor.apply(silence, pan=next_pan, send_end=0).data)
    assert np.any(outputs[0])
    assert not np.allclose(outputs[0][:, 0], outputs[0][:, 1])
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_segment_partitioning_preserves_continuous_wet_state():
    source = noise(5000)
    whole = VoiceSpatialProcessor(2, room(1)).apply(source, pan=0.25).data
    processor = VoiceSpatialProcessor(2, room(1))
    chunked = np.concatenate([
        processor.apply(Sound(source.data[a:b], SR), pan=0.25).data
        for a, b in [(0, 3), (3, 701), (701, 5000)]
    ])
    np.testing.assert_allclose(chunked, whole, atol=1e-7)


def test_break_effect_stays_centered_dry_and_does_not_feed_room():
    speech = noise(2000).data
    effect = np.full(500, 0.1, dtype=np.float32)
    outputs = []
    tails = []
    for suffix in [effect, np.zeros_like(effect)]:
        processor = VoiceSpatialProcessor(2, room(1))
        outputs.append(processor.apply(sound(np.concatenate([speech, suffix])), pan=0.25, send_end=len(speech)).data)
        tails.append(processor.apply(sound(np.zeros(3000)), pan=-1, send_end=0).data)
    delta = outputs[0] - outputs[1]
    np.testing.assert_allclose(delta[:len(speech)], 0, atol=1e-7)
    np.testing.assert_allclose(delta[len(speech):], 0.1, atol=1e-7)
    np.testing.assert_allclose(tails[0], tails[1], atol=1e-7)


def test_end_fade_excludes_panned_direct_sound():
    source = noise()
    output = VoiceSpatialProcessor(2, room(1)).apply(source, pan=0.25, is_last=True).data
    np.testing.assert_allclose(output[-1], source.data[-1] * np.array(center_unity_pan_gains(0.25)), atol=1e-7)


def test_clipping_only_after_final_sum_and_no_integer_wraparound():
    # Both the wet and panned direct path can exceed unity BEFORE cancellation.
    reverb = room()
    reverb._reverbs = [StreamingReverb(np.array([-2.0]), 0)]
    output = VoiceSpatialProcessor(2, reverb).apply(sound([0.8]), pan=1).data
    expected = 0.8 * np.array(center_unity_pan_gains(1)) - 1.6
    np.testing.assert_allclose(output[0], np.clip(expected, -1, 1), atol=1e-7)
    assert output[0, 1] == pytest.approx(-0.46862915)
    peak = VoiceSpatialProcessor(2).apply(sound([1, -1]), pan=1).data
    np.testing.assert_array_equal(peak[:, 1], [1, -1])
    # Mirror ffmpeg's final PCM conversion: clipped samples cannot wrap signs.
    pcm = (peak * 32767).astype(np.int16)
    np.testing.assert_array_equal(pcm[:, 1], [32767, -32767])


def test_wet_only_is_unclipped_and_legacy_still_clips():
    def loud_room():
        reverb = room()
        reverb._reverbs = [StreamingReverb(np.array([3.0]), 0)]
        return reverb
    source = sound([0.8])
    assert loud_room().process_wet(source)[0] == pytest.approx(2.4)
    assert loud_room().apply(source).data[0] == 1


@pytest.mark.parametrize("channels", [1, 2])
def test_empty_input_keeps_output_shape(channels):
    output = VoiceSpatialProcessor(channels, room()).apply(sound([]), is_last=True)
    assert output.data.shape == ((0,) if channels == 1 else (0, 2))


def test_mono_output_rejects_stereo_room():
    with pytest.raises(ValueError):
        VoiceSpatialProcessor(1, room(1))


def test_legacy_invalid_inputs_still_pass_through():
    for source in [sound([]), Sound(noise().data, 48_000), sound(np.zeros((5, 2)))]:
        assert room().apply(source) is source


@pytest.mark.parametrize("channels,width", [(1, 0), (2, 0), (2, 1)])
def test_drain_emits_remaining_decay_without_fading_speech(channels, width):
    source = noise()
    processor = VoiceSpatialProcessor(channels, room(width))
    speech = processor.apply(source, pan=0.25).data
    reference = VoiceSpatialProcessor(channels, room(width))
    expected_speech = reference.apply(source, pan=0.25).data
    expected_tail = reference.apply(sound(np.zeros(SR)), send_end=0).data
    tail = processor.drain_tail(dtype=source.data.dtype)
    assert tail is not None
    assert tail.sr == SR and tail.data.dtype == source.data.dtype
    assert tail.data.ndim == channels
    assert 0 < len(tail.data) < SR
    fade_n = round(RoomReverbProcessor.END_FADE_S * SR)
    np.testing.assert_array_equal(speech, expected_speech)
    np.testing.assert_allclose(tail.data[:-fade_n], expected_tail[:len(tail.data)-fade_n], atol=1e-7)
    np.testing.assert_allclose(tail.data[-fade_n:],
                               expected_tail[len(tail.data)-fade_n:len(tail.data)]
                               * (np.linspace(1, 0, fade_n) if channels == 1 else np.linspace(1, 0, fade_n)[:, None]),
                               atol=1e-7)
    assert np.any(tail.data != 0)
    np.testing.assert_array_equal(tail.data[-1], 0)
    assert processor.drain_tail() is None
    np.testing.assert_array_equal(processor.apply(sound(np.zeros(SR)), send_end=0).data, 0)


def test_drain_off_or_empty_does_not_add_silence():
    assert VoiceSpatialProcessor(2).drain_tail() is None
    assert VoiceSpatialProcessor(2, room(1)).drain_tail() is None


def test_drain_after_pause_only_emits_residual_tail_not_full_room_length():
    def impulse_room():
        reverb = room()
        reverb._reverbs = [StreamingReverb(np.array([0.0, .4, .3, .2, .1]), 0)]
        return reverb
    processor = VoiceSpatialProcessor(1, impulse_room())
    processor.apply(sound([1]))
    processor.apply(sound([0, 0]), send_end=0)
    tail = processor.drain_tail()
    assert tail is not None
    # Two of the four pending frames already played during the pause.
    assert len(tail.data) == 2
    assert processor.drain_tail() is None
