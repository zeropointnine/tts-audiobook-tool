import multiprocessing
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
import sounddevice as sd

from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream


def _callback(
    stream: SoundDeviceStream,
    frames: int = 8192,
    dac_time: float = 1.0,
    underflow: bool = False,
) -> np.ndarray:
    """Drives the stream callback directly, bypassing the audio device."""
    out = np.full((frames, 1), np.nan, dtype=np.float32)
    stream._callback(
        out,
        frames,
        SimpleNamespace(outputBufferDacTime=dac_time),
        cast(sd.CallbackFlags, SimpleNamespace(output_underflow=underflow)),
    )
    return out


@pytest.mark.parametrize("initially_paused", [False, True])
def test_pause_predicate_preserves_and_resumes_samples(initially_paused: bool) -> None:
    pause_event = multiprocessing.Event()
    if initially_paused:
        pause_event.set()
    predicate = pause_event.is_set
    stream = SoundDeviceStream(sample_rate=48000, pause_requested=predicate)
    assert stream.pause_requested is predicate
    samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    stream.add_data(samples)
    first_audio_output = Mock()
    stream.set_first_audio_output_callback(0, first_audio_output)

    consumed = 0
    if not initially_paused:
        np.testing.assert_array_equal(_callback(stream, frames=2)[:, 0], samples[:2])
        consumed = 2
        first_audio_output.assert_called_once_with()
        pause_event.set()

    anchor = (stream.last_dac_time, stream.last_dac_consumed, stream.last_audio_dac_end)
    for dac_time in (2.0, 3.0):
        np.testing.assert_array_equal(_callback(stream, frames=2, dac_time=dac_time), 0)
        np.testing.assert_array_equal(stream.buffer, samples[consumed:])
        assert stream.played_samples == consumed
        assert stream.total_samples_added == len(samples)
        assert (stream.last_dac_time, stream.last_dac_consumed, stream.last_audio_dac_end) == anchor
    if initially_paused:
        first_audio_output.assert_not_called()

    pause_event.clear()
    np.testing.assert_array_equal(
        _callback(stream, frames=8, dac_time=4.0)[:, 0],
        np.concatenate((samples[consumed:], np.zeros(8 - len(samples) + consumed))),
    )
    assert stream.played_samples == len(samples)
    assert len(stream.buffer) == 0
    first_audio_output.assert_called_once_with()


@pytest.mark.parametrize("with_predicate", [False, True])
def test_local_pause_preserves_samples_even_when_predicate_is_false(with_predicate: bool) -> None:
    pause_event = multiprocessing.Event()
    stream = SoundDeviceStream(pause_requested=pause_event.is_set if with_predicate else None)
    samples = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    stream.add_data(samples)

    stream.pause()
    np.testing.assert_array_equal(_callback(stream, frames=2), 0)
    np.testing.assert_array_equal(stream.buffer, samples)
    assert stream.played_samples == 0

    stream.unpause()
    np.testing.assert_array_equal(_callback(stream, frames=4)[:, 0], [0.25, 0.5, 0.75, 0])
    assert stream.played_samples == len(samples)


@pytest.mark.parametrize("configured", [False, True])
def test_start_uses_default_or_configured_output_settings(configured: bool) -> None:
    stream = (
        SoundDeviceStream(48000, blocksize=4096, latency="low")
        if configured else SoundDeviceStream(48000)
    )
    expected_blocksize = 4096 if configured else 8192
    expected_latency = "low" if configured else "high"
    assert stream.blocksize == expected_blocksize
    assert stream.latency == expected_latency
    assert stream.pause_requested is None
    device = Mock()
    with patch(
        "tts_audiobook_tool.sound.sound_device_stream.sd.OutputStream", return_value=device
    ) as output_stream:
        assert stream.start() is True

    output_stream.assert_called_once_with(
        samplerate=48000,
        channels=1,
        callback=stream._callback,
        dtype=np.float32,
        blocksize=expected_blocksize,
        latency=expected_latency,
    )
    device.start.assert_called_once_with()
    device.stop.assert_not_called()
    device.close.assert_not_called()


def test_played_samples_is_a_pure_consumption_count() -> None:
    stream = SoundDeviceStream(sample_rate=48000)
    assert stream.played_samples == 0

    stream.add_data(np.zeros(100, dtype=np.float32))
    assert stream.played_samples == 0

    _callback(stream)
    assert stream.played_samples == 100  # consumed only what was buffered

    # Underflow (empty buffer) must not advance the count.
    _callback(stream, underflow=True)
    _callback(stream, dac_time=1.1, underflow=True)
    assert stream.played_samples == 100

    # New data continues where the count left off.
    stream.add_data(np.zeros(50, dtype=np.float32))
    assert stream.played_samples == 100
    _callback(stream, dac_time=1.2)
    assert stream.played_samples == 150


def test_played_plus_buffer_equals_total_added() -> None:
    stream = SoundDeviceStream(sample_rate=48000)
    stream.add_data(np.zeros(48000, dtype=np.float32))

    _callback(stream)
    assert stream.played_samples + len(stream.buffer) == 48000

    _callback(stream, dac_time=1.1)
    _callback(stream, dac_time=1.2, underflow=True)
    assert stream.played_samples + len(stream.buffer) == 48000

    stream.add_data(np.zeros(50, dtype=np.float32))
    assert stream.played_samples + len(stream.buffer) == 48050


def test_clear_buffer_keeps_scheduled_tail_for_silence_detection() -> None:
    stream = SoundDeviceStream(sample_rate=48000)
    stream.add_data(np.zeros(8192, dtype=np.float32))

    # One full callback consumes the chunk; it is audible until the DAC end
    # time of this chunk (~10.171s at 48kHz).
    _callback(stream, dac_time=10.0)

    stream.clear_buffer()
    assert len(stream.buffer) == 0

    # The dropped chunk's frames are still scheduled at the device, so
    # playback is not complete until the stream clock passes the tail.
    stream.stream = cast(sd.OutputStream, SimpleNamespace(time=10.1))
    assert stream.is_playback_complete is False
    stream.stream = cast(sd.OutputStream, SimpleNamespace(time=10.2))
    assert stream.is_playback_complete is True


def test_clear_buffer_without_scheduled_audio_is_immediately_silent() -> None:
    stream = SoundDeviceStream(sample_rate=48000)
    stream.stream = cast(sd.OutputStream, SimpleNamespace(time=5.0))

    stream.clear_buffer()

    assert stream.is_playback_complete is True
