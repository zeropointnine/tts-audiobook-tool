from types import SimpleNamespace
from typing import cast

import numpy as np
import sounddevice as sd

from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream


def _callback(
    stream: SoundDeviceStream,
    frames: int = 8192,
    dac_time: float = 1.0,
    underflow: bool = False,
) -> None:
    """Drives the stream callback directly, bypassing the audio device."""
    out = np.zeros((frames, 1), dtype=np.float32)
    stream._callback(
        out,
        frames,
        SimpleNamespace(outputBufferDacTime=dac_time),
        cast(sd.CallbackFlags, SimpleNamespace(output_underflow=underflow)),
    )


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
