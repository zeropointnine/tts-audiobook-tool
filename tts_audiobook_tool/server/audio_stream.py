import threading
from collections import deque
from typing import Callable

import numpy as np
import sounddevice as sd

from tts_audiobook_tool.app_types import Sound

SAMPLE_RATE = 48000
BLOCKSIZE = 4096 # ~85ms latency

class AudioStream:
    """
    Manages real-time audio playback via a sounddevice OutputStream.

    Incoming audio (Sound objects at any sample rate) is resampled to SAMPLE_RATE,
    mixed to mono, and queued in an internal deque. The OutputStream callback drains
    the deque block-by-block on a background audio thread.

    Supports:
    - Muting without stopping the stream (set_is_mute)
    - A playback listener hook for tapping the live PCM output (e.g. HTTP streaming)
    - Querying remaining buffered duration (get_seconds_left)
    - Flushing the buffer mid-stream (clear)
    """

    def __init__(self):
        self._data_buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._playback_listener: Callable[[np.ndarray, int], None] | None = None
        self._is_mute: bool = False
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            latency="high", # extra tolerance to protect against 'cpu starvation'
            callback=self._callback,
        )
        self._stream.start()

    def set_playback_listener(self, listener: Callable[[np.ndarray, int], None]) -> None:
        """Register a callable invoked with (pcm_block, sample_rate) for each played block."""
        self._playback_listener = listener

    def set_is_mute(self, mute: bool) -> None:
        self._is_mute = mute

    def _callback(self, outdata: np.ndarray, frames: int, _time, _status):
        outdata.fill(0)
        remaining = frames
        write_pos = 0
        with self._lock:
            while remaining > 0 and self._data_buffer:
                chunk = self._data_buffer[0]
                take = min(len(chunk), remaining)
                outdata[write_pos:write_pos + take, 0] = chunk[:take]
                write_pos += take
                remaining -= take
                if take == len(chunk):
                    self._data_buffer.popleft()
                else:
                    self._data_buffer[0] = chunk[take:]
        # Notify HTTP feed outside the lock — copy the slice of outdata that was filled.
        # Do this before zeroing outdata so the HTTP stream always receives real audio.
        if write_pos > 0 and self._playback_listener is not None:
            self._playback_listener(outdata[:write_pos, 0].copy(), SAMPLE_RATE)
        if self._is_mute:
            outdata.fill(0)

    def append(self, sound: Sound):
        data = sound.data
        if data.ndim > 1:
            data = data.mean(axis=0)  # mix to mono (channel-first)
        data = data.astype(np.float32)
        if sound.sr != SAMPLE_RATE:
            old_len = len(data)
            new_len = int(old_len * SAMPLE_RATE / sound.sr)
            data = np.interp(
                np.linspace(0, old_len - 1, new_len),
                np.arange(old_len),
                data,
            ).astype(np.float32)
        with self._lock:
            self._data_buffer.append(data)

    def get_seconds_left(self) -> float:
        with self._lock:
            return sum(len(chunk) for chunk in self._data_buffer) / SAMPLE_RATE

    def clear(self):
        with self._lock:
            self._data_buffer.clear()
