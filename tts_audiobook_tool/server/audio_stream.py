import threading
from collections import deque

import numpy as np
import sounddevice as sd

from tts_audiobook_tool.app_types import Sound

SAMPLE_RATE = 48000
BLOCKSIZE = 4096 # ~85ms latency

class AudioStream:

    def __init__(self):
        self._data_buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            latency="high", # extra tolerance to protect against 'cpu starvation'
            callback=self._callback,
        )
        self._stream.start()

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
