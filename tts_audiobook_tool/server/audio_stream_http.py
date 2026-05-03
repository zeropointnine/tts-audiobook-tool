import io
import json
import queue
import struct
import threading
import time

import numpy as np
import soundfile as sf

_QUEUE_MAXSIZE = 20
# Accumulate this many played samples before encoding a WAV frame (~0.5 s at 48 kHz).
# Smaller = lower latency to HTTP clients; larger = fewer encode calls.
_ENCODE_BLOCK_SAMPLES = 24000


def _encode_pcm(data: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, data, sr, format='WAV', subtype='FLOAT')
    return buf.getvalue()


class AudioStreamHttp:

    def __init__(self):
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []
        self._pcm_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._accumulator = np.empty(0, dtype=np.float32)
        self._accum_sr: int = 48000
        self._sequence = 0
        self._timeline_sample_index = 0
        threading.Thread(target=self._encode_worker, daemon=True).start()

    # ── called from AudioStream._callback (audio thread) ──────────────────

    def on_audio_played(self, data: np.ndarray, sr: int) -> None:
        """Fast path: just enqueue; encoding happens in the background thread."""
        self._pcm_queue.put_nowait((data, sr))

    # ── encode / broadcast thread ──────────────────────────────────────────

    def _encode_worker(self) -> None:
        while True:
            try:
                data, sr = self._pcm_queue.get(timeout=0.2)
            except queue.Empty:
                self._flush()   # audio went silent — push the tail to clients
                continue
            if len(self._accumulator) > 0 and sr != self._accum_sr:
                self._flush()   # sample-rate changed, start a new frame
            self._accumulator = (
                np.concatenate([self._accumulator, data])
                if len(self._accumulator) else data.copy()
            )
            self._accum_sr = sr
            if len(self._accumulator) >= _ENCODE_BLOCK_SAMPLES:
                self._flush()

    def _flush(self) -> None:
        if len(self._accumulator) == 0:
            return
        wav_bytes = _encode_pcm(self._accumulator, self._accum_sr)
        sample_count = len(self._accumulator)
        header = {
            "type": "audio",
            "sequence": self._sequence,
            "sampleRate": self._accum_sr,
            "timelineStartSample": self._timeline_sample_index,
            "sampleCount": sample_count,
            "serverTime": time.time(),
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        body = struct.pack('>I', len(header_bytes)) + header_bytes + wav_bytes
        encoded = struct.pack('>I', len(body)) + body
        self._sequence += 1
        self._timeline_sample_index += sample_count
        self._accumulator = np.empty(0, dtype=np.float32)
        with self._lock:
            snapshot = list(self._clients)
        for q in snapshot:
            try:
                q.put_nowait(encoded)
            except queue.Full:
                pass

    # ── client management ──────────────────────────────────────────────────

    def connect(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        with self._lock:
            self._clients.append(q)
        return q

    def disconnect(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def clear(self) -> None:
        """Drain in-flight PCM and all client queues so server clear() takes effect promptly."""
        # Drain the PCM queue so the encode worker stops accumulating
        while True:
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                break
        self._accumulator = np.empty(0, dtype=np.float32)
        # Drain each client's delivery queue
        with self._lock:
            snapshot = list(self._clients)
        for q in snapshot:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)
