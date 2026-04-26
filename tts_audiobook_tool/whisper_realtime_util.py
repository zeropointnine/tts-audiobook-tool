import collections
import queue
import signal
from contextlib import contextmanager
import threading
import time
from typing import Callable, Iterator

import numpy as np
import sounddevice as sd
from faster_whisper.transcribe import Segment

from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.stt import Stt


@contextmanager
def block_sigint_during_stream_start() -> Iterator[None]:
    """
    On POSIX, block SIGINT while creating the PortAudio stream so the spawned
    audio thread inherits the mask and Ctrl-C stays on the main thread.

    Windows does not expose signal.pthread_sigmask, so stream creation there is
    already the best available behavior and should proceed without masking.
    """
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:
        yield
        return

    old_mask = pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, old_mask)


class WhisperRealTimeUtil:
    """
    Captures the default microphone and fires a callback with transcribed Segment
    objects as each utterance completes.

    Adaptive energy-based chunking: a chunk is dispatched when energy falls back
    near the recent ambient floor for at least `silence_duration_s` seconds after
    speech, or when an utterance exceeds `max_chunk_duration_s`.
    """

    BLOCKSIZE = 1024  # frames per PortAudio callback (~64ms at 16kHz)

    def __init__(
        self,
        prefs: Prefs,
        on_transcription: Callable[[list[Segment]], None],
        language: str | None = None,
        word_timestamps: bool = True,
        silence_threshold: float = 0.01,
        silence_duration_s: float = 0.5,
        max_chunk_duration_s: float = 8.0,
        min_chunk_duration_s: float = 1.5,
        noise_window_s: float = 6.0,
        speech_start_noise_ratio: float = 1.5,
        silence_noise_ratio: float = 1.4,
        peak_silence_ratio: float = 0.6,
        pre_speech_pad_s: float = 0.5,
        on_chunk_dispatched: Callable[[float], None] | None = None,
        debug_vad: bool = False,
        debug_vad_interval_s: float = 0.5,
    ):
        self.prefs = prefs
        self.on_transcription = on_transcription
        self.language = language
        self.word_timestamps = word_timestamps
        self.silence_threshold = silence_threshold
        self.silence_duration_s = silence_duration_s
        self.max_chunk_duration_s = max_chunk_duration_s
        self.min_chunk_duration_s = min_chunk_duration_s
        self.noise_window_s = noise_window_s
        self.speech_start_noise_ratio = speech_start_noise_ratio
        self.silence_noise_ratio = silence_noise_ratio
        self.peak_silence_ratio = peak_silence_ratio
        self.pre_speech_pad_s = pre_speech_pad_s
        self.debug_vad = debug_vad
        self.debug_vad_interval_s = debug_vad_interval_s
        self.on_chunk_dispatched = on_chunk_dispatched

        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._paused_event = threading.Event()
        self._stream: sd.InputStream | None = None
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()

        Stt.set_variant(self.prefs.stt_variant)
        Stt.set_config(self.prefs.stt_config)

        self._worker_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
        )
        self._worker_thread.start()

        with block_sigint_during_stream_start():
            self._stream = sd.InputStream(
                samplerate=WHISPER_SAMPLERATE,
                channels=1,
                dtype=np.float32,
                blocksize=self.BLOCKSIZE,
                callback=self._audio_callback,
            )
            self._stream.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._audio_queue.put(None)  # unblock worker
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
            self._worker_thread = None

    @property
    def is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def flush(self) -> None:
        """Discard all buffered mic audio and reset processing state."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._flush_event.set()

    def pause(self) -> None:
        """
        Stop accepting/processing mic audio until resumed.

        Blocks until any in-flight realtime transcribe finishes. Without this
        wait, a transcribe already running on the worker thread can overlap
        with a transcribe started elsewhere (e.g. the response-turn phrase
        timing path) on the same shared faster-whisper model — concurrent
        access to one CTranslate2 model segfaults natively.
        """
        self._paused_event.set()
        self.flush()
        with Stt.inference_lock:
            pass

    def resume(self) -> None:
        """Resume mic audio capture/processing with fresh state."""
        self.flush()
        self._paused_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._paused_event.is_set()

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        if self._stop_event.is_set():
            raise sd.CallbackStop
        if self._paused_event.is_set():
            return
        self._audio_queue.put_nowait(indata[:, 0].copy())

    def _processing_loop(self) -> None:
        buffer = np.array([], dtype=np.float32)
        silence_samples = 0
        speech_detected = False
        speech_peak_rms = 0.0

        silence_samples_needed = int(self.silence_duration_s * WHISPER_SAMPLERATE)
        min_samples = int(self.min_chunk_duration_s * WHISPER_SAMPLERATE)
        max_samples = int(self.max_chunk_duration_s * WHISPER_SAMPLERATE)
        pre_speech_pad_samples = int(self.pre_speech_pad_s * WHISPER_SAMPLERATE)
        noise_window_blocks = max(1, int(self.noise_window_s * WHISPER_SAMPLERATE / self.BLOCKSIZE))
        min_noise_blocks = max(3, int(0.3 * WHISPER_SAMPLERATE / self.BLOCKSIZE))
        recent_rms: collections.deque[float] = collections.deque(maxlen=noise_window_blocks)
        last_debug_at = 0.0

        def recent_noise_floor() -> float:
            """
            Return a robust estimate of current ambient input level.

            The old endpointing used a fixed RMS threshold, so a mic/noise-floor
            change could make ordinary room tone look like continuous speech.  A
            low percentile tracks background hum/fan noise while mostly ignoring
            brief speech peaks.
            """
            if not recent_rms:
                return max(1e-6, self.silence_threshold / self.speech_start_noise_ratio)
            return max(1e-6, float(np.percentile(np.array(recent_rms), 20)))

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:  # stop sentinel
                break

            if self._flush_event.is_set() or self._paused_event.is_set():
                buffer = np.array([], dtype=np.float32)
                silence_samples = 0
                speech_detected = False
                speech_peak_rms = 0.0
                recent_rms.clear()
                self._flush_event.clear()
                if self._paused_event.is_set():
                    continue

            buffer = np.concatenate((buffer, chunk))

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            noise_floor = recent_noise_floor()

            # Confirm that the buffer actually contains speech rather than a
            # steady elevated noise floor.  The absolute threshold preserves the
            # previous quiet-room sensitivity; the relative threshold adapts when
            # the mic/background level changes.
            has_noise_history = len(recent_rms) >= min_noise_blocks
            speech_start_threshold = max(
                self.silence_threshold if has_noise_history else self.silence_threshold * 2.0,
                noise_floor * self.speech_start_noise_ratio,
            )
            if not speech_detected and rms >= speech_start_threshold:
                speech_detected = True
                speech_peak_rms = rms
                if len(buffer) > pre_speech_pad_samples:
                    buffer = buffer[-pre_speech_pad_samples:]
            elif speech_detected:
                speech_peak_rms = max(speech_peak_rms, rms)

            # Once speech has been seen, treat return-to-background as silence.
            # The gate follows the ambient floor but is capped relative to the
            # utterance peak so speech-polluted noise history cannot make quiet
            # parts of the utterance look like silence.
            silence_gate = max(
                self.silence_threshold,
                min(
                    noise_floor * self.silence_noise_ratio,
                    speech_peak_rms * self.peak_silence_ratio,
                ),
            )

            if speech_detected and rms <= silence_gate:
                silence_samples += len(chunk)
            elif speech_detected:
                silence_samples = 0

            should_dispatch = (
                (speech_detected and silence_samples >= silence_samples_needed
                 and len(buffer) >= min_samples)
                or (speech_detected and len(buffer) >= max_samples)
            )

            if self.debug_vad:
                now = time.perf_counter()
                if now - last_debug_at >= self.debug_vad_interval_s or should_dispatch:
                    last_debug_at = now
                    print(
                        "[VAD] "
                        f"rms={rms:.5f} "
                        f"floor={noise_floor:.5f} "
                        f"start>={speech_start_threshold:.5f} "
                        f"silence<={silence_gate:.5f} "
                        f"peak={speech_peak_rms:.5f} "
                        f"silence={silence_samples / WHISPER_SAMPLERATE:.2f}s "
                        f"buffer={len(buffer) / WHISPER_SAMPLERATE:.2f}s "
                        f"history={len(recent_rms)}/{noise_window_blocks} "
                        f"speech={speech_detected} "
                        f"dispatch={should_dispatch}",
                        flush=True,
                    )

            if should_dispatch and len(buffer) > 0:
                if self.on_chunk_dispatched:
                    self.on_chunk_dispatched(len(buffer) / WHISPER_SAMPLERATE)
                self._transcribe_and_callback(buffer)
                buffer = np.array([], dtype=np.float32)
                silence_samples = 0
                speech_detected = False
                speech_peak_rms = 0.0
            elif not speech_detected and len(buffer) >= max_samples:
                # Drop old non-speech audio so waiting quietly before speaking
                # does not force the next utterance to wait for the full window
                # or send a giant leading-silence buffer to Whisper.
                if pre_speech_pad_samples > 0:
                    buffer = buffer[-pre_speech_pad_samples:]
                else:
                    buffer = np.array([], dtype=np.float32)

            recent_rms.append(rms)

        if len(buffer) > 0 and speech_detected:
            if self.on_chunk_dispatched:
                self.on_chunk_dispatched(len(buffer) / WHISPER_SAMPLERATE)
            self._transcribe_and_callback(buffer)

    def _transcribe_and_callback(self, audio: np.ndarray) -> None:
        try:
            with Stt.inference_lock:
                segments, _ = Stt.get_whisper().transcribe(
                    audio,
                    word_timestamps=self.word_timestamps,
                    language=self.language,
                )
                segments_list = list(segments)
            if segments_list:
                self.on_transcription(segments_list)
        except Exception as e:
            print(f"WhisperRealTimeUtil transcription error: {e}")
