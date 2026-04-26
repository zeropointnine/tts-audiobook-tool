import signal
from contextlib import contextmanager
import numpy as np
import sounddevice as sd
import threading
from numpy import ndarray
from typing import Iterator, Optional
from tts_audiobook_tool.util import *


class SoundDeviceStream:
    """
    A class to stream audio data to the default output device in a separate thread.
    This class manages a buffer of audio data and uses the sounddevice library
    to play it back in realtime.

    # TODO using this causes app exit to take a long time?
    """

    @staticmethod
    @contextmanager
    def block_sigint_during_stream_start() -> Iterator[None]:
        """
        On POSIX, block SIGINT while creating the PortAudio stream so the spawned
        audio thread inherits the mask and Ctrl-C stays on the main thread.

        Windows does not provide signal.pthread_sigmask, so skip masking there.
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

    def __init__(self, sample_rate: int):
        """
        Initializes the AudioStreamer.

        Args:
            sample_rate (int): The sample rate for the output audio stream (e.g., 44100).
        """
        # This is the sample_rate for the output audio stream.
        self.sample_rate = sample_rate

        # This is the data buffer from which the audio stream callback function draws from to
        # stream the audio in realtime.
        # A numpy array of float32 is highly appropriate for audio processing.
        self.buffer = np.array([], dtype=np.float32)

        # Cumulative count of samples ever appended via add_data().
        # Together with len(buffer), gives frames already consumed by the callback.
        self.total_samples_added = 0

        # Clock-anchored playback tracking, updated each callback that consumes audio.
        # last_dac_time: outputBufferDacTime from the most recent non-silent callback.
        # last_dac_consumed: cumulative samples consumed *before* that callback's chunk.
        # last_audio_dac_end: DAC time when the last audio sample of that chunk will be heard.
        # Initialise last_dac_time to inf so play_position_samples returns 0 before first callback.
        self.last_dac_time: float = float('inf')
        self.last_dac_consumed: int = 0
        self.last_audio_dac_end: float = 0.0

        # A lock is crucial to ensure thread-safe access to the buffer from both the
        # main thread (adding data) and the audio callback thread (consuming data).
        self.lock = threading.Lock()
        self._stop_requested = threading.Event()

        # The sounddevice stream object. It's None until start() is called.
        self.stream: Optional[sd.OutputStream] = None

        # State flags
        self.is_paused = False

    def _callback(self, outdata: ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        """
        The heart of the audio streamer, called by sounddevice in a separate thread.
        It pulls data from the buffer and sends it to the audio output.
        """
        if self._stop_requested.is_set():
            outdata.fill(0)
            raise sd.CallbackStop

        if status.output_underflow:
            # This can happen if the buffer runs out of data.
            # It's good practice to log or print a warning.
            print("Warning: Output underflow")

        with self.lock:
            if self.is_paused:
                # If paused, stream silence.
                outdata.fill(0)
                return

            # Determine how much data to pull from the buffer
            chunk_size = min(len(self.buffer), frames)

            # Copy data from our buffer to the output buffer
            # Reshape is necessary as the output expects a 2D array (frames, channels)
            outdata[:chunk_size] = self.buffer[:chunk_size].reshape(-1, 1)

            # Remove the copied data from our buffer
            self.buffer = self.buffer[chunk_size:]

            # If our buffer ran out before filling the whole output, fill the rest with silence
            if chunk_size < frames:
                outdata[chunk_size:].fill(0)

            # Anchor playback position to the stream clock so play_position_samples
            # advances continuously and is accurate regardless of audio backend.
            if chunk_size > 0:
                self.last_dac_consumed = self.total_samples_added - len(self.buffer) - chunk_size
                self.last_dac_time = time.outputBufferDacTime
                self.last_audio_dac_end = time.outputBufferDacTime + chunk_size / self.sample_rate

    def add_data(self, data: ndarray) -> tuple[int, int]:
        """
        Adds audio data to the buffer for streaming.

        This method handles converting the data to the required float32 format
        and converting stereo to mono if necessary.

        Args:
            data (ndarray): A numpy array of audio data. Can be int16, float64, etc.
                            and can be mono or stereo.

        Returns:
            (start_sample, end_sample): the half-open range, in cumulative-stream
            sample indices, that this chunk occupies. Useful for callers that
            want to correlate later playback positions with what was added.
        """
        # --- Handle data shape (convert stereo to mono) ---
        if data.ndim > 1 and data.shape[1] > 1:
            # Average the channels to create a mono signal
            data = data.mean(axis=1)

        # --- Handle data type conversion ---
        # If data is integer (like int16), normalize it to float32 in the range [-1.0, 1.0]
        if np.issubdtype(data.dtype, np.integer):
            # The standard for PCM audio is to divide by the max possible value for the bit depth.
            # For int16, this is 32768.
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        # If data is already float but not float32, convert it
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        # Now, append the processed data to the buffer in a thread-safe manner
        with self.lock:
            start = self.total_samples_added
            self.buffer = np.concatenate((self.buffer, data))
            self.total_samples_added += len(data)
            end = self.total_samples_added
        return start, end

    def start(self) -> None:
        """
        Starts the audio stream.

        If there is no data in the buffer, the stream will start and play silence
        until data is added.
        """
        if self.stream and self.stream.active:
            print("Stream is already running.")
            return

        # Create and start the output stream. We assume a mono output (channels=1).
        # Note big block size and latency=high
        try:
            # On POSIX, block SIGINT before creating the stream so PortAudio's
            # audio thread inherits the mask — prevents SIGINT from being
            # delivered to that thread and crashing inside a C++ recursive_mutex.
            with SoundDeviceStream.block_sigint_during_stream_start():
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self._callback,
                    dtype=np.float32,  # We work with float32 internally
                    blocksize=8192,
                    latency="high"
                )
                self.stream.start()
        except Exception as e:
            printt(f"{COL_ERROR}Couldn't open sounddevice output stream: {e}")

    def shut_down(self) -> None:
        """
        Stops and closes the audio stream, releasing all resources.
        The buffer is also cleared.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            # Also clear the buffer on shutdown
            with self.lock:
                self.buffer = np.array([], dtype=np.float32)
                self.total_samples_added = 0
                self.last_dac_time = float('inf')
                self.last_dac_consumed = 0
                self.last_audio_dac_end = 0.0

    @property
    def output_latency(self) -> float:
        """Returns the actual output latency of the PortAudio stream in seconds, or 0 if not started."""
        if self.stream is not None:
            return self.stream.latency  # type: ignore
        return 0.0

    @property
    def buffer_duration(self) -> float:
        """
        Returns the current duration of the audio in the 'external' buffer, in seconds.

        Returns:
            float: The duration of the buffered audio in seconds.
        """
        with self.lock:
            # Duration = number of samples / sample rate
            return len(self.buffer) / self.sample_rate

    @property
    def play_position_samples(self) -> int:
        """
        Current playback cursor in cumulative-stream sample indices — i.e. which
        sample is audible right now.  Advances continuously using the PortAudio
        stream clock (stream.time) anchored against outputBufferDacTime recorded
        in the callback, so it is accurate regardless of audio backend (ALSA,
        PulseAudio, PipeWire, etc.) and does not rely on the reported latency.
        Returns 0 before the first callback has fired.
        """
        stream = self.stream
        if stream is None:
            return 0
        with self.lock:
            dac_time = self.last_dac_time
            base = self.last_dac_consumed
        if dac_time == float('inf'):
            return 0  # no callback has fired yet
        try:
            return max(0, base + int((stream.time - dac_time) * self.sample_rate))
        except sd.PortAudioError:
            # Stream may be stopping/closed concurrently while another thread
            # is sampling playback position during Ctrl-C teardown.
            return 0

    @property
    def play_position_seconds(self) -> float:
        return self.play_position_samples / self.sample_rate

    @property
    def is_playback_complete(self) -> bool:
        """
        True once the last audio sample has actually been heard — i.e. the
        internal buffer is empty AND the stream clock has passed the DAC end
        time of the last chunk.  Use this instead of buffer_duration + sleep
        to wait for playback to truly finish.
        """
        stream = self.stream
        if stream is None:
            return True
        with self.lock:
            buf_empty = len(self.buffer) == 0
            dac_end = self.last_audio_dac_end
        try:
            return buf_empty and stream.time >= dac_end
        except sd.PortAudioError:
            # Treat a concurrently torn-down stream as effectively done for
            # shutdown/wait logic.
            return True

    def pause(self) -> None:
        """
        Pauses the audio stream. While paused, the stream will output silence.
        """
        print("Pausing stream.")
        # This flag is checked by the callback function.
        # Setting a boolean is an atomic operation in Python, but using a lock is best practice
        # for consistency and to avoid subtle race conditions.
        with self.lock:
            self.is_paused = True

    def unpause(self) -> None:
        """
        Resumes the audio stream after it has been paused.
        """
        print("Unpausing stream.")
        with self.lock:
            self.is_paused = False
