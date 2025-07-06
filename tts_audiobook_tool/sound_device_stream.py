import time
import numpy as np
import sounddevice as sd
import threading
from numpy import ndarray
from typing import Optional
from tts_audiobook_tool.util import *


class SoundDeviceStream:
    """
    A class to stream audio data to the default output device in a separate thread.
    This class manages a buffer of audio data and uses the sounddevice library
    to play it back in realtime.

    # TODO using this causes app exit to take a long time?
    """

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

    def add_data(self, data: ndarray) -> None:
        """
        Adds audio data to the buffer for streaming.

        This method handles converting the data to the required float32 format
        and converting stereo to mono if necessary.

        Args:
            data (ndarray): A numpy array of audio data. Can be int16, float64, etc.
                            and can be mono or stereo.
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
            self.buffer = np.concatenate((self.buffer, data))

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


# ==============================================================================
# Demonstration of the AudioStreamer class
# ==============================================================================
if __name__ == '__main__':
    SAMPLE_RATE = 44100

    # 1. Create an instance of the streamer
    streamer = SoundDeviceStream(sample_rate=SAMPLE_RATE)

    # 2. Start the stream. It will play silence for now.
    streamer.start()

    # --- Generate some test audio data (a sine wave) ---
    def generate_sine_wave(frequency, duration, sample_rate):
        t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
        # Using int16 as an example of data that needs conversion
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        return data.astype(np.int16)

    print("\nAdding 2 seconds of a 440 Hz tone...")
    tone_440hz = generate_sine_wave(440, 2.0, SAMPLE_RATE)
    streamer.add_data(tone_440hz)

    # --- Simulate a real-time scenario ---
    # We will add short bursts of audio every 0.5 seconds
    try:
        for i in range(5):
            print(f"Current buffer duration: {streamer.buffer_duration:.2f} seconds")
            time.sleep(1)

        # Demonstrate pause and unpause
        print("\nPausing for 3 seconds...")
        streamer.pause()
        time.sleep(3)

        # Add more data while paused
        print("Adding a 880 Hz tone while paused...")
        tone_880hz = generate_sine_wave(880, 2.0, SAMPLE_RATE)
        streamer.add_data(tone_880hz)
        print(f"Buffer duration after adding data while paused: {streamer.buffer_duration:.2f} seconds")

        print("\nUnpausing...")
        streamer.unpause()
        time.sleep(1)

        # Wait for the buffer to empty before shutting down
        print("\nWaiting for the buffer to finish playing...")
        while streamer.buffer_duration > 0:
            print(f"Buffer remaining: {streamer.buffer_duration:.2f}s")
            time.sleep(0.5)

        print("Buffer is empty.")
        time.sleep(1) # a little extra wait to ensure the last callback runs

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # 3. Important: always shut down the stream
        streamer.shut_down()
        print("Demonstration finished.")