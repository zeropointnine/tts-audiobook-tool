import threading
import uuid

import numpy as np
import sounddevice as sd

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


class PlaySoundUtil:
    """
    Provides asynchronous playback and cancellation for sound files and in-memory audio.
    Manages a single helper thread, allows for a single sound at a time.
    """

    _current_playback_thread: threading.Thread | None = None
    _stop_playback_event: threading.Event | None = None
    _current_sound_id = ""
    _playback_operation_lock = threading.RLock()
    _playback_state_lock = threading.Lock()

    @staticmethod
    def play_sound_file_async(path: str) -> tuple[str, str]:
        """Plays a sound file. Returns its sound id and any load error."""
        result = SoundFileUtil.load(path, target_sr=APP_SAMPLE_RATE)
        if isinstance(result, str):
            return "", result

        return PlaySoundUtil.play_sound_async(result), ""

    @staticmethod
    def play_sound_async(sound: Sound) -> str:
        """
        Plays in-memory sound data asynchronously.
        A new playback will cancel the previous one.
        Eats errors silently.
        Returns an id representing this playback.
        """
        with PlaySoundUtil._playback_operation_lock:
            PlaySoundUtil.stop_sound_async()
            sound_id = uuid.uuid4().hex
            stop_event = threading.Event()

            def play_stream_from_data(sound_data: np.ndarray, samplerate: int, channels: int):
                try:
                    stream_finished_event = threading.Event()
                    current_frame = 0

                    def callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
                        nonlocal current_frame
                        if stop_event.is_set():
                            raise sd.CallbackStop

                        # Calculate how many frames are left in the sound_data
                        frames_to_read = min(frames, len(sound_data) - current_frame)

                        if frames_to_read > 0:
                            # Copy data from sound_data to outdata
                            outdata[:frames_to_read] = sound_data[current_frame : current_frame + frames_to_read]
                            # Fill remaining part of outdata with zeros if less data was read than requested
                            if frames_to_read < frames:
                                outdata[frames_to_read:] = 0
                            current_frame += frames_to_read
                        else:
                            # No more data to play
                            raise sd.CallbackStop

                    def set_event_on_finish():
                        stream_finished_event.set()

                    with sd.OutputStream(
                        samplerate=samplerate,
                        channels=channels,
                        callback=callback,
                        dtype='float32',
                        finished_callback=set_event_on_finish
                    ):
                        stream_finished_event.wait() # Wait for playback to complete or be cancelled

                except sd.CallbackStop:
                    pass # Normal exit from the stream
                except Exception:
                    pass # Ignore other errors for now
                finally:
                    with PlaySoundUtil._playback_state_lock:
                        if PlaySoundUtil._current_sound_id == sound_id:
                            PlaySoundUtil._current_sound_id = ""
                            PlaySoundUtil._current_playback_thread = None
                            PlaySoundUtil._stop_playback_event = None

            # Ensure sound_data is 2D for sounddevice (frames, channels)
            # If it's 1D (mono), convert it to 2D
            sound_data_2d = sound.data.reshape(-1, 1) if sound.data.ndim == 1 else sound.data
            channels = sound_data_2d.shape[1]

            new_thread = threading.Thread(target=play_stream_from_data, args=(sound_data_2d, sound.sr, channels), daemon=True)
            with PlaySoundUtil._playback_state_lock:
                PlaySoundUtil._current_sound_id = sound_id
                PlaySoundUtil._current_playback_thread = new_thread
                PlaySoundUtil._stop_playback_event = stop_event
                new_thread.start()
            return sound_id

    @staticmethod
    def current_sound_id() -> str:
        """Returns the active sound id, or an empty string if no sound is playing."""
        with PlaySoundUtil._playback_state_lock:
            thread = PlaySoundUtil._current_playback_thread
            if thread is None or not thread.is_alive():
                PlaySoundUtil._current_sound_id = ""
            return PlaySoundUtil._current_sound_id

    @staticmethod
    def stop_sound_async() -> bool:
        """
        Stops a sound that is playing using `play_sound_async()`.
        Returns True if there was a sound to be stopped.
        """

        with PlaySoundUtil._playback_operation_lock:
            with PlaySoundUtil._playback_state_lock:
                thread = PlaySoundUtil._current_playback_thread
                stop_event = PlaySoundUtil._stop_playback_event

            # Signal previous thread to stop if it exists and is alive
            if thread is not None and thread.is_alive():
                if stop_event is not None:
                    stop_event.set() # Set the event to signal stopping
                thread.join(timeout=0.5) # Wait a short time for it to stop
                return True
            return False
