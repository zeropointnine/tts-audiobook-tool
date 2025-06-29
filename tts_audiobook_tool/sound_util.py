import librosa
import numpy as np
from numpy import ndarray
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.app_types import Sound

class SoundUtil:
    """
    """

    @staticmethod
    def resample(sound: Sound, resample_rate: int) -> Sound:
        new_data = librosa.resample(sound.data, orig_sr=sound.sr, target_sr=resample_rate)
        return Sound(new_data, resample_rate)

    @staticmethod
    def resample_for_whisper(sound: Sound) -> Sound:

        data = sound.data
        data = np.nan_to_num(sound.data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -1.0, 1.0)
        data = librosa.resample(data, orig_sr=sound.sr, target_sr=WHISPER_SAMPLERATE)
        return Sound(data, WHISPER_SAMPLERATE)

    @staticmethod
    def transcribe(whisper_model, sound: Sound) -> dict | str:
        """
        Transcribes the audio data.
        Makes temporary resampled audio if necessary.
        Returns error string on fail
        """
        if sound.sr != WHISPER_SAMPLERATE:
            temp_sound = SoundUtil.resample_for_whisper(sound)
        else:
            temp_sound = sound

        whisper_data = whisper_model.transcribe(temp_sound.data, word_timestamps=True, language=None)

        # Minor validation
        if not "text" in whisper_data or not isinstance(whisper_data["text"], str):
            return "Whisper data missing expected values"

        return whisper_data

    @staticmethod
    def transcribe_file(whisper_model, path: str) -> dict | str:
        """ Returns error string on fail """
        try:
            whisper_data = whisper_model.transcribe(path, word_timestamps=True, language=None)
        except Exception as e:
            return str(e)
        # Minor validation
        if not "text" in whisper_data or not isinstance(whisper_data["text"], str):
            return "Whisper data missing expected values"
        return whisper_data

    @staticmethod
    def trim(sound: Sound, start_time: float | None, end_time: float | None) -> Sound:

        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = sound.duration
        if end_time > sound.duration:
            # TODO: something is reporting an end_time that is longer than the sound duration :/
            end_time = sound.duration

        start_samples = int(start_time * sound.sr)
        end_samples = int(end_time * sound.sr)
        if end_samples > len(sound.data):
            end_samples = len(sound.data)

        # Ensure valid trim range
        if not (0 <= start_samples < end_samples <= len(sound.data)):
            raise ValueError(f"Invalid trim range - start {start_time} end {end_time} duration {sound.duration}")

        trimmed_data = sound.data[start_samples : end_samples]
        trimmed_data = np.copy(trimmed_data) # Create a copy to ensure it's not a view

        return Sound(trimmed_data, sound.sr)

    @staticmethod
    def normalize(arr: ndarray, headroom_db: float = 0.0) -> np.ndarray:
        """
        Normalizes a 1D NumPy array to a range of [-1, 1] with optional headroom.
        """
        # Ensure the input is a NumPy array
        arr = np.asarray(arr)

        # Find the minimum and maximum values of the array
        arr_min = np.min(arr)
        arr_max = np.max(arr)

        # --- Edge Case Handling ---
        # If all elements are the same, the range is 0.
        # To avoid division by zero, we return an array of zeros.
        if arr_min == arr_max:
            return np.zeros_like(arr)

        # --- Min-Max Scaling Formula ---
        # 1. Scale to the [0, 1] range: (arr - min) / (max - min)
        # 2. Scale to the [-1, 1] range: 2 * [scaled_to_0_1] - 1
        # normalized_arr = 2 * (arr - arr_min) / (arr_max - arr_min) - 1

        divisor = max(arr_max, abs(arr_min)) * 1.0
        normalized_arr = arr / divisor

        if headroom_db > 0:
            # Calculate the amplitude scaling factor from dB headroom
            # A positive headroom_db means the peak amplitude should be reduced.
            # For example, 3dB headroom means the peak should be 10^(-3/20) of its original value.
            amplitude_scale_factor = 10**(-headroom_db / 20.0)
            normalized_arr *= amplitude_scale_factor

        return normalized_arr

    @staticmethod
    def is_data_invalid(sound: Sound) -> list[str]:
        """ Returns list of 'reasons' why data is invalid """

        if not isinstance(sound.data, np.ndarray):
            return [f"Data is not a NumPy array, but {type(sound.data)}"]

        # Check for empty data or incorrect dimensions
        if sound.data.size == 0 or sound.data.ndim not in [1, 2]:
            return [f"Invalid shape or empty data. Shape: {sound.data.shape}"]

        reasons = []

        if np.isnan(sound.data).any():
            reasons.append("Data contains NaN value/s")

        if np.isinf(sound.data).any():
            reasons.append("Data contains Inf value/s")

        if np.max(np.abs(sound.data)) > 1.0:
            reasons.append(f"Value/s out of range, max value found: {np.max(np.abs(sound.data)):.2f}")

        return reasons


    @staticmethod
    def add_silence(sound: Sound, duration: float) -> Sound:
        """
        Returns error message on fail or empty string
        """
        silence = np.zeros(int(sound.sr * duration), dtype=sound.data.dtype) # Match dtype for concatenation

        new_data = np.concatenate([sound.data, silence])
        new_sound = Sound(new_data, sound.sr)
        return new_sound
