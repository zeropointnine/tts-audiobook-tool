import librosa
import numpy as np
from numpy import ndarray
import numpy
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *
from tts_audiobook_tool.phrase import Reason

class SoundUtil:
    """
    "Core" sound utility functions
    """

    @staticmethod
    def resample_if_necessary(sound: Sound, target_sr: int) -> Sound:
        """
        Returns new Sound with target sample rate (or identity if unnecessary)
        """        
        if sound.sr == target_sr:
            return sound
        new_data = librosa.resample(sound.data, orig_sr=sound.sr, target_sr=target_sr)
        return Sound(new_data, target_sr)

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
        Does peak normalization of audio, with specified headroom in dB (use positive number)
        """
        # Convert dB to linear scale
        if headroom_db < 0:
            headroom_db = 0
        headroom_linear = 10 ** (-headroom_db / 20)

        # Peak normalize to 1.0, then apply headroom
        normalized_arr = librosa.util.normalize(arr, norm=np.inf) * headroom_linear

        return normalized_arr

    @staticmethod
    def attenuate_if_necessary(arr: ndarray, headroom_db: float = 0.0) -> np.ndarray:
        """
        Only attenuates audio if peak exceeds allowed ceiling.
        Does not amplify quieter audio.
        """
        if headroom_db < 0:
            headroom_db = 0
        headroom_linear = 10 ** (-headroom_db / 20)

        peak = np.max(np.abs(arr))
        if peak <= 0 or peak <= headroom_linear:
            return arr

        return arr * (headroom_linear / peak)

    @staticmethod
    def is_data_invalid(sound: Sound) -> list[str]:
        """ Returns list of 'reasons' why sound data is invalid """

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
        silence = np.zeros(int(sound.sr * duration), dtype=sound.data.dtype) # Match dtype for concatenation
        new_data = np.concatenate([sound.data, silence])
        new_sound = Sound(new_data, sound.sr)
        return new_sound

    @staticmethod
    def make_silence_sound(seconds: float, sr: int, dtype: np.dtype) -> Sound:
        data = np.zeros(int(sr * seconds), dtype=dtype)
        return Sound(data, sr)

    @staticmethod
    def append_sound_using_path(base_sound: Sound, appended_sound_path: str) -> Sound:
        """
        Concatenates a Sound using the specified file path to `base_sound`.
        Resamples the loaded sound to match the base sound.
        On error, prints feedback and simply returns the base_sound.
        """

        load_result = SoundFileUtil.load(appended_sound_path)
        if isinstance(load_result, str):
            printt(f"Couldn't load sound {appended_sound_path} {load_result}")
            return base_sound

        appended_sound = load_result
        appended_sound = SoundUtil.resample_if_necessary(appended_sound, base_sound.sr)

        new_data = numpy.concatenate((base_sound.data, appended_sound.data))
        return Sound(new_data, base_sound.sr)

    @staticmethod
    def append_pause_or_section_effect(
        sound: Sound,
        reason: Reason,
        use_section_sound_effect: bool,
    ) -> Sound:
        if reason == Reason.SECTION and use_section_sound_effect:
            return SoundUtil.append_sound_using_path(sound, SECTION_SOUND_EFFECT_PATH)
        if reason.pause_duration:
            return SoundUtil.add_silence(sound, reason.pause_duration)
        return sound

