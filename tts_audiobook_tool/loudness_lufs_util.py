import os
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from tts_audiobook_tool.util import *


class LoudnessLufsUtil:
    """
    """

    @staticmethod
    def calculate_integrated_loudness(audio: np.ndarray, sample_rate) -> float | None:
        """
        Measures "integrated loudness" (EBU R128 / ITU-R BS.1770 algorithm) in LUFS
        """

        # Ensure mono audio for loudness measurement if stereo
        if audio.ndim > 1 and audio.shape[1] > 1:
            # Average channels for loudness calculation, but keep original channels for output
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        if len(audio_mono) < sample_rate * 0.4:
            # Skip, needs at least 400ms for reliable measurement
            return None

        # Measure loudness
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio_mono)
        return loudness


    @staticmethod
    def calculate_integrated_loudness_file(file_path: str) -> float | None | str:
        """ Returns float or None if sample too short or str if error message """
        try:
            audio, sample_rate = sf.read(file_path, dtype='float32')
        except Exception as e:
            return make_error_string(e)
        return LoudnessLufsUtil.calculate_integrated_loudness(audio, sample_rate)

    @staticmethod
    def print_integrated_loudness_directory(dir_path: str):
        """ For development"""
        count = 0
        sum = 0
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            result = LoudnessLufsUtil.calculate_integrated_loudness_file(file_path)
            if isinstance(result, float):
                count += 1
                sum += result
                result = f"{result:.2f}"
            print(result, file_path)

        print("\navg:", (sum / count))

    # ---

    @staticmethod
    def calculate_loudness_rms(file_path: str) -> float:
        """
        Returns RMS loudness (0.0 = silence, ~1.0 = max loudness for normalized audio)
        """
        # Load FLAC (automatically converts to float32 in [-1.0, 1.0])
        audio, _ = sf.read(file_path, dtype="float32")
        # Handle stereo by averaging channels (if needed)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        return np.sqrt(np.mean(audio**2))

