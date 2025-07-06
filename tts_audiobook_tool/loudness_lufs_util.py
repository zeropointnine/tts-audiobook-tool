import os
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


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
    def normalize_and_overwrite(wav_file_path: str) -> str:
        """
        Normalizes the loudness of a WAV file to a target LUFS and overwrites the original file.
        This is a more or less linear transformation.
        Returns empty string on success or no-action, else error string on fail

        (Not currently using)
        """
        try:
            audio, sample_rate = sf.read(wav_file_path, dtype='float32')

            loudness = LoudnessLufsUtil.calculate_integrated_loudness(audio, sample_rate)
            print("loudness", loudness)
            if loudness is None:
                # Skip, sample too short
                return ""
            if loudness == -np.inf:
                # Skip, avoid division by zero or extreme gain if loudness is -inf (silence)
                return ""

            # Target loudness (LUFS)
            target_lufs = -10.0

            # TODO: skip if loudness is close to target_lufs

            gain_db = target_lufs - loudness
            gain_linear = 10 ** (gain_db / 20.0)
            print("gain_db", gain_db)
            print("gain_linear", gain_linear)

            # Apply gain
            normalized_audio = audio * gain_linear

            # Peak normalize to prevent clipping (-1.0 dBTP ceiling is common, 0.0 is max)
            # Use pyloudnorm's peak normalization
            normalized_audio = pyln.normalize.peak(normalized_audio, -1.0)

            # Write the normalized audio back to the original file
            # Use soundfile to preserve original subtype if possible (e.g., PCM_16, PCM_24, FLOAT)
            sf.write(wav_file_path, normalized_audio, sample_rate) # sf handles dtype conversion based on file subtype

            return ""

        except Exception as e:
            return f"Error normalizing {wav_file_path}: {e}"

    @staticmethod
    def calculate_integrated_loudness_file(file_path: str) -> float | None | str:
        """ Returns float or None if sample too short or str if error message """
        try:
            audio, sample_rate = sf.read(file_path, dtype='float32')
        except Exception as e:
            return str(e)
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

