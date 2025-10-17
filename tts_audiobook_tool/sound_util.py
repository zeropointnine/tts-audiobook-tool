import librosa
import numpy as np
from numpy import ndarray
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import make_error_string, printt

class SoundUtil:
    """
    """

    @staticmethod
    def resample_if_necessary(sound: Sound, target_sr: int) -> Sound:
        """
        Returns new Sound with target sample rate
        But returns identity if not needed
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

    @staticmethod
    def speed_up_audio(sound: Sound, multiplier: float) -> Sound | str:
        """
        Uses wsola algorithm to speed up (or slow down) audio.
        Sounds quite good for voice clone reference clip use case.
        If error, returns error string.
        """

        from audiotsm import wsola
        from audiotsm.io.array import ArrayReader, ArrayWriter

        y, sr = sound.data, sound.sr

        try:

            # Set up the reader and writer
            reader = ArrayReader(y.reshape(1, -1)) # audiotsm expects a 2D array
            writer = ArrayWriter(channels=1)

            # The WSOLA processor
            # The frame_length and synthesis_hop parameters can be tuned, but defaults are often good.
            tsm = wsola(channels=1, speed=multiplier)

            # Process the audio
            tsm.run(reader, writer)
            y_fast = writer.data.flatten() # Get the processed audio back as a 1D array

            # Save the output
            return Sound(y_fast, sr)

        except Exception as e:
            return make_error_string(e)

    @staticmethod
    def find_local_minima(
        sound: Sound,
        target_timestamp_s: float,
        search_window_ms: int = 250,
        energy_window_ms: int = 20,
    ) -> float | str:
        """
        Finds the quietest point in a sound object within a search window around a target timestamp.

        This function calculates the root-mean-square (RMS) energy over a small, sliding window
        to identify the local minimum, which is often the best place to split audio between words.

        Args:
            sound (Sound): The sound object to analyze.
            target_timestamp_s (float): The center of the search window, in seconds.
            search_window_ms (int): The width of the search window on each side of the target, in milliseconds.
            energy_window_ms (int): The width of the sliding window for RMS energy calculation, in milliseconds.

        Returns:
            float | str: The timestamp of the local minimum in seconds, or an error string.
        """
        try:
            # [1] Convert times to sample indices
            sr = sound.sr
            target_sample = int(target_timestamp_s * sr)
            search_window_samples = int(search_window_ms / 1000 * sr)
            energy_window_samples = int(energy_window_ms / 1000 * sr)

            # Ensure the analysis window for RMS is at least 1 sample
            if energy_window_samples < 1:
                energy_window_samples = 1
            # Ensure it's an odd number for centering
            if energy_window_samples % 2 == 0:
                energy_window_samples += 1

            # [2] Define the search region
            start_sample = target_sample - search_window_samples
            end_sample = target_sample + search_window_samples

            # [3] Clamp search region to the bounds of the audio data
            start_sample = max(0, start_sample)
            end_sample = min(len(sound.data), end_sample)

            if start_sample >= end_sample:
                return "Search window is outside the audio data range."

            search_region = sound.data[start_sample:end_sample]

            # [4] Calculate RMS energy over the search region
            # hop_length=1 gives the highest resolution for finding the minimum
            rms_energy = librosa.feature.rms(
                y=search_region,
                frame_length=energy_window_samples,
                hop_length=1,
                center=True, # Pad the signal for centered frames
            )[0]

            # [5] Find the minimum energy point
            # The RMS array is smaller than the input region due to framing.
            # We need to find the minimum and map its index back to the original sample space.
            min_energy_index_in_rms = np.argmin(rms_energy)

            # The index corresponds to the center of the frame in the search_region
            min_energy_index_in_search_region = min_energy_index_in_rms

            # [6] Convert back to an absolute sample index and then to a timestamp
            absolute_min_sample = start_sample + min_energy_index_in_search_region
            best_timestamp_s = absolute_min_sample / sr

            return float(best_timestamp_s)

        except Exception as e:
            return f"Error finding local minima: {type(e).__name__} - {e}"

    @staticmethod
    def get_local_minima(sound: Sound, target_timestamp: float) -> float:
        """
        Wrapper which always returns a value
        """
        local_minima = SoundUtil.find_local_minima(sound, target_timestamp)
        if isinstance(local_minima, str):
            printt(f"Couldn't get local minima for target timestamp: {target_timestamp}")
            local_minima = target_timestamp

        return local_minima

    @staticmethod
    def save_local_minima_visualization(sound: Sound, target_timestamp: float, local_minima: float, dest_path_png: str):
        """
        Creates waveform image that is +/-500ms from target_timestamp
        Red line is target_timestamp. Green line is local_minima.
        For use while debugging, etc.
        """

        from PIL import Image
        from PIL import Image, ImageDraw

        # Image parameters
        width, height = 1200, 600
        background_color = (255, 255, 255)
        waveform_color = (0, 0, 255)
        target_line_color = (255, 0, 0)
        minima_line_color = (0, 255, 0)
        center_line_color = (200, 200, 200)

        # Create a new image
        img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # Define the window to display
        display_window_ms = 500
        display_window_s = display_window_ms / 1000

        # Calculate sample indices for the display window
        start_s = target_timestamp - display_window_s
        end_s = target_timestamp + display_window_s
        start_sample = int(start_s * sound.sr)
        end_sample = int(end_s * sound.sr)

        # Clamp to audio boundaries
        start_sample = max(0, start_sample)
        end_sample = min(len(sound.data), end_sample)

        # Extract the waveform segment
        waveform_segment = sound.data[start_sample:end_sample]

        # --- Drawing ---

        # Draw center line
        center_y = height // 2
        draw.line([(0, center_y), (width, center_y)], fill=center_line_color, width=1)

        # Draw waveform
        num_samples = len(waveform_segment)
        if num_samples > 1:
            # Downsample for performance if necessary, or draw all points
            # For simplicity, we'll map each sample to a horizontal position
            for i in range(num_samples - 1):
                x1 = int(i / num_samples * width)
                y1 = center_y - int(waveform_segment[i] * center_y * 0.9) # 0.9 to avoid touching edges

                x2 = int((i + 1) / num_samples * width)
                y2 = center_y - int(waveform_segment[i+1] * center_y * 0.9)

                draw.line([(x1, y1), (x2, y2)], fill=waveform_color, width=2)

        # Draw target timestamp indicator
        target_pos_s = target_timestamp - start_s
        target_x = int((target_pos_s / (end_s - start_s)) * width)
        draw.line([(target_x, 0), (target_x, height)], fill=target_line_color, width=2)

        # Draw local minima indicator
        minima_pos_s = local_minima - start_s
        minima_x = int((minima_pos_s / (end_s - start_s)) * width)
        draw.line([(minima_x, 0), (minima_x, height)], fill=minima_line_color, width=3)

        # Save
        img.save(dest_path_png)
        print(f"Waveform visualization saved to: {dest_path_png}")

