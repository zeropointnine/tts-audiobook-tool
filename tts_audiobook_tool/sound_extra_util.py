import librosa
import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.util import *


class SoundExtraUtil:

    @staticmethod
    def high_shelf_eq(sound: Sound, strength: float, boost_start_hz: float, q_like: float = 1.0) -> Sound:
        """
        Applies a simple high-shelf EQ style boost to improve clarity in muffled speech.

        Args:
            sound: Input sound.
            strength: Multiplier controlling boost amount (0 = no boost).
                Practical range is roughly 0.0 to 2.0.
            boost_start_hz: Frequency at which upper spectrum boosting begins.
            q_like: Higher values create a steeper, narrower transition band;
                lower values create a gentler, wider transition. This is
                Q-inspired behavior, not a strict biquad Q value.

        Returns:
            A new Sound with boosted treble and peak normalization applied as needed.
        """

        # Guard rails
        strength = max(0.0, float(strength))
        nyquist = sound.sr / 2
        if nyquist <= 0:
            return sound

        # Keep start frequency valid and below Nyquist
        boost_start_hz = max(20.0, float(boost_start_hz))
        boost_start_hz = min(boost_start_hz, max(25.0, nyquist * 0.98))

        # Q-like control (higher => sharper transition)
        q_like = max(0.1, float(q_like))

        # No-op path for explicit "disabled"
        if strength == 0:
            return sound

        data = sound.data.astype(np.float32, copy=False)
        is_mono = data.ndim == 1
        data_2d = data.reshape(-1, 1) if is_mono else data

        n_samples = data_2d.shape[0]
        if n_samples < 2:
            return sound

        # Build frequency-domain shelf profile
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sound.sr)

        # Maximum shelf gain in dB, controlled by strength.
        # Example: strength=1.0 => +6 dB; strength=2.0 => +12 dB
        max_boost_db = 6.0 * strength
        max_boost_linear = 10 ** (max_boost_db / 20.0)

        shelf = np.ones_like(freqs, dtype=np.float32)
        # Previous default was roughly boost_start_hz * 0.35.
        # q_like=1.0 preserves that baseline. Higher q_like narrows the band.
        transition_width_hz = max(80.0, (boost_start_hz * 0.35) / q_like)

        # Smooth transition into boosted upper band using smoothstep.
        transition_start = max(20.0, boost_start_hz - transition_width_hz)
        transition_end = min(nyquist, boost_start_hz + transition_width_hz)

        if transition_end <= transition_start:
            shelf[freqs >= boost_start_hz] = max_boost_linear
        else:
            x = (freqs - transition_start) / (transition_end - transition_start)
            x = np.clip(x, 0.0, 1.0)
            smooth = x * x * (3 - 2 * x)
            shelf = 1.0 + (max_boost_linear - 1.0) * smooth

        # Apply EQ per channel
        output = np.empty_like(data_2d, dtype=np.float32)
        for ch in range(data_2d.shape[1]):
            channel = data_2d[:, ch]
            spectrum = np.fft.rfft(channel)
            eq_spectrum = spectrum * shelf
            output[:, ch] = np.fft.irfft(eq_spectrum, n=n_samples)

        # Only attenuate if EQ boost pushed peaks above the target ceiling.
        output = SoundUtil.attenuate_if_necessary(output, headroom_db=NORMALIZATION_HEADROOM_DB)

        new_data = output[:, 0] if is_mono else output
        return Sound(new_data.astype(sound.data.dtype, copy=False), sound.sr)

    @staticmethod
    def speed_up_audio(sound: Sound, multiplier: float) -> Sound | str:
        """
        Uses wsola algorithm to speed up (or slow down) audio.
        Sounds quite good for voice clone reference clip use case (when sped up).
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
        local_minima = SoundExtraUtil.find_local_minima(sound, target_timestamp)
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
        from PIL import Image, ImageDraw # type: ignore

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

