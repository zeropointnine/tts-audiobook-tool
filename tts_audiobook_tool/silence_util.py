import librosa
import numpy as np
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
import os

from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.util import *

class SilenceUtil:

    @staticmethod
    def trim_silence_if_necessary(file_a: str, file_b: str, max_duration: float=0.5) -> str | bool:
        """
        Prints to console if action taken

        Returns False if no action taken,
        True if file/s successfully modified,
        or error message string on fail
        """

        a_end_duration = SilenceUtil.get_silence_duration_end(file_a)
        if a_end_duration is None:
            return f"Could not get silence duration for {file_a}"

        b_start_duration = SilenceUtil.get_silence_duration_start(file_b)
        if b_start_duration is None:
            return f"Could not get silence duration for {file_b}"

        overage = (a_end_duration + b_start_duration) - max_duration
        epsilon = 0.02 # ... files that have previously been cut will result in overage being _around_ 0.0
        if overage < epsilon:
            return False

        duration_a = AudioMetaUtil.get_audio_duration(file_a)
        duration_b = AudioMetaUtil.get_audio_duration(file_b)

        if duration_a is None or duration_b is None:
            return "Could not get file duration"

        # Determine how much to trim from each file
        trim_from_a = 0.0
        trim_from_b = 0.0

        if overage <= a_end_duration:
            trim_from_a = overage
        elif overage <= b_start_duration:
            trim_from_b = overage
        else:
            trim_from_a = a_end_duration
            trim_from_b = overage - a_end_duration

        # Create temporary file paths
        temp_base_dir = str(Path(file_a).parent)
        temp_file_a = os.path.join(temp_base_dir, make_random_hex_string() + ".flac")
        temp_file_b = os.path.join(temp_base_dir, make_random_hex_string() + ".flac")

        # Trim file_a if necessary
        if trim_from_a > 0:
            printt(file_a)
            printt(f"Trim from end {trim_from_a:.2f}s")
            success_a = SoundFileUtil.trim_flac_file(
                source_flac_path=file_a,
                dest_file_path=temp_file_a,
                start_time_seconds=0.0,
                end_time_seconds=duration_a - trim_from_a
            )
            if not success_a:
                if os.path.exists(temp_file_a):
                    os.remove(temp_file_a)
                return f"Failed to trim {file_a}"
            os.replace(temp_file_a, file_a)

        # Trim file_b if necessary
        if trim_from_b > 0:
            printt(file_b)
            printt(f"Trim from start {trim_from_b:.2f}s")
            success_b = SoundFileUtil.trim_flac_file(
                source_flac_path=file_b,
                dest_file_path=temp_file_b,
                start_time_seconds=trim_from_b,
                end_time_seconds=duration_b
            )
            if not success_b:
                if os.path.exists(temp_file_b):
                    os.remove(temp_file_b)
                return f"Failed to trim {file_b}"
            os.replace(temp_file_b, file_b)

        # TODO print feedback
        printt()
        return True

    @staticmethod
    def add_silence_if_necessary(file_a: str, file_b: str, min_duration: float) -> str | bool:
        """
        Enforces a minimum duration of silence in the audio of two adjacent sound files
        by appending silence to file_a if necessary.

        Prints to console if action taken

        Returns False if no action taken,
        True if file successfully modified,
        or error message string on fail
        """

        a_end_duration = SilenceUtil.get_silence_duration_end(file_a)
        if a_end_duration is None:
            return f"Could not get silence duration for {file_a}"

        b_start_duration = SilenceUtil.get_silence_duration_start(file_b)
        if b_start_duration is None:
            return f"Could not get silence duration for {file_b}"

        underage = min_duration - (a_end_duration + b_start_duration)
        epsilon = 0.02
        if underage < epsilon:
            return False

        temp_dest_file = make_sibling_random_file_path(file_a)
        err = SoundFileUtil.add_silence_flac(file_a, temp_dest_file, underage)
        if err:
            return err
        err = swap_and_delete_file(temp_dest_file, file_a)
        if err:
            return err

        printt(f"Silence ({underage:.2f}s) added to end of {file_a}")
        printt()
        return True

    @staticmethod
    def get_silence_duration_start(path: str) -> float | None:
        try:
            silent_segments, _ = SilenceUtil.detect_silence(path)
            if silent_segments and silent_segments[0][0] == 0.0:
                return silent_segments[0][1] - silent_segments[0][0]
            return 0.0
        except Exception:
            return None

    @staticmethod
    def get_silence_duration_end(path: str) -> float | None:
        try:
            silent_segments, total_duration = SilenceUtil.detect_silence(path)
            if silent_segments and silent_segments[-1][1] == total_duration:
                return silent_segments[-1][1] - silent_segments[-1][0]
            return 0.0
        except Exception:
            return None

    @staticmethod
    def detect_silence(
        audio_path: str,
        threshold_db_relative_to_peak: float=-30.0, # How many dB below the peak to consider silence
        min_silence_duration_ms: int=150,       # Minimum duration for a silence segment
        frame_length_ms: int=30,                # Frame length for RMS calculation
        hop_length_ms: int=10                   # Hop length for RMS calculation
    ) -> tuple[list[tuple[float, float]], float]:
        """
        Detects silence in an audio file based on a relative RMS threshold.

        Args:
            audio_path (str): Path to the audio file.
            threshold_db_relative_to_peak (float): Threshold in dB relative to the audio's peak RMS.
                                                Segments below this are considered silent.
            min_silence_duration_ms (int): Minimum duration (in ms) for a segment to be
                                        classified as silence. Shorter silences are ignored.
            frame_length_ms (int): The length of each frame for analysis (in ms).
            hop_length_ms (int): The step size between frames (in ms).

        Returns:
            list of tuples: A list where each tuple contains (start_time, end_time)
                            of a detected silence segment in seconds.
        """
        try:
            y, sr = librosa.load(audio_path, sr=None) # Load audio, sr=None preserves original sample rate
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return [], 0.0

        # Convert ms to samples
        frame_length = ms_to_samples(frame_length_ms, sr)
        hop_length = ms_to_samples(hop_length_ms, sr)

        # Calculate RMS energy for each frame
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        if len(rms_frames) == 0:
            print("Warning: No RMS frames computed. Audio might be too short or parameters incorrect.")
            # If audio is shorter than frame_length, librosa.feature.rms can return empty
            # Treat as entirely silent or non-silent based on your preference, or return empty
            if len(y) / sr * 1000 < min_silence_duration_ms:
                return [], 0.0 # Too short to have meaningful silence
            else: # Treat as one long silence if it meets min_duration criteria
                return [(0.0, len(y) / sr)], len(y) / sr


        # --- "Intelligent" Threshold ---
        # Convert RMS to dB. We use max RMS as reference (0 dB)
        # Any RMS value of 0 will result in -inf dB, handle this by replacing with a very small number
        epsilon = 1e-10
        rms_db_frames = librosa.amplitude_to_db(np.maximum(epsilon, rms_frames), ref=np.max)

        # The threshold is now absolute in dBFS (where 0dBFS is the peak of the signal)
        silence_threshold_db = threshold_db_relative_to_peak # e.g., -40 dB

        # Identify frames below the threshold
        silent_frames_mask = rms_db_frames < silence_threshold_db

        # Convert frame indices to time
        # librosa.frames_to_time gives the time of the *center* of each frame
        # We'll adjust to get start/end times of segments more accurately
        frame_times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_length)

        silent_segments = []
        in_silence = False
        silence_start_time = 0

        # Add a sentinel False at the end to ensure the last segment is processed
        # And a sentinel False at the beginning if the audio starts with silence
        processing_mask = np.concatenate(([False], silent_frames_mask, [False]))

        # Find where silence starts and ends
        # diff will be 1 where silence starts, -1 where it ends
        diff = np.diff(processing_mask.astype(int))

        silence_start_indices = np.where(diff == 1)[0]
        silence_end_indices = np.where(diff == -1)[0]

        # Ensure we have pairs of start/end
        if len(silence_start_indices) == 0: # No silence detected
            return [], 0.0

        for i in range(len(silence_start_indices)):
            start_idx = silence_start_indices[i]
            # End_idx corresponds to the *frame before* sound starts again
            # or the end of the audio if silence continues till the end.
            end_idx = silence_end_indices[i] if i < len(silence_end_indices) else len(rms_frames) -1


            # Convert frame indices to time
            # Start time is the beginning of the first silent frame
            current_silence_start_time = frame_times[start_idx] - (hop_length_ms / 2000.0)
            current_silence_start_time = max(0, current_silence_start_time)  # type: ignore  # Ensure not negative

            # End time is the end of the last silent frame
            # (center of last silent frame + half hop_length)
            # or if it's the very last frame, it's the end of audio
            if end_idx < len(frame_times):
                # Time of the *start* of the frame *after* the last silent frame
                # This frame (end_idx) is the first *non-silent* frame *after* the silence.
                # So, the silence ends at the start of this frame.
                current_silence_end_time = frame_times[end_idx] - (hop_length_ms / 2000.0)
            else: # Silence goes to the end of the audio
                current_silence_end_time = len(y) / sr

            current_silence_end_time = max(current_silence_start_time, current_silence_end_time)


            duration = current_silence_end_time - current_silence_start_time

            if duration * 1000 >= min_silence_duration_ms:
                silent_segments.append((current_silence_start_time, current_silence_end_time))

        # A simpler alternative using librosa.effects.split if you want non-silent intervals
        # And then derive silent intervals from that.
        # `top_db` in `split` is equivalent to `abs(threshold_db_relative_to_peak)` if peak is 0dB.
        # non_silent_intervals_samples = librosa.effects.split(y,
        #                                              top_db=abs(threshold_db_relative_to_peak),
        #                                              frame_length=frame_length,
        #                                              hop_length=hop_length)
        # non_silent_intervals_sec = librosa.samples_to_time(non_silent_intervals_samples, sr=sr)
        # print("Non-silent intervals from librosa.effects.split:", non_silent_intervals_sec)
        # You would then calculate silences based on gaps between these, and before first/after last.


        float_tuples = [tuple(float(x) for x in segment) for segment in silent_segments]
        duration = len(y) / sr
        return float_tuples, duration  # type: ignore


def ms_to_samples(ms, sr):
    """Converts milliseconds to samples"""
    return int(ms * sr / 1000)

