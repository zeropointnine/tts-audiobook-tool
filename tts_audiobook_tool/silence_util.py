import librosa
import numpy as np
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.util import *
from tts_audiobook_tool.sound_util import SoundUtil

class SilenceUtil:

    @staticmethod
    def trim_silence(sound, end_only=False) -> tuple[Sound, float, float]:
        """
        Returns trimmed Sound and the start and end times of the trim
        """
        start, end = SilenceUtil.get_start_and_end_silence(sound)
        if not start and not end:
            return Sound( np.copy(sound.data), sound.sr ), 0.0, sound.duration
        if end_only and not end:
            return Sound( np.copy(sound.data), sound.sr ), 0.0, sound.duration

        if end_only:
            start = 0
        else:
            start = start or 0
        end = end or sound.duration
        result = SoundUtil.trim(sound, start, end)
        return result, start, end

    @staticmethod
    def get_start_and_end_silence(sound: Sound) -> tuple[float | None, float | None]:
        # eats errors
        start_silence = SilenceUtil.get_start_silence(sound) or None
        end_silence = SilenceUtil.get_end_silence(sound) or None
        return start_silence, end_silence

    @staticmethod
    def get_start_silence(
        sound: Sound,
        max_seconds: float = 5.0,
        threshold_db_relative_to_peak: float = -42.0,
        min_silence_duration_ms: int = 100,
        frame_length_ms: int = 30,
        hop_length_ms: int = 10
    ) -> float | None:
        """
        Optimized version of get_start_silence_end_time that only processes
        the first `max_seconds` of audio and breaks early.

        Args:
            sound: The audio clip
            max_seconds: Maximum duration to analyze (in seconds)
            threshold_db_relative_to_peak:
                Threshold in dB relative to the audio's peak RMS.
                Segments below this are considered silent.
            min_silence_duration_ms:
                Minimum duration (in ms) for a segment to be
                classified as silence. Shorter silences are ignored.
            frame_length_ms:
                The length of each frame for analysis (in ms).
            hop_length_ms:
                The step size between frames (in ms).

        Returns:
            The end time of the initial silence, or None if no start
            silence detected.
        """
        try:
            if sound.data.size == 0:
                return None

            # Truncate data to only process the first max_seconds
            max_samples = int(max_seconds * sound.sr)
            truncated_data = sound.data[:max_samples]
            truncated_duration = len(truncated_data) / sound.sr

            # Convert ms to samples
            frame_length = ms_to_samples(frame_length_ms, sound.sr)
            hop_length = ms_to_samples(hop_length_ms, sound.sr)

            # Calculate RMS energy for each frame
            rms_frames = librosa.feature.rms(
                y=truncated_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            if rms_frames.size == 0:
                return None

            # Calculate peak RMS and the silence threshold
            peak_rms = np.max(rms_frames)
            if peak_rms == 0:  # Complete silence in the truncated window
                return min(truncated_duration, sound.duration)

            threshold = peak_rms * (10 ** (threshold_db_relative_to_peak / 20))

            # Identify frames below the threshold
            is_silent = rms_frames < threshold

            # Check if the first frame is silent
            if not is_silent[0]:
                return None

            # Minimum silence duration in frames (use ms ratio to avoid samplerate-dependent truncation errors)
            min_silence_frames = min_silence_duration_ms / hop_length_ms

            # Find first non-silent frame (early break)
            silent_run_end = 1  # Start from frame 1 since frame 0 is silent
            while silent_run_end < len(is_silent) and is_silent[silent_run_end]:
                silent_run_end += 1

            # Convert the end index to time
            end_time = float(librosa.frames_to_time(silent_run_end, sr=sound.sr, hop_length=hop_length))
            end_time = min(end_time, sound.duration)

            # Check if the silence duration meets the minimum
            silence_duration_frames = float(silent_run_end)
            if silence_duration_frames >= min_silence_frames:
                return end_time

            return None
        except Exception:
            return None

    @staticmethod
    def get_end_silence(
        sound: Sound,
        max_seconds: float = 5.0,
        threshold_db_relative_to_peak: float = -42.0,
        min_silence_duration_ms: int = 100,
        frame_length_ms: int = 30,
        hop_length_ms: int = 10
    ) -> float | None:
        """
        Detects trailing silence by reversing the tail of the audio and
        delegating to get_start_silence.

        Accepts the same parameters as get_start_silence.
        Returns the start time of the trailing silence in the original
        audio, or None if no qualifying trailing silence is found.
        """
        try:
            if sound.data.size == 0:
                return None

            # Truncate and reverse the last max_seconds of audio
            max_samples = int(max_seconds * sound.sr)
            tail_samples = min(max_samples, len(sound.data))
            tail_data = sound.data[-tail_samples:]
            reversed_tail_data = tail_data[::-1]
            reversed_sound = Sound(reversed_tail_data, sound.sr)

            # Use get_start_silence to find silence at the "start" (originally the end)
            result = SilenceUtil.get_start_silence(
                reversed_sound,
                max_seconds=max_seconds,
                threshold_db_relative_to_peak=threshold_db_relative_to_peak,
                min_silence_duration_ms=min_silence_duration_ms,
                frame_length_ms=frame_length_ms,
                hop_length_ms=hop_length_ms,
            )

            if result is None:
                return None

            # Convert the silence duration from the reversed domain back to
            # the original: silence_start = sound.duration - silence_duration
            silence_start = sound.duration - result
            return silence_start
        except Exception:
            return None

    @staticmethod
    def detect_silences(
        sound: Sound,
        threshold_db_relative_to_peak: float=-42.0, # How many dB below the peak to consider silence
        min_silence_duration_ms: int=100, # Minimum duration for a silence segment
        frame_length_ms: int=30, # Frame length for RMS calculation
        hop_length_ms: int=10 # Hop length for RMS calculation
    ) -> list[tuple[float, float]]:
        """
        Detects silence in an audio clip based on a relative RMS threshold, returning time ranges.

        Args:
            sound (Sound):
                The audio clip
            threshold_db_relative_to_peak (float):
                Threshold in dB relative to the audio's peak RMS.
                Segments below this are considered silent.
            min_silence_duration_ms (int):
                Minimum duration (in ms) for a segment to be
                classified as silence. Shorter silences are ignored.
            frame_length_ms (int):
                The length of each frame for analysis (in ms).
            hop_length_ms (int):
                The step size between frames (in ms).

        Returns:
            list of tuples:
                A list where each tuple contains (start_time, end_time)
                of a detected silence segment in seconds.
        """

        # Convert ms to samples
        frame_length = ms_to_samples(frame_length_ms, sound.sr)
        hop_length = ms_to_samples(hop_length_ms, sound.sr)

        # Calculate RMS energy for each frame
        rms_frames = librosa.feature.rms(y=sound.data, frame_length=frame_length, hop_length=hop_length)[0]

        try:
            if sound.data.size == 0:
                return []

            # Calculate peak RMS and the silence threshold
            peak_rms = np.max(rms_frames)
            if peak_rms == 0:  # Handle complete silence
                return [(0, sound.duration)] if sound.duration > 0 else []

            threshold = peak_rms * (10 ** (threshold_db_relative_to_peak / 20))

            # Identify frames below the threshold
            is_silent = rms_frames < threshold

            # Pad with False at both ends to correctly detect silence at the very beginning or end
            is_silent_padded = np.concatenate(([False], is_silent, [False]))

            # Find where silence begins and ends
            diff = np.diff(is_silent_padded.astype(int))
            silence_starts_indices = np.where(diff == 1)[0]
            silence_ends_indices = np.where(diff == -1)[0]

            # Minimum silence duration in frames (use ms ratio to avoid samplerate-dependent truncation errors)
            min_silence_frames = min_silence_duration_ms / hop_length_ms

            silence_segments: list[tuple[float, float]] = []
            for start_frame, end_frame in zip(silence_starts_indices, silence_ends_indices):
                duration_frames = end_frame - start_frame
                if duration_frames >= min_silence_frames:
                    # Convert frame indices to time in seconds
                    start_time = librosa.frames_to_time(start_frame, sr=sound.sr, hop_length=hop_length)
                    end_time = librosa.frames_to_time(end_frame, sr=sound.sr, hop_length=hop_length)

                    # Ensure end_time does not exceed sound duration
                    end_time = min(end_time, sound.duration)

                    if start_time < end_time:
                        silence_segments.append((start_time, end_time))

            return silence_segments
        except Exception:
            return []

    @staticmethod
    def is_silent_around(
        sound: Sound,
        target_timestamp_s: float,
        width: float = 0.10,
        silence_threshold_db: float = -40
    ) -> bool:
        """
        Calculates the smallest RMS of a contiguous `width` that encompasses `target_timestamp_s`,
        and returns True if that value is less than `silence_threshold_db`. 
        """
        ...

def ms_to_samples(ms, sr):
    """Converts milliseconds to samples"""
    return int(ms * sr / 1000)
