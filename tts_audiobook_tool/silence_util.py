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
        start_silence = SilenceUtil.get_start_silence_end_time(sound) or None
        end_silence = SilenceUtil.get_end_silence_start_time(sound) or None
        return start_silence, end_silence

    @staticmethod
    def get_start_silence_end_time(sound: Sound) -> float | None:
        segments = SilenceUtil.detect_silences(sound)
        if not segments:
            return None
        if segments[0][0] > 0.0:
            return None
        return segments[0][1]

    @staticmethod
    def get_end_silence_start_time(sound) -> float | None:

        segments = SilenceUtil.detect_silences(sound)
        if not segments:
            return None

        last_segment = segments[-1]

        end_segment_end = last_segment[1]
        if end_segment_end < sound.duration:
            return None

        end_segment_start = last_segment[0]
        if end_segment_start >= sound.duration:
            return None
        return end_segment_start

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

            # Minimum silence duration in frames
            min_silence_frames = ms_to_samples(min_silence_duration_ms, sound.sr) / hop_length

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
