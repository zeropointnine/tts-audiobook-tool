import math
import os

import librosa
import numpy as np
import soundfile

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.audio_convert import describe_error, transcode_to_aac
from tts_audiobook_tool.sound.audio_decoder import AudioDecodeError, read_audio
from tts_audiobook_tool.sound.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *

class SoundFileUtil:

    MAX_PREVIEW_SECONDS = 60.0

    @staticmethod
    def load_range(path: str, start: float, end: float) -> Sound | str:
        """
        Decodes at most one minute of an audiobook from `start` seconds, for
        playback: mono, APP_SAMPLE_RATE, default audio track. Seeks rather
        than decoding from the beginning.
        """
        if not all(math.isfinite(value) for value in (start, end)) or start < 0 or end <= start:
            return "No playable audio interval is available"
        duration = min(end - start, SoundFileUtil.MAX_PREVIEW_SECONDS)
        try:
            data, sr = read_audio(path, APP_SAMPLE_RATE, 1, start, duration)
        except AudioDecodeError as e:
            return str(e)
        except Exception as e:
            return make_error_string(e)
        if not len(data):
            return "No audio was found in the selected interval"
        return Sound(data, sr)

    @staticmethod
    def load(path: str, target_sr: int=0) -> Sound | str:
        """
        Returns data as normalized np32 floats, mono
        """
        try:
            data, sr = SoundFileUtil.read_mono(path)
        except Exception as e:
            return make_error_string(e)

        if target_sr != 0 and target_sr != sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
            return Sound(data, target_sr)
        else:
            return Sound(data, sr)

    @staticmethod
    def read_mono(path: str) -> tuple[np.ndarray, int]:
        """
        Whole file as mono float32 (channel mean) at its own sample rate.

        Formats libsndfile reads (WAV, FLAC, OGG, MP3, ...) go through soundfile,
        which gives the same samples librosa.load did; anything else (eg AAC/M4A)
        through the PyAV decoder. Never an external decoder.
        """
        try:
            data, sr = soundfile.read(path, dtype="float32", always_2d=True)
        except soundfile.LibsndfileError:
            data, sr = read_audio(path, sample_rate=None, channels=1)
            return data, sr
        mono = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1, dtype=np.float32)
        return np.ascontiguousarray(mono), int(sr)

    @staticmethod
    def save_flac(sound: Sound, flac_path: str) -> str:
        """ Return error string on fail """
        try:
            soundfile.write(
                flac_path,
                sound.data,
                sound.sr,
                format="FLAC",
                subtype="PCM_16"
            )
            return ""
        except Exception as e:
            return f"Error saving sound file {flac_path}: {type(e)} {e}"

    @staticmethod
    def is_valid_sound_file(path: str) -> str:
        """ Returns error string or empty string"""
        from mutagen._file import File
        try:
            result = File(path)
        except Exception as e:
            return f"Not a valid audio file: {e}"
        if result is None:
            return "Not a valid audio file"
        return ""

    @staticmethod
    def concatenate_flacs(
        file_paths: list[str],
        dest_flac_path: str
    ) -> str:
        """
        Concatenates a list of flac files into one.
        Empty file paths in list are ignored.
        Returns error message or empty string.
        """

        dest_base_path = str( Path(dest_flac_path).parent )
        os.makedirs(dest_base_path, exist_ok=True)

        # [1] Make temp text file with list of input WAVs for ffmpeg's concat demuxer

        temp_text_path = os.path.join(dest_base_path, PROJECT_CONCAT_TEMP_TEXT_FILE_NAME)
        try:
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                for path in file_paths:
                    # Must be absolute paths
                    # Important: Escape backslashes for ffmpeg on all platforms within the list file
                    # And wrap the path in single quotes for robustness (handles spaces etc.)
                    if not path:
                        continue
                    escaped_path = path.replace('\\', '\\\\')
                    f.write(f"file '{escaped_path}'\n")
        except Exception as e:
            delete_silently(temp_text_path)
            return make_error_string(e)

        # [2] Do concat

        printt(f"Concatenating {len(file_paths)} files...")
        printt()

        # -y: Overwrite output file without asking
        # -f concat: Use the concatenation demuxer
        # -safe 0: Allow unsafe file paths (useful if paths are complex, though quoting helps)
        # -i list_filename: Input file is the list we created
        partial_command = [
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_text_path
        ]
        partial_command.extend(FFMPEG_ARGUMENTS_OUTPUT_FLAC)

        err = FfmpegUtil.make_file(partial_command, dest_flac_path, use_temp_file=True)
        delete_silently(temp_text_path)

        return err

    @staticmethod
    def transcode_to_aac_at(
            source_file_path: str,
            dest_file_path: str,
            bitrate: str=AAC_BITRATE_DEFAULT,
    ) -> str:
        """
        Re-encodes the default audio track (the one enhance transcription uses)
        to AAC at `dest_file_path`, keeping container tags and chapters.
        Returns error string on fail.
        """
        try:
            transcode_to_aac(source_file_path, dest_file_path, bitrate)
        except Exception as e:
            return describe_error(e)
        return ""

    @staticmethod
    def transcode_to_aac(source_file_path: str, suffix: str=".m4b") -> tuple[str, str]:
        """
        Transcodes an audio file to AAC/MP4 format.
        Returns saved file name and error string, mutually exclusive
        """

        # TODO i refactored things, re-verify

        path = Path(source_file_path)
        if path.suffix in AAC_SUFFIXES:
            return "", "Is already an mp4/m4a file"

        dest_file_path = str(path.with_suffix(suffix))
        dest_file_path = make_unique_file_path(dest_file_path)

        partial_command = FFMPEG_TYPICAL_OPTIONS[:]
        partial_command.extend(["-i", source_file_path])
        partial_command.extend(make_ffmpeg_arguments_output_aac())
        partial_command.append("-")

        err = FfmpegUtil.make_file(partial_command, dest_file_path, False)
        if err:
            return "", err

        return dest_file_path, ""
