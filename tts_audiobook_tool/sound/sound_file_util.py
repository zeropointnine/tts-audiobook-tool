import math

import librosa
import numpy as np
import soundfile

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.audio_convert import describe_error, transcode_to_aac
from tts_audiobook_tool.sound.audio_decoder import AudioDecodeError, read_audio
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
