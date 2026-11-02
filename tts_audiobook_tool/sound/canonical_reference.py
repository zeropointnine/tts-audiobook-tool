"""
Canonical voice references for local TTS models.

Model libraries load reference audio with their own file loaders (librosa,
torchaudio, soundfile, ...). Some of those reach system FFmpeg for formats
libsndfile can't read: librosa through audioread, torchaudio through its FFmpeg
backend or TorchCodec. Handing them only PCM WAV/FLAC files that libsndfile
reads keeps them on soundfile, so no model needs system FFmpeg just to read a
voice sample.

Voice samples imported through the app are already canonical (decoded by the
app and saved as 16-bit FLAC, see `ProjectVoiceUtil._save_voice_sample_file`),
so they pass through unchanged. Anything else (eg an MP3 or M4A copied into an
older project by a pre-FLAC version of the app) is decoded once with the app's
PyAV decoder into a cached FLAC, keyed by the source's path, mtime and size.
A changed source yields a new cached file (and a new path), so model voice-clone
caches keyed by (path, mtime, size) are invalidated exactly as they are for the
source itself. An unchanged source reuses the cached file as is.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import soundfile

# libsndfile container formats and sample encodings every model loader handles
CANONICAL_FORMATS = frozenset({"WAV", "WAVEX", "FLAC"})
CANONICAL_SUBTYPES = frozenset({"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"})

CANONICAL_CACHE_SUBDIR = "voice_references"


class VoiceReferenceError(ValueError):
    """A voice reference file could not be turned into a canonical one."""


def is_canonical_reference(path: str) -> bool:
    """Whether libsndfile reads `path` as PCM/float WAV or FLAC."""
    try:
        info = soundfile.info(path)
    except Exception:
        return False
    return info.format in CANONICAL_FORMATS and info.subtype in CANONICAL_SUBTYPES


def default_cache_dir() -> str:
    from tts_audiobook_tool.app_support.app_paths import get_app_temp_dir
    return os.path.join(get_app_temp_dir(), CANONICAL_CACHE_SUBDIR)


def canonical_reference_path(path: str, cache_dir: str | None = None) -> str:
    """
    A PCM WAV/FLAC path with the same audio as `path`: `path` itself when it
    already is one, else a cached FLAC decoded from it (native sample rate, at
    most two channels).

    Raises `FileNotFoundError` for a missing source and `VoiceReferenceError`
    when it can't be decoded. Never falls back to an external decoder.
    """
    if is_canonical_reference(path):
        return path

    source = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    stat = os.stat(source)
    source_key = _digest(source)
    version_key = _digest(f"{source}\0{stat.st_mtime_ns}\0{stat.st_size}")
    cache_dir = cache_dir or default_cache_dir()
    dest = os.path.join(cache_dir, f"{source_key}-{version_key}.flac")
    if is_canonical_reference(dest):
        return dest

    from tts_audiobook_tool.sound.audio_decoder import read_audio

    try:
        data, sample_rate = read_audio(path, sample_rate=None, channels=None)
    except Exception as e:
        raise VoiceReferenceError(f"Couldn't decode voice sample {path}: {e}") from e
    if not len(data):
        raise VoiceReferenceError(f"Voice sample has no audio: {path}")

    os.makedirs(cache_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".flac", dir=cache_dir)
    os.close(fd)
    try:
        soundfile.write(
            temp_path, np.clip(data, -1.0, 1.0), sample_rate, format="FLAC", subtype="PCM_16"
        )
        os.replace(temp_path, dest)
        temp_path = ""
    except Exception as e:
        raise VoiceReferenceError(f"Couldn't save canonical copy of voice sample {path}: {e}") from e
    finally:
        if temp_path:
            _remove_quietly(temp_path)

    # Older versions of the same source are stale. A copy still open elsewhere
    # (Windows) is left for a later run.
    prefix = f"{source_key}-"
    for name in os.listdir(cache_dir):
        if name.startswith(prefix) and name != os.path.basename(dest):
            _remove_quietly(os.path.join(cache_dir, name))
    return dest


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
