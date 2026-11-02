"""Shared helpers for the per-model *_voice_clone_cache tests."""

import os


def cache_key_prefix(path: str) -> str:
    """The normalized voice path that TtsBaseModel's voice-clone cache keys start with."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def write_voice_file(path, content: bytes = b"voice") -> None:
    """
    A tiny real PCM voice file (WAV or FLAC, by suffix) standing in for a
    sample. Models get voice paths through the canonical-reference layer,
    which reads the file, so placeholder bytes no longer do. Different
    `content` gives different audio and size, as different bytes did.
    """
    import numpy as np
    import soundfile

    rng = np.random.default_rng(sum(content))
    samples = (0.1 * rng.standard_normal(100 * (len(content) + 1))).astype(np.float32)
    soundfile.write(str(path), samples, 16000, subtype="PCM_16")
