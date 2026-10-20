"""Shared helpers for the per-model *_voice_clone_cache tests."""

import os


def cache_key_prefix(path: str) -> str:
    """The normalized voice path that TtsBaseModel's voice-clone cache keys start with."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))
