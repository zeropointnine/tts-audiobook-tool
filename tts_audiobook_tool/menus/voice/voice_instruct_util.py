"""OmniVoice voice-design instruction validation, shared by its local and audio.cpp menus.

Kept free of menu, project and TTS imports so either backend's menu can use the
same best-effort resolver.
"""
from __future__ import annotations

from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import make_error_string


def validate_instruct(instruct: str) -> tuple[str, str]:
    """
    Best-effort pre-validation using OmniVoice's own instruct resolver, so users
    get feedback at menu time instead of waiting for inference.

    Returns:
        (error_message, normalized_instruct)

    If OmniVoice isn't importable in the current environment, validation is
    skipped and the original value is returned unchanged. That is the normal
    case for the audio.cpp entry, whose server validates its own vocabulary.
    """
    try:
        from omnivoice.models.omnivoice import _resolve_instruct  # type: ignore
    except Exception as e:
        L.e(f"{instruct} - {e}")
        return "", instruct

    try:
        normalized = _resolve_instruct(instruct)
    except ValueError as e:
        return make_error_string(e), ""
    except Exception:
        # Can't validate, just allow
        return "", instruct

    if normalized is None:
        return "", ""
    return "", normalized
