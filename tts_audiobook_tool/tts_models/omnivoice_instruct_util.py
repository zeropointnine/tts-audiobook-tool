"""OmniVoice voice-design instruction validation for the instructions editor.

Runs in the interactive main process, so OmniVoice is imported lazily: the
first call pays its import cost (no model or CUDA context is created).
"""
from __future__ import annotations

from tts_audiobook_tool.l import L


def validate_omnivoice_instruct(instruct: str) -> str:
    """
    Best-effort validation using OmniVoice's own instruct resolver.
    Returns an error message, or empty string when the value is acceptable.

    Validation only; callers keep the user's text. OmniVoice re-normalizes it
    at generation time (eg to Chinese items when the prompt text is Chinese).

    If OmniVoice isn't importable in the current environment, validation is
    skipped; generation still reports an invalid value.
    """
    try:
        from omnivoice.models.omnivoice import _resolve_instruct  # type: ignore
    except Exception as e:
        L.e(f"OmniVoice instruct validation unavailable: {e}")
        return ""

    try:
        normalized = _resolve_instruct(instruct)
    except ValueError as e:
        return str(e)
    except Exception as e:
        # Can't validate, just allow
        L.e(f"OmniVoice instruct validation failed: {e}")
        return ""

    if not normalized:
        return "No instruction items found"
    return ""
