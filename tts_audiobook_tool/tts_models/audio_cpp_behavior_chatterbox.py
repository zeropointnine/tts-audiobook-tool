"""Chatterbox behavior: resolves the language against the audio.cpp session's list.

The audio.cpp Chatterbox session lowercases the language, cuts it at the first
"-", treats empty as English, and raises a generation error for anything not in
its own list. That list is narrower than the Python library's (no ja, zh, he,
ru), so an unlisted code resolves UNSUPPORTED and readiness blocks it instead
of failing mid-run. The catalog's "normalized" policy hands over the base code.
"""
from __future__ import annotations

from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior
from tts_audiobook_tool.tts_models.language_resolution import LanguageResolution, LanguageStatus

# Mirrors supported_chatterbox_language_codes() in audio.cpp
# src/models/chatterbox/text_tokenizer.cpp; the server does not report it.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "hi", "it",
    "ko", "ms", "nl", "no", "pl", "pt", "sv", "sw", "tr",
})


class ChatterboxBehavior(AudioCppModelBehavior):

    def resolve_language(self, language: str) -> LanguageResolution:
        if not language:
            # Field omitted: the session treats it as English
            return LanguageResolution(LanguageStatus.NOT_SET)
        if language in SUPPORTED_LANGUAGES:
            return LanguageResolution(LanguageStatus.MAPPED, language)
        return LanguageResolution(LanguageStatus.UNSUPPORTED)
