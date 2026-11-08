"""Chatterbox behavior: blocks project languages the audio.cpp session rejects.

The audio.cpp Chatterbox session lowercases the language, cuts it at the first
"-", treats empty as English, and raises a generation error for anything not in
its own list. That list is narrower than the Python library's (no ja, zh, he,
ru), so readiness checks it up front instead of failing mid-run. The catalog's
"normalized" policy already sends the base code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.text_ops.language_util import normalize_language_code
from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior, ParameterValues

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project

# Mirrors supported_chatterbox_language_codes() in audio.cpp
# src/models/chatterbox/text_tokenizer.cpp; the server does not report it.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "hi", "it",
    "ko", "ms", "nl", "no", "pl", "pt", "sv", "sw", "tr",
})


class ChatterboxBehavior(AudioCppModelBehavior):

    def get_blocking_issues(self, project: Project, values: ParameterValues) -> list[ReadinessIssue]:
        # Empty is fine: the session treats it as English
        code = normalize_language_code(project.language_code or "")
        if code and code not in SUPPORTED_LANGUAGES:
            return [ReadinessIssue(
                "language", f"Chatterbox (audio.cpp) does not support project language {project.language_code.strip()!r}")]
        return []
