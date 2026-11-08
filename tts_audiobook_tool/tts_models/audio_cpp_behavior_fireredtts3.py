"""FireRedTTS3 (audio.cpp): project language -> the model's language tag.

The session builds its prompt from an exact language tag ("English",
"Chinese", ...), rejects anything else, and defaults to Chinese when none is
sent. The project stores a free-form language code, so this behavior maps its
normalized base code onto the tag list (``resolve_language``) and sends the
result as ``options.language``. A project language with no mapping blocks generation
rather than silently generating under the wrong frontend.

The tag is a Base-model feature (it goes into Base's prompt). If the server
serves the Instruct package instead, the request is still accepted but the tag
only selects the text normalizer; the server does not reveal which variant is
loaded, so the unmapped-language block applies either way.

Chinese dialect tags (``ZH_*``) are deliberately unreachable: ``zh`` always
maps to plain ``Chinese``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.text_ops.language_util import map_language
from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior, ParameterValues

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project

# Base code -> FireRedTTS3 tag. Aliases ("eng", "german", ...) and region
# suffixes are resolved by `language_util.map_language`.
_TAG_BY_CODE: dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}

# Tags whose text frontend is a full native normalizer; the others only get
# whitespace cleanup, and audio.cpp reports weaker quality for them.
_WELL_SUPPORTED_TAGS = frozenset({"English", "Chinese", "Cantonese"})


class FireRedTts3Behavior(AudioCppModelBehavior):

    @staticmethod
    def get_language_tag(language_code: str) -> str | None:
        """FireRedTTS3 language tag for a project language code, or None."""
        return map_language(language_code or "", _TAG_BY_CODE)

    def resolve_language(self, language: str) -> str:
        # The catalog's "normalized" policy hands over the base code; send the
        # exact tag the session requires. Readiness blocks unmapped codes first;
        # "" would omit the field rather than send an unknown tag.
        return map_language(language, _TAG_BY_CODE) or ""

    def get_blocking_issues(self, project: Project, values: ParameterValues) -> list[ReadinessIssue]:
        if self.get_language_tag(project.language_code) is None:
            shown = (project.language_code or "").strip() or "(empty)"
            return [ReadinessIssue(
                "language", f"FireRedTTS3 does not support project language {shown!r}")]
        return []

    def get_warning_issues(self, project: Project, values: ParameterValues) -> list[str]:
        tag = self.get_language_tag(project.language_code)
        if tag is None:
            return []
        if tag in _WELL_SUPPORTED_TAGS:
            return []
        return [f"FireRedTTS3 output for {tag} may be weaker than for English or Chinese"]
