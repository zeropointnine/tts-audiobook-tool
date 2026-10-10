"""FireRedTTS3 (audio.cpp): project language -> the model's language tag.

The session builds its prompt from an exact language tag ("English",
"Chinese", ...), rejects anything else, and defaults to Chinese when none is
sent. The project stores a free-form language code, so this behavior maps its
normalized base code onto the tag list and sends the result as
``options.language``. A project language with no mapping blocks generation
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
from tts_audiobook_tool.text_ops.language_util import normalize_language_code
from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior, ParameterValues

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project

# Normalized base code (ISO 639-1, plus a few ISO 639-2/3 and English-name
# spellings that free-form project codes often use) -> FireRedTTS3 tag.
_TAG_BY_CODE: dict[str, str] = {
    "ar": "Arabic", "arabic": "Arabic",
    "cs": "Czech", "ces": "Czech", "cze": "Czech", "czech": "Czech",
    "de": "German", "deu": "German", "ger": "German", "german": "German",
    "el": "Greek", "ell": "Greek", "gre": "Greek", "greek": "Greek",
    "en": "English", "eng": "English", "english": "English",
    "es": "Spanish", "spa": "Spanish", "spanish": "Spanish",
    "fi": "Finnish", "fin": "Finnish", "finnish": "Finnish",
    "fr": "French", "fra": "French", "fre": "French", "french": "French",
    "hi": "Hindi", "hin": "Hindi", "hindi": "Hindi",
    "id": "Indonesian", "ind": "Indonesian", "indonesian": "Indonesian",
    "it": "Italian", "ita": "Italian", "italian": "Italian",
    "ja": "Japanese", "jpn": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "kor": "Korean", "korean": "Korean",
    "nl": "Dutch", "nld": "Dutch", "dut": "Dutch", "dutch": "Dutch",
    "pl": "Polish", "pol": "Polish", "polish": "Polish",
    "pt": "Portuguese", "por": "Portuguese", "portuguese": "Portuguese",
    "ro": "Romanian", "ron": "Romanian", "rum": "Romanian", "romanian": "Romanian",
    "ru": "Russian", "rus": "Russian", "russian": "Russian",
    "th": "Thai", "tha": "Thai", "thai": "Thai",
    "tr": "Turkish", "tur": "Turkish", "turkish": "Turkish",
    "uk": "Ukrainian", "ukr": "Ukrainian", "ukrainian": "Ukrainian",
    "vi": "Vietnamese", "vie": "Vietnamese", "vietnamese": "Vietnamese",
    "yue": "Cantonese", "cantonese": "Cantonese",
    "zh": "Chinese", "zho": "Chinese", "chi": "Chinese", "chinese": "Chinese",
}

# Tags whose text frontend is a full native normalizer; the others only get
# whitespace cleanup, and audio.cpp reports weaker quality for them.
_WELL_SUPPORTED_TAGS = frozenset({"English", "Chinese", "Cantonese"})


class FireRedTts3Behavior(AudioCppModelBehavior):

    @staticmethod
    def get_language_tag(language_code: str) -> str | None:
        """FireRedTTS3 language tag for a project language code, or None."""
        return _TAG_BY_CODE.get(normalize_language_code(language_code or ""))

    def adjust_payload(self, payload: dict, values: ParameterValues) -> None:
        # The catalog's "normalized" policy put the base code in
        # `options.language`; swap it for the exact tag the session requires.
        # Readiness blocks unmapped codes first, so a miss here is a bug.
        options = payload["options"]
        options["language"] = _TAG_BY_CODE[options["language"]]

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
