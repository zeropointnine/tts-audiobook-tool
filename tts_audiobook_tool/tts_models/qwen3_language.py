"""
Qwen3-TTS language values, shared by the local model and the audio.cpp behavior.

Qwen3-TTS takes English language names rather than ISO codes (names also work
on older audio.cpp servers that reject ISO codes). Look up project codes with
`resolve_qwen3_language`, which normalizes first (see `language_util.map_language`).
"""

from tts_audiobook_tool.text_ops.language_util import map_language

# Base code -> Qwen3-TTS language value. May need updating if the model's
# supported languages change with future releases or finetunes.
QWEN3_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}

QWEN3_AUTO_LANGUAGE = "auto"


def resolve_qwen3_language(language_code: str) -> str | None:
    """Qwen3-TTS language value for a free-form project code, or None when unmapped/empty."""
    return map_language(language_code, QWEN3_LANGUAGE_NAMES)
