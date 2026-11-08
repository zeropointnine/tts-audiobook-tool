"""
OmniVoice language ids, shared by the local model and the audio.cpp behavior.

OmniVoice (local library and audio.cpp alike) accepts an exact id from its own
~650-entry table — ISO 639-1 where one exists, otherwise ISO 639-3 (eg "yue",
"nan", "nn") — or an English language name. It has no region/script subtags,
so "en-US" is unknown to it: the audio.cpp session rejects the request and the
local library silently drops to language-agnostic mode.

`resolve_omnivoice_language` therefore normalizes the project code to its base
code, then corrects the few base codes that are not OmniVoice ids.
"""

from tts_audiobook_tool.text_ops.language_util import normalize_language_code

# Base codes OmniVoice spells differently (its table has no "ar" or "tl").
_OMNIVOICE_ID_BY_BASE: dict[str, str] = {
    "ar": "arb",  # Standard Arabic
    "tl": "fil",  # Filipino
}

# Codes the shared alias table folds into "no", but which OmniVoice keeps
# distinct as Norwegian Bokmål.
_BOKMAL_CODES = frozenset({"nb", "nob"})


def resolve_omnivoice_language(language_code: str) -> str | None:
    """OmniVoice language id for a free-form project code, or None when empty."""
    subtag = (language_code or "").strip().lower().replace("_", "-").split("-", 1)[0]
    if subtag in _BOKMAL_CODES:
        return "nb"
    base = normalize_language_code(language_code or "")
    if not base:
        return None
    return _OMNIVOICE_ID_BY_BASE.get(base, base)
