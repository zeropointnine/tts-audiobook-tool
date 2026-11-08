"""
Shared language-code normalization and language-class predicates.

Project language codes are free-form: the menu stores whatever the user typed
(lowercased and stripped), so values like "en", "en-US", "en_GB", "eng", or
"English" can all end up on a project. Features that key off a language
normalize on read rather than at input.

Any logic that gives a language special treatment ("English-only rule",
"skip validation for CJK", etc.) should test the raw project code through the
predicates here (`is_english`, `is_spanish`, `is_cjk`,
`is_validation_supported`) rather than comparing strings directly. Code that
hands a language to an external library that only understands base codes
(pysbd, Whisper, num2words) should pass `normalize_language_code(...)`. Code
that translates the project code into a TTS model's own language vocabulary
should use `map_language` with a table keyed by base code.

Keeping the alias table and base-code rule here, in one place, stops those
features from drifting apart. See docs-dev/language-handling.md.
"""

from collections.abc import Mapping
from typing import TypeVar

_T = TypeVar("_T")


# Base code -> its ISO 639-2 (T and B) / 639-3 codes and English names.
# Covers every language some TTS model's language table knows, so those tables
# can be keyed by base code only (see `map_language`).
_ALIASES_BY_BASE: dict[str, tuple[str, ...]] = {
    "ar": ("ara", "arabic"),
    "cs": ("ces", "cze", "czech"),
    "da": ("dan", "danish"),
    "de": ("deu", "ger", "german"),
    "el": ("ell", "gre", "greek"),
    "en": ("eng", "english"),
    "es": ("spa", "spanish"),
    "fa": ("fas", "per", "persian", "farsi"),
    "fi": ("fin", "finnish"),
    "fr": ("fra", "fre", "french"),
    "he": ("heb", "hebrew"),
    "hi": ("hin", "hindi"),
    "hu": ("hun", "hungarian"),
    "id": ("ind", "indonesian"),
    "it": ("ita", "italian"),
    "ja": ("jpn", "japanese"),
    "ko": ("kor", "korean"),
    "mk": ("mkd", "mac", "macedonian"),
    "ms": ("msa", "may", "malay"),
    "nl": ("nld", "dut", "dutch"),
    # Bokmål is the standard written Norwegian; Whisper and Chatterbox only know "no"
    "no": ("nor", "nob", "nb", "norwegian"),
    "pl": ("pol", "polish"),
    "pt": ("por", "portuguese"),
    "ro": ("ron", "rum", "romanian"),
    "ru": ("rus", "russian"),
    "sv": ("swe", "swedish"),
    "sw": ("swa", "swahili"),
    "th": ("tha", "thai"),
    "tl": ("tgl", "tagalog"),
    "tr": ("tur", "turkish"),
    "uk": ("ukr", "ukrainian"),
    "vi": ("vie", "vietnamese"),
    "yue": ("cantonese",),
    "zh": ("zho", "chi", "cmn", "chinese", "mandarin"),
}

# Alias -> base code. Keeps language-keyed features from being silently
# disabled by a name- or three-letter-qualified project language code.
_LANGUAGE_ALIASES: dict[str, str] = {
    alias: base for base, aliases in _ALIASES_BY_BASE.items() for alias in aliases
}

# Languages exempted from whitespace-based word logic (word-error validation,
# word-count batching). Chinese/Japanese/Cantonese are written without spaces
# between words. Korean is space-delimited but has always been grouped here;
# whether it should be is an open question.
CJK_LANGUAGES: frozenset[str] = frozenset({"zh", "ja", "ko", "yue"})


def normalize_language_code(language_code: str) -> str:
    """Returns the base, lowercase language code (eg ``en-US`` -> ``en``).

    Accepts BCP-47 region/script variants (``en-US``, ``en_GB``), ISO 639-2
    three-letter codes (``eng``), and common full names (``English``).
    Empty or whitespace-only input returns ``""`` ("no language").
    """
    code = language_code.strip().lower()
    if not code:
        return ""
    base = code.replace("_", "-").split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(base, base)


def is_english(language_code: str) -> bool:
    """True when the (free-form) code denotes English. Empty is not English."""
    return normalize_language_code(language_code) == "en"


def is_spanish(language_code: str) -> bool:
    """True when the (free-form) code denotes Spanish."""
    return normalize_language_code(language_code) == "es"


def is_cjk(language_code: str) -> bool:
    """True when the (free-form) code denotes a language in `CJK_LANGUAGES`."""
    return normalize_language_code(language_code) in CJK_LANGUAGES


def is_validation_supported(language_code: str) -> bool:
    """
    False when Whisper-based word-error validation cannot work for the language
    (currently: the CJK languages), so validation is skipped entirely.
    Empty ("no language") is supported.
    """
    return not is_cjk(language_code)


def map_language(language_code: str, table: Mapping[str, _T]) -> _T | None:
    """
    Looks up a free-form project code in a table keyed by base code, eg a TTS
    model's ``{"en": "English", ...}``. Returns None when the code is empty or
    the language is not in the table.

    Tables should hold base codes only: aliases, region/script suffixes, and
    case are resolved here, so ``en-US``, ``eng`` and ``English`` all reach the
    ``"en"`` entry.
    """
    code = normalize_language_code(language_code)
    return table.get(code) if code else None
