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
(pysbd, Whisper, num2words) should pass `normalize_language_code(...)`.

Keeping the alias table and base-code rule here, in one place, stops those
features from drifting apart. See docs-dev/language-handling.md.
"""


# Common full names and ISO 639-2/3 codes -> primary 2-letter (or Whisper-style) code.
# Keeps language-keyed features from being silently disabled by a
# region/script/name-qualified project language code.
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en",
    "eng": "en",
    "spanish": "es",
    "spa": "es",
    "chinese": "zh",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",  # Mandarin
    "japanese": "ja",
    "jpn": "ja",
    "korean": "ko",
    "kor": "ko",
    "cantonese": "yue",
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
