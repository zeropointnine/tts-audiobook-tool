"""
Shared language-code normalization.

Project language codes are free-form: the menu stores whatever the user typed
(lowercased and stripped), so values like "en", "en-US", "en_GB", "eng", or
"English" can all end up on a project. Features that key off a language
normalize on read rather than at input.

Keeping the alias table and base-code rule here, in one place, stops those
features (the common-words whitelist, word equivalence) from drifting apart.
"""


# Common full names and ISO 639-2 codes -> primary 2-letter code.
# Keeps language-keyed features from being silently disabled by a
# region/script/name-qualified project language code.
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en",
    "eng": "en",
    "spanish": "es",
    "spa": "es",
}


def normalize_language_code(language_code: str) -> str:
    """Returns the base, lowercase language code (eg ``en-US`` -> ``en``).

    Accepts BCP-47 region/script variants (``en-US``, ``en_GB``), ISO 639-2
    three-letter codes (``eng``), and common full names (``English``).
    """
    code = language_code.strip().lower()
    if not code:
        return ""
    base = code.replace("_", "-").split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(base, base)
