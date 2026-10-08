import re

from num2words import num2words

from tts_audiobook_tool.text_ops import language_util

"""
Spelling out of "bare Roman numeral" headings (eg, a chapter heading line
that is just "XI") so the TTS model speaks "eleven" and the text comparison
step expects "eleven".

This module is deliberately pure text logic: it knows nothing about phrases
or reasons. Deciding *whether* a piece of text is eligible (ie, that it
stands alone as a paragraph-or-higher segment) is the caller's job, see
`PhraseGroup.spoken_text()`.

See docs-dev/roman-numeral-headings.md
"""

# Strict, canonical Roman numerals from 1 to 999 (no "M", so nothing >= 1000).
# Upper case only: lower case "xi" or "mix" is far more likely to be a word.
_ROMAN_NUMERAL_RE = re.compile(
    r"(?=[IVXLCD])"                # non-empty
    r"(?:CM|CD|D?C{0,3})"          # hundreds
    r"(?:XC|XL|L?X{0,3})"          # tens
    r"(?:IX|IV|V?I{0,3})"          # units
)

_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# A lone "I" is much more likely to be the pronoun than the number one
_EXCLUDED = {"I"}


def parse_roman_numeral(token: str) -> int | None:
    """
    Returns the value of `token` if it is a strict, upper-case Roman numeral
    from 1 to 999 (and not an excluded token like a lone "I"), else None.
    """
    if token in _EXCLUDED or not _ROMAN_NUMERAL_RE.fullmatch(token):
        return None
    total = 0
    for i, char in enumerate(token):
        value = _VALUES[char]
        # Subtractive notation (eg, the "I" in "IV")
        if i + 1 < len(token) and value < _VALUES[token[i + 1]]:
            total -= value
        else:
            total += value
    return total


def expand_bare_roman_numeral(text: str, language_code: str) -> str | None:
    """
    If `text` consists of nothing but a Roman numeral (see `parse_roman_numeral()`),
    optionally followed by a single period, returns it spelled out in the
    language, eg "XI" -> "Eleven", "XLII." -> "Forty-two.".
    Leading and trailing whitespace (eg, a trailing line feed) is preserved.

    Returns None when the text does not qualify, or when the language is
    empty or unsupported by num2words (in which case the text should be left alone).
    """
    stripped = text.strip()
    if not stripped:
        return None
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]

    has_period = stripped.endswith(".")
    token = stripped[:-1] if has_period else stripped

    value = parse_roman_numeral(token)
    if value is None:
        return None

    code = language_util.normalize_language_code(language_code)
    if not code:
        return None
    try:
        words = num2words(value, lang=code)
    except NotImplementedError:
        return None
    if not isinstance(words, str) or not words:
        return None

    words = words[0].upper() + words[1:]
    return f"{leading}{words}{'.' if has_period else ''}{trailing}"
