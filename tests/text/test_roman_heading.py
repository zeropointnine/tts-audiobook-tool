"""
Tests for the "bare Roman numeral heading" rule: a segment that is nothing
but an upper-case Roman numeral (eg, a chapter heading line "XI") is spoken
as a number word ("Eleven"), and the text comparison step expects that word.

See docs-dev/roman-numeral-headings.md
"""

import pytest

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.roman_heading import (
    expand_bare_roman_numeral,
    parse_roman_numeral,
)
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer


# ---
# parse_roman_numeral

@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("II", 2),
        ("IV", 4),
        ("V", 5),
        ("IX", 9),
        ("XI", 11),
        ("XIV", 14),
        ("XLII", 42),
        ("XC", 90),
        ("CD", 400),
        ("CMXCIX", 999),  # largest supported
    ],
)
def test_parse_valid_numerals(token: str, value: int) -> None:
    assert parse_roman_numeral(token) == value


@pytest.mark.parametrize(
    "token",
    [
        "I",       # excluded: far more likely the pronoun
        "",
        "xi",      # lower case: probably a word, not a heading
        "Xi",
        "M",       # 1000 and up is out of range
        "MI",
        "IIII",    # not canonical
        "VX",      # not canonical
        "IC",      # not canonical (99 is XCIX)
        "XIIV",
        "MIX",     # a word (and over the range)
        "DIM",
        "CIVIL",
        "LIVID",
        "MILD",
        "XI1",
        "X I",
    ],
)
def test_parse_rejects_non_numerals(token: str) -> None:
    assert parse_roman_numeral(token) is None


# ---
# expand_bare_roman_numeral

@pytest.mark.parametrize(
    ("text", "language_code", "expected"),
    [
        ("XI", "en", "Eleven"),
        ("XI.", "en", "Eleven."),
        ("IV", "en", "Four"),
        ("XLII", "en", "Forty-two"),
        ("CXXIII", "en", "One hundred and twenty-three"),
        ("XI\n", "en", "Eleven\n"),        # surrounding whitespace is preserved
        ("\n  XI\n\n", "en", "\n  Eleven\n\n"),
        ("XI", "en-US", "Eleven"),         # region codes are fine
        ("XI", "es", "Once"),
        ("XXI", "es", "Veintiuno"),
    ],
)
def test_expand_positive(text: str, language_code: str, expected: str) -> None:
    assert expand_bare_roman_numeral(text, language_code) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I",                       # excluded
        "I.",
        "Henry VIII",              # numeral is part of other text
        "VIII was the last of them",
        "Chapter XI",
        "World War II",
        "XI XII",
        "XI.1",
        "XI!",
        "(XI)",
        "XI:",
        "xi",
        "Mix",
        "MMXXIV",                  # out of range
        "",
        "   ",
    ],
)
def test_expand_negative(text: str) -> None:
    assert expand_bare_roman_numeral(text, "en") is None


def test_expand_leaves_text_alone_when_language_unsupported_or_missing() -> None:
    assert expand_bare_roman_numeral("XI", "") is None
    assert expand_bare_roman_numeral("XI", "zz") is None


# ---
# PhraseGroup.spoken_text (the reason policy)

def _group(*phrases: tuple[str, Reason]) -> PhraseGroup:
    return PhraseGroup([Phrase(text, reason) for text, reason in phrases])


@pytest.mark.parametrize(
    "reason", [Reason.PARAGRAPH, Reason.SPACE_BREAK, Reason.SECTION_BREAK]
)
def test_spoken_text_expands_standalone_heading(reason: Reason) -> None:
    group = _group(("XI", reason))
    assert group.spoken_text("en") == "Eleven"
    assert group.text == "XI"  # the stored text itself is untouched


@pytest.mark.parametrize(
    "reason",
    [Reason.UNDEFINED, Reason.WORD, Reason.PHRASE, Reason.SENTENCE],
)
def test_spoken_text_leaves_numeral_below_paragraph_alone(reason: Reason) -> None:
    group = _group(("XI", reason))
    assert group.spoken_text("en") == "XI"


def test_spoken_text_leaves_merged_group_alone() -> None:
    # Eg, a short sentence merged into a neighbor: not a standalone line
    group = _group(("XI.", Reason.SENTENCE), ("The rain fell.", Reason.PARAGRAPH))
    assert group.spoken_text("en") == "XI.The rain fell."


def test_spoken_text_leaves_ordinary_text_alone() -> None:
    group = _group(("Henry VIII was king.", Reason.PARAGRAPH))
    assert group.spoken_text("en") == "Henry VIII was king."
    group = _group(("I", Reason.PARAGRAPH))
    assert group.spoken_text("en") == "I"


# ---
# End to end: the comparison step agrees with what was spoken
#
# These assert on the normalized forms the validator compares. (Asserting on
# Validator word errors would be misleading here: a lone number token is not
# in the common-word whitelist, so the validator gives it a free "uncommon
# word" wildcard pass against any single transcript word, right or wrong.)

def _normalized(source: str, transcript: str, language_code: str) -> tuple[str, str]:
    return TextNormalizer.normalize_source_and_transcript(
        source, transcript, language_code=language_code
    )


@pytest.mark.parametrize("transcript", ["Eleven.", "Eleven", "eleven", "11"])
def test_comparison_accepts_spoken_heading_transcript(transcript: str) -> None:
    source = _group(("XI", Reason.PARAGRAPH)).spoken_text("en")
    normalized_source, normalized_transcript = _normalized(source, transcript, "en")
    assert normalized_source == normalized_transcript == "11"


def test_comparison_without_the_rule_would_not_match() -> None:
    # Documents why the rule exists: raw "XI" never equals a spoken "Eleven."
    assert _normalized("XI", "Eleven.", "en") == ("xi", "11")


def test_comparison_still_distinguishes_a_wrong_number() -> None:
    source = _group(("XI", Reason.PARAGRAPH)).spoken_text("en")
    normalized_source, normalized_transcript = _normalized(source, "Twelve.", "en")
    assert normalized_source != normalized_transcript


def test_comparison_spanish_heading() -> None:
    source = _group(("XI", Reason.PARAGRAPH)).spoken_text("es")
    assert source == "Once"
    assert _normalized(source, "Once.", "es") == ("11", "11")


# ---
# The English number normalizer no longer depends on attached punctuation

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Eleven", "11"),
        ("Eleven.", "11"),
        ("Eleven,", "11"),
        ("I have forty-two.", "i have 42"),
        ("Chapter twenty-one: Home", "chapter 21 home"),
        ("He paid $5.50.", "he paid 5 50"),  # decimals are not split
    ],
)
def test_english_numbers_normalize_regardless_of_trailing_punctuation(
    text: str, expected: str
) -> None:
    assert TextNormalizer.normalize_common(text, "en") == expected
