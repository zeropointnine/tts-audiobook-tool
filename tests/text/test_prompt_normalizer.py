import pytest

from tts_audiobook_tool.text_ops.prompt_normalizer import PromptNormalizer


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Normally formatted sentence. Sanity check.",
         "Normally formatted sentence. Sanity check."),
        ("  Trim ends of    string and  normalize extra white space  ",
         "Trim ends of string and normalize extra white space"),
        # ellipsis
        ("Ellipsis character… So.", "Ellipsis character... So."),
        ("Ellipsis character with space before … So.",
         "Ellipsis character with space before ... So."),
        ("Triple-dot... So.", "Triple-dot... So."),
        ("Triple-dot with space before ... So.",
         "Triple-dot with space before ... So."),
        ("Alternate format for ellipsis, not uncommon . . . Ugh.",
         "Alternate format for ellipsis, not uncommon ... Ugh."),
        ("Fancy apost: I think it’s . . . unseemly.",
         "Fancy apost: I think it's ... unseemly."),
        # German quotes: low-9 opening quotes like the other fancy quotes
        ("„Fred?“ Er sagte: ‚Ja.‘", "\"Fred?\" Er sagte: 'Ja.'"),
        # Unicode classifies the ornamental forms as symbols, even inside dialogue.
        ("❝Don❜t say ❛never❜.❞", '"Don\'t say \'never\'."'),
        ("❠He said ❟yes❜.❞", '"He said \'yes\'."'),
        ("\U0001f676Yes\U0001f677. \U0001f678No\U0001f677.", '"Yes". "No".'),
        # NFKC also expands ℃ to °C. Keep both the degree and the digit boundaries.
        ("30°C or 30℃", "thirty°C or thirty°C"),
        ("12°30′", "twelve°thirty′"),
        ("Multiple dots...........", "Multiple dots..."),
        ("Multiple dots with space ...........", "Multiple dots with space ..."),
        ("Multiple ellipsis characters…………", "Multiple ellipsis characters..."),
        ("Multiple ellipsis characters with space …………",
         "Multiple ellipsis characters with space ..."),
        ("This is a pause...... and another one…… and a mixed one…...",
         "This is a pause... and another one... and a mixed one..."),
        # dashes
        ("Normal dash usage: Higgs-Boson", "Normal dash usage: Higgs-Boson"),
        ("Em-dash\u2014okay?", "Em-dash\u2014okay?"),
        ("Em-dash bounded by spaces \u2014 okay?",
         "Em-dash bounded by spaces \u2014 okay?"),
        ("En-dash\u2013here", "En-dash\u2013here"),
        ("En-dash bounded by spaces \u2013 here",
         "En-dash bounded by spaces \u2013 here"),
        # U+2500 is a Unicode symbol, but can be used as a dash in prose.
        ("Waited─then", "Waited─then"),
        ("Waited ─ then", "Waited ─ then"),
        ("Waited ─── then", "Waited then"),
        ("Dash bounded by spaces - here", "Dash bounded by spaces - here"),
        ("Garbage punctuation: ,.,.,---... ", "Garbage punctuation:"),
        ("Garbage punctuation in two consecutive 'words': ,.,.,-- -... ",
         "Garbage punctuation in two consecutive 'words':"),
    ],
)
def test_normalize_prompt_en(source: str, expected: str) -> None:
    assert PromptNormalizer.normalize_prompt(source, language_code="en") == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Grouping commas don't change whether a number is spelled out.
        ("It is 3,200 miles away", "It is 3200 miles away"),
        ("It is 3200 miles away", "It is 3200 miles away"),
        ("Total: 1,200,000.", "Total: 1200000."),
        ("It was 3,200, give or take.", "It was 3200, give or take."),
        # Keep list punctuation and the 999 cutoff for spelling out numbers.
        ("3, 200", "three, two hundred"),
        ("999 and 1,000", "nine hundred and ninety-nine and 1000"),
        # Don't speak only part of a compound number, malformed grouping, or ID.
        ("3,200.50", "3,200.50"),
        ("1,234,56", "1,234,56"),
        ("$3,200 and € 3,200", "$3,200 and € 3,200"),
        ("$ 3,200 or 3,200 $", "$3,200 or 3,200$"),
        ("3,200% or 3,200 %", "3,200% or 3,200%"),
        ("ID-3,200 and -3,200", "ID-3,200 and -3,200"),
    ],
)
def test_grouped_english_integers_in_prompt(source: str, expected: str) -> None:
    assert PromptNormalizer.normalize_prompt(source, language_code="en") == expected


def test_grouped_integer_rule_uses_english_language_aliases() -> None:
    # A comma need not be a thousands separator in another language.
    assert PromptNormalizer.normalize_prompt("3,200", language_code="en-US") == "3200"
    assert PromptNormalizer.normalize_prompt("3,200", language_code="es") == "tres,doscientos"


@pytest.mark.parametrize(
    ("source", "language_code", "un_all_caps", "expected"),
    [
        # Only a standalone ampersand is verbalized; keep names and query strings intact.
        ("Smith & Sons, AT&T, R&D, https://example.com/?a=x&b=y", "en", False,
         "Smith and Sons, AT&T, R&D, https://example.com/?a=x&b=y"),
        ("Smith ＆ Sons", "en-US", False, "Smith and Sons"),
        ("A && B", "en", False, "A B"),
        ("Smith & Sons", "fr", False, "Smith & Sons"),
        ("Smith & Sons", "", False, "Smith & Sons"),
        # The inserted lowercase "and" must not interrupt the all-caps rewrite.
        ("OLIVER SMITH & SONS", "en", True, "oliver smith and sons"),
    ],
)
def test_standalone_ampersand_in_prompt(
    source: str, language_code: str, un_all_caps: bool, expected: str
) -> None:
    assert PromptNormalizer.normalize_prompt(source, language_code, un_all_caps) == expected


PROMPT_WORD_SUBSTITUTIONS = {"Ariekei": "AriaKay", "kilohour": "kilo hour"}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # "Identity"
        ("Identity example", "Identity example"),
        # Should also change nothing
        ("It's a kilo hour", "It's a kilo hour"),
        # Typical example
        ("The Ariekei are a species.", "The AriaKay are a species."),
        # Lower-to-uppercase (because source word is uppercase)
        ("Kilohour is the unit of measurement.",
         "Kilo hour is the unit of measurement."),
        # Noun suffix plural
        ("It is five kilohours away", "It is five kilo hours away"),
        # Noun suffix + lower-to-upper
        ("Kilohours off my life", "Kilo hours off my life"),
        # Noun suffix possessive
        ("The Ariekei's morphology is kinda",
         "The AriaKay's morphology is kinda"),
    ],
)
def test_prompt_word_substitutions_en(source: str, expected: str) -> None:
    result = PromptNormalizer.apply_prompt_word_substitutions(
        source, PROMPT_WORD_SUBSTITUTIONS, "en"
    )
    assert result == expected


@pytest.mark.parametrize("language_code", ["en-US", "en_GB", "eng", "English"])
def test_prompt_word_substitutions_accepts_first_class_language_aliases(
    language_code: str,
) -> None:
    result = PromptNormalizer.apply_prompt_word_substitutions(
        "It is five kilohours away", PROMPT_WORD_SUBSTITUTIONS, language_code
    )
    assert result == "It is five kilo hours away"


def test_prompt_word_substitutions_skips_suffixes_for_other_languages() -> None:
    result = PromptNormalizer.apply_prompt_word_substitutions(
        "It is five kilohours away", PROMPT_WORD_SUBSTITUTIONS, "fr"
    )
    assert result == "It is five kilohours away"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("This is a normal ", "This is a normal "),
        ("TYPICAL EXAMPLE OF THE START of a new section",
         "typical example of the start of a new section"),
        ("TYPICAL EXAMPLE OF THE START of a new section, and here's an ACRONYM",
         "typical example of the start of a new section, and here's an ACRONYM"),
        ("UPPER UP ... UP, they said.", "upper up ... up, they said."),
        ("Nonuppercase UPPER UPPER UPPER", "Nonuppercase UPPER UPPER UPPER"),
    ],
)
def test_un_caps(source: str, expected: str) -> None:
    assert PromptNormalizer.un_all_caps_prompt(source) == expected