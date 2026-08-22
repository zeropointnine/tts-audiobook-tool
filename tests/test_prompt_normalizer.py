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
        ("En-dash\u2014here", "En-dash\u2014here"),
        ("En-dash bounded by spaces \u2014 here",
         "En-dash bounded by spaces \u2014 here"),
        ("Dash bounded by spaces - here", "Dash bounded by spaces - here"),
        ("Garbage punctuation: ,.,.,---... ", "Garbage punctuation:"),
        ("Garbage punctuation in two consecutive 'words': ,.,.,-- -... ",
         "Garbage punctuation in two consecutive 'words':"),
    ],
)
def test_normalize_prompt_en(source: str, expected: str) -> None:
    assert PromptNormalizer.normalize_prompt(source, language_code="en") == expected


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