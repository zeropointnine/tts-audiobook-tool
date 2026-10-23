import pytest

from tts_audiobook_tool.app_support import app_text


@pytest.mark.parametrize(
    ("raw_word", "expected"),
    [
        # No trailing or leading ws/punc
        ("hello", ("", "hello", "")),
        ("don't", ("", "don't", "")),
        # Simple whitespace
        ("hello  ", ("", "hello", "  ")),
        ("  hello", ("  ", "hello", "")),
        # Fancy quote
        ("“Hello”", ("“", "Hello", "”")),
        ("‘Hello’", ("‘", "Hello", "’")),
        # Leading and/or trailing punctuation and whitespace
        ('hello", ... ?!', ("", "hello", '", ... ?!')),
        ('", said', ('", ', "said", "")),
        ('", and…', ('", ', "and", "…")),
    ],
)
def test_split_raw_word(raw_word: str, expected: tuple[str, str, str]) -> None:
    assert app_text.split_raw_word(raw_word) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Typographical scene-break ornaments
        ("◇", True),
        ("* * *\n\n", True),
        ("---", True),
        (". . .", True),
        # Quote-led non-vocalizable lines are silent dialog beats, not ornaments
        ('"..."', False),
        ("“…”\n\n", False),
        ("‘—’", False),
        ("«...»", False),
        # Vocalizable or empty text is never ornamental
        ('"Well?"', False),
        ("", False),
        ("\n\n", False),
    ],
)
def test_is_ornamental_line(text: str, expected: bool) -> None:
    assert app_text.is_ornamental_line(text) is expected


def test_has_trailing_ornamental_line_ignores_quoted_beat() -> None:
    # A trailing `"..."` line is diegetic, so it does not record a space break.
    assert app_text.has_trailing_ornamental_line('"Well?"\n\n◇\n\n')
    assert not app_text.has_trailing_ornamental_line('"Well?"\n\n"..."\n\n')