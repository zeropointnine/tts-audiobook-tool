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