"""Conservative typography equivalences for every Textual Find consumer."""

import pytest

from tts_audiobook_tool.textual.find_text import normalize_find_text


@pytest.mark.parametrize(
    ("variants", "expected"),
    [
        ("'‘’‚‛‹›", "'"),
        ('"“”„‟«»', '"'),
        ("-‐‑‒–—", "-"),
        (" \u00a0\u202f", " "),
    ],
)
def test_character_families(variants: str, expected: str) -> None:
    for character in variants:
        assert normalize_find_text(character) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Wait…", "WAIT..."),
        ("CAFÉ", "cafe\u0301"),
        ("Straße", "STRASSE"),
        ("«Don’t—stop…»\u00a0Café", '"DON\'T-STOP..." CAFE\u0301'),
    ],
)
def test_equivalences_are_symmetric_and_idempotent(left: str, right: str) -> None:
    # Normalizing both sides also supports typographic variants in the query.
    normalized = normalize_find_text(left)
    assert normalized == normalize_find_text(right)
    assert normalize_find_text(normalized) == normalized


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("café", "cafe"),
        ("a-b", "ab"),
        ("a b", "a  b"),
        ("a\tb", "a b"),
        ("a\nb", "a b"),
        ("a\u200bb", "ab"),
        ("−", "-"),  # Mathematical minus is not a typographic dash.
        ("′", "'"),  # Prime is not an apostrophe.
        ("①", "1"),  # No broad Unicode compatibility folding.
        ('"', "'"),
    ],
)
def test_other_distinctions_stay_significant(left: str, right: str) -> None:
    assert normalize_find_text(left) != normalize_find_text(right)


def test_empty_text_stays_empty() -> None:
    assert normalize_find_text("") == ""
