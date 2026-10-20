import pytest

from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Exactly three vocalizable words qualify; fewer remain attached.
        (
            "They liked it (but I didn't).",
            ["They liked it ", "(but I didn't)."],
        ),
        (
            "Version (beta, release) changed.",
            ["Version (beta, release) changed."],
        ),
        # Qualifying asides split on both sides when surrounded by prose.
        (
            "They left (without saying goodbye) after dinner.",
            ["They left ", "(without saying goodbye) ", "after dinner."],
        ),
        (
            "(For reasons nobody explained) the gate closed.",
            ["(For reasons nobody explained) ", "the gate closed."],
        ),
        (
            "One (with several words here) two (and several more here) end.",
            [
                "One ",
                "(with several words here) ",
                "two ",
                "(and several more here) ",
                "end.",
            ],
        ),
        # Nested parentheses are treated as one top-level aside.
        (
            "It changed (in a way (few expected) at the time) after launch.",
            [
                "It changed ",
                "(in a way (few expected) at the time) ",
                "after launch.",
            ],
        ),
        # Ordinary phrase punctuation remains active inside qualifying prose.
        (
            "He stayed (despite rain, wind, and snow) until noon.",
            [
                "He stayed ",
                "(despite rain, ",
                "wind, ",
                "and snow) ",
                "until noon.",
            ],
        ),
        # An unbalanced sentence gets no parenthetical boundaries.
        (
            "They left (without saying goodbye.",
            ["They left (without saying goodbye."],
        ),
        (
            "They left without saying goodbye) afterward.",
            ["They left without saying goodbye) afterward."],
        ),
        # Empty and punctuation-only parentheses remain attached.
        ("Pause () briefly.", ["Pause () briefly."]),
        ("Pause (---) briefly.", ["Pause (---) briefly."]),
        # Citation/reference punctuation must not create incidental breaks.
        (
            "The claim (Smith et al., 2020) remains disputed.",
            ["The claim (Smith et al., 2020) remains disputed."],
        ),
        (
            "Several studies (Smith 2020; Jones 2021) agree.",
            ["Several studies (Smith 2020; Jones 2021) agree."],
        ),
        (
            "Look there (see chapter 12, for details) before continuing.",
            ["Look there (see chapter 12, for details) before continuing."],
        ),
        (
            "Use the listed entries (12, 14, 16) in order.",
            ["Use the listed entries (12, 14, 16) in order."],
        ),
        (
            "Use the listed entries (iv, vi, ix) in order.",
            ["Use the listed entries (iv, vi, ix) in order."],
        ),
    ],
)
def test_sentence_to_phrases_parenthetical_policy(
    source: str,
    expected: list[str],
) -> None:
    result = PhraseSegmenter.sentence_string_to_phrase_strings(source, "en")

    assert result == expected
    assert "".join(result) == source


def test_non_english_parenthetical_does_not_apply_english_citation_detection() -> None:
    source = "Die Behauptung (Smith et al., 2020) bleibt umstritten."

    result = PhraseSegmenter.sentence_string_to_phrase_strings(source, "de")

    assert result == [
        "Die Behauptung ",
        "(Smith et al., ",
        "2020) ",
        "bleibt umstritten.",
    ]
    assert "".join(result) == source
