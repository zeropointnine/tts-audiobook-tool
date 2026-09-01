import pytest

from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter


def test_text_to_phrases_downgrades_immediate_consecutive_sections() -> None:
    text = "Chapter 1\n\n\nThe Beginning\n\n\nProse starts here."

    result = PhraseSegmenter.text_to_phrases(text, 40, "en")

    assert result == [
        Phrase("Chapter 1\n\n\n", Reason.SPACE_BREAK),
        Phrase("The Beginning\n\n", Reason.PARAGRAPH),
        Phrase("Prose starts here.", Reason.SENTENCE),
    ]


def test_text_to_phrases_keeps_non_consecutive_sections() -> None:
    text = "Chapter 1\n\n\nProse starts here.\n\nNext section\n\n\nMore prose."

    result = PhraseSegmenter.text_to_phrases(text, 40, "en")

    assert result == [
        Phrase("Chapter 1\n\n\n", Reason.SPACE_BREAK),
        Phrase("Prose starts here.\n\n", Reason.PARAGRAPH),
        Phrase("Next section\n\n\n", Reason.SPACE_BREAK),
        Phrase("More prose.", Reason.SENTENCE),
    ]


def test_text_to_phrases_promotes_merged_ornament_to_space_break() -> None:
    text = (
        "Take good care of Traitre in my absence.\n\n◇\n\n"
        "We discussed a few other topics."
    )

    result = PhraseSegmenter.text_to_phrases(text, 40, "en")

    assert result == [
        Phrase(
            "Take good care of Traitre in my absence.\n\n◇\n\n",
            Reason.SPACE_BREAK,
        ),
        Phrase("We discussed a few other topics.", Reason.SENTENCE),
    ]


def test_text_to_phrases_smoke_tricky_inputs() -> None:
    # Robustness smoke test: ornamental separators, dangling punctuation, and
    # whitespace-only lines must segment without crashing or emitting blank
    # phrases. (Expected boundaries here are pysbd-version sensitive, so this
    # asserts shape, not exact output; the dedicated tests above pin behavior.)
    result = PhraseSegmenter.text_to_phrases(
        "And you can . . .\nWell?", 40, "en"
    )
    assert len(result) >= 1
    assert all(phrase.text.strip() for phrase in result)

    result = PhraseSegmenter.text_to_phrases(SAMPLE1, 40, "en")

    assert len(result) >= 1
    assert all(phrase.text.strip() for phrase in result)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Hello", ["Hello"]),
        ("An item. Another item. Third",
         ["An item. ", "Another item. ", "Third"]),
        ("A paragraph\nAnother paragraph\nThird item",
         ["A paragraph\n", "Another paragraph\n", "Third item"]),
        ("Simple example. Hello. Ends with ellipsis... Item",
         ["Simple example. ", "Hello. ", "Ends with ellipsis... ", "Item"]),
        ("Dangling word test ... Hello.",
         ["Dangling word test ... ", "Hello."]),
        ("Non-vocalizable word after paragraph.\n* * *\nHello.",
         ["Non-vocalizable word after paragraph.\n* * *\n", "Hello."]),
        ("Non-vocalizable word after paragraph ...\n* * *\nHello.",
         ["Non-vocalizable word after paragraph ...\n* * *\n", "Hello."]),
    ],
)
def test_string_to_sentence_strings(source: str, expected: list[str]) -> None:
    assert PhraseSegmenter.string_to_sentence_strings(source, "en") == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("If this, then that.", ["If this, ", "then that."]),
        ("The reason: Because", ["The reason: ", "Because"]),
        ("The reason: Because... \n", ["The reason: ", "Because... \n"]),
        ("They liked it (but I didn't).",
         ["They liked it ", "(but I didn't)."]),
        ("They liked it; I didn't", ["They liked it; ", "I didn't"]),
        ('I was like, "Yo"', ['I was like, ', '"Yo"']),
        ('"Alright then," she said.', ['"Alright then," ', "she said."]),
        ("“Alright then,” she said.", ["“Alright then,” ", "she said."]),
        ("Is it Steins;Gate or Re:Zero?", ["Is it Steins;Gate or Re:Zero?"]),
        ("Malformed,,,:::;;; text", ["Malformed,,,:::;;; ", "text"]),
        ("Malformed,,,:::;;;text", ["Malformed,,,:::;;;text"]),
    ],
)
def test_sentence_to_phrases(source: str, expected: list[str]) -> None:
    assert PhraseSegmenter.sentence_string_to_phrase_strings(source) == expected


SAMPLE1 = """
Paragraph followed by multiple empty lines.


Paragraph followed by multiple whitespace-only lines.
  
  
Paragraph followed by empty lines plus ornamental separator line a.

•••

Paragraph followed by empty lines plus ornamental separator line b.

◆ ◆ ◆

Paragraph followed by single ornamental separator line.
~
End

Phrase, followed by multiple ornamental separator lines, sanity check
============
◆◆◆
End
"""