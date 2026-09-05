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


def test_text_to_phrases_merges_leading_ornament_into_first_content_phrase() -> None:
    # Regression: a leading ornamental line used to survive as its own
    # ornament-only phrase (and therefore an ornament-only first PhraseGroup
    # of a section). It must ride with the first content phrase instead.
    text = "✦\n\nThe chapter begins in earnest."

    result = PhraseSegmenter.text_to_phrases(text, 40, "en")

    assert result == [
        Phrase("✦\n\nThe chapter begins in earnest.", Reason.SENTENCE),
    ]


def test_merge_ornamental_lines_prepends_leading_ornaments_to_first_content_phrase() -> None:
    phrases = [
        Phrase("◇\n\n", Reason.PARAGRAPH),
        Phrase("* * *\n\n\n", Reason.SPACE_BREAK),
        Phrase("Content phrase.\n\n", Reason.PARAGRAPH),
    ]

    result = PhraseSegmenter.merge_ornamental_lines(phrases)

    assert result == [
        Phrase("◇\n\n* * *\n\n\nContent phrase.\n\n", Reason.PARAGRAPH),
    ]


def test_merge_ornamental_lines_keeps_phrases_when_no_content_exists() -> None:
    phrases = [
        Phrase("◇\n\n", Reason.PARAGRAPH),
        Phrase("◆ ◆ ◆\n\n", Reason.PARAGRAPH),
    ]

    result = PhraseSegmenter.merge_ornamental_lines(phrases)

    assert result == phrases


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
        ('"One. Two."', ['"One. ', 'Two."']),
        (
            'He said, "One. Two." Then left.',
            ['He said, "One. ', 'Two." ', 'Then left.'],
        ),
        (
            '"One? Two!" she said.',
            ['"One? ', 'Two!" she said.'],
        ),
        ('"One." "Two."', ['"One." ', '"Two."']),
        (
            '“Outer one. ‘Inner one. Inner two.’ Outer two.”',
            ['“Outer one. ', '‘Inner one. ', 'Inner two.’ ', 'Outer two.”'],
        ),
        ('«One. Two.» Next.', ['«One. ', 'Two.» ', 'Next.']),
        ('「One。 Two。」 Next.', ['「One。 ', 'Two。」 ', 'Next.']),
        ("'One. Two.' Next.", ["'One. ", "Two.' ", 'Next.']),
        ("It's Bob's book. Next.", ["It's Bob's book. ", 'Next.']),
        (
            'He typed " without closing, then "Hello." Afterwards.',
            ['He typed " without closing, then "Hello." ', 'Afterwards.'],
        ),
        ('  "One. Two."  ', ['  "One. ', 'Two."  ']),
        ('And you can . . .\nYes?', ['And you can . . .\n', 'Yes?']),
    ],
)
def test_quote_aware_sentence_segmentation_preserves_source(
    source: str,
    expected: list[str],
) -> None:
    result = PhraseSegmenter.string_to_sentence_strings(source, "en")

    assert result == expected
    assert "".join(result) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("If this, then that.", ["If this, ", "then that."]),
        ("The reason: Because", ["The reason: ", "Because"]),
        ("The reason: Because... \n", ["The reason: ", "Because... \n"]),
        ("They liked it; I didn't", ["They liked it; ", "I didn't"]),
        ('I was like, "Yo"', ['I was like, ', '"Yo"']),
        ('"Alright then," she said.', ['"Alright then," ', "she said."]),
        ("“Alright then,” she said.", ["“Alright then,” ", "she said."]),
        ("Is it Steins;Gate or Re:Zero?", ["Is it Steins;Gate or Re:Zero?"]),
        ("Malformed,,,:::;;; text", ["Malformed,,,:::;;; ", "text"]),
        ("Malformed,,,:::;;;text", ["Malformed,,,:::;;;text"]),
        ("", []),
        ("   ", ["   "]),
        ("Wait, ", ["Wait, "]),
        ("Alpha, ; Beta", ["Alpha, ; ", "Beta"]),
        ('He answered "yes" and left.', ['He answered "yes" ', "and left."]),
        ("He answered ”yes” and left.", ["He answered ”yes” ", "and left."]),
        ("First–second", ["First–", "second"]),
        ("First – second", ["First – ", "second"]),
        ("First—second", ["First—", "second"]),
        ("First — second", ["First — ", "second"]),
        # Double normal dash is a phrase break when bounded by
        # vocalizable ("content") characters, with optional
        # whitespace on either side of the dash pair
        ("Hello--what are you doing?", ["Hello--", "what are you doing?"]),
        ("Hello -- what are you doing?", ["Hello -- ", "what are you doing?"]),
        ("Hello-- what are you doing?", ["Hello-- ", "what are you doing?"]),
        ("Hello --what are you doing?", ["Hello --", "what are you doing?"]),
        ("Numbers 3--4 work", ["Numbers 3--", "4 work"]),
        # Not breaks: single dash, longer dash run, missing content bounds
        ("A well-known fact", ["A well-known fact"]),
        ("Wait---triple stays whole", ["Wait---triple stays whole"]),
        ("Ends with--", ["Ends with--"]),
        ("--Starts with", ["--Starts with"]),
        ("Punct...--;;bounded", ["Punct...--;;bounded"]),
    ],
)
def test_sentence_to_phrases(source: str, expected: list[str]) -> None:
    result = PhraseSegmenter.sentence_string_to_phrase_strings(source, "en")

    assert result == expected
    assert "".join(result) == source


def test_text_to_phrases_double_dash_break() -> None:
    result = PhraseSegmenter.text_to_phrases("Hello--what are you doing?", 40, "en")

    assert result == [
        Phrase("Hello--", Reason.PHRASE),
        Phrase("what are you doing?", Reason.SENTENCE),
    ]


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