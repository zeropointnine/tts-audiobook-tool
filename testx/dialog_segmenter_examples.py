"""
Manual examples for the dialog-segmentation second pass.

Run from the repository root:
    python testx/dialog_segmenter_examples.py

Most examples deliberately begin as one pre-split group so this script isolates
dialog segmentation from the normal sentence/phrase/max-length rules.
"""

from dataclasses import dataclass
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.dialog_segmenter import (
    DialogSegmenter,
)


@dataclass(frozen=True)
class Example:
    name: str
    initial_segments: list[str]


EXAMPLES = [
    Example(
        "Dialog followed by attribution",
        ['"Hello," the man said.'],
    ),
    Example(
        "Narration followed by dialog",
        ['I told them, "What is up?"'],
    ),
    Example(
        "Free-standing quoted paragraph",
        [
            '"This is me talking in a free-standing paragraph. '
            'This quote may span more than one phrase group, but yea."'
        ],
    ),
    Example(
        "Curly quotes",
        ["He said “Hello,” then left."],
    ),
    Example(
        "Punctuation outside the closing quote",
        ['Did he say "Hello"? Then he shrugged.'],
    ),
    Example(
        "Leading dialog punctuation",
        ['He paused. "...Well, yes." Then he left.'],
    ),
    Example(
        "Lowercase inline quote is ignored",
        ['That was "too much" for me.'],
    ),
    Example(
        "Long lowercase dialog is accepted",
        ['He called it “a phrase with several ordinary words” yesterday.'],
    ),
    Example(
        "Short lowercase dialog after comma",
        ['She said, “yes.”'],
    ),
    Example(
        "Short lowercase dialog at paragraph start",
        ['“yes,” she replied.'],
    ),
    Example(
        "Short lowercase dialog with exclamation mark",
        ['He shouted “go!”'],
    ),
    Example(
        "Short lowercase dialog beginning with dialog dash",
        ['(aside) “—yes...” Then he stopped.'],
    ),
    Example(
        "Other short lowercase inline quote is ignored",
        ['He chose “red” over blue.'],
    ),
    Example(
        "Short lowercase inline quote with period is ignored",
        ['The answer was “yes”.'],
    ),
    Example(
        "Lowercase dialog spans existing segments",
        [
            "thought Alice to herself, “after such a fall as this, ",
            "I shall think nothing of tumbling down stairs! ",
            "Why, I wouldn’t say anything about it, even if I fell ",
            "off the top of the house!” (Which was very likely true.)",
        ],
    ),
    Example(
        "Punctuation-only quote is ignored",
        ['I was all like: "?!!" Then I left.'],
    ),
    Example(
        "Unmatched quote is ignored; later pair still works",
        ['He typed " without closing, then "Hello." Afterwards.'],
    ),
    Example(
        "Quotes do not pair across paragraphs",
        ['Before "Hello.\nAfter." End.'],
    ),
    Example(
        "Multiple dialog passages",
        ['He said "Hello." Then she said “Goodbye.” Finally.'],
    ),
    Example(
        "Dialog spans existing segments without recombination",
        [
            'Before "This spans ',
            "several existing ",
            'segments." After.',
        ],
    ),
]


def make_groups(segments: list[str]) -> list[PhraseGroup]:
    return [
        PhraseGroup([Phrase(segment, Reason.SENTENCE)])
        for segment in segments
    ]


def print_segments(label: str, segments: list[str]) -> None:
    print(f"  {label} ({len(segments)}):")
    for index, segment in enumerate(segments, start=1):
        print(f"    {index:>2}. {segment!r}")


def main() -> None:
    for number, example in enumerate(EXAMPLES, start=1):
        groups = make_groups(example.initial_segments)
        result = DialogSegmenter.segment_groups(groups)

        print(f"\n{number:>2}. {example.name}")
        print_segments("before", example.initial_segments)
        print_segments("after", [group.text for group in result])


if __name__ == "__main__":
    main()
