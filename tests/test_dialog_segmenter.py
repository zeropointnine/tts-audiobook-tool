import unittest

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.text_ops.dialog_segmenter import (
    DialogSegmenter,
)


def make_group(text: str, voice_index: int = -1) -> PhraseGroup:
    return PhraseGroup(
        [Phrase(text, Reason.SENTENCE)],
        voice_index=voice_index,
    )


def segment_one(text: str) -> list[PhraseGroup]:
    return DialogSegmenter.segment_groups([make_group(text)])


class TestDialogSegmenter(unittest.TestCase):
    def test_splits_straight_quoted_dialog_from_attribution(self):
        groups = segment_one('"Hello," the man said.')

        self.assertEqual(
            [group.text for group in groups],
            ['"Hello," ', "the man said."],
        )

    def test_splits_narration_before_straight_quoted_dialog(self):
        groups = segment_one('I told them, "What is up?"')

        self.assertEqual(
            [group.text for group in groups],
            ["I told them, ", '"What is up?"'],
        )

    def test_supports_curly_double_quotes(self):
        groups = segment_one("He said “Hello,” then left.")

        self.assertEqual(
            [group.text for group in groups],
            ["He said ", "“Hello,” ", "then left."],
        )

    def test_supports_mixed_double_quote_glyphs(self):
        groups = segment_one('First “Hello." Then "Goodbye.” End.')

        self.assertEqual(
            [group.text for group in groups],
            ["First ", '“Hello." ', "Then ", '"Goodbye.” ', "End."],
        )

    def test_keeps_adjacent_punctuation_and_boundary_whitespace_with_quote(self):
        groups = segment_one('He asked, "Ready")?! Then he waited.')

        self.assertEqual(
            [group.text for group in groups],
            ["He asked, ", '"Ready")?! ', "Then he waited."],
        )

    def test_assigns_voice_to_multiple_supported_quotes_and_attached_punctuation(self):
        groups = DialogSegmenter.segment_groups(
            [make_group('He asked, "Ready")?! Then “Go.” Finally.')],
            dialog_voice_index=1,
        )

        self.assertEqual(
            [group.text for group in groups],
            ["He asked, ", '"Ready")?! ', "Then ", "“Go.” ", "Finally."],
        )
        self.assertEqual(
            [group.voice_index for group in groups],
            [-1, 1, -1, 1, -1],
        )

    def test_accepts_leading_dialog_punctuation_before_capital(self):
        groups = segment_one('He paused. "...Well, yes." Then he left.')

        self.assertEqual(
            [group.text for group in groups],
            ["He paused. ", '"...Well, yes." ', "Then he left."],
        )

    def test_ignores_short_lowercase_quote_without_dialog_signals(self):
        text = 'That was "too much" for me.'

        groups = segment_one(text)

        self.assertEqual([group.text for group in groups], [text])

    def test_accepts_lowercase_quote_longer_than_three_words(self):
        groups = segment_one(
            'He called it “a phrase with several ordinary words” yesterday.'
        )

        self.assertEqual(
            [group.text for group in groups],
            [
                "He called it ",
                "“a phrase with several ordinary words” ",
                "yesterday.",
            ],
        )

    def test_accepts_short_lowercase_quote_after_introductory_punctuation(self):
        cases = (
            ('She said, “yes.”', "She said, "),
            ('She answered: “yes.”', "She answered: "),
            ('She answered— “yes.”', "She answered— "),
            ('She answered– “yes.”', "She answered– "),
        )
        for text, narration in cases:
            with self.subTest(text=text):
                groups = segment_one(text)
                self.assertEqual(
                    [group.text for group in groups],
                    [narration, "“yes.”"],
                )

    def test_accepts_short_lowercase_quote_at_paragraph_start(self):
        groups = segment_one('“yes,” she replied.')

        self.assertEqual(
            [group.text for group in groups],
            ["“yes,” ", "she replied."],
        )

    def test_accepts_short_lowercase_quote_with_question_or_exclamation(self):
        groups = segment_one('He shouted “go!” Then asked “ready?”')

        self.assertEqual(
            [group.text for group in groups],
            ["He shouted ", "“go!” ", "Then asked ", "“ready?”"],
        )

    def test_accepts_short_lowercase_quote_beginning_with_dialog_dash(self):
        groups = segment_one('(aside) “—yes...” Then he stopped.')

        self.assertEqual(
            [group.text for group in groups],
            ["(aside) ", "“—yes...” ", "Then he stopped."],
        )

    def test_ignores_other_short_lowercase_inline_quotes(self):
        for text in (
            'He chose “red” over blue.',
            'The answer was “yes”.',
        ):
            with self.subTest(text=text):
                groups = segment_one(text)
                self.assertEqual([group.text for group in groups], [text])

    def test_ignores_quote_without_vocalizable_text(self):
        text = 'I was all like: "?!!" Then I left.'

        groups = segment_one(text)

        self.assertEqual([group.text for group in groups], [text])

    def test_short_numeric_quote_without_dialog_signal_is_ignored(self):
        text = 'The code was "123" that day.'

        groups = segment_one(text)

        self.assertEqual([group.text for group in groups], [text])

    def test_unmatched_quote_does_not_hide_later_well_formed_pair(self):
        text = 'He typed " without closing, then "Hello." Afterwards.'

        groups = segment_one(text)

        self.assertEqual(
            [group.text for group in groups],
            [
                'He typed " without closing, then ',
                '"Hello." ',
                "Afterwards.",
            ],
        )

    def test_does_not_pair_quotes_across_paragraph_boundary(self):
        text = 'Before "Hello.\nAfter." End.'

        groups = segment_one(text)

        self.assertEqual([group.text for group in groups], [text])

    def test_handles_multiple_independent_pairs_left_to_right(self):
        text = 'He said "Hello." Then she said “Goodbye.” Finally.'

        groups = segment_one(text)

        self.assertEqual(
            [group.text for group in groups],
            [
                "He said ",
                '"Hello." ',
                "Then she said ",
                "“Goodbye.” ",
                "Finally.",
            ],
        )

    def test_spans_existing_groups_without_combining_them(self):
        original_groups = [
            make_group('Before "This spans ', voice_index=4),
            make_group("several existing ", voice_index=5),
            make_group('segments." After.', voice_index=6),
        ]

        groups = DialogSegmenter.segment_groups(original_groups)

        self.assertEqual(
            [group.text for group in groups],
            [
                "Before ",
                '"This spans ',
                "several existing ",
                'segments." ',
                "After.",
            ],
        )
        self.assertEqual(
            [group.voice_index for group in groups],
            [4, 4, 5, 6, 6],
        )
        self.assertIs(groups[2], original_groups[1])

    def test_accepts_lowercase_dialog_spanning_existing_groups(self):
        original_groups = [
            make_group(
                "thought Alice to herself, “after such a fall as this, "
                "I shall think nothing of tumbling down stairs! "
            ),
            make_group(
                "Why, I wouldn’t say anything about it, even if I fell "
                "off the top of the house!” (Which was very likely true.)"
            ),
        ]

        groups = DialogSegmenter.segment_groups(
            original_groups,
            dialog_voice_index=1,
        )

        self.assertEqual(
            [group.text for group in groups],
            [
                "thought Alice to herself, ",
                "“after such a fall as this, I shall think nothing of "
                "tumbling down stairs! ",
                "Why, I wouldn’t say anything about it, even if I fell "
                "off the top of the house!” ",
                "(Which was very likely true.)",
            ],
        )
        self.assertEqual(
            [group.voice_index for group in groups],
            [-1, 1, 1, -1],
        )

    def test_assigns_voice_sample_two_to_dialog_fragments(self):
        original_group = make_group(
            'Before "Hello." After.',
            voice_index=4,
        )

        groups = DialogSegmenter.segment_groups(
            [original_group],
            dialog_voice_index=1,
        )

        self.assertEqual(
            [group.text for group in groups],
            ["Before ", '"Hello." ', "After."],
        )
        self.assertEqual([group.voice_index for group in groups], [4, 1, 4])
        self.assertEqual(original_group.voice_index, 4)

    def test_assigns_dialog_voice_to_wholly_quoted_existing_group(self):
        original_group = make_group('"Hello."', voice_index=4)

        groups = DialogSegmenter.segment_groups(
            [original_group],
            dialog_voice_index=1,
        )

        self.assertEqual([group.text for group in groups], ['"Hello."'])
        self.assertEqual([group.voice_index for group in groups], [1])
        self.assertEqual(original_group.voice_index, 4)
        self.assertIsNot(groups[0], original_group)

    def test_assigns_dialog_voice_across_existing_groups(self):
        original_groups = [
            make_group('Before "This spans ', voice_index=4),
            make_group("several existing ", voice_index=5),
            make_group('segments." After.', voice_index=6),
        ]

        groups = DialogSegmenter.segment_groups(
            original_groups,
            dialog_voice_index=1,
        )

        self.assertEqual(
            [group.voice_index for group in groups],
            [4, 1, 1, 1, 6],
        )
        self.assertEqual(
            [group.voice_index for group in original_groups],
            [4, 5, 6],
        )
        self.assertIsNot(groups[2], original_groups[1])

    def test_short_inline_quote_not_classified_as_dialog_keeps_existing_voice(self):
        original_group = make_group('That was "too much" for me.', voice_index=4)

        groups = DialogSegmenter.segment_groups(
            [original_group],
            dialog_voice_index=1,
        )

        self.assertEqual([group.voice_index for group in groups], [4])
        self.assertIs(groups[0], original_group)

    def test_preserves_all_source_characters(self):
        text = 'Before “Hello.”?!  After "Goodbye."\nNext paragraph.'

        groups = segment_one(text)

        self.assertEqual("".join(group.text for group in groups), text)

    def test_text_to_groups_option_runs_after_normal_grouping(self):
        text = 'I told them, "What is up?" Then I left.'

        without_option = PhraseGrouper.text_to_groups(
            text,
            max_words=100,
            strategy=SegmentationStrategy.MAX_LEN,
            pysbd_lang="en",
        )
        with_option = PhraseGrouper.text_to_groups(
            text,
            max_words=100,
            strategy=SegmentationStrategy.MAX_LEN,
            pysbd_lang="en",
            dialog_segmentation=True,
        )

        self.assertEqual([group.text for group in without_option], [text])
        self.assertEqual(
            [group.text for group in with_option],
            ["I told them, ", '"What is up?" ', "Then I left."],
        )
        self.assertEqual(
            [group.voice_index for group in without_option],
            [-1],
        )
        self.assertEqual(
            [group.voice_index for group in with_option],
            [-1, 1, -1],
        )


if __name__ == "__main__":
    unittest.main()
