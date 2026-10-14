import unittest

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper


class TestPhraseGrouper(unittest.TestCase):

    def test_sentence_strategy_keeps_short_sentences_separate(self):
        text = "Hi. Go. This is longer."

        groups = PhraseGrouper.text_to_groups(text, 20, SegmentationStrategy.SENTENCE, "en")

        self.assertEqual([group.text for group in groups], ["Hi. ", "Go. ", "This is longer."])

    def test_sentence_plus_strategy_merges_short_sentences(self):
        text = "Hi. Go. This is longer."

        groups = PhraseGrouper.text_to_groups(text, 20, SegmentationStrategy.SENTENCE_PLUS, "en")

        self.assertEqual([group.text for group in groups], ["Hi. Go. ", "This is longer."])

    def test_multi_sentence_strategy_combines_sentences_up_to_max_words(self):
        text = "One two. Three four. Five six."

        groups = PhraseGrouper.text_to_groups(text, 4, SegmentationStrategy.MULTI_SENTENCE, "en")

        self.assertEqual([group.text for group in groups], ["One two. Three four. ", "Five six."])

    def test_max_len_strategy_respects_paragraph_boundaries(self):
        text = "One two. Three four.\nNext part here."

        groups = PhraseGrouper.text_to_groups(text, 20, SegmentationStrategy.MAX_LEN, "en")

        self.assertEqual([group.text for group in groups], ["One two. Three four.\n", "Next part here."])

    def test_oversized_first_phrase_does_not_create_empty_group(self):
        oversized_phrase = Phrase("one two three four five", Reason.SENTENCE)

        groups = PhraseGrouper.group_to_groups_by_max_words(
            PhraseGroup([oversized_phrase]),
            max_words=4,
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].phrases, [oversized_phrase])
        self.assertEqual(groups[0].text, oversized_phrase.text)

    def test_text_ingestion_normalizes_line_endings_and_trailing_whitespace(self):
        text = (
            "One. \n \t\n"
            "Two. \r\n\t \r\n"
            "Three. \r  \r"
            "Four."
        )

        groups = PhraseGrouper.text_to_groups(
            text,
            20,
            SegmentationStrategy.SENTENCE,
            "en",
        )

        self.assertEqual(
            "".join(group.text for group in groups),
            "One.\n\nTwo.\n\nThree.\n\nFour.",
        )

    def test_phrase_quote_end_reason_orders_between_word_and_phrase(self):
        self.assertEqual(Reason.PHRASE_QUOTE_END.json_value, "isqe")
        self.assertIs(Reason.from_json_value("isqe"), Reason.PHRASE_QUOTE_END)
        self.assertLess(Reason.WORD, Reason.PHRASE_QUOTE_END)
        self.assertLess(Reason.PHRASE_QUOTE_END, Reason.PHRASE)
        self.assertLess(Reason.PHRASE_QUOTE_END, Reason.SENTENCE)

    def test_sentence_and_dialog_passes_split_multi_sentence_attribution(self):
        text = '"One? Two!" she said.'

        groups = PhraseGrouper.text_to_groups(
            text,
            40,
            SegmentationStrategy.SENTENCE,
            "en",
            dialog_segmentation=True,
        )

        self.assertEqual(
            [group.text for group in groups],
            ['"One? ', 'Two!" ', 'she said.'],
        )
        self.assertEqual([group.voice_index for group in groups], [1, 1, -1])
        self.assertEqual(
            [group.last_reason for group in groups],
            [Reason.SENTENCE, Reason.PHRASE_QUOTE_END, Reason.SENTENCE],
        )

    def test_dialog_pass_voices_each_embedded_quoted_sentence(self):
        text = 'He said, "One. Two." Then left.'

        groups = PhraseGrouper.text_to_groups(
            text,
            40,
            SegmentationStrategy.SENTENCE,
            "en",
            dialog_segmentation=True,
        )

        self.assertEqual(
            [group.text for group in groups],
            ['He said, ', '"One. ', 'Two." ', 'Then left.'],
        )
        self.assertEqual([group.voice_index for group in groups], [-1, 1, 1, -1])
        self.assertEqual("".join(group.text for group in groups), text)

    def test_dialog_segmentation_marks_quote_end_pieces_with_phrase_quote_end_reason(self):
        text = 'He muttered, "Never mind," she answered quietly.'

        groups = PhraseGrouper.text_to_groups(
            text,
            40,
            SegmentationStrategy.SENTENCE_PLUS,
            "en",
            dialog_segmentation=True,
        )

        self.assertEqual(
            [group.text for group in groups],
            ["He muttered, ", '"Never mind," ', "she answered quietly."],
        )
        self.assertEqual(
            [[phrase.reason for phrase in group.phrases] for group in groups],
            [[Reason.PHRASE], [Reason.PHRASE_QUOTE_END], [Reason.SENTENCE]],
        )
        self.assertEqual([group.voice_index for group in groups], [-1, 1, -1])

    def test_trailing_ornamental_line_merges_into_last_group(self):
        # A trailing ornament has no trailing line breaks, so its reason stays
        # SENTENCE and the segmenter's backward merge skips it. The grouper
        # must still fold the resulting ornament-only group backward.
        text = "Prose here we go.\n\nMore prose follows.\n\n✦"

        groups = PhraseGrouper.text_to_groups(
            text,
            40,
            SegmentationStrategy.SENTENCE_PLUS,
            "en",
        )

        self.assertEqual(
            [group.text for group in groups],
            ["Prose here we go.\n\n", "More prose follows.\n\n✦"],
        )
        self.assertEqual(groups[-1].last_reason, Reason.SPACE_BREAK)

    def test_ornamental_group_reisolated_by_dialog_split_merges_backward(self):
        # The ornament backward-merges into the quote's phrase, then the
        # dialog pass splits that phrase at the quote's span end, isolating
        # an ornament-only group mid-text. It must fold back into the quote
        # group.
        text = "He spoke at length.\n\n“First quote.”\n\n✦\n\nMore text follows."

        groups = PhraseGrouper.text_to_groups(
            text,
            40,
            SegmentationStrategy.SENTENCE_PLUS,
            "en",
            dialog_segmentation=True,
        )

        self.assertEqual(
            [group.text for group in groups],
            ["He spoke at length.\n\n", "“First quote.”\n\n✦\n\n", "More text follows."],
        )
        self.assertEqual(groups[1].last_reason, Reason.SPACE_BREAK)
        self.assertEqual("".join(group.text for group in groups), text)

    def test_leading_ornamental_group_merges_forward(self):
        # With dialog segmentation, the dialog pass can re-split the leading
        # ornament (previously merged into the first content phrase) back out
        # at an opening quote. It must re-attach forward.
        text = "✦\n\n“Hello,” she said.\n\nMore prose follows here."

        groups = PhraseGrouper.text_to_groups(
            text,
            40,
            SegmentationStrategy.SENTENCE_PLUS,
            "en",
            dialog_segmentation=True,
        )

        self.assertEqual(
            [group.text for group in groups],
            ["✦\n\n“Hello,” ", "she said.\n\n", "More prose follows here."],
        )
        self.assertEqual([group.voice_index for group in groups], [1, -1, -1])
        self.assertEqual("".join(group.text for group in groups), text)

    def test_merge_ornamental_groups_keeps_groups_when_no_content_exists(self):
        groups = [
            PhraseGroup([Phrase("✦\n\n", Reason.PARAGRAPH)]),
            PhraseGroup([Phrase("◆ ◆ ◆\n\n", Reason.PARAGRAPH)]),
        ]

        result = PhraseGrouper.merge_ornamental_groups(groups)

        self.assertEqual([group.text for group in result], [group.text for group in groups])


if __name__ == '__main__':
    unittest.main()