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


if __name__ == '__main__':
    unittest.main()