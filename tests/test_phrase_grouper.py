import unittest

from tts_audiobook_tool.app_types import SegmentationStrategy
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


if __name__ == '__main__':
    unittest.main()
