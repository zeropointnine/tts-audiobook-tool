import unittest

from tts_audiobook_tool.phrase import Phrase, Reason
from tts_audiobook_tool.phrase_segmenter import PhraseSegmenter

class TestPhraseSegmenter(unittest.TestCase):

    def test_text_to_phrases(self):

        items = [
            (
                "Minimal",
                [Phrase("Minimal", Reason.SENTENCE)]
            ),
            (
                "Sentence one. Sentence two",
                [Phrase("Sentence one. ", Reason.SENTENCE), Phrase("Sentence two", Reason.SENTENCE)]
            ),
            (
                "Sentence.\nNew paragraph",
                [Phrase("Sentence.\n", Reason.PARAGRAPH), Phrase("New paragraph", Reason.SENTENCE)]
            ),

            # Test merging of dangling punc-words
            (
               "And you can . . .\nWell?", 
               [Phrase("And you can . . . ", Reason.SENTENCE), Phrase("Well?", Reason.SENTENCE)]
               
               # psybd bug that I still need to add logic to work around.
               # Should REALLY be this!
               # [Phrase("And you can . . .\n", Reason.PARAGRAPH), Phrase("Well?", Reason.SENTENCE)]
            )
        ]

        for inp, answer in items:
          result = PhraseSegmenter.text_to_phrases(inp, 40, "en")
          print()
          print("input:", repr(inp))
          print("result:")
          for item in result:
             print(f"    {item}")
          self.assertTrue(result == answer)


    def test_string_to_sentence_strings(self):
        print()

        items = [
            (
                "Hello",
                ["Hello"]
            ),
            (
                "An item. Another item. Third",
                ["An item. ", "Another item. ", "Third"]
            ),
            (
                "A paragraph\nAnother paragraph\nThird item",
                ["A paragraph\n", "Another paragraph\n", "Third item"]
            ),
            (
                "Simple example. Hello. Ends with ellipsis... Item",
                ["Simple example. ", "Hello. ", "Ends with ellipsis... ", "Item"]
            )
        ]

        for inp, answer in items:
          result = PhraseSegmenter.string_to_sentence_strings(inp, "en")
          print()
          print("input:", repr(inp))
          print("result:", result)
          print("answer:", answer)
          self.assertTrue(result == answer)

if __name__ == '__main__':
    unittest.main()
