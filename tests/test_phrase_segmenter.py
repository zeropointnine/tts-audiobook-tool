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
            ),
            (
                "Dangling word test ... Hello.",
                ["Dangling word test ... ", "Hello."]
            ),
            (
                "Non-vocalizable word after paragraph.\n* * *\nHello.",
                ["Non-vocalizable word after paragraph.\n* * *\n", "Hello."]
            ),
            (
                "Non-vocalizable word after paragraph ...\n* * *\nHello.",
                ["Non-vocalizable word after paragraph ...\n* * *\n", "Hello."]
            ),
        ]

        for inp, answer in items:
          result = PhraseSegmenter.string_to_sentence_strings(inp, "en")
          print()
          print("input:", repr(inp))
          print("result:", result)
          print("answer:", answer)
          self.assertTrue(result == answer)

    def test_sentence_to_phrases(self):
       print()

       items = [
          (
              "If this, then that.", 
             ["If this, ", "then that."]
          ),
          (
              "The reason: Because", 
             ["The reason: ", "Because"]
          ),
          (
              "The reason: Because... \n", 
             ["The reason: ", "Because... \n"]
          ),
          (
              "They liked it (but I didn't).", 
             ["They liked it ", "(but I didn't)."] 
          ),
          (
              "They liked it; I didn't", 
             ["They liked it; ", "I didn't"] 
          ),
          (
             "I was like, \"Yo\"", 
             ["I was like, ", "\"Yo\""] 
          ),          
          (
              "\"Alright then,\" she said.", 
             ["\"Alright then,\" ", "she said."] 
          ),          
          (
              "“Alright then,” she said.", 
             ["“Alright then,” ", "she said."] 
          ),          
          (
              "Is it Steins;Gate or Re:Zero?", 
             ["Is it Steins;Gate or Re:Zero?"]
          ),
          (
              "Malformed,,,:::;;; text",
             ["Malformed,,,:::;;; ",  "text"]
          ),
          (
              "Malformed,,,:::;;;text",
             ["Malformed,,,:::;;;text"]
          )
       ]

       for inp, answer in items:
          result = PhraseSegmenter.sentence_string_to_phrase_strings(inp)
          print()
          print("input:", repr(inp))
          print("result:", result)
          print("answer:", answer)
          self.assertTrue(result == answer)
          

if __name__ == '__main__':
    unittest.main()
