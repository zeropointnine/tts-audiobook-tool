import unittest

from tts_audiobook_tool.text_util import TextUtil

class TestTextUtil(unittest.TestCase):

    def test_split_raw_word(self):
        
        items = [
            # No trailing or leading ws/punc
            ( "hello", ("", "hello", "") ), 
            ( "don't", ("", "don't", "") ), 

            # Simple whitespace
            ( "hello  ", ("", "hello", "  ") ), 
            ( "  hello", ("  ", "hello", "") ),

            # Fancy quote
            ( "“Hello”", ("“", "Hello", "”") ),
            ( "‘Hello’", ("‘", "Hello", "’") ),

            # Leading and/or trailing punctuation and whitespace
            ( "hello\", ... ?!", ("", "hello", "\", ... ?!") ),
            ( "\", said", ("\", ", "said", "") ),
            ( "\", and…", ("\", ", "and", "…") )
        ]
        
        for raw_word, answer in items:
            result = TextUtil.split_raw_word(raw_word)
            print("raw word:", repr(raw_word))
            print("result  :", repr(result))
            print("answer  :", repr(answer))
            print()
            self.assertEqual(result, answer)

if __name__ == '__main__':
    unittest.main()
