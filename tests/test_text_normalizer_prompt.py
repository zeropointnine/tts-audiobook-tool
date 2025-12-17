import re
import unittest

from tts_audiobook_tool.text_normalizer import TextNormalizer

class TestTextNormalizerPrompt(unittest.TestCase):

    def test_normalize_prompt_en(self):
        
        print()
        items = [
            (
                "Normally formatted sentence. Sanity check.", 
                "Normally formatted sentence. Sanity check."
            ),
            (
                "  Trim ends of    string and  normalize extra white space  ", 
                "Trim ends of string and normalize extra white space"
            ),
            
            # ellipsis
            (
                "Ellipsis character… So.", 
                "Ellipsis character… So."
            ),
            (
                "Ellipsis character with space before … So.", 
                "Ellipsis character with space before … So."
            ),
            (
                "Triple-dot... So.", 
                "Triple-dot... So."
            ),
            (
                "Triple-dot with space before ... So.", 
                "Triple-dot with space before ... So."
            ),
            (
                "Alternate format for ellipsis, not uncommon . . . Ugh.", 
                "Alternate format for ellipsis, not uncommon ... Ugh."
            ),
            (
                "Fancy apost: I think it’s . . . unseemly.", 
                "Fancy apost: I think it’s ... unseemly." # remains unchanged atm
            ),
            (
                "Multiple dots...........", 
                "Multiple dots..."
            ),
            (
                "Multiple dots with space ...........", 
                "Multiple dots with space ..."
            ),
            (
                "Multiple ellipsis characters…………", 
                "Multiple ellipsis characters…"
            ),
            (
                "Multiple ellipsis characters with space …………", 
                "Multiple ellipsis characters with space …"
            ),
            (
                "This is a pause...... and another one…… and a mixed one…...", 
                "This is a pause... and another one… and a mixed one…..." # weird, but leave it for now
            ),

            # dashes
            ("Normal dash usage: Higgs-Boson", "Normal dash usage: Higgs-Boson"),
            ("Em-dash\u2014okay?", "Em-dash\u2014okay?"),
            ("Em-dash bounded by spaces \u2014 okay?", "Em-dash bounded by spaces \u2014 okay?"),
            ("En-dash\u2014here", "En-dash\u2014here"),
            ("En-dash bounded by spaces \u2014 here", "En-dash bounded by spaces \u2014 here"),
            ("Dash bounded by spaces - here", "Dash bounded by spaces - here"),

            ("Garbage punctuation: ,.,.,---... ", "Garbage punctuation:"),
            ("Garbage punctuation in two consecutive 'words': ,.,.,-- -... ", "Garbage punctuation in two consecutive 'words':"),
        ]
        for source, answer in items:
            result = TextNormalizer.normalize_prompt_common(source, language_code="en")
            print("source:", source)
            print("result:", result)
            print("answer:", answer)
            print()
            self.assertTrue(result == answer)

if __name__ == '__main__':
    unittest.main()
