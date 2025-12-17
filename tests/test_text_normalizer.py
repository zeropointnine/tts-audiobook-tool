import re
import unittest

from tts_audiobook_tool.text_normalizer import TextNormalizer

class TestTextNormalizer(unittest.TestCase):

    def test_normalize_common(self):
        
        items = [
            (
                "Excess  white   space  ", 
                "excess white space"
            ),
            (
                "Single quote: Here's Johnny and 'single scare quote phrase'", 
                "single quote heres johnny and single scare quote phrase"
            ),
            (
                "Weird punctuation: . ... x.x...,!a", 
                "weird punctuation x x a"
            ),
            (
                "Underscore: filenames should start with \"test_\" or end with _test", 
                "underscore filenames should start with test or end with test"
            ),
            (
                "Dashes: dashed-word emdash——emdash ... –endash––endash–", 
                "dashes dashed word emdash emdash endash endash"
            ),
            (
                "“This is too much, my love!”", 
                "this is too much my love"
            ),
            (
                "Random emojis: 😉 in the Read😉Me😉? Well... 🙂‍↔️, why not?", 
                "random emojis in the readme well why not"
            ),
            (
                "Café au lait", 
                "café au lait"
            ),
            (
                "Регулярные выражения", # ensure non-roman characters don't get stripped
                "регулярные выражения" 
            ) 
        ]
        for source, answer in items:
            result = TextNormalizer.normalize_common(source)
            print("input :",source)
            print("result:", result)
            print("answer:", answer)
            print()
            self.assertEqual(result, answer)

    def test_normalize_common_numbers_en(self):
        """ 
        Numbers-related transformations, utilizing TextNormalizer.normalize_numbers_en()
        """
        
        items = [
            ("Hello 19", "hello 19"),
            ("hello nineteen", "hello 19"),
            ("ninety", "90"),
            ("twenty one", "21"),
            ("twenty-one", "21"),

            ("99% sure", "99 sure"),
            ("ninety-nine percent sure", "99 sure"),
            ("ninety nine percent sure", "99 sure"),

            # ("I have $2005", "I have $2005"),
            # ("I have two thousand and five dollars", "I have $2005"),
            # ("I have two thousand five dollars", "I have $2005"),

            # ("There were about twenty of us: five Ambassadors, a handful of Staff, and we two.", ""),
            # ("I’d spent thousands of hours in the immer. I’d been to ports on tens of countries on tens of worlds", ""),
            # ("It was the third sixteenth of September, a Dominday.", ""),
            

            # ("I have two thousand five dollars", "I have $2005"),
        ]
        for inp, answer in items:
            result = TextNormalizer.normalize_common(inp, language_code="en")
            print("input :",inp)
            print("result:", result)
            print("answer:", answer)
            print()
            self.assertTrue(result == answer) 

    def test_sounds_the_same(self):
        
        items = [
            ("color", "colour", True),
            ("analyze", "analyse", True),
            ("been", "bean", True),
            ("been", "bean", True),
            ("one", "Juan", False),
            ("also", "Oslo", False),
            ("apple", "orange", False),
            ("embassy town", "embassytown", True)
        ]
        for a, b, answer in items:
            result = TextNormalizer.sounds_the_same_en(a, b)
            self.assertEqual(result, answer)

    def test_normalize_spacing(self):
        items = [
            # STT breaks a compound word ("fire fly" -> "firefly")
            (
                "Look at that firefly glow.",
                "Look at that fire fly glow.",
                "Look at that firefly glow."
            ),
            # STT merges two words ("highschool" -> "high school")
            (
                "I went to high school yesterday.",
                "I went to highschool yesterday.",
                "I went to high school yesterday."
            ),
            # Non-space-related difference (should not be fixed)
            (
                "The quick brown fox.",
                "The quick fox.", 
                "The quick fox."
            )
        ]
        for source, transcript, answer in items:
          result = TextNormalizer.normalize_spacing(source=source, transcript=transcript)
          self.assertEqual(result, answer)


if __name__ == '__main__':
    unittest.main()
