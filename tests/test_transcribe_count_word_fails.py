import unittest

from tts_audiobook_tool.validate_util import ValidateUtil

class TestTranscribeGranular(unittest.TestCase):

    def test_granular(self):

        items = [
            (
                "happy path and all dictionary words",
                "happy path and all dictionary words",
                0
            ),
            (
                "an example where transcript is missing words",
                "an example",
                5
            ),
            (
                "an example",
                "an example where transcript hallucinated words on the end",
                7
            ),
            (
                "wildcard mechanics test asadfwxx yes",
                "wildcard mechanics test blergh yes",
                0
            ),
            (
                "wildcard mechanics two words test angroo hello",
                "wildcard mechanics two words test anne grew hello",
                0            
            ),
            (
                "source of truth",
                "no match what so ever",
                5
            )
        ]

        for a, b, answer in items:
            failure_codes = ValidateUtil.get_word_errors(a, b, language_code="en", verbose=True)
            num_fail_words = len(failure_codes)
            self.assertTrue(num_fail_words == answer)


if __name__ == '__main__':
    unittest.main()
