import unittest

from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.whitelist import Whitelist

class TestTranscribeGranular(unittest.TestCase):

    def setUp(self):
        Whitelist().set_language_code("en")

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

    def test_spanish_supported_whitelist(self):
        Whitelist().set_language_code("es")

        self.assertTrue(Whitelist().has("hola"))
        self.assertFalse(Whitelist().has("qwertyasdfz"))

        failure_codes = ValidateUtil.get_word_errors(
            "hola qwertyasdfz",
            "hola algo distinto",
            language_code="es",
            verbose=False,
        )
        self.assertEqual(failure_codes, [])

    def test_unsupported_language_disables_whitelist_wildcards(self):
        Whitelist().set_language_code("fr")

        failure_codes = ValidateUtil.get_word_errors(
            "motinvente bonjour",
            "autrechose bonjour",
            language_code="fr",
            verbose=False,
        )
        self.assertEqual(len(failure_codes), 1)


if __name__ == '__main__':
    unittest.main()
