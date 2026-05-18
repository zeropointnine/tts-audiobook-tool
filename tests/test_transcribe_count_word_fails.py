import unittest

from tts_audiobook_tool.segment_stt_info_util import SegmentSttInfoUtil
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.text_ops.whitelist import Whitelist
from tts_audiobook_tool.segment_stt_info import SegmentSttInfo
from tts_audiobook_tool.util import strip_ansi_codes

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

    def test_word_error_alignment(self):
        alignment = ValidateUtil.get_word_error_alignment(
            "one two three",
            "one blah bonus",
            language_code="en",
        )

        self.assertEqual(
            [(item.action, item.source_text, item.transcript_text) for item in alignment],
            [
                ("match_direct", "one", "one"),
                ("mismatch_sub", "two", "blah"),
                ("mismatch_sub", "three", "bonus"),
            ]
        )

    def test_word_error_visualization(self):
        info = SegmentSttInfo(
            version=SegmentSttInfoUtil.VERSION,
            type=SegmentSttInfoUtil.TYPE,
            language_code="en",
            index_1b=1,
            source="one two three",
            prompt="one two three",
            transcript="one blah three extra",
            normalized_source="one two three",
            normalized_transcript="one blah three extra",
            generation_word_error_count=3,
            timed_phrases=[],
            transcript_words=[],
            exception=None,
        )

        visualization = strip_ansi_codes(SegmentSttInfoUtil.make_word_error_visualization(info))

        self.assertIn("one", visualization)
        self.assertIn("[=/=: two/blah]", visualization)
        self.assertIn("[+: extra]", visualization)

    def test_word_error_visualization_missing(self):
        info = SegmentSttInfo(
            version=SegmentSttInfoUtil.VERSION,
            type=SegmentSttInfoUtil.TYPE,
            language_code="en",
            index_1b=1,
            source="one two three",
            prompt="one two three",
            transcript="one three",
            normalized_source="one two three",
            normalized_transcript="one three",
            generation_word_error_count=1,
            timed_phrases=[],
            transcript_words=[],
            exception=None,
        )

        visualization = strip_ansi_codes(SegmentSttInfoUtil.make_word_error_visualization(info))

        self.assertIn("one", visualization)
        self.assertIn("[x: two]", visualization)
        self.assertIn("three", visualization)


if __name__ == '__main__':
    unittest.main()
