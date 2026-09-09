import operator
import unittest

from tts_audiobook_tool.text_ops.word_equivalence import WordEquivalence
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.text_ops.whitelist import Whitelist


class TestWordEquivalence(unittest.TestCase):

    def setUp(self):
        Whitelist().set_language_code("en")

    def test_is_equivalent_two_way(self):
        self.assertTrue(WordEquivalence.is_equivalent("all right", "alright", "en"))
        self.assertTrue(WordEquivalence.is_equivalent("alright", "all right", "en"))
        self.assertTrue(WordEquivalence.is_equivalent("Cannot", "can not", "en"))
        self.assertFalse(WordEquivalence.is_equivalent("all right", "alright", "es"))
        self.assertFalse(WordEquivalence.is_equivalent("all right", "wrong", "en"))
        # Same variant maps to the same group (identity); harmless since
        # direct matches take precedence in the alignment DP
        self.assertTrue(WordEquivalence.is_equivalent("alright", "alright", "en"))
        self.assertFalse(WordEquivalence.is_equivalent("everything", "everything", "en"))

    def test_max_phrase_length(self):
        self.assertEqual(WordEquivalence.max_phrase_length("en"), 2)
        self.assertEqual(WordEquivalence.max_phrase_length("es"), 0)

    def test_language_code_is_normalized(self):
        self.assertTrue(WordEquivalence.supports_language("EN-US"))
        self.assertEqual(WordEquivalence.max_phrase_length(" en-US "), 2)
        self.assertEqual(
            Validator.get_word_errors("all right", "alright", language_code="en-US"),
            [],
        )

    def test_lookup_is_read_only(self):
        lookup = WordEquivalence.get_lookup("en")
        with self.assertRaises(TypeError):
            operator.setitem(lookup, "all right", frozenset({"corrupted"}))
        self.assertTrue(WordEquivalence.is_equivalent("all right", "alright", "en"))

    def test_equivalent_phrase(self):
        result = WordEquivalence.equivalent_phrase(
            ["it", "was", "alright"], 2,
            ["it", "was", "all", "right"], 2,
            "en",
        )
        self.assertEqual(result, (1, 2))

        result = WordEquivalence.equivalent_phrase(
            ["all", "right", "then"], 0,
            ["alright", "then"], 0,
            "en",
        )
        self.assertEqual(result, (2, 1))

        self.assertIsNone(WordEquivalence.equivalent_phrase(
            ["alright"], 0, ["wrong"], 0, "en"
        ))

    def test_phrase_equivalence_in_word_errors(self):
        # "alright" (source) transcribed as "all right" should be a full match
        failure_codes = Validator.get_word_errors(
            "everything is alright now",
            "everything is all right now",
            language_code="en",
        )
        self.assertEqual(failure_codes, [])

        # And the reverse direction
        failure_codes = Validator.get_word_errors(
            "everything is all right now",
            "everything is alright now",
            language_code="en",
        )
        self.assertEqual(failure_codes, [])

    def test_phrase_equivalence_alignment_action(self):
        alignment = Validator.get_word_error_alignment(
            "everything is alright now",
            "everything is all right now",
            language_code="en",
        )
        self.assertEqual(
            [(item.action, item.source_text, item.transcript_text) for item in alignment],
            [
                ("match_direct", "everything", "everything"),
                ("match_direct", "is", "is"),
                ("match_equivalent", "alright", "all right"),
                ("match_direct", "now", "now"),
            ]
        )

    def test_equivalence_precedes_uncommon_word_pass(self):
        # "gimme" is an uncommon word spanning 2 transcript words, so without
        # equivalence data this would align via the uncommon_pass_2 free pass.
        alignment = Validator.get_word_error_alignment(
            "gimme",
            "give me",
            language_code="en",
        )
        self.assertEqual(
            [(item.action, item.source_text, item.transcript_text) for item in alignment],
            [("match_equivalent", "gimme", "give me")],
        )

    def test_no_false_positives(self):
        # Similar-looking but non-equivalent phrases still count as errors.
        # "all" vs "alright" is a substitution and "wrong" a deletion.
        failure_codes = Validator.get_word_errors(
            "everything is all wrong now",
            "everything is alright now",
            language_code="en",
        )
        self.assertEqual(len(failure_codes), 2)


if __name__ == '__main__':
    unittest.main()
