import operator
import unittest
from unittest.mock import patch

from tts_audiobook_tool import constants
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer
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
        # Identical indexed variants retain the query API's compatibility behavior;
        # direct matches take precedence in the alignment DP.
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


class TestExplicitWordEquivalence(unittest.TestCase):

    @staticmethod
    def make_equivalence(pairs=None):
        # Own all tables and cache: synthetic ambiguous entries must never leak
        # into the production English data or another test's lazy index.
        class SyntheticEquivalence(WordEquivalence):
            _EQUIVALENCE_PAIRS = {} if pairs is None else pairs
            _LOOKUP = {}

        return SyntheticEquivalence

    @staticmethod
    def alignment(source, transcript):
        return [
            (step.action, step.source_text, step.transcript_text)
            for step in Validator.get_word_error_alignment(source, transcript, "en")
        ]

    def setUp(self):
        Whitelist().set_language_code("en")

    def test_new_pairs_are_explicit_and_bidirectional(self):
        for expanded, contracted in [("would have", "wouldve"), ("did not", "didnt")]:
            with self.subTest(expanded=expanded):
                self.assertIn((expanded, contracted), WordEquivalence._EQUIVALENCE_PAIRS["en"])
                for source, transcript in [(expanded, contracted), (contracted, expanded)]:
                    self.assertTrue(WordEquivalence.is_equivalent(source, transcript, "en"))
                    self.assertIn(transcript, WordEquivalence.get_lookup("en")[source])
                    self.assertEqual(
                        WordEquivalence.equivalent_phrase(
                            source.split(), 0, transcript.split(), 0, "en"
                        ),
                        (len(source.split()), len(transcript.split())),
                    )

    def test_new_pairs_are_language_scoped_and_accept_english_variants(self):
        self.assertEqual(dict(WordEquivalence.get_lookup("fr")), {})
        self.assertEqual(WordEquivalence.max_phrase_length("fr"), 0)
        for expanded, contracted in [("would have", "wouldve"), ("did not", "didnt")]:
            for source, transcript in [(expanded, contracted), (contracted, expanded)]:
                with self.subTest(source=source, transcript=transcript):
                    self.assertFalse(WordEquivalence.is_equivalent(source, transcript, "fr"))
                    self.assertIsNone(WordEquivalence.equivalent_phrase(
                        source.split(), 0, transcript.split(), 0, "fr"
                    ))
                    french_path = Validator.get_word_error_alignment(source, transcript, "fr")
                    self.assertTrue(all(step.action in {
                        "mismatch_sub", "skip_source", "skip_transcript"
                    } for step in french_path))
                    self.assertEqual(len(Validator.get_word_errors(source, transcript, "fr")), 2)
                    for language_code in ["en-US", " EN-US "]:
                        self.assertTrue(WordEquivalence.is_equivalent(
                            source, transcript, language_code
                        ))
                        self.assertEqual([
                            (step.action, step.source_text, step.transcript_text)
                            for step in Validator.get_word_error_alignment(
                                source, transcript, language_code
                            )
                        ], [("match_equivalent", source, transcript)])
                        self.assertEqual(
                            Validator.get_word_errors(source, transcript, language_code), []
                        )

    def test_new_pairs_have_exact_equivalence_alignment_in_both_directions(self):
        for expanded, contracted in [("would have", "wouldve"), ("did not", "didnt")]:
            for source, transcript in [(expanded, contracted), (contracted, expanded)]:
                for contextual in [False, True]:
                    with self.subTest(source=source, transcript=transcript, contextual=contextual):
                        expected = [("match_equivalent", source, transcript)]
                        if contextual:
                            expected = [("match_direct", "we", "we")] + expected + [
                                ("match_direct", "stayed", "stayed")
                            ]
                            source_text = f"we {source} stayed"
                            transcript_text = f"we {transcript} stayed"
                        else:
                            source_text, transcript_text = source, transcript
                        self.assertEqual(self.alignment(source_text, transcript_text), expected)
                        self.assertEqual(
                            Validator.get_word_errors(source_text, transcript_text, "en"), []
                        )

    def test_raw_apostrophes_normalize_into_exact_equivalence_paths(self):
        for expanded, contracted, raw in [
            ("would have", "wouldve", "would've"),
            ("did not", "didnt", "didn't"),
            ("would have", "wouldve", "would’ve"),
            ("did not", "didnt", "didn’t"),
        ]:
            for reverse in [False, True]:
                with self.subTest(raw=raw, reverse=reverse):
                    source, transcript = (raw, expanded) if reverse else (expanded, raw)
                    source_phrase, transcript_phrase = (
                        (contracted, expanded) if reverse else (expanded, contracted)
                    )
                    normalized = TextNormalizer.normalize_source_and_transcript(
                        f"We {source} stayed.", f"We {transcript} stayed.", "en"
                    )
                    self.assertEqual(
                        normalized,
                        (f"we {source_phrase} stayed", f"we {transcript_phrase} stayed"),
                    )
                    self.assertEqual(self.alignment(*normalized), [
                        ("match_direct", "we", "we"),
                        ("match_equivalent", source_phrase, transcript_phrase),
                        ("match_direct", "stayed", "stayed"),
                    ])
                    self.assertEqual(Validator.get_word_errors(*normalized, "en"), [])

    def test_overlapping_pairs_merge_only_direct_neighbors(self):
        equivalence = self.make_equivalence(pairs={"en": [
            ("hed", "he had"),
            ("hed", "he would"),
            ("hed", "he had"),  # Duplicate edge.
            ("alpha", "beta"),
            ("alpha", "gamma"),
            ("beta", "gamma"),
            ("alpha", "delta"),
            ("beta", "delta"),
            ("delta", "beta"),  # Duplicate reversed edge.
        ]})
        expected = {
            "hed": frozenset({"he had", "he would"}),
            "he had": frozenset({"hed"}),
            "he would": frozenset({"hed"}),
            "alpha": frozenset({"beta", "gamma", "delta"}),
            "beta": frozenset({"alpha", "gamma", "delta"}),
            "gamma": frozenset({"alpha", "beta"}),
            "delta": frozenset({"alpha", "beta"}),
        }
        lookup = equivalence.get_lookup(" EN-US ")
        self.assertEqual(dict(lookup), expected)
        self.assertIs(lookup, equivalence.get_lookup("en"))
        for source, neighbors in expected.items():
            self.assertIsInstance(lookup[source], frozenset)
            self.assertNotIn(source, lookup[source])
            for transcript in expected:
                self.assertEqual(
                    equivalence.is_equivalent(source, transcript, "en"),
                    source == transcript or transcript in neighbors,
                    (source, transcript),
                )
        self.assertTrue(equivalence.is_equivalent("  HE\tHAD\n", " HED ", "en"))
        self.assertFalse(equivalence.is_equivalent("unindexed", "unindexed", "en"))
        self.assertFalse(equivalence.is_equivalent("", "hed", "en"))
        with self.assertRaises(TypeError):
            operator.setitem(lookup, "hed", frozenset({"corrupt"}))
        with self.assertRaises(AttributeError):
            lookup["hed"].add("corrupt")
        self.assertEqual(dict(equivalence.get_lookup("en")), expected)

    def test_synthetic_ambiguous_pairs_align_without_a_transitive_bridge(self):
        equivalence = self.make_equivalence(pairs={"en": [
            ("hed", "he had"), ("hed", "he would"),
        ]})
        with patch("tts_audiobook_tool.validator.WordEquivalence", equivalence), \
                patch.object(Whitelist, "supports_language", return_value=False):
            for expanded in ["he had", "he would"]:
                for source, transcript in [("hed", expanded), (expanded, "hed")]:
                    with self.subTest(source=source, transcript=transcript):
                        self.assertEqual(self.alignment(source, transcript), [
                            ("match_equivalent", source, transcript)
                        ])
            for source, transcript in [("he had", "he would"), ("he would", "he had")]:
                self.assertFalse(equivalence.is_equivalent(source, transcript, "en"))
                self.assertIsNone(equivalence.equivalent_phrase(
                    source.split(), 0, transcript.split(), 0, "en"
                ))
                self.assertEqual(self.alignment(source, transcript), [
                    ("match_direct", "he", "he"),
                    ("mismatch_sub", source.split()[1], transcript.split()[1]),
                ])
                self.assertEqual(Validator.get_word_errors(source, transcript, "en"), [
                    f"s:{source.split()[1]}/{transcript.split()[1]}"
                ])

    def test_can_variants_preserve_all_pairwise_relationships(self):
        # The former three-member declaration required all three explicit pairs,
        # not just two edges that would depend on forbidden transitive matching.
        variants = {"can not", "cannot", "cant"}
        lookup = WordEquivalence.get_lookup("en")
        for source in variants:
            with self.subTest(source=source):
                self.assertEqual(lookup[source], frozenset(variants - {source}))
                for transcript in variants - {source}:
                    self.assertTrue(WordEquivalence.is_equivalent(source, transcript, "en"))

    def test_language_and_max_length_use_lookup_keys(self):
        equivalence = self.make_equivalence(pairs={
            "en": [("alpha", "beta"), ("alpha", "a longer phrase")],
            "zz": [("one", "two words")],
        })
        self.assertTrue(equivalence.supports_language("zz"))
        self.assertTrue(equivalence.is_equivalent("one", "two words", "ZZ"))
        self.assertEqual(equivalence.max_phrase_length("en"), 3)
        self.assertEqual(equivalence.max_phrase_length("zz"), 2)
        self.assertEqual(equivalence.max_phrase_length("unsupported"), 0)

    def test_empty_data_has_empty_read_only_lookup_and_zero_max_length(self):
        for pairs in [{}, {"en": []}]:
            with self.subTest(pairs=pairs):
                equivalence = self.make_equivalence(pairs)
                lookup = equivalence.get_lookup("en")
                self.assertEqual(dict(lookup), {})
                self.assertEqual(equivalence.max_phrase_length("en"), 0)
                self.assertFalse(equivalence.is_equivalent("same", "same", "en"))
                with self.assertRaises(TypeError):
                    operator.setitem(lookup, "same", frozenset({"other"}))

    def test_invalid_phrases_are_rejected_in_pairs(self):
        invalid_phrases = [
            "", " ", "Alpha", "straße", " alpha", "alpha ", "alpha  beta",
            "alpha\tbeta", "alpha\nbeta", "alpha\u00a0beta", "don't", "alpha-beta",
            "alpha_beta", "alpha/beta", "alpha.", "alpha🙂",
        ]
        for phrase in invalid_phrases:
            for entry in [("valid", phrase), (phrase, "valid")]:
                with self.subTest(entry=entry):
                    equivalence = self.make_equivalence(pairs={"en": [entry]})
                    with self.assertRaises(ValueError):
                        equivalence.get_lookup("en")
                    self.assertNotIn("en", equivalence._LOOKUP)

    def test_malformed_pairs_are_rejected(self):
        for entry in [(), ("alpha",), ("alpha", "beta", "gamma"), ("alpha", "alpha")]:
            with self.subTest(entry=entry):
                equivalence = self.make_equivalence(pairs={"en": [entry]})
                with self.assertRaises(ValueError):
                    equivalence.get_lookup("en")
                self.assertNotIn("en", equivalence._LOOKUP)

    def test_failed_lazy_build_never_caches_a_partial_language(self):
        for malformed_entry in [("bad phrase", "BAD"), ("bad",), ("bad", "bad")]:
            with self.subTest(entry=malformed_entry):
                equivalence = self.make_equivalence(pairs={
                    "en": [("alpha", "beta"), ("gamma", "delta")],
                    "zz": [("one", "two")],
                })
                table = equivalence._EQUIVALENCE_PAIRS
                table["en"].append(malformed_entry)
                valid_other_language = equivalence.get_lookup("zz")
                for query in [equivalence.get_lookup, equivalence.max_phrase_length,
                              equivalence.get_lookup]:
                    with self.assertRaises(ValueError):
                        query("en")
                    self.assertNotIn("en", equivalence._LOOKUP)
                    self.assertIs(equivalence.get_lookup("zz"), valid_other_language)
                table["en"].pop()
                self.assertEqual(dict(equivalence.get_lookup("en")), {
                    "alpha": frozenset({"beta"}), "beta": frozenset({"alpha"}),
                    "gamma": frozenset({"delta"}), "delta": frozenset({"gamma"}),
                })

    def test_equivalence_feature_is_enabled_by_default(self):
        self.assertIs(constants.ENABLE_WORD_EQUIVALENCE, True)

    def test_disabled_flag_skips_lookup_and_removes_only_equivalence_allowance(self):
        # Disabling an already prepared index must not change or poison its cache.
        prepared_lookup = WordEquivalence.get_lookup("en")
        pairs = [("would have", "wouldve"), ("did not", "didnt"), ("all right", "alright")]
        for source, transcript in pairs:
            self.assertEqual(self.alignment(source, transcript), [
                ("match_equivalent", source, transcript)
            ])
        with patch.object(constants, "ENABLE_WORD_EQUIVALENCE", False), \
                patch.object(WordEquivalence, "get_lookup", side_effect=AssertionError(
                    "disabled equivalence must not prepare a lookup"
                )) as lookup, \
                patch.object(Whitelist, "supports_language", return_value=False):
            for expanded, contracted in pairs:
                for source, transcript in [(expanded, contracted), (contracted, expanded)]:
                    with self.subTest(source=source, transcript=transcript):
                        path = self.alignment(source, transcript)
                        self.assertNotIn("match_equivalent", [step[0] for step in path])
                        self.assertTrue(Validator.get_word_errors(source, transcript, "en"))
            self.assertEqual(self.alignment("same", "same"), [
                ("match_direct", "same", "same")
            ])
            self.assertEqual(self.alignment("sea", "see"), [
                ("match_homophone", "sea", "see")
            ])
            self.assertEqual(Validator.get_word_errors("sea", "see", "en"), [])
            lookup.assert_not_called()
        self.assertIs(WordEquivalence.get_lookup("en"), prepared_lookup)
        for expanded, contracted in pairs:
            for source, transcript in [(expanded, contracted), (contracted, expanded)]:
                self.assertEqual(self.alignment(source, transcript), [
                    ("match_equivalent", source, transcript)
                ])
                self.assertEqual(Validator.get_word_errors(source, transcript, "en"), [])

    def test_disabled_flag_keeps_both_uncommon_word_passes(self):
        # Use the real dictionary and phonetics rather than mocking the passes.
        self.assertFalse(Whitelist().has("qzxvplk"))
        with patch.object(constants, "ENABLE_WORD_EQUIVALENCE", False), \
                patch.object(WordEquivalence, "get_lookup", side_effect=AssertionError(
                    "disabled equivalence must not prepare a lookup"
                )) as lookup:
            for transcript, action in [("banana", "uncommon_pass_1"),
                                       ("banana apple", "uncommon_pass_2")]:
                with self.subTest(action=action):
                    self.assertEqual(self.alignment("qzxvplk", transcript), [
                        (action, "qzxvplk", transcript)
                    ])
                    self.assertEqual(Validator.get_word_errors("qzxvplk", transcript, "en"), [])
            lookup.assert_not_called()

    def test_unrelated_phrases_still_error_next_to_an_equivalence(self):
        with patch.object(Whitelist, "supports_language", return_value=False):
            for source, transcript, expected_errors in [
                ("would have", "wood table", ["s:would/wood", "s:have/table"]),
                ("did not", "bright cat", ["s:did/bright", "s:not/cat"]),
            ]:
                with self.subTest(source=source, transcript=transcript):
                    self.assertEqual(Validator.get_word_errors(source, transcript, "en"), expected_errors)
                    self.assertEqual(self.alignment(source, transcript), [
                        ("mismatch_sub", a, b)
                        for a, b in zip(source.split(), transcript.split())
                    ])
                    self.assertEqual(
                        Validator.get_word_errors(
                            f"{source} would have stayed", f"{transcript} wouldve stayed", "en"
                        ), expected_errors,
                    )
                    self.assertEqual(
                        self.alignment(f"{source} would have stayed", f"{transcript} wouldve stayed"),
                        [("mismatch_sub", a, b) for a, b in zip(source.split(), transcript.split())]
                        + [("match_equivalent", "would have", "wouldve"),
                           ("match_direct", "stayed", "stayed")],
                    )


if __name__ == '__main__':
    unittest.main()
