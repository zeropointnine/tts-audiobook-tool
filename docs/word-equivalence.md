# Word/Phrase Equivalence System

This document describes the word/phrase equivalence system used by word-error
validation: a small, curated, per-language table of variants (single words or
multi-word phrases) that are treated as matches of each other during
validation, even though they are neither exact, homophone, nor
spacing-repairable matches.

**Implementation:** `tts_audiobook_tool/text_ops/word_equivalence.py`
(`WordEquivalence`)
**Validator integration:** `tts_audiobook_tool/validator.py`
(`Validator.get_word_error_alignment()`)
**Tests:** `tests/test_word_equivalence.py`

---

## The problem it solves

Validation already has two kinds of leniency that can absorb transcript
divergence, but both are single-word-only or alignment-based:

1. **Source-guided spacing repair** (`TextNormalizer.normalize_spacing_en`)
   can align a split/merged compound **only when the letters join up
   exactly**, eg source `good bye` ↔ transcript `goodbye`, or source
   `highschool` ↔ transcript `high school`.
2. **Double Metaphone homophone matching**
   (`TextNormalizer.sounds_the_same_en`) compares single words
   phonetically, eg `smith` ↔ `smyth`.

Neither mechanism can relate a phrase to a single word whose letters differ,
eg:

- source `all right` ↔ transcript `alright` (`allright` ≠ `alright`)
- source `will not` ↔ transcript `wont` (from `won't`, apostrophe stripped)
- source `toward` ↔ transcript `towards` (Metaphone primary codes `TRT` vs
  `TRTS` differ)

The equivalence table exists precisely for these leftovers.

---

## Data model

Equivalence groups are declared in `WordEquivalence._EQUIVALENCE_GROUPS`,
a per-language list of tuples. Each tuple is one group of variants; any two
members of the same group are mutual matches:

```python
_EQUIVALENCE_GROUPS: dict[str, list[tuple[str, ...]]] = {
    "en": [
        ("all right", "alright"),
        ("do not", "dont"),          # from "don't", apostrophe stripped
        ("toward", "towards"),
        ...
    ],
}
```

Entry format rules:

- lowercase (input is casefolded by `TextNormalizer` before comparison),
- single-spaced multi-word phrases (keys are joined with single spaces),
- **normalized form**, ie no punctuation. Contractions appear in their
  post-normalization shape: `don't` → `dont`, `it's` → `its`, `I'm` → `im`.
  This is because `TextNormalizer.normalize_common()` strips apostrophes
  from both source and transcript before comparison.

The class builds a read-only two-way lookup (variant → its frozen group),
cached per language. `get_lookup()` returns a `MappingProxyType` so callers
cannot corrupt the process-global cache; `is_equivalent()` is the
convenience API on top of it.

### Curation policy: what belongs in the table

An entry earns its place only if **no other mechanism already aligns it**.
Before adding a pair, check:

1. If joining the split side reproduces the merged side letter-for-letter
   (`a lot`/`alot`, `for ever`/`forever`, `good bye`/`goodbye`,
   `can not`/`cannot`), spacing repair already handles it — do **not** add
   it.
2. If the two words share a Double Metaphone primary code, homophone
   matching already handles it — do **not** add it (eg `toward`/`towards`
   have differing codes `TRT`/`TRTS`, so it *is* added).

Kept entries therefore fall into categories like: split/merged forms whose
letters differ, irregular contractions whose expansion is a separate word
sequence, informal dialogue reductions (`got to`/`gotta`), and spelling
variants Metaphone does not unify (`judgment`/`judgement`, `-ward`/`-wards`,
`while`/`whilst`).

Note that some entries are deliberately tolerant of genuine ambiguity: eg
`("it is", "its")` also grants a free pass between `it is` and the
possessive `its`, and `cant`/`wont` are rare real words. This is accepted
as the cost of a lenient validator.

---

## Language handling

`WordEquivalence.normalize_language_code()` accepts BCP-47 region/script
variants (`en-US`, `en_GB`), ISO 639-2 codes (`eng`), and common full names
(`English`), normalizing all of them to the base two-letter code. This
prevents a region- or name-qualified project language code from silently
disabling the feature.

The same alias table is mirrored in `Whitelist._LANGUAGE_ALIASES` so both
subsystems normalize language codes identically everywhere.

Languages without equivalence data are fully supported: lookup returns an
empty mapping and `max_phrase_length()` returns `0`.

---

## Validator integration

`Validator.get_word_error_alignment()` uses the equivalence data as an
additional zero-cost alignment option in its dynamic-programming table:

- For each DP cell, source windows of up to `max_phrase_length()` words are
  tested against transcript windows of up to the same length. When the
  joined phrases belong to the same equivalence group, the cell gets a
  zero-cost transition labeled `match_equivalent`.
- Phrase strings per `(start, length)` window are precomputed once per
  side, so the per-cell scan avoids repeated string joins.
- Precedence at equal cost: direct/homophone match, then equivalence
  match, then uncommon-word wildcard passes. Ordering matters only for
  diagnostics — it determines which action label the reconstructed
  alignment reports for a zero-cost step.

The new alignment action is `match_equivalent`
(eg `alright` ↔ `all right`), surfaced in verbose validation output and in
`SegmentTranscriptUtil`'s colored transcript rendering the same way
homophone matches are.

---

## Relationship to the larger validation system

Equivalence matching is one of several comparison allowances applied inside
the word-error DP alignment. For the surrounding normalization pipeline,
the DP alignment algorithm, the other leniency mechanisms (spacing repair,
Double Metaphone, whitelist wildcards), and threshold computation, see
`docs/tts-validation-architecture.md`.
