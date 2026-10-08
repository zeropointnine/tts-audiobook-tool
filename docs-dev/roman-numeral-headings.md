# Bare Roman Numeral Headings

Last updated: 2026-10-08

## Purpose

Many books mark chapters with a line that is nothing but a Roman numeral:

```text
XI
```

Left alone, a TTS model reads that unpredictably (a letter sequence, "eks eye", "eleven", or skips it). This rule makes the app speak it as a number word ("Eleven"), and makes the text-comparison (validation) step expect that word.

## The rule

A `PhraseGroup` is rewritten for speech when **all** of these hold:

1. The group contains exactly **one** phrase.
2. That phrase's reason is **`>= Reason.PARAGRAPH`** (`PARAGRAPH`, `SPACE_BREAK`, `SECTION_BREAK`).
3. The text, ignoring surrounding whitespace and at most one trailing period, is a **strict, upper-case Roman numeral from 1 to 999**.
4. The numeral is not the lone letter **`I`** (far more likely the pronoun).
5. `num2words` supports the project language.

If any condition fails the text is left exactly as is. The stored project text is never modified; only what is spoken and compared changes.

Examples:

| Text | Reason | Spoken as |
|---|---|---|
| `XI` | `PARAGRAPH` | `Eleven` |
| `XLII.` | `SPACE_BREAK` | `Forty-two.` |
| `XI` (language `es`) | `PARAGRAPH` | `Once` |
| `XI` | `SENTENCE` | `XI` (unchanged) |
| `XI.` merged with the next sentence | n/a (two phrases) | unchanged |
| `I`, `xi`, `Mix`, `MMXXIV`, `Chapter XI`, `Henry VIII` | any | unchanged |

### Why these constraints

- **Reason `>= PARAGRAPH`**: a line break after the text means it stood alone on its own line, so it is a heading, not a numeral inside a sentence (`Henry VIII`, `World War II`). The grouper never merges a group ending at `PARAGRAPH` or higher with its neighbour (`PhraseGrouper.merge_short_sentences()` only merges across reasons below `PARAGRAPH`), so a real heading line is always its own group.
- **Single phrase**: cheap insurance against a group assembled from several phrases.
- **Upper case only, canonical form only**: `xi`, `mix`, `civil`, `IIII`, `VX` and similar are words or typos.
- **Up to 999**: matches the ceiling `PromptNormalizer.normalize_prompt()` already uses for digit expansion. `M` is not accepted.

## Implementation map

| Piece | Location | Role |
|---|---|---|
| Pure text logic | `tts_audiobook_tool/text_ops/roman_heading.py` | `parse_roman_numeral()` (strict parse, 1-999, excludes `I`) and `expand_bare_roman_numeral(text, language_code)` (returns spelled-out text or `None`). Knows nothing about phrases or reasons. |
| Policy | `PhraseGroup.spoken_text(language_code)` in `app_types/phrase.py` | Applies the single-phrase and reason conditions, then calls the pure helper. This is the only place that decides eligibility. |
| Call sites | `GenerateUtil.phrase_group_to_prompt()`, the `Validator.validate()` call in `GenerateUtil.generate_and_validate_batch()`, `SegmentTranscriptUtil.from_validation_result()` | Use `spoken_text()` instead of `as_flattened_phrase().text`. |

Real-time playback needs no change: it calls `GenerateUtil.generate_and_validate_batch()`, the same function as the file-generation flow. It still uses `as_flattened_phrase()` but only for `phrase.reason` (pauses and break effects).

## Design: one spoken text, used by both sides

The expanded text is computed once, from the group, and used both as the TTS prompt input and as the validation `source`. They cannot drift apart. Neither `PromptNormalizer` nor `TextNormalizer` contains any Roman-numeral logic, and the strings they take stay plain strings. Eligibility depends on `Reason`, which only exists at the `PhraseGroup` level, so the decision is made there and passed down as ordinary text rather than threading `Reason` through the normalizers.

```text
PhraseGroup ──spoken_text()──> "Eleven"
                                 ├─> prepare_text_for_inference() -> TTS model
                                 └─> Validator.validate(source) -> normalize_source()
                                          Whisper "Eleven." -> normalize_transcript()
                                          both sides -> "11"
```

The sidecar's `source` field now records the spoken text (`Eleven`), and `normalized_source` / `prompt` follow from it. Existing sidecars keep the values stored when they were written, and re-scoring uses those stored normalized values, so old segments are unaffected.

## Related change: English number normalization and attached punctuation

For the source `Eleven` and a Whisper transcript `Eleven.` to compare equal, the comparison normalizer must treat them the same way. It did not: `whisper_normalizer`'s `EnglishNumberNormalizer` converts `Eleven` to `11` but leaves `Eleven.` as `eleven.` (and turns `forty-two.` into `40 two.`), because punctuation attached to a number word defeats it.

`normalize_common_en_specific()` in `text_ops/text_normalizer.py` now runs the number normalizer **per clause**: the text is split after a letter followed by clause punctuation (`. , ; : ! ?`) and whitespace or end of text, and each piece is normalized separately. Decimals and currency (`5.50`, `$5`) are not split, since the punctuation there follows a digit. This improves number handling for all English text, not just headings: `I have forty-two.` -> `i have 42`, `Chapter twenty-one: Home` -> `chapter 21 home`.

Spanish already normalized `Once.` and `Once` identically, so it needed no change.

## Caveats and notes

- **The validator is lenient about lone numbers.** A bare number token such as `11` is not in the common-word whitelist, so `Validator.get_word_errors()` treats it as an uncommon word and gives it a free wildcard pass against any single transcript word. For a one-word heading, the validator therefore cannot really fail the segment either way. The practical benefit of this rule is the **prompt** (the model is told to say a word) and a source/transcript pair that is consistent in the sidecar and in diagnostics. The tests assert on the normalized strings for this reason.
- **Unsupported languages are untouched.** If `num2words` has no entry for the language, the text is left as-is, matching `normalize_prompt()`'s silent fallback.
- **Extending the rule**: change the eligibility conditions in `PhraseGroup.spoken_text()`; change what counts as a numeral in `roman_heading.py`.

## Tests

`tests/test_roman_heading.py` covers the parser (valid, invalid, range and excluded cases), the expander (positive/negative text, whitespace, language handling), the reason policy on `PhraseGroup.spoken_text()`, the source/transcript comparison, and the punctuation-independent English number normalization.

See also: `docs-dev/tts-validation-architecture.md`, `docs-dev/text-segmentation.md` (phrase reasons).
