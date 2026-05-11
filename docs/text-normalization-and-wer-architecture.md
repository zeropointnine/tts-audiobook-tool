# Architecture: Text Normalization and Word-Error Comparison Flow

## Document Purpose

This document describes the generation-time validation flow that compares:

1. The source text from a `PhraseGroup`
2. The Whisper transcription of the generated TTS audio

It focuses on how the app:

- starts from segmented source text,
- generates audio,
- transcribes that audio with Whisper,
- normalizes the source text and transcription using slightly different rules,
- compares the normalized texts word-by-word,
- and turns that comparison into validation results that drive retry, saving, and re-generation behavior.

---

## High-Level Architecture

At generation time, each text segment follows this conceptual pipeline:

```text
PhraseGroup
  -> flattened prompt text
  -> TTS audio generation
  -> audio post-processing
  -> Whisper transcription to timestamped words
  -> transcript flattening to text
  -> source/transcript normalization
  -> word-error comparison
  -> validation result (pass / word-error fail / trim / music fail / skip)
  -> save audio + timing/debug metadata
```

The key design idea is that the app does **not** compare raw source text directly against raw Whisper output. Instead, it compares two normalized representations that are intentionally made easier to align.

---

## Runtime Flow

### 1. Source text begins as a `PhraseGroup`

**File:** `tts_audiobook_tool/phrase.py`

`PhraseGroup` is the unit of text passed into TTS generation.

- A `PhraseGroup` wraps one or more `Phrase` objects.
- `PhraseGroup.text` concatenates the original phrase text without altering whitespace.
- `PhraseGroup.as_flattened_phrase().text` produces the exact prompt text used for generation.

This matters because the validator compares the generated audio against the **flattened group text**, not against each inner phrase separately.

Relevant implementation:

```python
def as_flattened_phrase(self) -> Phrase:
    reason = self.phrases[-1].reason if self.phrases else Reason.UNDEFINED
    return Phrase(self.text, reason)
```

### 2. Generation turns the `PhraseGroup` into a TTS prompt

**Files:**

- `tts_audiobook_tool/generate_util.py`
- `tts_audiobook_tool/tts_model/tts_base_model.py`
- `tts_audiobook_tool/prompt_normalizer.py`

`GenerateUtil.generate()` builds one prompt per selected segment:

```python
prompt = GenerateUtil.phrase_group_to_prompt(phrase_groups[index], project)
```

And today that prompt is simply:

```python
prompt = phrase_group.as_flattened_phrase().text
```

At the `GenerateUtil` layer, the prompt starts as the same flattened text that later serves as the validation source text.

However, that flattened text is **not necessarily the final exact text handed to the TTS model**. Before inference, each model runs the standard pre-inference prompt pipeline in `TtsBaseModel.prepare_text_for_inference()`:

1. project-level word substitutions via `PromptNormalizer.apply_prompt_word_substitutions()`
2. generic prompt normalization via `PromptNormalizer.normalize_prompt()`
3. model-specific prompt massage via `massage_for_inference()`

Relevant implementation:

```python
text = PromptNormalizer.apply_prompt_word_substitutions(
    text, project.word_substitutions, project.language_code
)
text = PromptNormalizer.normalize_prompt(
    text=text,
    language_code=project.language_code,
    un_all_caps=self.INFO.un_all_caps,
)
text = self.massage_for_inference(text)
```

That prompt-side normalization is intended to improve **TTS inference behavior**, not text comparison.

Examples of prompt-side transformations include:

- applying user-configured word substitutions,
- collapsing repeated ellipses,
- removing punctuation-only tokens except a small pause-related whitelist,
- optionally converting an all-caps opening span to lowercase,
- converting smaller digit strings to words using `num2words`,
- and applying model-specific punctuation substitutions such as replacing em-dashes or semicolons for models that handle them poorly.

This is related to the comparison flow only in the broad sense that both start from the same `PhraseGroup` text. Architecturally, they are different pipelines with different goals:

- **prompt normalization** tries to make the text easier for the TTS model to speak well,
- **comparison normalization** tries to make source text and Whisper transcript easier to compare fairly.

So the validation source text is structurally tied to the same flattened `PhraseGroup` text that enters generation, but the later WER comparison should not be thought of as comparing against the fully massaged inference prompt string.

### 3. Generated audio is post-processed before transcription

Still in `GenerateUtil.generate()`, the raw model output is cleaned up before any STT step:

- silence is trimmed at the ends via `SoundAppUtil.apply_generate_post_processing()`
- internal silence gaps may be limited via `SilenceUtil.limit_silence_gaps()`

This ordering is important: Whisper sees the **post-processed** audio, not the raw model output.

### 4. Whisper transcribes the generated audio into words

**Files:**

- `tts_audiobook_tool/generate_util.py`
- `tts_audiobook_tool/whisper_util.py`

`GenerateUtil.generate_and_validate_batch()` performs transcription after audio generation:

```python
gen_result = WhisperUtil.transcribe_to_words(
    sound, project.language_code, stt_variant, stt_config
)
```

`WhisperUtil.transcribe_to_words()` is the central wrapper for this:

- resamples audio to `WHISPER_SAMPLERATE` if necessary,
- configures the selected Whisper variant/config via `Stt`,
- suppresses unsupported language hints by clearing `language_code` if the model cannot accept it,
- calls Whisper with `word_timestamps=True`,
- materializes the returned segment generator,
- and flattens segments into a single list of timestamped `Word` objects.

Relevant call:

```python
segments, _ = Stt.get_whisper().transcribe(
    audio=sound.data,
    word_timestamps=True,
    language=language_code or None,
)
```

The comparison pipeline therefore starts from **timestamped Whisper words**, not from raw segment text.

### 5. Whisper words are flattened into transcript text

**File:** `tts_audiobook_tool/whisper_util.py`

Before normalization/comparison, the validator converts the `Word` list into one text string:

```python
text = " ".join([word.word.strip() for word in words])
```

This preserves Whisper's token text, punctuation, and capitalization as much as possible at this stage. Normalization happens later.

### 6. Validation compares source text vs transcript text

**File:** `tts_audiobook_tool/validate_util.py`

The main generation path calls:

```python
validation_result = ValidateUtil.validate(
    sound, text, transcribed_words, project.language_code, strictness=project.strictness
)
```

Inside `ValidateUtil.validate()`:

1. Whisper words are flattened into transcript text.
2. `ValidateUtil.get_word_error_fail()` computes normalized word errors and the failure threshold.
3. Optional music detection may override the result with `MusicFailResult`.
4. A `WordErrorResult` is created.
5. The validator optionally attempts to derive a `TrimmedResult` if the transcript semantically matches a trimmed subrange of the audio.

So the word-error comparison is the core analysis step, but it is not always the final returned result.

---

## Normalization Architecture

### Why there are two normalization paths

**File:** `tts_audiobook_tool/text_normalizer.py`

The app uses a shared normalization core plus a transcript-specific spacing correction step.

- `normalize_source()` normalizes source text for comparison.
- `normalize_transcript()` normalizes transcription text **relative to the already-normalized source**.

This asymmetry is intentional.

The source text is considered the canonical reference. The transcript side gets an extra adjustment step because Whisper may split or merge compounds differently from the source text, even when the underlying spoken content is effectively the same.

### Shared normalization (`normalize_common`)

Both source and transcript pass through `TextNormalizer.normalize_common()`.

Main transformations:

1. `TextUtil.normalize_text_general(text)` runs first.
2. `casefold()` lowercases aggressively for international text.
3. Apostrophes adjacent to word boundaries are stripped.
4. Language-specific transformations are applied via dispatch functions:
   - **English** (`normalize_common_en_specific()`): dashes replaced with spaces, then text-number normalization via `whisper_normalizer.english.EnglishNumberNormalizer` so forms like `ninety-nine` normalize reliably.
   - **Spanish** (`normalize_common_es_specific()`): calls `strip_spanish_diacritics_keep_enye()` which removes accented characters (e->e, etc.) while preserving ñ/Ñ via sentinel-based protect-decompose-restore.
5. Punctuation is replaced with spaces.
6. Remaining non-word/non-space characters are stripped.
7. `TextUtil.massage_post_normalize()` performs final cleanup.

Relevant implementation skeleton:

```python
text = TextUtil.normalize_text_general(text)
text = text.casefold().strip()
text = re.sub(r"\b'|'\b", "", text)

# Language-specific treatments
if language_code == "en":
    text = normalize_common_en_specific(text)
elif language_code == "es":
    text = normalize_common_es_specific(text)

text = re.sub(f"[{re.escape(punc)}]", " ", text)
text = re.sub(r'[^\w\s]', '', text)
text = TextUtil.massage_post_normalize(text)
```

### Source normalization

Source normalization is straightforward:

```python
normalized_source = TextNormalizer.normalize_source(source, language_code=language_code)
```

This becomes the canonical string used for:

- word counting,
- comparison alignment,
- threshold computation,
- and regen/failure reporting elsewhere in the app.

### Transcript normalization

Transcript normalization dispatches to language-specific functions after the shared `normalize_common()` pass:

```python
normalized_transcript = TextNormalizer.normalize_common(transcript, language_code)

if language_code == "en":
    normalized_transcript = normalize_transcript_en_specific(normalized_source, normalized_transcript)
elif language_code == "es":
    normalized_transcript = normalize_transcript_es_specific(normalized_source, normalized_transcript)
```

**English** (`normalize_transcript_en_specific`): delegates to `normalize_spacing_en()`, which uses `difflib.SequenceMatcher` to align the normalized transcript against the normalized source and selectively repair whitespace-only divergences. This reduces false mismatches such as:

- source: `high school` vs transcript: `highschool`
- source: `firefly` vs transcript: `fire fly`

Behavior by opcode:

- `equal`: keep transcript text
- `delete`: if source-only content is whitespace, restore it
- `insert`: if transcript-only content is whitespace, drop it
- `replace`: keep transcript text as-is

**Spanish** (`normalize_transcript_es_specific`): currently returns the transcript as-is without spacing repair. The spacing repair logic (`normalize_spacing_en`) is English-only for now.

This means the English transcript gets a **source-guided spacing repair** pass that the source does not need, while Spanish does not yet have an equivalent spacing repair step.

### Combined API used by validation

The main validator typically uses:

```python
normalized_source, normalized_transcript = \
    TextNormalizer.normalize_source_and_transcript(source, transcript, language_code=language_code)
```

That helper guarantees the correct ordering:

1. normalize source first,
2. then normalize transcript using the normalized source.

---

## Word-Error Comparison Architecture

### Entry point: `get_word_error_fail()`

**File:** `tts_audiobook_tool/validate_util.py`

This method is the main bridge between normalization and pass/fail logic.

It:

1. normalizes source and transcript,
2. computes detailed word-error codes,
3. counts normalized source words,
4. computes the allowed threshold from project strictness,
5. returns `(did_fail, word_errors, threshold)`.

Relevant implementation:

```python
normalized_source, normalized_transcript = \
    TextNormalizer.normalize_source_and_transcript(source, transcript, language_code=language_code)

word_errors = ValidateUtil.get_word_errors(normalized_source, normalized_transcript, language_code)
num_word_errors = len(word_errors)
num_words = TextUtil.get_word_count(normalized_source, vocalizable_only=True)
fail_threshold = ValidateUtil.compute_threshold(num_words, strictness)
```

### Core algorithm: dynamic-programming alignment

`ValidateUtil.get_word_errors()` compares the two normalized strings at the word level.

It first tokenizes by spaces:

```python
source_words = normalized_source.split()
transcript_words = normalized_transcript.split()
```

It then runs a dynamic-programming alignment similar to edit distance.

Supported outcomes:

- direct match
- homophone match (English only)
- uncommon-word free pass (language-dependent whitelist behavior)
- deletion (`d:word`)
- insertion (`i:word`)
- substitution (`s:source/transcript`)

The DP table minimizes total failure cost, and a parent table reconstructs the chosen path.

### Comparison rules beyond exact equality

The comparison layer is intentionally more permissive than raw string equality.

#### 1. Homophone matching for English

`TextNormalizer.sounds_the_same_en()` uses `doublemetaphone` and compares primary metaphone codes.

This allows some pronunciation-equivalent English mismatches to count as matches.

#### 2. Language-specific whitelist behavior

**Files:**

- `tts_audiobook_tool/validate_util.py`
- `tts_audiobook_tool/whitelist.py`
- `tts_audiobook_tool/state.py`

The uncommon-word wildcard logic uses the language-aware `Whitelist` class. The former `DictionaryEn` class has been removed entirely in favor of `Whitelist`, which supports multiple languages.

At runtime, `Whitelist` maintains a language-specific set of common words for the
current project language. Today it supports:

- `en` via `tts_audiobook_tool/assets/whitelist_english.txt`
- `es` via `tts_audiobook_tool/assets/whitelist_spanish.txt`

Language variants are normalized to a base code first:

- `en-US` -> `en`
- `en-GB` -> `en`
- `es-ES` -> `es`
- `es-MX` -> `es`

That normalization happens in `Whitelist.normalize_language_code()`. The active
project language is then synced into the whitelist singleton via
`Whitelist.set_language_code(self.project.language_code)` when `State.project` is set.

The `Whitelist.supports_language(language_code)` method is used by `is_uncommon_word()` to determine whether wildcard behavior is available for the current language. If the active language is unsupported, `supports_language()` returns `False` and the uncommon-word wildcard behavior is disabled entirely.

#### 3. Uncommon-word free passes

The `is_uncommon_word()` function inside `get_word_errors()` checks:

```python
def is_uncommon_word(word: str) -> bool:
    return not Whitelist.has(word) if Whitelist.supports_language(language_code) else False
```

This means uncommon-word wildcard passes are available for **any** language supported by `Whitelist`, not just English. If the language is unsupported, `is_uncommon_word()` always returns `False` and no wildcard passes are granted.

When a word is deemed uncommon, the validator treats it as a wildcard:

- 1 source word ↔ 1 transcript word at zero cost
- or, for uncommon words, 1 source word ↔ 2 transcript words at zero cost

This is primarily meant to avoid over-penalizing names, rare words, or words the STT model handles inconsistently.

Current behavior differs by language:

| Language | Whitelist file | Uncommon-word wildcard pass | Homophone matching | Extra `Whitelist.has()` behavior |
|---|---|---|---|---|
| `en` | `whitelist_english.txt` | yes | yes | also accepts some possessive/trailing-`s` forms |
| `es` | `whitelist_spanish.txt` | yes | no | also applies Spanish dynamic expansion heuristics via `WhitelistUtilEs` |
| unsupported languages | none | no | no | none |

This distinction matters because the comparison layer has **two separate kinds of
leniency**:

- English-only phonetic leniency via `sounds_the_same_en()`
- supported-language whitelist leniency via `Whitelist`

So Spanish currently benefits from uncommon-word wildcard passes, but not from the
English-specific homophone logic. It is also **not** limited to strict file
membership: `Whitelist.has()` can trust additional Spanish forms via
`WhitelistUtilEs`, including some noun/adjective gender-number variants and some
longer inflected forms that belong to a sufficiently represented whitelist stem
family.

When verbose mode is enabled, `ValidateUtil.format_source_with_uncommon_words()` highlights uncommon source words in brackets with error coloring so it is easy to see which words are receiving wildcard treatment during validation.

For more on how the Spanish whitelist was built and normalized, see:

- `docs/spanish_whitelist_methodology.md`

### Output format of word errors

The comparison returns a list of failure codes rather than just a number.

Formats:

- `d:word` — word in source missing from transcript
- `i:word` — extra word in transcript
- `s:source/transcript` — substitution mismatch

`WordErrorResult.num_errors` is simply `len(errors)`.

---

## Threshold and Pass/Fail Logic

### Threshold is based on normalized source word count

The allowed error threshold is computed from the normalized source word count, not from the raw source text and not from the transcript length.

**File:** `tts_audiobook_tool/validate_util.py`

```python
fail_threshold = ValidateUtil.compute_threshold(num_words, strictness)
```

Current formulas:

| Tolerance (`Strictness`) | Formula |
|---|---|
| `LOW` | `ceil(num_words / 10) + 1` |
| `MODERATE` | `ceil(num_words / 10)` |
| `HIGH` | `max(0, ceil(num_words / 10) - 1)` |
| `INTOLERANT` | `0` |

### `WordErrorResult` fail semantics

**File:** `tts_audiobook_tool/validation_result.py`

`WordErrorResult.is_fail` is defined as:

```python
return self.num_errors > self.threshold
```

So a segment only fails when it exceeds the threshold, not when it merely equals it.

### Other result types can supersede the plain word-error result

After word errors are computed:

- `MusicFailResult` wins if music is detected in the generated audio.
- `TrimmedResult` wins if trimming a subrange yields a semantic match.
- `SkippedResult` is used when STT validation is intentionally skipped.

That means the word-error comparison is foundational, but the final saved outcome can represent a corrected or overridden interpretation of the audio.

---

## Trimmed-Match Subflow

One important nuance is that a non-zero-error transcript does not always stay a word-error result.

`ValidateUtil.make_trimmed_result()` tries to detect whether the generated audio contains extra leading or trailing material, while some contiguous subset of the transcript matches the normalized source exactly.

The method:

1. normalizes the source,
2. computes the normalized transcript,
3. slides a source-length window across the transcribed words,
4. normalizes each sub-transcript against the normalized source,
5. reruns `get_word_errors()`,
6. and if the sub-transcript has zero errors, trims the audio to the matching time window.

This is an important architectural detail because it shows that the normalization/comparison stack is reused not only for pass/fail scoring, but also for **repairing** outputs that are semantically correct after trimming.

---

## Persistence and Downstream Consumers

### Saved sound segments encode validation outcome

**File:** `tts_audiobook_tool/generate_util.py`

After validation, `GenerateUtil.save_sound_and_timing_json()` saves the final audio and related metadata.

For transcript-bearing results (`TranscriptResult` and subclasses), it also force-aligns phrase timing metadata and writes a parallel `.json` file.

### Word error counts are persisted in filenames

The saved filename may include the word error count for later regen logic. That behavior is documented more fully in:

- `docs/sound-segment-and-regen-architecture.md`

The important connection here is that the persisted `num_errors` value ultimately comes from the normalization + comparison pipeline described in this document.

### Normalized source word counts are reused elsewhere

Other parts of the app reuse `TextNormalizer.normalize_source()` plus `TextUtil.get_word_count(..., vocalizable_only=True)` when they need a stable count of comparable words, such as:

- regen list display in `app_util.py`
- dynamic failure detection in `project_sound_segments.py`

This helps keep generation-time validation and later regen-time evaluation consistent.

---

## Related but Separate Flow: STT Alignment for Existing Audiobooks

There is a second, distinct STT flow in `tts_audiobook_tool/stt_flow.py` used for enhancing existing audiobooks.

That flow also starts from source text, runs Whisper, and compares/aligns source text with transcription, but its primary goal is **timing alignment and metadata generation**, not TTS output validation.

In that path:

- `PhraseGrouper.text_to_groups()` segments source text,
- `SttUtil.transcribe_to_words()` transcribes the source audiobook,
- `SttUtil.make_timed_phrases()` merges source phrases with transcribed words.

It is adjacent to this architecture, but separate from the generation-time normalization/WER validation flow described above.

---

## End-to-End Summary

For one generated segment, the practical comparison flow is:

1. Start with `PhraseGroup` source text.
2. Flatten it into the TTS prompt.
3. Generate audio.
4. Post-process audio.
5. Transcribe audio with Whisper into timestamped words.
6. Flatten Whisper words into transcript text.
7. Normalize source text with shared normalization rules.
8. Normalize transcript text with the same shared rules **plus source-guided spacing repair**.
9. Compare normalized word sequences using dynamic-programming alignment.
10. Apply language-appropriate comparison allowances where appropriate:
    - English homophone matching
    - whitelist-based uncommon-word wildcard passes for supported languages (currently English and Spanish)
11. Count failures and compare against the threshold derived from normalized source word count.
12. Convert that into a `ValidationResult` used for retry, saving, and re-generation logic.

That is the architectural core of the app's text normalization and WER comparison system.