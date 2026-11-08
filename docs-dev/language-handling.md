# Architecture: Language Handling (`project.language_code`), TODOs

This document traces how the project language code (`Project.language_code`) is used
through the full audiobook-creation process, in rough sequential order: project setup,
text import & segmentation, prompt preprocessing, TTS inference, transcription &
validation, and finally concatenation/export.

It closes with the default/fallback-rule analysis (notably pysbd's English fallback),
some musings on the problems and tradeoffs of relying on this single field as a
"source of truth" for everything it does, and the open questions that remain.

Related docs: [Project Spec v5](<project-spec-v5.md>) (project schema), [EPUB import flow](<epub-import-flow.md>)
(import path), [TTS validation architecture](<tts-validation-architecture.md>) (validation), and
[word equivalence](<word-equivalence.md>) (one of the language-keyed subsystems).

---

## The field itself

```python
# tts_audiobook_tool/project.py
language_code: str = PROJECT_DEFAULT_LANGUAGE   # "en"
```

- A free-form string on the pydantic `Project` model, persisted to `project.json` and
  round-tripped verbatim on save/load.
- Defaults to `"en"` (`PROJECT_DEFAULT_LANGUAGE` in `constants_config.py`).
- The intent (per the project menu prompt) is a two-letter code like `"en"`, `"es"`,
  `"zh"`, but nothing enforces that beyond a very light syntax check. Values like
  `"en-US"`, `"en_GB"`, or `"eng"` can legitimately be entered; longer values such as
  `"English"` or `"zh-Hant"` exceed the menu's 5-character limit but can still arrive
  via a hand-edited `project.json` (which is also not lowercased on load).
- The canonical record of the code actually used when the current text was segmented
  is `Book.segmentation_settings.language_code` (see "Text import & segmentation"
  below). The old serialized `applied_language_code` key is accepted only as a legacy
  deserialization input; it is no longer a `Project` attribute. The same is true of the
  other old `applied_*` segmentation keys.
- The project menu also lets the user clear the code entirely (empty string), meaning
  "no language hint".

### Normalization: normalize-on-read, not at input

Input-side handling is minimal: the project menu normalizer only does
`value.strip().lower()`. Everything downstream normalizes on read instead, via
[`language_util.py`](<../tts_audiobook_tool/text_ops/language_util.py>):

```python
normalize_language_code("en-US")   == "en"
normalize_language_code("en_GB")   == "en"
normalize_language_code("eng")     == "en"
normalize_language_code("English") == "en"
normalize_language_code("zho")     == "zh"
normalize_language_code("")        == ""
```

It lowercases, strips, maps `_` to `-`, takes the subtag before the first `-`, and runs
an alias table (`_ALIASES_BY_BASE`): ISO 639-2 T/B and 639-3 codes plus English names
for every language some TTS model's language table knows (eg `deu`/`ger`/`german ->
de`, `zho`/`chi`/`cmn`/`chinese -> zh`, `nb`/`nob`/`norwegian -> no`,
`cantonese -> yue`). Endonyms (`deutsch`, `español`) are not aliased.

### The rule: language-class predicates

Any logic that gives a language special treatment must test the raw project code
through one of these predicates, never by comparing strings:

| Predicate | True for | Used by |
|---|---|---|
| `is_english(code)` | base `en` | text normalizer English pass, sounds-the-same, recommended strictness, prompt noun-suffix substitution, citation-parenthetical and dialog-attribution heuristics, heading chapter-prefix rule, enhance alignment |
| `is_spanish(code)` | base `es` | text normalizer Spanish pass |
| `is_cjk(code)` | base in `CJK_LANGUAGES` = `zh`, `ja`, `ko`, `yue` | word-count batching opt-out |
| `is_validation_supported(code)` | `not is_cjk(code)` (empty is supported) | STT/validation skip gate, project-menu CJK hint and strictness exemption |
| `Whitelist.supports_language(code)` | base in `Whitelist.LANGUAGES` (`en`, `es`) — "first-class" | uncommon-word detection, tolerance hints, uncommon-words inspector |

The predicates normalize internally, so callers pass the stored value as-is; hot loops
should resolve the predicate once outside the loop (as `Validator.get_word_errors`
does). Code that hands the language to an external library that only understands base
codes — pysbd, Whisper, num2words — passes `normalize_language_code(code)`. Code that
translates the code into a TTS model's own vocabulary uses
`map_language(code, table)` with a base-keyed table (see §6).

`Whitelist.normalize_language_code` and `WordEquivalence.normalize_language_code`
delegate to the shared function. [`test_language_util.py`](<../tests/text/test_language_util.py>)
pins a table of free-form codes against every predicate at once (including prefix
traps like Javanese `jav` and Konkani `kok`, which are *not* Japanese/Korean), plus
consumer wiring tests showing region-qualified codes behave like their base code.

---

## Sequential walkthrough

### 1. Project setup: setting the language code

The Project menu exposes "Language code" (`menus/project_menu.py`). The UI help text
summarizes the duties honestly — the code is a *hint* used by:

- semantic segmentation of imported text,
- prompt pre-processing,
- Whisper transcription,
- TTS inference (Chatterbox Multilingual, MOSS).

The input validator does four things:

1. **Syntax check** — at least one alphabetic char, max length 5. Anything else is
   "Bad value".
2. **Model requirement** — if the current model is Chatterbox Multilingual, the code
   must be one of the model's `supported_languages_multi` (queried by inspecting the
   loaded model).
3. **CJK warning** — if `not is_validation_supported(code)`, a hint warns that
   Whisper-based validation will be skipped.
4. **Strictness downgrade side effect** — if the language is not first-class
   (`not Whitelist.supports_language(code)`), validation is supported, and strictness
   isn't already LOW, it is forced to LOW and saved.

After saving, the code is pushed into the `Whitelist` singleton
(`Whitelist().set_language_code(...)`), which loads the bundled common-words
dictionary for the language if one exists.

### 2. Project load: re-syncing the singleton

`State.project` setter (`state.py`) re-syncs global/static state whenever the project
object changes — including `Whitelist().set_language_code(self.project.language_code)`.
This matters because `Whitelist` is a process-wide singleton whose "current language"
must track whichever project is active. The project menu's set/clear paths do the same
sync manually.

### 3. Text import & segmentation

When text is imported (manual entry, plain text file, or EPUB), it is segmented into
`PhraseGroup`s via `PhraseGrouper` / `PhraseSegmenter`, and the project language code
is one of the inputs:

- `pysbd` (the sentence-boundary detector) receives the language as `pysbd_lang`.
  `PhraseSegmenter.string_to_sentence_strings` normalizes it to the base code first
  (so `es-MX` gets pysbd's Spanish rules), then falls back to `"en"` when pysbd has no
  rules for it — including `""`. See "Default and fallback rules" below.
- Several heuristics are English-only via `is_english` and short-circuit for anything
  else: citation/reference-like parentheticals (`PhraseSegmenter`), the
  name-plus-verb dialog attribution rule (`DialogSegmenter`), and the EPUB heading
  chapter-prefix rule (`heading_end_detector`). Bare Roman-numeral headings are
  spelled out through `num2words` with the base code (`roman_heading`).

**The snapshot pattern.** The settings used for segmentation are recorded canonically
on the book (`BookSegmentationSettings.language_code`). Legacy serialized
`applied_language_code` values are folded into this book setting while older projects
are deserialized, but no project-level runtime mirror is retained. This is deliberate:
if the user later changes `language_code`, the phrase groups still remember which
language they were segmented under. UI surfaces (`segmentation_info_dialog.py`, the
text menu's post-import summary) and the Textual text editor read
`book.segmentation_settings.language_code`; edited text is therefore re-segmented using
the canonical book snapshot.

Note that segmentation always consumes the *live* `state.project.language_code` at
import time; `book.segmentation_settings.language_code` is the historical record.

Also of note: EPUB import reads `state.project.language_code` like any other text
source — the EPUB's own `dc:language` metadata is not authoritative for the project
(see [EPUB import flow](<epub-import-flow.md>), "Existing segmentation settings remain
authoritative"), and the EPUB section-skip heuristics currently bake in
English-language keywords regardless of the project language (no code check at all).

After segmentation, the text menu shows the first-class-tolerance hint when
`Whitelist.supports_language(state.project.language_code)`.

### 4. Pre-generation text flows (enhance / conversation)

The LLM enhance flow and the streaming conversation flow both carry the project
language into their text chunking:

- `enhance/enhance_flow.py` passes `language_code=state.project.language_code` into
  its chunking/segmentation calls.
- `conversation/conversation.py` builds `ChunkingConfig(language_code=...)` once from
  the project, and `conversation_internals.py` uses it for pysbd-driven sentence
  splitting of streamed LLM output.

### 5. Prompt preprocessing

Before text goes to a TTS model it passes through `PromptNormalizer`
(`text_ops/prompt_normalizer.py`), which is language-aware in two places:

- **Number expansion** — integers up to 999 are expanded to words via `num2words(...,
  lang=normalize_language_code(code))` so models don't verbalize digit strings
  naively. The base code matters for aliases: num2words' own fallback truncates to
  two letters, which turns `spa` into the unknown `sp`. (Region variants such as
  num2words' `es_CO` are not reachable anyway, since its keys are case-sensitive and
  project codes are lowercased.) Unsupported locales fail silently (digits pass
  through untouched); an empty code skips expansion.
- **Prompt word substitutions** — the English noun-suffix rule is gated on
  `is_english`, so `"en-US"` reaches it.

### 6. TTS inference

Behaviour varies sharply by model: the code is a hint for some, a contract for one
(Chatterbox Multilingual), and irrelevant to the rest.

**Model language mappings are translation, not branching.** Asking "what does this
model call this language?" is answered by a per-model lookup table, not by the
predicates. The rule:

```python
model_value = map_language(project.language_code, MODEL_TABLE)  # None if empty/unmapped
```

`language_util.map_language` normalizes, then looks up. Model tables are keyed by
**base code only**; aliases, region suffixes and case are the shared pipeline's job, so
`en-US`, `eng` and `English` always reach the `"en"` entry.
[`test_model_language_mappings.py`](<../tests/models/test_model_language_mappings.py>)
asserts every table is base-keyed.

Predicates still apply where model code genuinely *branches* on a language. The one
known case, AuK's English-vs-Chinese duration estimate in `auk_duration.py`, still has
its own `startswith` normalizer that mirrors upstream.

| Behaviour | Models | Resolver |
|---|---|---|
| Normalized table lookup | MOSS: local, both audio.cpp variants, both SGLang-Omni variants | `MossBaseModel.get_language_name` → `LANGUAGE_NAMES_BY_CODE` |
| | Qwen3: local and audio.cpp, one shared table | `qwen3_language.resolve_qwen3_language` → `QWEN3_LANGUAGE_NAMES`; `"auto"` when unmapped |
| | FireRedTTS3 (audio.cpp) | `FireRedTts3Behavior.get_language_tag` → `_TAG_BY_CODE`; readiness blocks unmapped codes |
| Base code plus a few id corrections | OmniVoice: local and audio.cpp | `omnivoice_language.resolve_omnivoice_language` (`ar` → `arb`, `tl` → `fil`, `nb`/`nob` kept as `nb`); audio.cpp via `OmniVoiceBehavior` (`language_policy = "project_code"`, so the behavior sees the raw code) |
| Normalized code sent as-is | audio.cpp Dots, IndexTTS2, Chatterbox (`language_policy = "normalized"`); Dots local (`DotsBaseModel.resolve_language`) | `AudioCppModelSupport.resolve_request_language` / `DotsBaseModel.resolve_language` |
| Contract on the normalized code | Chatterbox Multilingual (local) | `ChatterboxBaseModel.resolve_language_id`, compared against the library's base-code `SUPPORTED_LANGUAGES` by readiness and the project menu, and sent as `language_id` |
| | Chatterbox (audio.cpp) | `ChatterboxBehavior` readiness checks the server's own 19-code list (mirrored from audio.cpp; the server doesn't report it) |
| No language sent | audio.cpp `language_policy = "omit"`: CosyVoice3, Fish S2 / Drama3, GLM, and Breeze, Echo, Higgs v3 (which never read the field) | n/a |

No model passes the raw project code through any more. The `project_code` policy
remains in the catalog schema, but its only user (OmniVoice) has a behavior that
resolves the code; the base behavior's identity `resolve_language` would send it
verbatim.

**Do any models use region/script subtags?** No. Checked against the audio.cpp source
(`/d/p/audio.cpp-repo`) and the local libraries, for every model that used to
receive the raw code:

| Model | Reads the language? | What it accepts | `en-US` before |
|---|---|---|---|
| Breeze, Echo (audio.cpp) | no | — | ignored |
| Higgs v3 (audio.cpp) | no (declares only `"Auto"`) | — | ignored |
| Chatterbox (audio.cpp) | yes | 19 base codes; lowercases, cuts at `-`, empty means `en`, otherwise errors | worked; `en_us`/`eng` and `zh`/`ja`/`ru`/`he` were generation errors |
| OmniVoice (audio.cpp) | yes | ~650 exact ids (ISO 639-1, else 639-3, eg `yue`, `nan`, `nn`, `nb`, `fil`, `arb`) or English names; otherwise errors | generation error |
| OmniVoice (local) | yes | same vocabulary; otherwise warns and goes language-agnostic | silently dropped |
| Dots (local) | yes | normalizes with `langcodes` itself (strips region, prefers macrolanguage) | worked |

OmniVoice's three-letter ids (`nan` Min Nan, `nn` Nynorsk, …) are separate languages,
not region subtags. Normalization keeps them, because unknown three-letter codes
pass through unchanged. Only `nb` conflicts with the shared aliases, hence the
Bokmål exception. `test_omnivoice_resolved_ids_exist_in_library` (run under
`venv-omni`) checks every aliased base code resolves to a real OmniVoice id.

Every model that passes a language reports it in the pre-flight warnings through
`tts_models/model_support.make_language_value_note`, which names the value
actually sent and the project code it came from, eg
`Language value passed to model: English (project language code = "en-US")`
("none" when no language reaches the model). The shown value must come from the
same resolver as the request (`resolve_request_language`, `get_language_name`,
`resolve_language_value`, `resolve_language_id`) so the two cannot drift.

### 7. Voice-clone reference transcription

Voice-clone flows that transcribe a reference audio sample (e.g.
`menus/voice/voice_menu_shared.py`) print the language code and pass
`state.project.language_code` to `Transcriber.transcribe_to_words`, so Whisper is
hinted the same language it will later validate in.

### 8. Generation-time transcription & validation

This is the heaviest concentration of language-dependent logic; see
[TTS validation architecture](<tts-validation-architecture.md>) for the full flow.

**Whisper transcription** (`transcriber.py`, `model_worker.py`):
- The code is normalized to its base code at both entry points
  (`Transcriber.transcribe_to_words` and `ModelWorker.transcribe_audio_blocking`, the
  latter also covering enhance alignment and realtime conversation transcription).
  Empty means `language=None` (Whisper auto-detects).
- If the base code still isn't in Whisper's `supported_languages`, it is *silently
  dropped* rather than raising; likewise, in the interactive-worker path, if the
  backend reports the language unsupported, transcription is retried with
  `language=None`.
- `Stt.should_skip` short-circuits the whole validation stage with "Unsupported
  language" when `not is_validation_supported(state.project.language_code)`.

**Validation** (`validator.py`, `text_ops/text_normalizer.py`,
`text_ops/word_equivalence.py`). Callers pass the raw project code; every branch
below resolves it through a predicate:
- `TextNormalizer.normalize_common` applies English-specific contraction/number
  normalization when `is_english`, Spanish rules when `is_spanish`; anything else gets
  only the language-independent pass. `normalize_transcript` has parallel branches.
- `Validator.get_word_errors` combines the `Whitelist` (uncommon-word detection,
  first-class languages only), `WordEquivalence` lookups (per-language phrase
  equivalence tables), and `sounds_the_same_en` (only when `is_english`).
- `ExcessiveDurationResult.is_excessively_long` normalizes the source with the
  language code and computes words-per-second against a rate; language affects the
  word count via normalization.
- `Strictness.get_recommended_default(language_code)` recommends MODERATE when
  `is_english`, else LOW.

**Batching** (`generate_util.py`): when `is_cjk`, word-count sorting/bucketing of
batch items is disabled, since whitespace word counts are meaningless for those
scripts.

**Enhance alignment** (`enhance/enhance_anchored_alignment.py`) normalizes the code
and enables phonetic matching and English ordinal expansion for English — and, as
the one remaining exception, also treats `""` as English (see open questions).

The per-segment transcript metadata (`SegmentTranscriptData`) records the
`language_code` used at transcription time, so saved debug artifacts are
self-describing.

### 9. Concatenation & export

The concatenation stage itself is **language-agnostic**: ordering, gap limiting,
normalization, chapter markers, and break sound effects never consult the language
code. The only language touchpoint at the end of the pipeline is metadata: the ABR
export's `project_snapshot` embeds `language_code` verbatim (see
[ABR metadata spec](<abr-metadata-spec.md>)), preserving the hint alongside the delivered audio.

---

## Consumer summary

| Stage | Consumer | How the code is resolved | Effect when absent/unsupported |
|---|---|---|---|
| Project setup | `menus/project_menu.py` | raw (lowercased at input); predicates for hints | required for Chatterbox-ML; strictness forced LOW |
| Project load | `state.py` → `Whitelist` singleton | normalized | whitelist stays empty for non-first-class languages |
| Segmentation | `PhraseSegmenter` / pysbd | base code; `"en"` fallback | English sentence rules; English-only heuristics disabled |
| Segmentation record | `book.segmentation_settings.language_code` snapshot | raw snapshot | shows "(none)" in dialogs |
| Enhance/conversation | `ChunkingConfig` → pysbd | base code (inside `PhraseSegmenter`) | as Segmentation |
| Prompt prep | `PromptNormalizer` (num2words / substitutions) | base code / `is_english` | digits unexpanded; rules skipped |
| TTS: model mappings | per-model language tables / resolvers | base code via `map_language` or a model resolver (see §6) | `None` / empty / `auto`; readiness block for FireRedTTS3 and Chatterbox |
| TTS: Chatterbox ML | readiness check, `language_id` | base code vs the model's list | readiness failure |
| Whisper STT | `Transcriber`, `ModelWorker.transcribe_audio_blocking` | base code | auto-detect (`language=None`) |
| Validation | `TextNormalizer`, `Whitelist`, `WordEquivalence`, `Validator` | predicates | language-independent pass only |
| Skip gate | `Stt.should_skip` | `is_validation_supported` | validation skipped entirely |
| Batching | `generate_util` | `is_cjk` | word-count sorting disabled |
| Concat/export | concat (none); ABR metadata | verbatim snapshot | n/a |

---

## Default and fallback rules (considered, not yet acted on)

This section records the analysis of what the app does when the code is empty or
names a language a subsystem has no rules for — mainly the English fallback in
sentence segmentation. Normalization was done first, on purpose, and none of the
policy options below have been implemented yet.

### Where an unknown or empty code gets English

| Site | Trigger | Effect |
|---|---|---|
| pysbd, `PhraseSegmenter.string_to_sentence_strings` | base code pysbd has no rules for (`pt`, `sv`, `bn`, `th`, …), or `""` | English sentence rules |
| `PhraseGrouper.text_to_groups(pysbd_lang="en")` default argument | none: every caller passes a value | n/a |
| Enhance alignment `_expand_token` / `_number_to_words` | `""` | English ordinals (`1st`); numbers spelled out in English |
| `PROJECT_DEFAULT_LANGUAGE = "en"` | code never set | English everywhere, incl. Whisper told `language="en"` |

### Two fallback philosophies

The app answers "what if we don't know the language?" in two different ways:

- **Assume English:** pysbd and the enhance alignment tokenizer (the table above).
- **Assume nothing** (no language-specific rules): text normalization (only the
  language-independent pass), the Whitelist, sounds-the-same, word equivalence,
  num2words in prompts (digits left alone), the English-only segmentation heuristics
  (skipped), and Whisper (auto-detect).

pysbd is the odd one out for a practical reason: it has to be given *some*
language. The enhance case has no such constraint.

### pysbd in detail

**What "English" means in pysbd.** In pysbd 0.3.4, `English` is just `Common` +
`Standard` (the generic base ruleset) with an English `SENTENCE_STARTERS` list. So the
fallback really means "pysbd's generic rules":

- **Terminal punctuation:** `. ! ?` plus CJK full-width `。！？．`.
- **Abbreviations:** an English-centric list (titles, US states, months, and words
  like `no`, `is`, `me`, `la`, `que`, `con`, `del`).
- **How abbreviations work:** after a known abbreviation followed by a capitalised
  word, pysbd splits *unless* the abbreviation is in the "prepositive" list (`mr`,
  `dr`, `prof`, `st`, …). Number abbreviations (`no`, `art`, `pp`, …) only protect
  their period before a digit. A title abbreviation pysbd doesn't know (`Sra.`, `Sig.`,
  `Hr.`) is just a word ending a sentence.

**Languages pysbd 0.3.4 supports:** am, ar, bg, da, de, el, en, es, fa, fr, hi, hy, it,
ja, kk, mr, my, nl, pl, ru, sk, ur, zh. Notably missing: pt, sv, no, fi, cs, ro, hu,
tr, uk, ko, vi, th, bn, ta, he, id.

**Measured behaviour** (through `PhraseSegmenter.string_to_sentence_strings`, which
adds the app's quote-interior boundary logic on top of pysbd):

| Text | English rules | Native rules |
|---|---|---|
| es `Llegó el Sr. García con la Sra. López.` | splits after `Sr.` **and** `Sra.` | splits after `Sr.` only — pysbd's Spanish rules aren't perfect either |
| de `Er kam z.B. am 3. Mai an.` | splits after `3.` | correct |
| it `Il Sig. Rossi è arrivato.` | splits after `Sig.` | correct |
| ru `Он пришёл в 5 ч. вечера.` | splits after `ч.` | correct |
| zh / ja quoted dialogue | same result as native | raw pysbd differs (native zh splits inside quotes; English keeps the quote together), but the app's quote logic evens it out |
| hi `।`, el `;`, ar `؟` (supported; shown to illustrate the English rules) | **no sentence splits at all** | correct |
| pt `O Sr. Silva chegou.` (no pt rules) | splits after `Sr.` | n/a |
| bn `।`, th (no bn/th rules) | no splits | n/a |
| es with accented capitals (`Dr. Álvarez`, `Él`) | correct | correct |

**The two failure modes:**

- **Extra splits** after abbreviations the English list doesn't know (Latin and
  Cyrillic scripts). This is the worse failure for audio: a sentence-length pause in
  the middle of `Sr. | García`, and possibly a group boundary there. Merging short
  sentences into groups may soften it, depending on segmentation strategy.
- **Missed splits** in scripts with other terminal punctuation. pysbd covers several
  of these natively (hi, mr, el, ar, fa, ur, am, hy, my), so since normalization this
  hits only unsupported ones, eg Bengali/Nepali/Assamese (danda `।`) or Tigrinya
  (`።`). Phrase splitting on
  `, ; :` and the `max_words` limit catch some of it, but every sentence pause
  (`Reason.SENTENCE`) is lost. For scripts without spaces, `max_words` does nothing
  either, since word counts are whitespace-based.

**What normalization already changed.** Before normalization, every region or alias
form (`es-mx`, `zh-cn`, `ja-jp`, `en-us`, `spa`) also fell into the fallback:
`es-mx` got English sentence rules even though pysbd has Spanish rules. That was a
bug, not a policy, and it is fixed. Now only genuinely unsupported languages and `""`
reach the fallback.

Migration consequence: for an existing project with such a code, text re-segmented
after the change gets native rules, while untouched text keeps its English-rule
boundaries. Re-segmentation happens on new imports, and on text-editor edits, which
re-segment with `book.segmentation_settings.language_code`. Nothing reports the mix.

### Pros and cons of the English last resort

**Pros**
- It is effectively pysbd's generic ruleset, the base that the other languages'
  rules extend.
- Many abbreviations are shared across languages (`Dr.`, `St.`, `e.g.`, `etc.`), and
  multi-period forms like `z.B.` are protected generically.
- Its punctuation set covers Latin scripts and CJK, so for most unsupported
  Latin-script languages the output is mostly right, with local errors.
- Predictable and maintenance-free; `""` needs *some* answer and English is a sane
  one.
- The app layer covers the biggest gaps: quote handling, phrase splitting, word
  limits, short-sentence merging.

**Cons**
- **Silent.** Nothing tells the user that sentence splitting used English rules, so
  wrong pauses look like a model problem.
- **Inconsistent** with the "assume nothing" philosophy used everywhere else.
- **Wrong for some scripts** — those that end sentences with marks pysbd's generic
  rules don't know (see missed splits above).

### Options

1. **Keep `"en"` as the last resort** (current behaviour).
2. **Report the fallback.** Show something like "sentence rules: en (fallback; no
   pysbd rules for `pt`)" in the segmentation info dialog and the post-import
   summary. Cheap, and works alongside any other option. Implementation notes:
   - Put the decision in one helper (eg `effective_pysbd_language(code) ->
     (lang, is_fallback)`) used by both `PhraseSegmenter` and the UI, so the
     message can't drift from the behaviour.
   - The book's segmentation snapshot stores the user's code, not the rules
     language actually used. Recording the effective language there would be a
     schema change. Computing it at display time is simpler, but a pysbd upgrade
     that adds a language would make old segmentation look native when it wasn't.
3. **Map unsupported languages to a related pysbd language,** eg `pt/ca/gl → es`,
   `bn/ne/as → hi` (danda), `uk/be → ru`, `no/sv → da`, `cs → sk`. Mainly useful for
   scripts with other terminal punctuation. It's a hand-maintained table that
   guesses, it surprises when it guesses wrong, and a related language's
   abbreviations can mislead too.
4. **A "neutral" custom ruleset** (generic rules minus the English abbreviations and
   sentence starters), registered with pysbd. It would need overriding pysbd
   internals (`LANGUAGE_CODES`), which is fragile across pysbd updates. It would
   split after *every* abbreviation, probably worse than English for Latin text. And
   it can't add terminal punctuation across the board, because Greek `;` is a
   question mark in Greek but a semicolon everywhere else.

**Current leaning:** 1 + 2 (keep English, make it visible). Leave 3 for later, only
for the punctuation-mismatched scripts, if a user actually needs one. Skip 4.

### Empty code in enhance alignment

`_expand_token` treats `""` like `en` (English ordinals, num2words `"en"`); it is
marked with a TODO.

- **Why it's mostly harmless:** expansion runs on both the source and transcript
  sides, so digits expand identically on both. It only matters when one side has
  digits and the other has words. For a non-English project with a cleared code,
  English words won't match native words — but not expanding wouldn't match either.
- **Why change it:** it's the only place where `""` means English and isn't forced
  to (unlike pysbd). Making it neutral would align it with the rest of the app.
- **Cost of changing it:** an English project whose code was cleared would lose
  number and ordinal matching during alignment.
- **Leaning:** make it neutral (only expand when the normalized code is `en`).

### The project default `"en"`

A separate product question. It is right for most users, and English is the only
language with full first-class treatment (whitelist, sounds-the-same, MODERATE
recommended strictness). But a user who never touches the setting gets English
everywhere, including Whisper being told `language="en"` instead of auto-detecting.

Alternatives considered:
- **Default to `""`:** Whisper auto-detects, but English users silently lose
  first-class validation.
- **Ask for the language when creating a project.**

**Leaning:** leave it as is.

### Sequencing

Normalization (the predicates and base codes for pysbd, Whisper and num2words) was
deliberately done before any fallback-policy change:

- **Every option above assumes normalized input.** Option 1 *is* "normalize, then
  fall back".
- **It shrank the problem:** most fallback cases were region or alias forms.
- **It keeps behaviour changes attributable:** normalization changes were bug fixes;
  fallback changes are deliberate policy.

The `""` → English behaviour in enhance alignment was kept as-is during
normalization, for the same reason.

---

## Musings: problems and tradeoffs

**One field, many masters.** `language_code` simultaneously serves as: a segmentation
hint (pysbd), a text-normalization dialect selector, a whitelist/equivalence table
key, a Whisper hint, a num2words locale, a TTS model parameter, and — for exactly one
model — a hard contract. These duties have genuinely different vocabularies. pysbd
wants its own language identifiers; Whisper wants its two-letter codes; num2words
wants locale tags; Chatterbox Multilingual wants its private list of exact strings;
the whitelist only knows `en` and `es`. A single free-form string is the lowest common
denominator that lets all of them be wrong in different ways. The saving grace is that
for most consumers it is explicitly a *hint* with sane fallbacks (auto-detect,
language-independent normalization, silently skipped rules). The design leans on
graceful degradation, and mostly achieves it — but the degradation is silent.

**Raw vs normalized comparisons.** App logic that gives a language special treatment
now goes through the `language_util` predicates, and the base code is what reaches
pysbd, Whisper, and num2words, so `"en-US"`, `"eng"`, `"es-mx"`, and `"zh_CN"` behave
like their base codes throughout text handling and validation. Before this, a single
validation call for `"en-US"` ran the English whitelist (normalized) without English
normalization or sounds-the-same (raw), and the two CJK detectors disagreed with
each other (`zh_CN`, `zho`) and misfired on prefixes (`jav`, `kok` skipped validation;
`yue` didn't). Model language mappings now resolve the base code too (§6). What remains raw is the
input validator's 5-character limit.

**Empty string is overloaded.** `""` means "no hint" (Whisper auto-detects,
normalization goes language-independent) — except in pysbd and enhance alignment,
where it means English (see "Default and fallback rules"). It is also the default value of
the canonical `book.segmentation_settings.language_code` snapshot before any
segmentation has happened. The `or None` / `or ""` coercions scattered at call sites
make "unset" and "deliberately cleared" indistinguishable. The default of `"en"` for
`language_code` itself is also a quiet opinion: a user who never touches the setting
gets English hints everywhere, including Whisper being told `language="en"` rather
than auto-detecting.

**Should input normalize instead?** Normalizing at input (canonicalizing to a base
two-letter code, or even a small structured type) would make the predicates
unnecessary by construction. The counterargument is real, though: region/script
information would be thrown away, and some consumers genuinely want it — Whisper
dialect hints and some models' language inputs can distinguish things the base code
collapses (Chatterbox Multilingual's list, it turns out, is base codes only). The current design is a compromise:
store what the user said, resolve it through `language_util` wherever app logic or a
base-code library consumes it. The cost is that every new language-keyed feature must
use the predicates rather than a string comparison — which `test_language_util.py`
helps police only for the consumers it covers.

**The whitelist singleton is hidden global state.** Language correctness for
validation depends on `Whitelist().set_language_code(...)` having been called with the
right value at the right moments (project load, project menu set/clear). The
`state.py` setter covers the normal paths, but any code path that constructs or swaps
a project without going through `State` risks a stale singleton — validation would
quietly run against the wrong language's common-words dictionary. A purist would pass
the language explicitly into each validator call (as most functions already do with
`language_code` parameters) and demote the singleton to a cache; the current hybrid
means both mechanisms coexist and must agree.

**Snapshot vs live drift.** `book.segmentation_settings.language_code` records
segmentation provenance, which is the right instinct — text segmented under one
language shouldn't silently claim another. But only segmentation is snapshotted. Transcription,
normalization, and TTS inference all read the live `language_code`, so a mid-project language change
leaves audio validated under one language sitting next to audio validated under
another, with only the per-segment `SegmentTranscriptData` to tell them apart. A
"which language was this segment validated with?" audit requires opening debug
artifacts; nothing at concat time surfaces the mixed provenance (concat is
language-blind by design, which is both a strength and a blind spot).

**Several support matrices.** "Which languages work" is answered separately by
`CJK_LANGUAGES` (validation and batching), `Whitelist.LANGUAGES` (first-class),
Whisper's `supported_languages`, pysbd's identifier set, num2words' converters, and
Chatterbox's requirement. The first two now live behind shared predicates; the rest
are external and consulted with the base code.

## Open questions

- **English fallback policy** — see "Default and fallback rules" (leaning: keep
  English, report it).
- **`""` in enhance alignment** — same section (leaning: make it neutral). Marked
  with a TODO in `_expand_token`.
- **Project default `"en"`** — same section (leaning: leave it).
- **Korean in `CJK_LANGUAGES`** — Korean is space-delimited, so whitespace word
  logic could plausibly apply; it is grouped with zh/ja for historical reasons.
- **OmniVoice readiness** — the audio.cpp session still errors at generation time for
  a base code OmniVoice doesn't know (local OmniVoice just goes language-agnostic).
  A readiness check would need OmniVoice's id list in the main process.
- **AuK duration estimate** — its own `startswith` language normalizer could use the
  shared pipeline, but it deliberately mirrors upstream.
- **Input limit** — the 5-character validator rejects `english` and `zh-Hant` even
  though the alias table understands them.
