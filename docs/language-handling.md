# Architecture: Language Handling (`project.language_code`), TODOs

This document traces how the project language code (`Project.language_code`) is used
through the full audiobook-creation process, in rough sequential order: project setup,
text import & segmentation, prompt preprocessing, TTS inference, transcription &
validation, and finally concatenation/export.

It closes with some musings on the problems and tradeoffs of relying on this single
field as a "source of truth" for everything it does, and on the question of
language code normalization.

Related docs: `docs/project-spec-v2.md` (project schema), `docs/epub-import-flow.md`
(import path), `docs/tts-validation-architecture.md` (validation), and
`docs/word-equivalence.md` (one of the language-keyed subsystems).

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
  `"en-US"`, `"en_GB"`, `"eng"`, or `"English"` can legitimately end up stored.
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
`tts_audiobook_tool/text_ops/language_util.py`:

```python
normalize_language_code("en-US") == "en"
normalize_language_code("en_GB") == "en"
normalize_language_code("eng")   == "en"
normalize_language_code("English") == "en"
normalize_language_code("")      == ""
```

It lowercases, strips, takes the subtag before the first `-`, and runs a small alias
table (`english/eng -> en`, `spanish/spa -> es`). The docstring there states the
philosophy explicitly: features that key off a language normalize on read rather than
at input. `Whitelist.normalize_language_code` and `WordEquivalence` delegate to this
same function, and `tests/test_language_util.py` asserts the subsystems share it.

Not every consumer normalizes, though — several compare the raw stored value (see the
musings at the end).

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
3. **CJK warning** — if the code is a validation-unsupported language (`zh`, `ja`,
   `ko` per `VALIDATION_UNSUPPORTED_LANGUAGES`), a hint warns that Whisper-based
   validation will be skipped.
4. **Strictness downgrade side effect** — if the language is not "first-class"
   (no whitelist dictionary, i.e. not `en`/`es`) and strictness isn't already LOW, it
   is forced to LOW and saved.

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

- `pysbd` (the sentence-boundary detector) receives the language as `pysbd_lang`;
  `PhraseSegmenter` guards against invalid pysbd language identifiers.
- `PhraseSegmenter.sentence_string_to_phrase_strings(..., language_code)` and the
  parenthetical heuristics (`is_phrase_worthy_parenthetical`, citation/reference
  detection) branch on the code — several of these are English-only and
  short-circuit for anything else.

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
(see `docs/epub-import-flow.md`, "Existing segmentation settings remain
authoritative"), and the EPUB section-skip heuristics currently bake in
English-language assumptions regardless of the project language.

After segmentation, the text menu shows the first-class-tolerance hint when
`state.project.language_code in ("en", "es")` — a raw-string comparison, not a
normalized one.

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
  lang=language_code)` so models don't verbalize digit strings naively. Unsupported
  `num2words` locales fail silently (`NotImplementedError` is swallowed, digits pass
  through untouched).
- **Prompt word substitutions** — noun-suffix rules etc.; this path *does* normalize
  the code first via `language_util.normalize_language_code` so `"en-US"` reaches the
  English rule.

### 6. TTS inference

Behavior varies sharply by model:

- **Dots** — passes `language=project.language_code.strip() or None` straight to
  `generate()`.
- **MOSS** — maps the code to a language *name* via
  `MossBaseModel.get_language_name(...)` when non-empty.
- **Chatterbox Multilingual** — the odd one out: the code is a hard requirement.
  Readiness inspection fails with a `ReadinessIssue` if
  `project.language_code not in instance.supported_languages_multi()` (exact, raw
  comparison), and the project menu enforces the same list at input time.
- **Most other models** — ignore the language entirely.

So for inference, the code is a hint for some models, a contract for one, and
irrelevant to the rest.

### 7. Voice-clone reference transcription

Voice-clone flows that transcribe a reference audio sample (e.g.
`menus/voice/voice_menu_shared.py`) print the language code and pass
`state.project.language_code` to `Transcriber.transcribe_to_words`, so Whisper is
hinted the same language it will later validate in.

### 8. Generation-time transcription & validation

This is the heaviest concentration of language-dependent logic; see
`docs/tts-validation-architecture.md` for the full flow.

**Whisper transcription** (`transcriber.py`):
- The code is passed as `language=language_code or None` (empty string = let Whisper
  auto-detect).
- If the code isn't in Whisper's `supported_languages`, it is *silently dropped*
  rather than raising; likewise, in the interactive-worker path, if the backend
  reports the language unsupported, transcription is retried with `language=None`.
- `stt.py` short-circuits the whole validation stage with "Unsupported language" when
  `Validator.is_unsupported_language_code(state.project.language_code)` — i.e. the
  code starts with `zh`, `ja`, or `ko` (a `startswith` match on the raw string).

**Validation** (`validator.py`, `text_ops/text_normalizer.py`,
`text_ops/word_equivalence.py`):
- `TextNormalizer.normalize_common` branches on the *raw* code: `== "en"` applies
  English-specific contraction/number normalization, `== "es"` Spanish rules,
  anything else gets only the language-independent pass. `normalize_transcript` has
  parallel `en`/`es` branches keyed the same way.
- `Validator.get_word_errors` combines the `Whitelist` (uncommon-word detection,
  first-class `en`/`es` only), `WordEquivalence` lookups (per-language phrase
  equivalence tables), and `sounds_the_same_en` — which is strictly `language_code
  == "en"`.
- `ExcessiveDurationResult.is_excessively_long` normalizes the source with the
  language code and computes words-per-second against a rate; language affects the
  word count via normalization.
- `Strictness.get_recommended_default(language_code)` recommends a default tolerance
  based on whether the language is English (raw `== "en"` comparison).

The per-segment transcript metadata (`SegmentTranscriptData`) records the
`language_code` used at transcription time, so saved debug artifacts are
self-describing.

### 9. Concatenation & export

The concatenation stage itself is **language-agnostic**: ordering, gap limiting,
normalization, chapter markers, and break sound effects never consult the language
code. The only language touchpoint at the end of the pipeline is metadata: the ABR
export's `project_snapshot` embeds `language_code` verbatim (see
`docs/abr-metadata-spec.md`), preserving the hint alongside the delivered audio.

---

## Consumer summary

| Stage | Consumer | Uses raw or normalized code? | Effect when absent/unsupported |
|---|---|---|---|
| Project setup | `menus/project_menu.py` | raw (lowercased at input) | required for Chatterbox-ML; strictness forced LOW |
| Project load | `state.py` → `Whitelist` singleton | normalized | whitelist stays empty for non-first-class languages |
| Segmentation | `PhraseSegmenter` / pysbd | raw, guarded | pysbd falls back / heuristics disabled |
| Segmentation record | `book.segmentation_settings.language_code` snapshot | raw snapshot | shows "(none)" in dialogs |
| Enhance/conversation | `ChunkingConfig` | raw | chunking degrades to language-neutral |
| Prompt prep | `PromptNormalizer` (num2words / substitutions) | normalized (substitutions), raw (num2words lang) | digits unexpanded; rules skipped |
| TTS: Dots / MOSS | model `generate()` | raw | `None` / empty language name |
| TTS: Chatterbox ML | readiness check | raw, exact match | readiness failure |
| Whisper STT | `Transcriber` | raw | auto-detect (`language=None`) |
| Validation | `TextNormalizer`, `Whitelist`, `WordEquivalence`, `Validator` | mixed (see musings) | language-independent pass only |
| Skip gate | `Validator.is_unsupported_language_code` (`zh/ja/ko`) | raw, `startswith` | validation skipped entirely |
| Concat/export | concat (none); ABR metadata | verbatim snapshot | n/a |

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
graceful degradation, and mostly achieves it — but the degradation is silent, so a
user who sets `"en-US"` gets subtly less validation than one who sets `"en"` (see
below), with no warning anywhere.

**Raw vs normalized comparisons are inconsistent.** The codebase has a canonical
normalizer (`language_util.normalize_language_code`) and a test asserting the
whitelist and word-equivalence subsystems share it. But plenty of call sites still
compare the raw stored value: `TextNormalizer.normalize_common` branches on
`language_code == "en"` / `"es"`; `sounds_the_same_en` requires exactly `"en"`;
`Strictness.get_recommended_default` compares raw; the text-menu hint checks
`in ("en", "es")`; Whisper membership checks and the CJK `startswith` skip-gate use
the raw string; Chatterbox Multilingual demands an exact member of its list. So
`"en-US"` and `"eng"` normalize to first-class English for the whitelist, yet bypass
the English text-normalizer branches, the sounds-the-same check, and the recommended
strictness. Conversely the CJK gate is a `startswith` on the *raw* code, so `"zh"`
skips validation but a hypothetical `"zh-Hant"`-style value behaves differently than
the alias-collapsed `"zh"` elsewhere. The normalize-on-read philosophy is only
half-applied, and the seams show precisely where behavior differs across the seam.

**Empty string is overloaded.** `""` means "no hint" (Whisper auto-detects,
normalization goes language-independent) — but it is also the default value of the
canonical `book.segmentation_settings.language_code` snapshot before any segmentation
has happened. The `or None` /
`or ""` coercions scattered at call sites make "unset" and "deliberately cleared"
indistinguishable. The default of `"en"` for
`language_code` itself is also a quiet opinion: a user who never touches the setting
gets English hints everywhere, including Whisper being told `language="en"` rather
than auto-detecting.

**Should input normalize instead?** Normalizing at input (canonicalizing to a base
two-letter code, or even a small structured type) would fix the raw-comparison
inconsistencies by construction. The counterargument is real, though: region/script
information would be thrown away, and some consumers genuinely want it — Whisper
dialect hints, num2words locales, and Chatterbox Multilingual's exact list can
distinguish things the base code collapses. The current snapshot/alias design is a
compromise: store what the user said, normalize where a subsystem only understands
base codes. The cost of that compromise is that every new language-keyed feature must
remember to normalize, and must decide which comparison style it's using — and the
test suite only pins the two subsystems that already do it right.

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

**Unsupported-language policy is ad hoc.** `zh/ja/ko` hard-skip validation because
Whisper tokenizes those languages wordlessly; that list lives in `constants.py` and is
matched with `startswith` on the raw code. It's a reasonable pragmatic gate, but it's
a second, independent notion of "which languages work" alongside Whitelist's
first-class list, Whisper's `supported_languages`, pysbd's identifier set, and
Chatterbox's requirement — five different support matrices, none derived from a
single source of truth. Any future work here (e.g. CJK word segmentation for
validation, or another first-class whitelist language) will have to touch several of
them, and the risk is that they drift — exactly the drift the shared
`normalize_language_code` was introduced to prevent for the two subsystems that
already share it.
