# Word Substitutions (Prompt Pipeline, Matching Rules, Menu, and Dialog Input)

Last updated: 2026-09-10


## Document Purpose

This document describes the app's **word substitution** feature: a project-level
table of reference words and their replacements that is applied to the TTS
prompt at inference time to improve pronunciation of proper names,
neologisms, and other words the model tends to mispronounce.

It covers:

- where substitutions sit in the pre-inference prompt pipeline,
- the exact matching rules (tokenization, punctuation peeling, case, English
  noun suffixes, precedence),
- the authoring constraints that follow from those rules,
- persistence and legacy normalization, and
- the dialog's user-input rules and the design decisions behind them.

### Implementation map

| Concern | Location |
| --- | --- |
| Prompt application | `tts_audiobook_tool/text_ops/prompt_normalizer.py` (`PromptNormalizer.apply_prompt_word_substitutions`) |
| Pipeline call site | `tts_audiobook_tool/tts_models/tts_base_model.py` (`TtsBaseModel.prepare_text_for_inference`) |
| Sidecar prompt record | `tts_audiobook_tool/project_support/segment_transcript_util.py` (`SegmentTranscriptUtil.make_inference_prompt`) |
| Tokenization helpers | `tts_audiobook_tool/app_support/app_text.py` (`get_words`, `split_raw_word`, `ws_punc_chars`) |
| Project field | `tts_audiobook_tool/project.py` (`Project.word_substitutions`) |
| Persistence | `tts_audiobook_tool/project_support/project_serialization_util.py` |
| Legacy parse / normalization | `tts_audiobook_tool/project_support/project_util.py` (`ProjectUtil.parse_word_substitutions_json_string`) |
| Editor app | `tts_audiobook_tool/textual/word_substitutions_app.py` (`WordSubstitutionsApp`) |
| Edit dialog | `tts_audiobook_tool/textual/word_substitutions_dialog.py` (`WordSubstitutionsDialog`) |
| Menu surface | `tts_audiobook_tool/menus/text_menu.py` (`TextMenu.word_substitutions_menu`) |
| Uncommon-words inspector | `tts_audiobook_tool/app_support/app_text.py` (`get_uncommon_words`) + `tts_audiobook_tool/text_ops/whitelist.py` |
| Language-code normalization | `tts_audiobook_tool/text_ops/language_util.py` (`normalize_language_code`) |
| Tests | `tests/test_prompt_normalizer.py`, `tests/test_word_substitutions_app.py`, `tests/test_word_substitutions_dialog.py`, `tests/test_text_menu.py`, `tests/test_language_util.py` |

---

## 1. Data model

The substitution table is a single ordered map on the project:

```python
# tts_audiobook_tool/project.py
word_substitutions: dict[str, str] = Field(default_factory=dict)
```

- **Key** = the "reference word" to look for in the prompt.
- **Value** = the text that replaces it.

Only one table exists per project, shared by every model and every segment. It
is applied to the prompt only: the project's stored text and the phrase text
used for validation are never rewritten. The generated segment's sidecar
`prompt` field *does* record the effective (post-pipeline) inference prompt, as
described in §2.1; that is a record of what was sent, not a rewrite of any
project or validation text.

The table is discarded when the project's source text is completely replaced
(**Text > Import from text file / Manually enter text / Import from EPUB file**),
along with generated sound segments and section markers; the replacement
confirmation names each of these, substitutions included. Incremental edits made
in the text editor leave it intact, since most of the source text still matches.

---

## 2. Where substitutions run in the prompt pipeline

Word substitutions are the **first** step of the standard pre-inference text
pipeline, in `TtsBaseModel.prepare_text_for_inference()`:

```text
PhraseGroup
  -> flattened phrase text            (GenerateUtil.phrase_group_to_prompt)
  -> [1] project word substitutions   (PromptNormalizer.apply_prompt_word_substitutions)
  -> [2] generic prompt normalization (PromptNormalizer.normalize_prompt)
  -> [3] model-specific massage       (TtsBaseModel.massage_for_inference)
  -> model.generate()
```

```python
# tts_audiobook_tool/tts_models/tts_base_model.py
# Both methods are classmethods (see §2.1)
text = PromptNormalizer.apply_prompt_word_substitutions(
    text, project.word_substitutions, project.language_code
)
text = PromptNormalizer.normalize_prompt(
    text=text,
    language_code=project.language_code,
    un_all_caps=cls.INFO.un_all_caps,
)
text = cls.massage_for_inference(text)
return text
```

Consequences of running **first**:

1. The replacement text is itself subject to later normalization — digits up
   to 999 are expanded to words (`num2words`), repeated ellipses are
   collapsed, punctuation-only tokens are dropped except a small
   pause-related whitelist, optionally a leading all-caps span is lowercased
   (`un_all_caps`, per model), and model-specific punctuation fixes apply.
2. The substitution step sees the *original* prompt words (before number
   expansion), so keys may include digits.

### 2.1 Recording the effective prompt

The segment sidecar (`SegmentTranscriptData`) stores both the `source` (project
text, used as the validation reference) and the `prompt` (what the model was
actually asked to say). `SegmentTranscriptUtil.from_validation_result()` derives
`prompt` with `SegmentTranscriptUtil.make_inference_prompt()`, which calls
`Tts.get_class().prepare_text_for_inference(project, source)` — the same
pipeline function used for inference. `prepare_text_for_inference()` and
`massage_for_inference()` are classmethods precisely so this can be computed
without loading a model instance. As a result, substitution is reflected
anywhere the sidecar prompt is shown (generation console details, the
generation editor's segment-info dialog), while `source` and the normalized
validation text still reflect the unmodified project text.

### Interaction with validation

The validator compares the Whisper transcript against the **original** source
text; `apply_prompt_word_substitutions()` is not called on the validation
source (`validator.py` and `generate_util.py` contain no reference to it).

Therefore a substitution changes what the model is asked to say but not what
it is expected to have said. A replacement that sounds substantially different
from the original word can increase reported word errors even though the audio
is correct. This is the main reason the feature's UI copy says it "works best
with a bit of experimentation."

---

## 3. Matching rules

The complete application routine (`PromptNormalizer.apply_prompt_word_substitutions`,
abridged only in that the numbered comments are ours):

```python
language_code = language_util.normalize_language_code(language_code)

prompt_raw_words_original = app_text.get_words(prompt)
prompt_raw_words = []

for prompt_raw_word in prompt_raw_words_original:

    before, prompt_word, after = app_text.split_raw_word(prompt_raw_word)
    prompt_word_lc = prompt_word.lower()

    for reference_word, substitute_word in substitutions.items():

        reference_word_lc = reference_word.lower()

        if prompt_word_lc == reference_word_lc:            # 1. exact match
            should_make_upper = substitute_word[0].islower() and prompt_word[0].isupper()
            if should_make_upper:
                substitute_word = substitute_word[0].upper() + substitute_word[1:]
            prompt_word = substitute_word
            break                                          # final; stops scanning keys

        if language_code == "en":                          # 2. English noun suffixes
            TREAT_AS_NOUN = True
            if TREAT_AS_NOUN:
                SUFFIXES = ["s", "es", "'s", "s'"]
                for suffix in SUFFIXES:
                    if prompt_word_lc.endswith(suffix) and len(prompt_word_lc) > len(suffix):
                        root_prompt_word_lc = prompt_word_lc[:-len(suffix)]
                        if root_prompt_word_lc == reference_word_lc:
                            should_make_upper = substitute_word[0].islower() and prompt_word[0].isupper()
                            if should_make_upper:
                                substitute_word = substitute_word[0].upper() + substitute_word[1:]
                            prompt_word = substitute_word + suffix
                            break                              # breaks suffix loop only

    prompt_raw_words.append(before + prompt_word + after)

new_prompt = "".join(prompt_raw_words)
return new_prompt
```

### 3.1 Tokenization

`app_text.get_words()` splits the prompt with `re.findall(r"\S+\s*|\s+", s.lstrip())`,
so each token is a run of non-whitespace characters plus any trailing
whitespace; whitespace-only runs are also returned as tokens. Nothing is
lowercased or otherwise altered at this stage.

### 3.2 Punctuation peeling

`app_text.split_raw_word(token)` returns `(before, middle, after)`:

- `before` = leading run of `ws_punc_chars`
- `after` = trailing run of `ws_punc_chars`
- `middle` = everything between them

`ws_punc_chars` is:

```python
string.whitespace + string.punctuation
+ "\u2026\u2013\u2014"          # ellipsis, em-dash, en-dash
+ "\u201C\u201D\u2018\u2019"    # fancy double/single quotes
```

Only `middle` is compared against keys. `before` and `after` are reattached
verbatim, so surrounding punctuation and spacing survive a substitution:

```text
"The Ariekei, however..."   +   {"Ariekei": "AriaKay"}
  internal token middle = "Ariekei"   (leading/trailing ws+punc peeled)
  -> "The AriaKay, however..."
```

Internal punctuation is *not* peeled, so `Higgs-Boson`, `don't`, `O'Brien`,
`e.g`, and `U.S.A` can all be matched as keys.

### 3.3 Case-insensitive matching

Comparison is `middle.lower() == reference_word.lower()`. Stored key casing
does not affect matching, and the replacement value's casing is used as
authored except for the first-letter rule below.

### 3.4 Capitalization transfer (first letter only)

If the value starts lowercase **and** the matched source word starts uppercase,
the value's first character is uppercased. Otherwise the value is used
verbatim. The source word here is the original `middle` (for suffix matches,
the original word, not the root).

Examples with the same table:

| Key → Value | Prompt | Result |
| --- | --- | --- |
| `kilohour → kilo hour` | `Kilohour is…` | `Kilo hour is…` |
| `kilohour → Kilo hour` | `kilohour is…` | `Kilo hour is…` (value casing kept) |
| `NASA → nasa` | `NASA and NASAs` | `Nasa and Nasas` |

### 3.5 English noun suffix handling

Only for English, and only if no exact match was found, the word is treated as
a possibly-inflected noun. `language_code` is normalized first via
`language_util.normalize_language_code()`, so alias and region codes (`en-US`,
`en_GB`, `eng`, `English`) enable the rule too, matching how the whitelist and
word equivalence resolve language codes. For each suffix in
`["s", "es", "'s", "s'"]`, if the lowercased token ends with the suffix, the
suffix is non-empty, and the remainder equals the key, then the key portion is
replaced and the suffix appended to the replacement.

This means a single root key covers plural and possessive forms:

| Table | Prompt | Result |
| --- | --- | --- |
| `kilohour → kilo hour` | `five kilohours away` | `five kilo hours away` |
| `Ariekei → AriaKay` | `Ariekei's morphology` | `AriaKay's morphology` |

### 3.6 Precedence — exact beats suffix; suffix conflicts are last-wins

The outer loop scans `substitutions.items()` in **insertion order**, but the
two match kinds behave differently:

- An **exact** match replaces the token and breaks the outer loop, so it is
  final and cannot be overridden by later keys — regardless of position.
- A **suffix** match only breaks the inner suffix loop; the outer loop keeps
  scanning, comparing each remaining key against the *original* lowercased
  token. A later key can therefore overwrite a suffix result.
  - A later key with an **exact** match overwrites it (exact always wins).
  - A later key with another **suffix** match overwrites it (last suffix
    match wins).

Verified cases:

| Table (insertion order) | Prompt | Result | Why |
| --- | --- | --- | --- |
| `kilohour→X, kilohours→Y` | `kilohours` | `Y` | suffix hits first, later exact overrides |
| `kilohours→Y, kilohour→X` | `kilohours` | `Y` | exact hits first and stops |
| `boxe→X, box→Y` | `boxes` | `Yes` | both suffix-match (`es` wins last) |
| `box→Y, boxe→X` | `boxes` | `Xs` | last suffix match wins |

In practice, authoring both a root and its inflected form as separate keys is
redundant and can be order-sensitive; prefer a single root key. (The editor
does not currently warn about this.)

### 3.7 What substitution does not do

- **No multi-word keys.** Matching happens against a single token whose
  `middle` never contains whitespace, so a key containing a space can never
  match.
- **No substring or word-internal matching.** `Ariekei` will not rewrite
  `AriekeiX`; the whole `middle` must equal the key (or key + handled suffix).
- **No fuzzy/phonetic matching.** Matching is exact after case folding and
  punctuation peeling.
- **No lowercasing of the prompt itself.** Only the matched token changes.
- **No effect on stored text, validation source, or exports.**

### 3.8 Worked example

Table: `{"Ariekei": "AriaKay", "kilohour": "kilo hour"}`

| Prompt | Result |
| --- | --- |
| `Identity example` | `Identity example` |
| `It's a kilo hour` | `It's a kilo hour` |
| `The Ariekei are a species.` | `The AriaKay are a species.` |
| `Kilohour is the unit of measurement.` | `Kilo hour is the unit of measurement.` |
| `It is five kilohours away` | `It is five kilo hours away` |
| `Kilohours off my life` | `Kilo hours off my life` |
| `The Ariekei's morphology is kinda` | `The AriaKay's morphology is kinda` |

---

## 4. Authoring constraint: the matchable-key predicate

Because of §3.1–§3.2, a key can only ever be reached if it is already in the
shape of a token `middle`. The exact predicate is:

```python
is_matchable = (
    bool(key)
    and not any(char.isspace() for char in key)     # whitespace splits tokens
    and app_text.split_raw_word(key)[1] == key      # already peeled
)
```

| Key | Matchable? | Note |
| --- | --- | --- |
| `Ariekei`, `Higgs-Boson`, `don't`, `O'Brien`, `e.g`, `U.S.A`, `42`, `café` | yes | internal punctuation/digits are fine |
| `e.g.`, `U.S.A.` | no | trailing `.` is peeled by the matcher |
| `'tis` | no | leading `'` is peeled; use `tis` |
| `kilo hour` | no | whitespace splits the token first |
| `...`, `!!!` | no | `middle` is empty |

Values have no such constraints: they are inserted verbatim, may contain
spaces (`kilo hour`), and are trimmed only of surrounding whitespace.

Note that `split_raw_word` peels `ws_punc_chars` only, so punctuation outside
that set (for example `« »`) is *not* peeled. Validation therefore tests
against `split_raw_word` rather than a hand-written punctuation list, so the
rule stays pinned to the matcher's actual behavior.

---

## 5. Persistence and legacy normalization

- **On save**, the map is serialized under the legacy key
  `word_substitutions_json_string` as `json.dumps(project.word_substitutions)`
  (`project_serialization_util.py`).
- **On load**, `normalize_loaded_project_dict()` pops that field and parses it
  with `ProjectUtil.parse_word_substitutions_json_string()`. If parsing fails,
  the project gets `{}`.
- The legacy parser **lowercases keys** and strips both keys and values, and
  rejects empty keys/values and non-string entries (returning an error string).

Round-trip consequence: the editor stores keys as authored (for display), but
a save→load cycle normalizes keys to lowercase. This is harmless for matching
(case-insensitive) and for the editor's case-insensitive sort/collision checks,
but it means displayed casing is not guaranteed to persist across reloads.

---

## 6. Menu surface: editor and uncommon-words inspector

`Text > Word substitutions` (`TextMenu.word_substitutions_menu`) exposes one or
two items, depending on project state:

| Item | Shown when | Action |
| --- | --- | --- |
| `Edit word substitutions (items: N)` | always | Runs `WordSubstitutionsApp`; on `EditorSaved` the menu exits one level |
| `Inspect project text for uncommon words` | `Whitelist.supports_language(project.language_code)` and the project has phrase groups | Prints the uncommon-words report (§6.1) |

`Whitelist.supports_language()` is the app's "first-class language" check: the
languages that have a bundled common-words dictionary (`Whitelist.LANGUAGES`,
currently `en` and `es`). It normalizes its argument first via
`text_ops/language_util.py`, so `en-US`, `eng`, and `English` all qualify, while
a project in any other language simply hides the inspector item. (This is the
same gate used by uncommon-word validation, so the inspector never offers words
the validator would not treat as uncommon.)

### 6.1 Uncommon-words report

The inspector is read-only; it never writes to the project. It flattens every
`phrase.words` entry across `project.phrase_groups` (tokens still include their
surrounding whitespace and punctuation) and calls
`app_text.get_uncommon_words()`, which:

- strips outer punctuation/whitespace with `split_raw_word` and lowercases the
  token (fancy apostrophes are folded to `'`),
- skips words present in the whitelist singleton's currently loaded language and
  words rejected by `is_vocalizable()`,
- groups remaining casings under one lowercase key (eg `yggdrasil` →
  `["yggdrasil", "Yggdrasil"]`),
- sorts by descending occurrence count.

The screen heading is `Uncommon words`; its subheading names the normalized
project language (the `%1` slot in `UNCOMMON_WORDS_DESC` is filled with
`language_util.normalize_language_code(project.language_code)`). The report
prints at most the top 25 entries as `<count>  <variant> | <variant>`, or
`None found`, then waits for ENTER.

---

## 7. Dialog: user input behavior and design decisions

`WordSubstitutionsDialog` is a modal with:

```text
<Add item | Edit item>          <right-justified, dim: [ENTER] Submit>
<blank line>
Original word:     [ input ]
Substitution word: [ input ]
<error line, empty by default>
```

- `TAB` / `SHIFT+TAB` and `DOWN` / `UP` toggle focus between the two inputs.
- `ENTER` in either input submits.
- `ESC` cancels (dismisses `None`).
- The dialog returns a `WordSubstitutionEdit(original, substitution, previous_original)`;
  `previous_original` is `None` for an add and the pre-edit key for an edit.

The dialog is the only input surface for the table. Its result is applied to the
editor's in-memory staged map (`staged`, snapshotted from
`project.word_substitutions` on construction); edits reach disk only when the
user confirms exit, and a failed `project.save()` restores the snapshot and
reports `EditorSaveFailed`.

### 7.1 Submit algorithm

`try_submit()` runs in this exact order:

1. **Clear the error line.**
2. Strip both fields and write the stripped values back to the inputs.
3. If either field is empty → return silently (dialog stays open).
4. Normalize the original key (see §7.2).
5. If normalization failed → show the error and return.
6. If `original == replacement` → return silently.
7. If `original.casefold()` collides with another existing key (excluding the
   row's own previous key) → show "Already exists" and return.
8. Otherwise dismiss with the edit.

`existing_keys` is precomputed case-folded, minus the row being edited, so
editing a row without renaming its key is not a self-collision.

### 7.2 Key normalization rule

```python
# tts_audiobook_tool/textual/word_substitutions_dialog.py
def normalize_original_key(value: str) -> tuple[str, str]:
    value = value.strip()
    if any(char.isspace() for char in value):
        return value, WHITESPACE_MESSAGE        # "Original word can't contain spaces"
    normalized = app_text.split_raw_word(value)[1]
    if not normalized:
        return value, EMPTY_KEY_MESSAGE         # "Original word must contain a word"
    return normalized, ""
```

This is exactly the matchable-key predicate from §4, reusing the matcher's own
`split_raw_word` so a stored key is always reachable. Examples:

| Typed key | Stored key | Outcome |
| --- | --- | --- |
| `Higgs-Boson`, `don't`, `O'Brien` | unchanged | commits |
| `  e.g.  ` | `e.g` | commits (peeled) |
| `U.S.A.` | `U.S.A` | commits (peeled) |
| `'tis` | `tis` | commits (peeled) |
| `kilo hour` | — | error: can't contain spaces |
| `kilo<tab>hour`, `kilo<nbsp>hour` | — | error: can't contain spaces |
| `...`, `!!!` | — | error: must contain a word |

The value is only stripped; it is not punctuation-normalized, because
replacement text is inserted verbatim and spaces are the point (`kilo hour`).

### 7.3 Validation messages

| Constant | Text | Trigger |
| --- | --- | --- |
| `WHITESPACE_MESSAGE` | `Original word can't contain spaces` | `any(char.isspace())` in the stripped key |
| `EMPTY_KEY_MESSAGE` | `Original word must contain a word` | key normalizes to empty (punctuation-only) |
| `ALREADY_EXISTS_MESSAGE` | `Already exists` | case-folded collision with another staged key |

### 7.4 Design decisions

- **Auto-normalize instead of reject for leading/trailing punctuation.**
  The matcher peels `ws_punc_chars` itself, so `e.g.` and `'tis` as typed
  could never match while their peeled forms can. Rewriting the field and
  committing the peeled form keeps the table useful without asking the user to
  understand the tokenizer. Rejecting would be technically correct but hostile
  for the common paste-a-word-with-punctuation case.
- **Hard-error on internal whitespace rather than trimming.** Trimming an
  internal space (`kilo hour` → `kilohour`) would silently invent a different
  word. The user likely intended a multi-word key, which the matcher cannot
  support, so an explicit message is safer than a guess.
- **Hard-error on an empty normalized key.** A punctuation-only entry (`...`)
  would otherwise be stored with an empty comparison token and silently never
  fire. Contrast with a wholly empty field, which is treated as "not finished
  typing" and ignored silently.
- **Internal punctuation is preserved.** `Higgs-Boson`, `don't`, and `O'Brien`
  are the canonical use cases for the feature; a blanket "no punctuation" rule
  would break them. Only leading/trailing punctuation is peeled, matching the
  matcher exactly.
- **Keys are not lowercased in the editor.** Matching is case-insensitive, so
  lowercasing would only destroy displayed casing. (The legacy load path still
  lowercases on reload; see §5.)
- **Equality is checked after normalization.** `same.` vs `same` is a no-op,
  consistent with what the matcher would do.
- **Collisions are case-insensitive and exclude the edited row.** This mirrors
  the matcher's case folding and avoids a false self-conflict when only the
  replacement value changes.
- **The error line clears on every submit attempt.** Otherwise a stale message
  (e.g. a previous collision) could persist while the user is editing, and
  silently-ignored submits would leave misleading feedback on screen.
- **Validation reuses `split_raw_word` rather than listing punctuation.**
  The peel set (`ws_punc_chars`) can change; deriving the rule from the helper
  keeps input validation and prompt matching from drifting apart.
- **No redundant-suffix warning (yet).** A key ending in `s`/`es`/`'s`/`s'`
  is redundant given §3.5 and can interact with precedence in surprising ways,
  but flagging it would require explaining the suffix rule in the error line.
  It is a candidate for a soft, non-blocking warning.

---

## 8. Invariants and test coverage

Behavioral invariants:

1. Substitution is inference-time and prompt-only.
2. A stored key always satisfies the matchable-key predicate.
3. Matching is case-insensitive and bounded to a single token.
4. An exact match always beats a suffix match.
5. Edits are staged until the exit confirmation, and a failed save rolls the
   in-memory project back.
6. A rendered table row is exactly the view width. An over-long word cell ends
   with `…` plus one blank margin cell, so the two word columns keep a
   one-character separation even when both are truncated.
7. A full source-text replacement clears the table (in
   `ProjectTextIOUtil.set_phrase_groups_and_save` and
   `set_phrase_groups_chapters_and_save`); structural text-editor edits do not.
8. The uncommon-words inspector is read-only and is offered only for first-class
   languages when the project has phrase groups.

Relevant tests:

- `tests/test_prompt_normalizer.py` — end-to-end matching, capitalization,
  plural/possessive suffixes (including alias language codes such as `en-US`).
- `tests/test_word_substitutions_app.py` — staged edits and commit/rollback.
- `tests/test_word_substitutions_dialog.py` — `normalize_original_key` cases,
  whitespace/punctuation-only errors, stale-error clearing, collision,
  equality-after-normalization, Tab/Enter/Esc behavior.
- `tests/test_text_menu.py` — the edit item's label and count, presence and
  absence of the uncommon-words inspector (phrase groups, language support,
  alias codes), the inspector's printed report, and the text-import
  confirmation.
- `tests/test_project_book_integration.py` — word-substitution clearing on
  text-file/manual and EPUB import.
- `tests/test_language_util.py` — the shared normalizer and the fact that the
  whitelist and word equivalence delegate to it.
