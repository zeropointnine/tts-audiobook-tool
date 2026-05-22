# Project Spec v2

## Purpose

This document describes:

- what changed in **project spec version 2**, and
- how the app migrates older projects to the current format.

The current project spec version is defined in `tts_audiobook_tool/constants.py`:

```python
PROJECT_SPEC_VERSION = 2
```

The behavior described here is implemented primarily in:

- `tts_audiobook_tool/project.py`
- `tts_audiobook_tool/project_support/project_util.py`
- `tts_audiobook_tool/app_types/book_serialization.py`
- `tests/test_project_book_integration.py`

---

## Summary of the v2 change

The main project-spec change is a **storage split**.

### Version 1

In v1, project text was stored inline in `project.json`.

This could appear as:

- `text`
- `text_segments`

### Version 2

In v2, project text is stored externally in `project_text.json`.

`project.json` now primarily stores project settings and metadata, while
`project_text.json` stores the actual text/book payload.

This is stated directly in `Project`:

- version 1: project text stored inline in `project.json`
- version 2: project text stored externally in `project_text.json`

---

## Canonical v2 file layout

### `project.json`

Under v2, `project.json` is the settings file. It stores values such as:

- `version`
- `language_code`
- segmentation options such as `segmentation_strategy` and `max_words`
- generation options such as `generate_range`
- section markers in `markers`
- export and post-processing settings
- model-specific settings for the active TTS backends

It does **not** canonically store the full phrase-group text payload anymore.

### `project_text.json`

Under v2, `project_text.json` is the canonical text payload file.

The preferred current format is:

```json
{
    "format": "book.v1",
    "book": {
        "title": "...",
        "text_source_kind": "...",
        "audio_source_kind": "...",
        "segmentation_settings": {
            "language_code": "...",
            "max_words_per_segment": 120,
            "strategy": "max_len"
        },
        "sections": [
            {
                "title": "Chapter 1",
                "phrase_groups": []
            }
        ]
    }
}
```

This format is produced by `book_to_project_text_json_dict(...)` in
`tts_audiobook_tool/app_types/book_serialization.py`.

---

## Supported text payload formats

Even though `book.v1` is the canonical saved format, the loader still accepts
older payloads for backward compatibility.

### Current canonical format

- `book.v1`

Example shape:

```json
{
    "format": "book.v1",
    "book": { ... }
}
```

### Older accepted formats

- `phrase_groups.v1`

Example shape:

```json
{
    "format": "phrase_groups.v1",
    "phrase_groups": [ ... ]
}
```

- bare legacy list

Example shape:

```json
[
    ... phrase groups ...
]
```

When older payloads are loaded successfully, the app upgrades them to the
canonical `book.v1` representation.

---

## Other important spec-shape changes

In addition to moving text out of `project.json`, the current v2 shape reflects
several normalization changes.

### `chapter_indices` became `markers`

Older projects may store chapter boundaries in `chapter_indices`.

The current project shape uses:

```json
"markers": [ ... ]
```

On load, if `markers` is missing but `chapter_indices` exists, the loader copies
that legacy data into `markers`.

### Legacy `applied_*` fields are compatibility-only

These older flat-text fields may still be present in legacy projects:

- `applied_language_code`
- `applied_strategy`
- `applied_max_words`

These are still read for compatibility, especially when reconstructing a `Book`
from legacy phrase-group data, but they are not part of the preferred canonical
saved shape once `book.v1` text data is available.

### Some model keys are remapped on load

`ProjectUtil.remap_legacy_keys(...)` normalizes some older field names, for
example:

- `fish_voice_file_name` → `fish_s1_voice_file_name`
- `fish_voice_text` → `fish_s1_voice_text`
- `fish_temperature` → `fish_s1_temperature`
- `fish_seed` → `fish_s1_seed`
- `vibevoice_model_path` → `vibevoice_target`
- `qwen3_path_or_id` → `qwen3_target`
- `indextts2_emo_voice_alpha` → `indextts2_emo_alpha` when applicable

These are part of compatibility handling rather than the central v2 storage
change, but they are part of how older projects are normalized.

---

## In-memory normalization model

The loader normalizes project text into the current in-memory `Project` / `Book`
representation.

### Flat legacy text becomes a `Book`

If legacy phrase groups exist without a `book`, the app wraps them into a
single-section `Book` using compatibility defaults such as:

- `text_source_kind="legacy_flat"`
- `audio_source_kind="unknown"`

Legacy segmentation fields are folded into `BookSegmentationSettings`:

- `applied_language_code`
- `applied_strategy`
- `applied_max_words`

This means older flat projects can still be used through the newer structured
book model.

### Flat compatibility is still preserved

The `Project` model still keeps compatibility access to flat phrase-group data,
for example through:

- `phrase_groups`
- `get_flat_phrase_groups()`
- `get_section_start_indices()`

This allows older flows to remain functional while the canonical persisted shape
uses structured book text.

---

## Migration flow

Project migration is handled automatically during project load.

The main entry point is:

```python
ProjectUtil.load_using_dir_path(dir_path)
```

That method, together with `Project.model_validate(...)`, performs detection,
normalization, and rewrite-to-canonical-format when needed.

### Step 1: load `project.json`

The app loads `project.json` first.

It also notes whether legacy flat compatibility fields are present, especially:

- `applied_language_code`
- `applied_strategy`
- `applied_max_words`

### Step 2: detect where the text currently lives

The loader distinguishes between several cases.

#### Case A: inline legacy text in `project.json`

If `project.json` contains either:

- `text`, or
- `text_segments`

then the project is treated as an inline-text legacy project.

#### Case B: external text in `project_text.json`

If there is no inline text and neither `phrase_groups` nor `book` is already
present in the loaded settings dict, the loader tries to read
`project_text.json`.

It accepts:

- `book.v1`
- `phrase_groups.v1`
- bare legacy lists

If the payload loads successfully, it is converted into a `Book` instance.

### Step 3: remap legacy keys and normalize values

Before final validation, the loader normalizes legacy names and values. This
includes:

- model key remapping via `ProjectUtil.remap_legacy_keys(...)`
- `word_substitutions_json_string` → `word_substitutions`
- `chapter_indices` → `markers`
- generate-range sentinel values such as `"all"` or `"a"` → normalized empty
  string
- validation/defaulting of many newer settings fields

### Step 4: build the current `Project` model

`Project.model_validate(...)` then converts the normalized dictionary into the
current `Project` model.

Important behaviors include:

- inline `text` becomes `phrase_groups`
- inline `text_segments` becomes `phrase_groups`
- if only flat phrase groups exist, the loader constructs a single-section
  `Book`
- if a `Book` already exists, flat compatibility fields are synchronized from it

### Step 5: rewrite the project if migration occurred

If a legacy format was detected, the loader saves the project back in canonical
form.

This is what actually performs migration on disk.

---

## Migration cases in detail

### 1. Inline text in `project.json`

If legacy text was found inline in `project.json`, the app saves the project with
`force_phrase_groups=True`.

That causes the text payload to be written out to `project_text.json`, and the
project becomes normalized to the split-file v2 layout.

Effectively:

- old inline text is extracted from `project.json`
- `project_text.json` is written
- `project.json` remains the settings file

### 2. Legacy `project_text.json` payloads

If `project_text.json` exists but is not already `book.v1`, the loader converts
it and then saves it back as `book.v1`.

This covers:

- `phrase_groups.v1`
- bare phrase-group lists

### 3. Already using `book.v1`, but still carrying stale legacy fields

If text is already stored as `book.v1` but `project.json` still contains old
`applied_*` fields, the loader rewrites `project.json` to remove those stale
compatibility fields.

This keeps the persisted project shape canonical even if the project had already
partly migrated in an earlier app version.

---

## Canonical outcome after migration

After migration, the expected steady-state layout is:

- `project.json`
  - settings and metadata only
  - uses `markers` instead of `chapter_indices`
  - does not need stale legacy `applied_*` fields when the structured text file
    exists
- `project_text.json`
  - stored as `{"format": "book.v1", "book": ...}`

This split is the defining feature of project spec v2.

---

## Test coverage and migration examples

`tests/test_project_book_integration.py` verifies several migration scenarios.

### Tested behaviors

- legacy `phrase_groups.v1` `project_text.json` payloads are migrated to
  `book.v1`
- bare-list `project_text.json` payloads are migrated to `book.v1`
- legacy `chapter_indices` is rewritten as `markers`
- stale `applied_language_code`, `applied_strategy`, and `applied_max_words`
  are removed from canonical `project.json`
- projects loaded from legacy flat phrase groups still preserve flat
  compatibility behavior in memory

These tests are the best executable reference for expected migration behavior.

---

## Relation to EPUB import and newer book-aware workflows

The v2 project split aligns with newer book-aware workflows such as EPUB import.

For example, the EPUB architecture documentation notes that:

- segmented text is serialized as `project_text.json`
- chapter boundaries are serialized as `markers` in `project.json`
- the structured text model remains separate from the general settings model

This separation is part of why the v2 spec is useful: text/book structure can
evolve independently from the rest of the project settings payload.

---

## Bottom line

Project spec v2 changes the project from a mostly single-file format into a
split format:

- `project.json` for settings and metadata
- `project_text.json` for text/book content

Migration is automatic and load-driven:

- legacy inline text is extracted out of `project.json`
- old external text formats are upgraded to `book.v1`
- legacy compatibility fields are folded into the structured book model and then
  removed from canonical saved output where appropriate

This lets older projects continue to load while moving all saved projects toward
one consistent v2 layout.