# Project Spec v2

Last updated: 2026-08-14

## Purpose

This document describes:

- what changed in **project spec version 2**, and
- how the app migrates older projects to the current format.

The current project spec version is defined in `tts_audiobook_tool/constants.py`:

```python
PROJECT_SPEC_VERSION = 2
```

The behavior described here is implemented primarily in:

- `tts_audiobook_tool/project.py` — `Project` model and in-memory normalization
- `tts_audiobook_tool/project_support/project_load_util.py` — load/migration
  orchestration (`ProjectLoadUtil.load_using_dir_path`, `remap_legacy_keys`)
- `tts_audiobook_tool/project_support/project_serialization_util.py` — loaded-dict
  normalization (`normalize_loaded_project_dict`) and canonical settings
  serialization (`to_project_json_dict`)
- `tts_audiobook_tool/project_support/project_text_io_util.py` —
  `project_text.json` persistence (`ProjectTextIOUtil.save_book`)
- `tts_audiobook_tool/project_support/project_book_util.py` — book ↔ flat
  compatibility bridging (`ProjectBookUtil`)
- `tts_audiobook_tool/app_types/book_serialization.py`
- `tts_audiobook_tool/app_types/__init__.py` — `Book`, `BookSection`,
  `BookSegmentationSettings`, `VoiceSelectMode`, and other shared structural
  types
- `tts_audiobook_tool/app_support/JsonSaveUtil.py` — atomic, locked persistence
  of the JSON artifacts
- `tts_audiobook_tool/project_support/project_transfer_util.py` — supporting
  project files (raw text, imported EPUB) during clone/transfer
- `tests/test_project_book_integration.py`

`ProjectUtil` (`tts_audiobook_tool/project_support/project_util.py`) remains the
public entry point, but its load/migration methods now delegate to
`ProjectLoadUtil`.

---

## Summary of the v2 changes

Project spec v2 introduced a **storage split**: project settings remain in
`project.json`, while book text is stored in `project_text.json`.

The current `project_text.json` format is `book.v2`. It adds a `voice_index`
to each phrase group so different groups can be associated with different
entries of a model's voice-clone sample list. The default voice index is `-1`.

This field is the storage-side counterpart of the `voice_select_mode` project
setting (`VoiceSelectMode` in `tts_audiobook_tool/app_types/__init__.py`):

- `user_defined` — each phrase group's `voice_index` selects its voice sample
- `auto_advance` — the sample cycles in order on each batch generation
- `disabled` — the first sample is used for every generation

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

- `dir_path`
- `version`
- `language_code`
- segmentation options such as `segmentation_strategy` and `max_words`
- generation options such as `generate_range`
- section markers in `markers`
- export and post-processing settings
- `voice_select_mode`
- model-specific settings for **every** supported TTS backend (not just the
  active one); `to_project_json_dict` always serializes all model blocks

It does **not** canonically store the full phrase-group text payload anymore.

### `project_text.json`

Under v2, `project_text.json` is the canonical text payload file.

The preferred current format is:

```json
{
    "format": "book.v2",
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
                "phrase_groups": [
                    {
                        "voice_index": -1,
                        "phrases": [
                            {
                                "text": "...",
                                "reason": "s"
                            }
                        ]
                    }
                ]
            }
        ]
    }
}
```

This format is produced by `book_to_project_text_json_dict(...)` in
`tts_audiobook_tool/app_types/book_serialization.py`.

### Optional sidecar text files

Projects may also carry additional text-related files in the project
directory. They are not part of the canonical two-file split, but they are
persisted alongside it and are treated as "supporting project files" by
`ProjectTransferUtil` when a project is cloned or transferred:

- `project_text_raw.txt` — the raw source text before segmentation. Written by
  `ProjectTextIOUtil.set_phrase_groups_and_save` /
  `set_phrase_groups_chapters_and_save`, readable via
  `ProjectTextIOUtil.load_raw_text`
- `project_text.epub` — a copy of the imported EPUB, written by
  `EpubExtractor.copy_epub_to_project`

---

## Supported text payload formats

Even though `book.v2` is the canonical saved format, the loader still accepts
older payloads for backward compatibility.

### Current canonical format

- `book.v2`

Example shape:

```json
{
    "format": "book.v2",
    "book": { ... }
}
```

### Older accepted formats

- `book.v1`
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
canonical `book.v2` representation. Older phrase groups did not have a speaker
index and therefore load with the default voice index of `-1`.

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
saved shape once `book.v2` text data is available.

### Voice-clone fields use hybrid string/list serialization

Multi-voice-sample support changed how voice-clone fields are stored in
`project.json`. In memory every voice-clone field is a `list[str]`, but on save
the fields serialize to the legacy string shape when zero or one sample is
configured, and to a JSON list only when multiple samples exist:

- 0 samples → `""`
- 1 sample → `"sample.flac"` (string, legacy shape)
- 2+ samples → `["one.flac", "two.flac"]`

This is implemented in
`ProjectSerializationUtil.serialize_voice_list_value(...)`, with the
applicable fields listed in `ProjectSerializationUtil.VOICE_LIST_FIELD_ALIASES`.
On load, `normalize_voice_list_value(...)` coerces either shape back to a list,
and legacy alias keys (e.g. `fish_s1_voice_text` for
`fish_s1_voice_transcript`) are normalized in the same pass.

The hybrid shape keeps old projects compatible while allowing new projects to
store multiple voice-clone samples per model.

### Some model keys are remapped on load

`ProjectUtil.remap_legacy_keys(...)` (implemented in
`ProjectLoadUtil.remap_legacy_keys`) normalizes some older field names, for
example:

- `fish_voice_file_name` → `fish_s1_voice_file_name`
- `fish_voice_text` → `fish_s1_voice_text`
- `fish_temperature` → `fish_s1_temperature`
- `fish_seed` → `fish_s1_seed`
- `higgs_v3_voice_text` → `higgs_v3_voice_transcript`
- `vibevoice_model_path` → `vibevoice_target`
- `qwen3_path_or_id` → `qwen3_target`
- `indextts2_emo_voice_alpha` → `indextts2_emo_alpha` when applicable

MOSS flat sampling fields are also split during normalization (in
`ProjectSerializationUtil.normalize_loaded_project_dict` rather than
`remap_legacy_keys`):

- `moss_temperature` → `moss_delay_temperature`
- `moss_top_p` → `moss_delay_top_p`
- `moss_top_k` → `moss_delay_top_k`

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

Flat phrase-group access is still preserved for older flows:

- `phrase_groups` is a `Project` property backed by the book's phrase groups
- `ProjectBookUtil.get_flat_phrase_groups(project)` returns the flat
  phrase-group list
- `ProjectBookUtil.get_section_start_indices(project)` returns the section
  start indices (from `book` when present, falling back to `markers`)

This allows older flows to remain functional while the canonical persisted shape
uses structured book text.

---

## Save mechanics: explicit save via JsonSaveUtil

Project changes remain in memory until `save()` is called explicitly; there is
no auto-save. `project.save()` and `ProjectTextIOUtil.save_book` both persist
through `JsonSaveUtil.save(...)` in
`tts_audiobook_tool/app_support/JsonSaveUtil.py`, which:

- serializes the full payload in memory before touching disk, so a failed
  payload build can never truncate a valid file
- writes to a uniquely named temp sibling, flushes, and `fsync`s before
  atomically replacing the destination via `os.replace`
- holds a reentrant, artifact-specific lock (`JsonArtifactType.PROJECT` vs
  `PROJECT_TEXT`) so concurrent saves of the same artifact cannot interleave

`Project.save()` also normalizes `version` to the current
`PROJECT_SPEC_VERSION` and coerces an invalid `chapter_mode` while holding the
save lock, so every rewrite produces a canonical `project.json`.

---

## Migration flow

Project migration is handled automatically during project load.

The main entry point is:

```python
ProjectUtil.load_using_dir_path(dir_path)
```

`ProjectUtil.load_using_dir_path` delegates to
`ProjectLoadUtil.load_using_dir_path` in
`tts_audiobook_tool/project_support/project_load_util.py`. That method,
together with `Project.model_validate(...)`, performs detection,
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

- `book.v2`
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

If legacy text was found inline in `project.json`,
`ProjectLoadUtil.load_using_dir_path` calls `ProjectTextIOUtil.save_book(project)`
followed by `project.save()`.

That writes the text payload out to `project_text.json` and re-saves
`project.json` without the inline text, normalizing the project to the
split-file v2 layout.

Effectively:

- old inline text is extracted from `project.json`
- `project_text.json` is written
- `project.json` remains the settings file

### 2. Legacy `project_text.json` payloads

If `project_text.json` exists but is not already `book.v2`, the loader converts
it and then saves it back as `book.v2`.

This covers:

- `book.v1`
- `phrase_groups.v1`
- bare phrase-group lists

### 3. Already using `book.v2`, but still carrying stale legacy fields

If text is already stored as `book.v2` but `project.json` still contains old
`applied_*` fields, the loader re-saves the project. The fields are not removed
by any explicit deletion: `ProjectSerializationUtil.to_project_json_dict`
simply never serializes them, so the rewritten `project.json` no longer
contains them.

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
  - stored as `{"format": "book.v2", "book": ...}`

This split is the defining feature of project spec v2.

---

## Test coverage and migration examples

`tests/test_project_book_integration.py` verifies several migration scenarios.

### Tested behaviors

- legacy `book.v1` and `phrase_groups.v1` `project_text.json` payloads are
  migrated to `book.v2`
- bare-list `project_text.json` payloads are migrated to `book.v2`
- legacy `chapter_indices` is rewritten as `markers`
- stale `applied_language_code`, `applied_strategy`, and `applied_max_words`
  are removed from canonical `project.json`
- projects loaded from legacy flat phrase groups still preserve flat
  compatibility behavior in memory
- legacy single-string voice values load as lists and round-trip back to the
  string shape, while multi-sample voice lists serialize as JSON lists
- `voice_select_mode` and per-model voice-clone settings normalize on load and
  serialize on save

These tests are the best executable reference for expected migration behavior.

---

## Relation to EPUB import and newer book-aware workflows

The v2 project split aligns with newer book-aware workflows such as EPUB import.

For example, the EPUB architecture documentation notes that:

- segmented text is serialized as `project_text.json`
- chapter boundaries are serialized as `markers` in `project.json`
- the structured text model remains separate from the general settings model
- the raw source text (`project_text_raw.txt`) and the imported EPUB itself
  (`project_text.epub`) are kept in the project directory as sidecar files

This separation is part of why the v2 spec is useful: text/book structure can
evolve independently from the rest of the project settings payload.

---

## Bottom line

Project spec v2 changes the project from a mostly single-file format into a
split format:

- `project.json` for settings and metadata (including multi-sample voice-clone
  lists and `voice_select_mode`)
- `project_text.json` for text/book content (with per-phrase-group
  `voice_index`)
- optional sidecars: `project_text_raw.txt` and `project_text.epub`

Migration is automatic and load-driven:

- legacy inline text is extracted out of `project.json`
- old external text formats are upgraded to `book.v2`
- legacy compatibility fields are folded into the structured book model and then
  removed from canonical saved output where appropriate

This lets older projects continue to load while moving all saved projects toward
one consistent v2 layout.
