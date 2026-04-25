# Architecture: Sound Segment File Management and Dynamic Failure Detection

## Document Purpose

This document describes:

1. How the app saves generated audio files to the local project directory, including the filename schema that encodes metadata.
2. How word error counts are embedded in sound segment filenames.
3. How the app determines whether a sound segment qualifies for "re-generation".

---

## Architecture

### 1. Where Sound Segment Files Are Saved

Generated audio for each text segment is saved as a `.flac` file in the project's `sound_segments/` subdirectory.

- **Directory path**: `{project.dir_path}/sound_segments/` (the subdirectory name is defined by the constant `PROJECT_SOUND_SEGMENTS_SUBDIR`).
- **Project model** (`project.py`): exposes this path via the `sound_segments_path` property.
- **Directory watcher** (`ProjectSoundSegments.__init__` in `project_sound_segments.py`): uses `watchdog` to monitor this directory for file creation/deletion/move events. When a `.flac` file that parses as a valid `SoundSegment` is added or removed, the in-memory catalog is marked dirty and rebuilt on next access.
- **Saving** (`GenerateUtil.save_sound_and_timing_json` in `generate_util.py`, line 463): calls `SoundFileUtil.save_flac()` to write the `.flac`, and optionally writes a parallel `.json` file containing timing/alignment metadata (generated via `ForceAlignUtil`).
- **Parallel files**: For each `[tags]...text.flac`, there can be a `[tags]...text.json` (timing data) and a `[tags]...text.debug.json` (debug info, if debug saving is enabled).

### 2. Sound Segment Filename Schema

Filenames are constructed by `SoundSegmentUtil.make_file_name()` (`sound_segment_util.py`, line 60). The format uses consecutive bracketed tags at the start of the filename:

**Normal (non-real-time) pattern:**
```
[{idx}] [{hash}] [{model}] [{voice}][{num_errors}]{text_preview}.flac
```

**Real-time pattern:**
```
[{timestamp}] [{idx}] [{model}] [{voice}][{num_errors}]{text_preview}.flac
```

**Tag descriptions:**

| Tag | Example | Description |
|-----|---------|-------------|
| `[idx]` | `[00024]` | 1-based index, zero-padded to 5 digits |
| `[hash]` | `[3ae0f21b9de65a3c]` | Hash of `"{index} {segment_text}"` — used to detect if source text has changed since generation |
| `[model]` | `[vibevoice]` | TTS model file tag (e.g., `chatterbox`, `oute`, `fish_s1`, `higgs`, `mira`, `qwen3tts`, `pocket`, `vibevoice`, `indextts2`, `glm`) |
| `[voice]` | `[sy_even_if_ch1_c]` | Voice tag derived from the voice file name used for cloning |
| `[{num_errors}]` | `[5]` | Number of word errors detected during validation (optional — only present for `WordErrorResult` and `MusicFailResult`) |
| `{text_preview}` | `With_Lord_knows_what_s_beyond` | First ~50 characters of the segment text, sanitized for filesystem safety |

**Example filename:**
```
[00024] [3ae0f21b9de65a3c] [vibevoice] [sy_even_if_ch1_c] [5] With_Lord_knows_what_s_beyond.flac
```

### 3. How Word Error Counts Are Encoded

In `SoundSegmentUtil.make_file_name()`:

```python
if isinstance(validation_result, WordErrorResult):
    num_fails_tag = f" [{validation_result.num_errors}]" 
elif isinstance(validation_result, MusicFailResult):
    num_fails_tag = f" [99]"  # sentinel value for music fails
else:
    num_fails_tag = ""
```

- **WordErrorResult**: The actual word error count is written (e.g., `[5]`).
- **MusicFailResult**: A sentinel value of `[99]` is used (since music detection failure has no word error count).
- **TrimmedResult / SkippedResult**: No error count tag is written (these are not failures).

**Parsing back** (`SoundSegment.from_file_name()`):

```python
num_errors = int(tags[4]) if len(tags) >= 5 and tags[4].isdigit() else -1
```

- `num_errors` defaults to `-1` if not present or not parseable.
- The `SoundSegment` NamedTuple carries: `file_name`, `idx`, `hash`, `num_errors`, `model`, `voice` — no `is_fail` field.

### 4. How the App Determines If a Segment Qualifies for Re-generation

**File eligibility for inclusion in the catalog** (`SoundSegmentUtil.make_sound_segments_map()`, line 22):

A `.flac` file is only included in the sound segments map if:

1. It has a valid `SoundSegment` structure (parsable tags).
2. Its index is within bounds of the current `phrase_groups` list.
3. Its `hash` matches the current hash computed from `"{index} {current_text}"`.

Requirement #3 means that if the source text for a segment changes, all previously generated files for that index become invisible to the catalog — they are effectively orphaned.

**Dynamic failure detection:**

The method `ProjectSoundSegments.is_segment_failed(index, item)` determines whether a sound segment is considered "failed" based on its `num_errors` and the project's current "Word error tolerance" setting:

```python
def is_segment_failed(self, index: int, item: SoundSegment) -> bool:
    if item.num_errors == -1:
        return False  # Unknown error count — treat as not-failed
    if item.num_errors == 99:
        return True   # Music-fail sentinel — always considered failed
    phrase_group = self.project.phrase_groups[index]
    normalized_source = TextNormalizer.normalize_source(
        phrase_group.text, self.project.language_code)
    num_words = TextUtil.get_word_count(
        normalized_source, vocalizable_only=True)
    threshold = ValidateUtil.compute_threshold(
        num_words, self.project.strictness)
    return item.num_errors > threshold
```

`ProjectSoundSegments.get_failed_indices_in_generate_range()` uses this to find all indices that need re-generation:

```python
def get_failed_indices_in_generate_range(self) -> set[int]:
    all_indices = self.project.get_indices_to_generate()
    failed_indices = set()
    for index in all_indices:
        items = self.sound_segments_map.get(index, [])
        if not items:
            continue
        for item in items:
            if self.is_segment_failed(index, item):
                failed_indices.add(index)
                break
    return failed_indices
```

Key behaviors:

- Failure is computed **dynamically** against the project's current "Word error tolerance" setting — changing tolerance can change which segments are considered failed without touching any files.
- If there are multiple files for the same index (multiple generation attempts), the index is considered "failed" if **any** of them exceeds the threshold.
- `get_best_item_for()` selects the file with the lowest `num_errors` for each index. A value of `-1` is treated as 9999 (worst possible) for ranking purposes.
- `get_word_error_counts_in_generate_range()` returns the `num_errors` for existing items — used during regen to track whether a retry improved the error count.
- `delete_redundants_for()` keeps only the best (lowest-error) file per index and deletes the rest.

**Threshold computation** (`ValidateUtil.compute_threshold()`, `validate_util.py`):

| Tolerance | Formula | Example (10-word segment) |
|-----------|---------|--------------------------|
| Low | `ceil(num_words / 10) + 1` | 2 errors allowed |
| Moderate | `ceil(num_words / 10)` | 1 error allowed |
| High | `ceil(num_words / 10) - 1` | 0 errors allowed |
| Intolerant | `0` | 0 errors always |

This method is shared between `get_word_error_fail()` (used during live generation) and `is_segment_failed()` (used for regen detection of existing files), ensuring consistent threshold logic in both paths.

---

## Implemented Changes

The following changes were made as of commit `c9d0266` and the subsequent copy-update session:

### Dynamic failure detection (replacing static `[fail]` tag)

| Location | Change |
|----------|--------|
| `SoundSegmentUtil.make_file_name()` | `fail_tag` variable and `[fail]` insertion removed |
| `SoundSegment.from_file_name()` | `is_fail` field removed; `"[fail]" in file_name` parsing removed |
| `SoundSegment` NamedTuple | `is_fail` field eliminated entirely |
| `ValidateUtil` | `compute_threshold(num_words, strictness)` extracted as standalone static method |
| `ProjectSoundSegments` | `is_segment_failed(index, item)` added; uses `num_errors` + project tolerance dynamically |
| `ProjectSoundSegments.get_failed_indices_in_generate_range()` | Uses `self.is_segment_failed()` instead of `item.is_fail` |

### UI copy ("Word error tolerance" rename and regen copy cleanup)

| Location | Change |
|----------|--------|
| `generate_menu.py` — `make_tolerance_label()` | "Transcript validation strictness" / "Validation strictness" → "Word error tolerance" |
| `generate_menu.py` — `make_regen_label()` | "Regenerate failed segments" → "Re-generate segments with errors"; `@ {level}` → `tolerance: {level}` |
| `generate_menu.py` — `make_retries_label()` | "Transcript validation max retries" → "Re-generation max retries" |
| `generate_menu.py` — `STRICTNESS_DESC` | Rewritten without jargon ("retry-on-generation", "regen-detection") |
| `generate_menu.py` — strictness option descriptions | HIGH/INTOLERANT descriptions rewritten; "regen list" / "flagged for regen" → "flagged for re-generation" |
| `generate_menu.py` — no-failures feedback | "No failed items to regenerate" → "No segments with errors to re-generate" |
| `generate_util.py` — generation summary | "tagged as failed" → "with word errors" |
| `constants.py` — `HINT_REGEN` | Title and body rewritten; "tagged as failed/having excessive errors" removed |
| `constants.py` — `HINT_FORCED_STRICTNESS_LOW` | Setting name updated to "Word error tolerance" |

### Migration path for legacy files

- Old files with `[fail]` tag but **without** `[num_errors]`: parsed with `num_errors == -1`, treated as not-failed (conservative default).
- Old files with `[fail]` tag **and** `[num_errors]`: the `[fail]` tag is simply ignored; failure is computed from the error count alone.
- Newly generated files: no `[fail]` tag — failure is computed dynamically from `[num_errors]` and project tolerance.
- Over time, as segments are regenerated, old `[fail]`-tagged files will be replaced by new ones. `delete_redundants_for()` helps accelerate this naturally.

---

## Open Questions and Remaining Work

### 1. `num_errors == 99` music-fail sentinel ambiguity

A segment with exactly 99 genuine word errors is indistinguishable from a music-fail. In practice this is extremely unlikely given typical segment lengths, but it is not formally resolved. Options:
- Accept the ambiguity (current approach).
- Replace `[99]` with a dedicated tag like `[music]` — would require a parsing change and migration consideration for existing `[99]` files.

### 2. `-1` / `9999` asymmetry for legacy files

`is_segment_failed()` treats `num_errors == -1` as **not-failed** (conservative). `get_best_item_for()` treats `-1` as **9999** (worst possible rank). A file with an unknown error count is therefore safe from re-generation but ranks last when multiple candidates exist for the same index. This asymmetry is intentional but could be surprising — worth a comment at the call sites.

### 3. Internal code still uses "strictness" naming

The `Strictness` enum, `project.strictness` attribute, and `GenerateMenu.strictness_menu()` method all use "strictness" internally while the UI now says "Word error tolerance". A full rename of internal identifiers (`project.strictness` → `project.word_error_tolerance`, `Strictness` enum → `WordErrorTolerance`) would affect serialization and many call sites — worth deciding whether to pursue for consistency or leave internal naming as-is.

### 4. `HIGH` threshold for zero-word segments

Fixed: `compute_threshold()` for `HIGH` now returns `max(0, ceil(num_words / 10) - 1)`, preventing a negative threshold for zero-word segments.
