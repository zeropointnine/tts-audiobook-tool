# ABR Section Metadata Plan

## Purpose

This document proposes the next Python-side ABR metadata change needed to support the newer
`BookSection` direction.

The goal is **not** to change the browser reader implementation yet. The goal is to update
the ABR metadata format and Python export logic so exported ABR files can carry explicit
structural section information.

## Scope

Included in this plan:

- incrementing the ABR metadata version,
- documenting that version change at the constant,
- adding a new top-level ABR metadata field for structural sections,
- generating that new field on the Python export side,
- updating the ABR metadata spec,
- adding tests for serialization and index-mapping behavior.

Explicitly out of scope for this plan:

- browser-player consumption of the new field,
- browser backward-compatibility handling details,
- changing the player into multiple section-owned document blocks,
- flat/manual text import section inference,
- broader cleanup of `Reason.SPACE_BREAK` or `section_dividers` compatibility logic.

## Design Decision: Flat Timeline with Sections Overlay

The intended model is:

- ABR remains a **flat playback/readback timeline**,
- `text_segments` remains the canonical monolithic ordered array,
- `sections` becomes a **structural overlay** on top of that flat array,
- browser presentation is still intended to be one continuous scrolling document flow.

This means section metadata should **not** redefine ABR as a nested or per-section document
format.

The conceptual shape is:

```json
{
  "version": 3,
  "text_segments": [...],
  "bookmarks": [...],
  "sections": [...]
}
```

Under this model:

- **bookmarks** are user-created point markers into `text_segments`,
- **sections** are book-structural ranges over `text_segments`.

This preserves the current continuous-reader/playback model while enabling explicit section
navigation and visible section boundaries later.

## Current State

Current ABR metadata version:

- `tts_audiobook_tool/constants.py`
- `ABR_VERSION = 2`

Current metadata writers:

- `tts_audiobook_tool/concat_util.py`
  - main generated-audiobook ABR export
- `tts_audiobook_tool/enhance/enhance_flow.py`
  - enhance-existing-audiobook ABR export

Current metadata serialization/parsing:

- `tts_audiobook_tool/app_types/app_metadata.py`

Current top-level payload shape is effectively:

```json
{
  "version": 2,
  "raw_text": "...",
  "bookmarks": [0, 12, 31],
  "text_segments": [...],
  "has_section_break_audio": true,
  "project_snapshot": {}
}
```

## Proposed Version Bump

Bump ABR metadata version from `2` to `3`.

Recommended constant comment in `tts_audiobook_tool/constants.py`:

```python
# ABR metadata version history:
# - 1: original timed text/bookmark payload; missing version implies 1
# - 2: adds project_snapshot
# - 3: adds structural sections metadata for reader/player navigation
ABR_VERSION = 3
```

That keeps the version intent close to the canonical writer constant.

## Proposed New Top-Level Field

Add a new top-level field:

```json
"sections": [
  {
    "title": "Chapter 1",
    "start_index": 0,
    "end_index": 42
  }
]
```

### Semantics

- `sections` is an array of structural section descriptors.
- Each section maps a `BookSection` onto the final exported ABR `text_segments` array.
- `title` is the `BookSection.title` value and may be an empty string.
- `start_index` is inclusive.
- `end_index` is exclusive.

Using half-open ranges is recommended because:

- `Book.section_ranges()` already uses half-open ranges,
- it avoids ambiguity at adjacent boundaries,
- it works well with Python slicing-style mental models.

## Why an Overlay Instead of Nested Section Text Arrays

This plan intentionally keeps `text_segments` flat and global.

That means:

- playback timing remains attached to one canonical sequence,
- bookmarks remain simple indices into that same sequence,
- the browser can remain a single scrolling text flow,
- section navigation can jump into the same document/timeline,
- phrase subdivision only needs boundary remapping, not nested restructuring.

This avoids creating new complexity such as:

- nested per-section timing models,
- section-local vs global bookmark index ambiguity,
- more complicated cross-section playback behavior,
- duplicate storage of flat and nested text structures.

## Python Data Model Proposal

Current `AppMetadata` should be extended to include section metadata.

Preferred direction: add a small typed helper structure near `AppMetadata`, for example:

```python
class AppMetadataSection(NamedTuple):
    title: str
    start_index: int
    end_index: int
```

And then:

```python
class AppMetadata(NamedTuple):
    ...
    sections: list[AppMetadataSection]
```

Serialization should emit top-level JSON field `sections`.

Parsing should:

- accept missing `sections` as `[]` for older ABR files,
- validate the type/shape when `sections` is present,
- keep older ABR version compatibility intact.

## Main Export Logic Proposal

### Primary writer: `tts_audiobook_tool/concat_util.py`

`ConcatUtil.make_file(...)` should become the main producer of ABR `sections` metadata for
generated audiobook exports.

The section source should come from the structured project/book model, not from browser-side
heuristics.

Likely sources:

- `state.project.get_section_ranges()`
- `state.project.book.sections`

### Required behavior

The Python writer should:

1. derive structural sections from project `BookSection` data,
2. intersect those sections with the exported phrase-group range,
3. convert them into indices relative to the exported ABR `text_segments`,
4. serialize them into the new top-level `sections` field.

### Export-range behavior

This matters because ABR export does not always write the whole project.

Expected behavior:

- full-book exports include all overlapping sections,
- chapter/file exports include only sections that overlap the exported range,
- resulting `start_index` / `end_index` values are relative to the exported metadata payload,
  not global project phrase indices.

## Subdivision / Index Mapping Implications

This is the most important implementation detail.

When `state.project.subdivide_phrases` is enabled, `text_segments` may become finer-grained
than the original exported phrase-group list.

Current behavior already remaps bookmarks in:

- `make_subdivided_timed_phrases(...)`

Section metadata must follow the same principle.

### Consequence

Section boundaries cannot be defined only in original phrase-group coordinates when written
into ABR metadata. They must be translated into final `text_segments` coordinates.

### Recommended helper approach

Extend the subdivision pipeline so Python metadata generation can recover a mapping from each
original exported phrase index to the first resulting final `text_segments` index.

For example, evolve:

```python
def make_subdivided_timed_phrases(...) -> tuple[list[TimedPhrase], list[int]]
```

into something like:

```python
def make_subdivided_timed_phrases(...)
    -> tuple[list[TimedPhrase], list[int], list[int]]
```

Where the third return value maps:

- exported phrase index
- -> first generated text-segment index

Then a helper can convert structural section ranges into ABR section ranges.

For example:

```python
def make_app_metadata_sections(
    project: Project,
    index_start: int,
    index_end: int,
    phrase_to_text_segment_start_indices: list[int],
    text_segment_count: int,
) -> list[AppMetadataSection]:
    ...
```

This helper should produce section ranges in final ABR coordinates.

## Enhance-Existing-Audiobook Flow

`tts_audiobook_tool/enhance/enhance_flow.py` also writes ABR metadata, but it does not
currently originate from a structured ebook-imported `Book` with meaningful sections.

Recommended behavior for now:

- write version `3`,
- write `sections: []`.

That keeps the schema consistent without inventing section structure where none exists.

## ABR Spec Update Proposal

Update `docs/abr-metadata-spec.md` to reflect version `3`.

The spec changes should include:

1. top-level schema example updated to include `sections`,
2. `version` section updated so the current value is `3`,
3. version history expanded:
   - v1: base timed text/bookmark format
   - v2: adds `project_snapshot`
   - v3: adds `sections`
4. new field-definition section for `sections`,
5. validation rules for `sections`,
6. recommended writer behavior updated to say writers should emit version `3` and include
   `sections` when known.

The spec should also explicitly state that `sections` overlays `text_segments` and does not
turn ABR into a nested per-section document format.

## Testing Plan

Recommended tests:

### Serialization / parsing

1. `AppMetadata.to_json_string()` writes `sections`.
2. `AppMetadata.get_from_json_string()` accepts missing `sections` as `[]`.
3. malformed `sections` values fail validation.

### Range mapping

4. full export produces expected section ranges.
5. partial export trims/rebases section ranges correctly.
6. subdivision remaps section starts correctly when one phrase expands into multiple
   `text_segments`.

### Flow coverage

7. concat-generated ABR metadata includes structural sections.
8. enhance flow writes a valid v3 payload with empty `sections`.

## Deliverables

When implementing this plan later, expected code/documentation deliverables are:

- `tts_audiobook_tool/constants.py`
  - `ABR_VERSION` bumped and documented
- `tts_audiobook_tool/app_types/app_metadata.py`
  - new `sections` support in data model + serialization/parsing
- `tts_audiobook_tool/concat_util.py`
  - Python-side generation of section overlay metadata
- `tts_audiobook_tool/enhance/enhance_flow.py`
  - emit version `3` with empty sections list
- `docs/abr-metadata-spec.md`
  - schema/spec update
- tests covering serialization and index mapping

## Bottom Line

The correct model is:

- ABR stays flat,
- `text_segments` remains canonical,
- `sections` is a structural overlay using index ranges,
- future browser presentation can stay one continuous scrolling document,
- section navigation becomes a jump layer over that monolithic flow rather than a separate
  per-section reader document model.

That is the design direction this sub-plan is intended to preserve.