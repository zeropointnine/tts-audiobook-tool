# ABR Metadata Spec

## Overview

ABR metadata is the custom metadata payload embedded by `tts-audiobook-tool` into final exported audiobook files.

Its purpose is to let the browser player reconstruct:

- the timed text shown during playback
- embedded bookmark locations
- whether section-break audio was inserted

It also carries export-time metadata for app compatibility, including:

- an ABR format `version`
- an optional `project_snapshot` used by the ABR-based new-project flow to import settings

The payload itself is JSON. It is stored inside the audio container as a custom metadata field/tag.

In practice, an `*.abr.flac`, `*.abr.m4a`, or `*.abr.m4b` file is just a normal
FLAC/MP4-family audio file with this extra metadata embedded.

---

## Where the metadata is written

There are two current producers of ABR metadata:

- `ConcatUtil.make_file()` writes metadata for normal project concat/export
- the enhance flow writes metadata when adding synchronized text to an existing audio file

At the end of concat/export, it builds an `AppMetadata` object using:

- `timed_phrases`: the full audiobook text segment sequence, with generated/exported audio segment durations where playable audio exists and zero timings where playable audio does not exist
- `bookmark_indices`: user-provided bookmark indices, possibly adjusted after subdivision
- `has_section_break_audio`: the project's break-sound-effect setting
- `title`: `project.book.title`
- `version`: the current `ABR_VERSION`
- `project_snapshot`: the serialized project settings, without book/text data
- `sections`: ranges derived from the project's structural book sections

The enhance flow builds the same payload shape from the supplied source text and an
alignment of that text against an existing audio file. It writes no bookmarks, sets
`has_section_break_audio` to `false`, uses an empty `project_snapshot`, and creates
`sections` only when the source import supplied section ranges (currently EPUB input).
Its title is the imported source title, falling back to the source audio filename stem.

That payload is then embedded into the final file:

- FLAC via `AppMetadata.save_to_flac()`
- AAC/MP4/M4B via `AppMetadata.save_to_mp4()`

During concat, the app can also write a standalone debug/dev JSON sidecar containing the same
payload shape that is embedded into the audio file.

- path pattern: `*.abr.metadata.json`
- location: the active timestamped subdirectory under `combined/`
- one sidecar is written per emitted ABR output file
- this happens only when `DEV` is enabled or `Prefs.save_debug_files` is `True`

The browser player reads the same payload through `browser_player/metadata-util.js`.

---

## Container-level storage

The JSON payload is stored differently depending on the container format.

### FLAC

Stored as a Vorbis comment field:

- field name: `TTS_AUDIOBOOK_TOOL`

### MP4 / M4A / M4B

Stored as a custom `----` atom with:

- mean: `tts-audiobook-tool`
- name/tag: `audiobook-data`

These identifiers are the stable container-level keys for the ABR payload.

---

## Payload format

The embedded value is a JSON object.

### Top-level schema

```json
{
  "title": "Example Book",
  "version": 3,
  "bookmarks": [0, 12, 31],
  "text_segments": [
    {
      "text": "Chapter 1",
      "time_start": 0.0,
      "time_end": 2.35
    },
    {
      "text": "It was a bright cold day in April.",
      "time_start": 2.35,
      "time_end": 6.92
    }
  ],
  "has_section_break_audio": true,
  "sections": [
    {
      "title": "Chapter 1",
      "start_index": 0,
      "end_index": 2
    }
  ],
  "project_snapshot": {
    "dir_path": "/abs/path/to/original/project",
    "version": 2,
    "language_code": "en"
  }
}
```

---

## Field definitions

### `title` (optional)

String book title.

Semantics:

- this is the top-level title of the imported/populated book
- it may come from EPUB metadata, a plain-text source filename, or be empty
- empty string is legal

Notes:

- this is separate from per-section `sections[].title`
- if missing, consumers should treat it as `""`

### `version` (optional in stored data, written by current producers)

Integer ABR metadata version.

Current value:

- `2`: includes `project_snapshot`
- `3`: includes structural `sections` metadata

Backward compatibility rule:

- if `version` is missing, the file should be treated as ABR version `1`
- ABR version `1` means there is no `project_snapshot`

Version `3` adds `sections`, which provide structural overlay ranges over the flat
`text_segments` array. They do not convert ABR into a nested per-section document model.

### `project_snapshot` (optional)

Object containing an export-time snapshot of project settings. Normal concat/export
writes the result of `ProjectSerializationUtil.to_snapshot_dict()`; the enhance flow
writes `{}` because it is not exporting a project.

Semantics:

- this is not used directly for browser playback
- it is used by the new-project flow to import settings from an ABR output file
- it is functionally based on the project's `project.json` data at export time
- book, phrase-group, and text-body data are not included here

Important notes:

- unlike a purely portable settings bundle, this snapshot intentionally includes `dir_path`
- `dir_path` can serve as a best-effort hint for locating related files such as voice clone audio during a future import flow
- consumers should treat `dir_path` as optional and untrusted; it may not exist or may be invalid on another machine
- model-specific voice filename fields may therefore be meaningful in combination with `dir_path`

The exact field set tracks `ProjectSerializationUtil.to_project_json_dict()`. It is a
settings snapshot rather than a complete project backup. In particular, the project
book and its sections/phrase groups are not part of this object; synchronized text and
section ranges are carried separately by `text_segments` and `sections`.

### `text_segments` (required)

Array of timed text items.

Each item has the form:

```json
{
  "text": "string",
  "time_start": 12.34,
  "time_end": 15.67
}
```

Semantics:

- `text`: text displayed by the player for that segment
- `time_start`: segment start time in seconds, or `0` when the segment has no playable audio in this ABR file
- `time_end`: segment end time in seconds, or `0` when the segment has no playable audio in this ABR file

Notes:

- This is the only field the browser player currently treats as required.
- `text` is presentation text, not a normalized token stream.
- `text_segments` represents the full audiobook text segment sequence, not only the rendered/exported audio span.
- Segment timing is derived from concatenated export timing, not from container chapter metadata.
- Some segments may have `time_start = 0` and `time_end = 0` when no playable audio exists for that item in this ABR file.
- Zero-timed segments can be outside the rendered/exported range, missing generated audio, or non-verbal/formatting-related text.

### `bookmarks` (optional, recommended)

Array of integers.

Each integer is an index into `text_segments`.

Semantics:

- bookmark `0` means the first text segment
- bookmark `12` means the thirteenth text segment

The browser player uses these as initial embedded bookmarks. They can seed the player bookmark state if no local bookmark state already exists.

If missing, consumers should treat it as an empty list.

### `raw_text` (legacy only)

Older ABR files may contain a `raw_text` field holding an encoded representation of the
original source text.

Important notes:

- Newer exporters no longer write this field.
- The browser player does not require it.
- Consumers should tolerate it being absent.
- Parsers may ignore it entirely when present.

### `has_section_break_audio` (optional)

Boolean.

Semantics:

- for normal concat/export, `true` means the project had break sound effects enabled
- the enhance flow always writes `false`

For concat exports, this value records the setting rather than auditing the rendered
audio for at least one effect. The audio pipeline only inserts an effect at eligible
boundaries and never appends one after the final playable segment. Consequently, a
`true` value does not guarantee that a particular file contains an audible effect.

The browser player uses this to decide whether to add section-divider treatment in the displayed text flow.

If missing, consumers should treat it as `false`.

### `sections` (optional)

Array of structural section descriptors.

Each item has the form:

```json
{
  "title": "Chapter 1",
  "start_index": 0,
  "end_index": 42
}
```

Semantics:

- `title`: human-readable section title, possibly empty
- `start_index`: inclusive index into `text_segments`
- `end_index`: exclusive index into `text_segments`

Important notes:

- `sections` is an overlay on the flat `text_segments` array
- it is intended for reader/player structure and navigation
- it does not imply a nested per-section text payload format
- current concat exports emit all project book sections, including split exports where
  sections outside the playable output range contain only zero-timed text segments
- the browser uses section ranges as separate text blocks only when the ranges are
  contiguous, non-empty, ordered, and together cover all of `text_segments`; otherwise
  text display falls back to one flat block
- the navigation panel exposes section navigation only when more than one valid section
  is present and targets the first playable segment in each section
- normal concat mirrors the project's section list without filtering empty sections, so
  a project containing an empty section can produce `start_index == end_index`; the
  browser retains that descriptor for navigation normalization but disables it when no
  playable segment exists, and flat text-block rendering falls back as described above

If missing, consumers should treat it as an empty list.

---

## How `text_segments` are produced

At export time:

1. the concat pipeline builds a phrase/path entry for every project phrase group
2. it renders playable audio in project order; entries outside the selected split-export
   range and entries with no generated audio keep a duration of zero
3. the complete duration list is converted into timed phrases
4. if phrase subdivision is enabled, the timed phrases may be replaced with finer-grained segments based on adjacent JSON alignment metadata

So the ABR payload carries the full text sequence plus the exported playback timeline where playable audio exists.

Subdivision is attempted only for entries with a sound path. If an entry has no sound,
its alignment sidecar is missing, or its sidecar cannot be parsed, the original segment
is retained. A bookmark on a subdivided phrase is moved to the first resulting segment.

This distinction matters because bookmark and section indices refer to the full
`text_segments` array actually written into the file, while playback position
restoration and navigability depend on the subset of segments with positive-duration
timing.

The enhance flow follows a different production path: it segments imported text and
aligns those segments to a transcription of an existing audio file. Unmatched segments
can likewise have zero timing. EPUB enhancement preserves imported section ranges;
plain-text enhancement currently writes no sections.

---

## Relationship to chapter metadata

ABR metadata is separate from audiobook chapter metadata.

- MP4/M4B chapter metadata is standard container metadata used by chapter-aware players
- ABR metadata is app-specific metadata used by the browser player

For AAC/M4B exports with multiple effective chapter sections, the file may contain both:

- standard chapter metadata
- ABR metadata

They serve different purposes and should not be conflated.

---

## Consumer expectations

### Browser player expectations

The browser player rejects the payload unless it is a JSON object with a non-empty
`text_segments` array. It otherwise normalizes fields permissively:

- `title` defaulting to `""`
- ABR `version` defaults to `1` when missing or not an integer greater than zero
- `bookmarks` defaulting to `[]`
- bookmark values are converted with JavaScript `parseInt`; values that do not produce
  an integer are discarded, and non-playable/out-of-range values are later discarded
  by the navigation panel
- legacy `raw_text` may be absent; when present, the player attempts URL-safe-base64
  decoding followed by zlib decompression and falls back to `""` on failure
- `has_section_break_audio` defaulting to `false`
- only the literal boolean `true` enables `has_section_break_audio`
- `sections` defaulting to `[]`; malformed or out-of-bounds section entries are dropped
- `project_snapshot` defaulting to `{}`

The browser does not currently validate the fields inside each `text_segments` item
before downstream code uses `text`, `time_start`, and `time_end`. Writers must therefore
still follow the stricter schema in this document.

### Python app reader expectations

`AppMetadata.get_from_json_string()` also requires a non-empty `text_segments` list. It:

- defaults missing `title`, `version`, `bookmarks`, `has_section_break_audio`,
  `project_snapshot`, and `sections`
- rejects an invalid `title`, `version`, top-level collection type,
  `project_snapshot`, or section descriptor
- converts bookmark values with Python `int()` but does not range-check them
- does not currently enforce a boolean type for `has_section_break_audio`
- validates section indices as non-negative integers with `end_index >= start_index`,
  but does not check `end_index` against the number of text segments

### Recommended writer behavior

Writers producing ABR-compatible files should:

- write `title` when known, using `""` when no title is available
- write `version` explicitly as `3` for the current format
- always include `text_segments`
- ensure `bookmarks`, if present, contain valid indices into `text_segments`
- write `has_section_break_audio` explicitly as a boolean
- write `sections` when structural section information is known
- write `project_snapshot` as the project settings snapshot when such data exists

---

## Validation rules

An ABR payload should satisfy the following:

- top-level value is a JSON object
- `title`, if present, is a string and may be empty
- `version`, if present, is an integer >= 1
- `text_segments` exists and is a non-empty array
- every `text_segments` item has:
  - `text` as a string
  - `time_start` as a number
  - `time_end` as a number
- `bookmarks`, if present, is an array of integers
- `has_section_break_audio`, if present, is a boolean
- `sections`, if present, is an array of objects with string `title` and integer
  `start_index`/`end_index` values where
  `0 <= start_index < end_index <= text_segments.length`
- `raw_text`, if present in legacy files, may be ignored by consumers
- `project_snapshot`, if present, is an object

For full browser text-block support, section ranges should additionally be ordered,
contiguous, non-overlapping, non-empty, and cover the complete `text_segments` array.
The normal concat producer can preserve empty project sections as empty ranges, while
the enhance producer filters empty ranges. The current browser-side and Python parsers
are intentionally permissive in different areas, but new third-party writers should
follow the stricter interpretation above.

---

## Compatibility notes

- FLAC and MP4-family ABR files carry the same JSON payload, only the container-level tag location differs.
- When loudness normalization converts a FLAC input containing valid ABR metadata to an
  AAC-family output, it reloads that payload from the FLAC and writes it to the MP4
  custom tag. FLAC-to-FLAC normalization relies on FFmpeg preserving the FLAC metadata.
- The optional `*.abr.metadata.json` debug sidecar is not part of the container format; it is only a developer/debug artifact mirroring the embedded payload.
- File naming such as `.abr.flac` or `.abr.m4b` is a project convention, not part of the metadata spec itself.
- Version 1 ABR files do not contain `project_snapshot`; missing `version` should be interpreted as version 1.
- Version 2 ABR files do not contain `sections`.
- The browser player's localStorage identity rules are documented separately in `docs/browser-player-identity.md`.

---

## Minimal example

```json
{
  "title": "Hello World",
  "version": 3,
  "bookmarks": [0],
  "text_segments": [
    {
      "text": "Hello world.",
      "time_start": 0.0,
      "time_end": 1.42
    }
  ],
  "has_section_break_audio": false,
  "sections": [],
  "project_snapshot": {}
}
```

---

## Summary

ABR metadata is:

- a JSON payload
- embedded into FLAC or MP4-family files using custom metadata storage
- optionally carrying a top-level book `title`
- centered around `text_segments`
- used primarily by the browser player for synchronized text and bookmarks
- extended in version 2 to also carry `project_snapshot` for project settings import
- extended in version 3 to also carry structural `sections` overlay metadata
- optionally mirrored during concat into a standalone debug JSON sidecar for inspection

The most important compatibility contract is the combination of:

- container tag location
- top-level JSON field names
- `text_segments` item structure
