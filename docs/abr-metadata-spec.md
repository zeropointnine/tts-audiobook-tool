# ABR Metadata Spec

## Overview

ABR metadata is the custom metadata payload embedded by `tts-audiobook-tool` into final exported audiobook files.

Its purpose is to let the browser player reconstruct:

- the timed text shown during playback
- embedded bookmark locations
- whether section-break audio was inserted

It also carries export-time metadata for app compatibility, including:

- an ABR format `version`
- an optional `project_snapshot` that can be used for future project import/reconstruction flows

The payload itself is JSON. It is stored inside the audio container as a custom metadata field/tag.

In practice, an `*.abr.flac`, `*.abr.m4a`, or `*.abr.m4b` file is just a normal FLAC/MP4-family audio file with this extra embedded metadata.

---

## Where the metadata is written

`ConcatUtil.make_file()` is the main producer of ABR metadata.

At the end of concat/export, it builds an `AppMetadata` object using:

- `timed_phrases`: the full audiobook text segment sequence, with generated/exported audio segment durations where playable audio exists and zero timings where playable audio does not exist
- `bookmark_indices`: user-provided bookmark indices, possibly adjusted after subdivision
- `has_section_break_audio`: whether section-break sound effects were included

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

### `version` (optional in stored data, required conceptually)

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

Object containing an export-time snapshot of project settings.

Semantics:

- this is not used directly for browser playback
- it is intended to support future flows that import settings from an ABR output file
- it is functionally based on the project's `project.json` data at export time
- phrase-group/text-body data is not included here

Important notes:

- unlike a purely portable settings bundle, this snapshot intentionally includes `dir_path`
- `dir_path` can serve as a best-effort hint for locating related files such as voice clone audio during a future import flow
- consumers should treat `dir_path` as optional and untrusted; it may not exist or may be invalid on another machine
- model-specific voice filename fields may therefore be meaningful in combination with `dir_path`

The exact field set is expected to track the app's current `project.json` settings shape, minus phrase group data.

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

- `true` means the export inserted explicit section-break audio between some adjacent segments
- `false` means it did not

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

If missing, consumers should treat it as an empty list.

---

## How `text_segments` are produced

At export time:

1. the concat pipeline renders the selected phrase audio in order
2. segment durations are collected during concatenation
3. those durations are converted into timed phrases
4. if phrase subdivision is enabled, the timed phrases may be replaced with finer-grained segments based on adjacent JSON alignment metadata

So the ABR payload carries the full text sequence plus the exported playback timeline where playable audio exists.

This distinction matters because bookmark indices refer to the full `text_segments` array actually written into the file, while playback position restoration depends on the subset of segments with positive-duration timing.

---

## Relationship to chapter metadata

ABR metadata is separate from audiobook chapter metadata.

- MP4/M4B chapter metadata is standard container metadata used by chapter-aware players
- ABR metadata is app-specific metadata used by the browser player

For AAC/M4B exports with bookmark-based chapters, the file may contain both:

- standard chapter metadata
- ABR metadata

They serve different purposes and should not be conflated.

---

## Consumer expectations

### Browser player expectations

The browser player currently expects:

- ABR `version` if present, otherwise it assumes version `1`
- a valid JSON payload
- a non-empty `text_segments` array

It also supports:

- `bookmarks` defaulting to `[]`
- legacy `raw_text` being absent, empty, or ignored
- `has_section_break_audio` defaulting to `false`
- `sections` defaulting to `[]`
- `project_snapshot` defaulting to `{}`

### Recommended writer behavior

Writers producing ABR-compatible files should:

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
- `version`, if present, is an integer >= 1
- `text_segments` exists and is a non-empty array
- every `text_segments` item has:
  - `text` as a string
  - `time_start` as a number
  - `time_end` as a number
- `bookmarks`, if present, is an array of integers
- `has_section_break_audio`, if present, is a boolean
- `sections`, if present, is an array of objects with string `title` and integer
  `start_index`/`end_index` values where `0 <= start_index <= end_index`
- `raw_text`, if present in legacy files, may be ignored by consumers
- `project_snapshot`, if present, is an object

The current browser-side parser is intentionally permissive in some areas, but new writers should follow the stricter interpretation above.

---

## Compatibility notes

- FLAC and MP4-family ABR files carry the same JSON payload, only the container-level tag location differs.
- The transcode flow preserves ABR metadata when converting an ABR FLAC file to AAC/M4A by copying the JSON payload into the MP4 custom tag.
- The optional `*.abr.metadata.json` debug sidecar is not part of the container format; it is only a developer/debug artifact mirroring the embedded payload.
- File naming such as `.abr.flac` or `.abr.m4b` is a project convention, not part of the metadata spec itself.
- Version 1 ABR files do not contain `project_snapshot`; missing `version` should be interpreted as version 1.
- Version 2 ABR files do not contain `sections`.
- The browser player's localStorage identity rules are documented separately in `docs/browser-player-identity.md`.

---

## Minimal example

```json
{
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
- centered around `text_segments`
- used primarily by the browser player for synchronized text and bookmarks
- extended in version 2 to also carry `project_snapshot` for possible future project import/recovery flows
- extended in version 3 to also carry structural `sections` overlay metadata
- optionally mirrored during concat into a standalone debug JSON sidecar for inspection

The most important compatibility contract is the combination of:

- container tag location
- top-level JSON field names
- `text_segments` item structure
