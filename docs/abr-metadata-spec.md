# ABR Metadata Spec

## Overview

ABR metadata is the custom metadata payload embedded by `tts-audiobook-tool` into final exported audiobook files.

Its purpose is to let the browser player reconstruct:

- the timed text shown during playback
- embedded bookmark locations
- whether section-break audio was inserted
- optionally, the source raw text

It also carries export-time metadata for app compatibility, including:

- an ABR format `version`
- an optional `project_snapshot` that can be used for future project import/reconstruction flows

The payload itself is JSON. It is stored inside the audio container as a custom metadata field/tag.

In practice, an `*.abr.flac`, `*.abr.m4a`, or `*.abr.m4b` file is just a normal FLAC/MP4-family audio file with this extra embedded metadata.

---

## Where the metadata is written

`ConcatUtil.make_file()` is the main producer of ABR metadata.

At the end of concat/export, it builds an `AppMetadata` object using:

- `timed_phrases`: generated from the exported audio segment durations
- `raw_text`: loaded from the project text source
- `bookmark_indices`: user-provided bookmark indices, possibly adjusted after subdivision
- `has_section_break_audio`: whether section-break sound effects were included

That payload is then embedded into the final file:

- FLAC via `AppMetadata.save_to_flac()`
- AAC/MP4/M4B via `AppMetadata.save_to_mp4()`

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
  "version": 2,
  "raw_text": "<encoded representation of the original raw text>",
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

Backward compatibility rule:

- if `version` is missing, the file should be treated as ABR version `1`
- ABR version `1` means there is no `project_snapshot`

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
- `time_start`: segment start time in seconds
- `time_end`: segment end time in seconds

Notes:

- This is the only field the browser player currently treats as required.
- `text` is presentation text, not a normalized token stream.
- Segment timing is derived from concatenated export timing, not from container chapter metadata.
- Some segments may have `time_start = 0` and `time_end = 0` when no real audio timing exists for that item.

### `bookmarks` (optional, recommended)

Array of integers.

Each integer is an index into `text_segments`.

Semantics:

- bookmark `0` means the first text segment
- bookmark `12` means the thirteenth text segment

The browser player uses these as initial embedded bookmarks. They can seed the player bookmark state if no local bookmark state already exists.

If missing, consumers should treat it as an empty list.

### `raw_text` (legacy / optional in practice)

Semantically, this field is the original source text for the book.

Within the metadata payload, that text is stored in encoded form:

1. UTF-8 text
2. zlib-compressed bytes
3. base64url-encoded ASCII string

Important notes:

- This field is still written by the exporter.
- The browser player no longer requires it for normal operation.
- The browser player tolerates it being missing or empty.
- After decoding and decompressing, the consumer gets back the real raw source text.

Base64 details:

- URL-safe base64 alphabet is used
- `-` and `_` may appear in place of `+` and `/`

### `has_section_break_audio` (optional)

Boolean.

Semantics:

- `true` means the export inserted explicit section-break audio between some adjacent segments
- `false` means it did not

The browser player uses this to decide whether to add section-divider treatment in the displayed text flow.

If missing, consumers should treat it as `false`.

---

## How `text_segments` are produced

At export time:

1. the concat pipeline renders the selected phrase audio in order
2. segment durations are collected during concatenation
3. those durations are converted into timed phrases
4. if phrase subdivision is enabled, the timed phrases may be replaced with finer-grained segments based on adjacent JSON alignment metadata

So the ABR payload represents the final exported playback timeline, not just the original project phrase list.

This distinction matters because bookmark indices refer to the final `text_segments` array actually written into the file.

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
- `raw_text` being absent or empty
- `has_section_break_audio` defaulting to `false`
- `project_snapshot` defaulting to `{}`

### Recommended writer behavior

Writers producing ABR-compatible files should:

- write `version` explicitly as `2` for the current format
- always include `text_segments`
- ensure `bookmarks`, if present, contain valid indices into `text_segments`
- write `raw_text` using the existing compressed/base64url form for compatibility
- write `has_section_break_audio` explicitly as a boolean
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
- `raw_text`, if present, is a string in the encoded form described above
- `project_snapshot`, if present, is an object

The current browser-side parser is intentionally permissive in some areas, but new writers should follow the stricter interpretation above.

---

## Compatibility notes

- FLAC and MP4-family ABR files carry the same JSON payload, only the container-level tag location differs.
- The transcode flow preserves ABR metadata when converting an ABR FLAC file to AAC/M4A by copying the JSON payload into the MP4 custom tag.
- File naming such as `.abr.flac` or `.abr.m4b` is a project convention, not part of the metadata spec itself.
- Version 1 ABR files do not contain `project_snapshot`; missing `version` should be interpreted as version 1.

---

## Minimal encoded example

```json
{
  "version": 2,
  "raw_text": "eJzzSM3JyVcozy_KSVEEAB0JBF4=",
  "bookmarks": [0],
  "text_segments": [
    {
      "text": "Hello world.",
      "time_start": 0.0,
      "time_end": 1.42
    }
  ],
  "has_section_break_audio": false,
  "project_snapshot": {}
}
```

In that example, the `raw_text` value shown is not a hash and not arbitrary binary junk semantically; it is the encoded storage form of the original raw text.

If you decode base64url and then zlib-decompress it, you recover the actual source text.

---

## Summary

ABR metadata is:

- a JSON payload
- embedded into FLAC or MP4-family files using custom metadata storage
- centered around `text_segments`
- used primarily by the browser player for synchronized text and bookmarks
- extended in version 2 to also carry `project_snapshot` for possible future project import/recovery flows

The most important compatibility contract is the combination of:

- container tag location
- top-level JSON field names
- `text_segments` item structure
