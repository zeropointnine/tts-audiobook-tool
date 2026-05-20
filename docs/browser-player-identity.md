# Browser Player Playback Identity

## Overview

The browser player persists local playback state in `localStorage`. Two persisted state types are tied to an opened ABR audiobook:

- resumed playback position (`audio.currentTime`)
- locally edited bookmark indices

These two state types do **not** have identical identity requirements, so the browser player derives two separate IDs from ABR metadata:

- `positionId`: strict identity for restoring playback position
- `textId`: document/text identity for bookmark storage

Older browser-player versions used a single `fileId` for both concerns. The current player intentionally does **not** match or recover state from those old localStorage keys because the old identity semantics were overloaded.

---

## ABR text/timeline model

ABR metadata `text_segments` represents the full audiobook text segment sequence. It is not limited to the currently rendered or exported audio span.

Segments with playable audio have positive duration:

```text
time_end > time_start
```

Segments without playable audio use zero timing:

```text
time_start = 0
time_end = 0
```

Zero-timed segments may be:

- outside the rendered/exported range
- missing generated audio
- non-verbal or formatting-related text

The browser player infers the playable audio span from the first and last positive-duration `text_segments` entries.

---

## Bookmark identity

Bookmarks are stored as indices into `text_segments`.

Therefore bookmark identity is based on the full text segment sequence, ignoring audio timing.

Conceptually:

```js
textId = hash({
  kind: "browser-player-bookmark-text-v1",
  texts: textSegments.map(segment => segment.text || "")
})
```

This means local bookmarks survive regenerated audio, changed durations, or partial/complete render changes as long as the text segment sequence remains stable.

---

## Position identity

Restoring `audio.currentTime` is stricter. The saved second offset should only be applied when it probably lands in the same audiobook neighborhood.

Position identity currently requires:

- the full text segment sequence to match
- inferred playable start index to match
- inferred playable end index to match
- rounded total playable timeline duration to match

Conceptually:

```js
positionId = hash({
  kind: "browser-player-position-timeline-v1",
  textIdentity,
  playableStartIndex,
  playableEndIndex,
  duration
})
```

`playableStartIndex` is the first index whose `time_end > time_start`.

`playableEndIndex` is exclusive: one past the last positive-duration segment index.

`duration` is the maximum positive `time_end`, rounded to two decimal places for identity stability.

Per-segment timings are intentionally not part of `positionId` yet. Including every timing would make regenerated audio with tiny timing changes lose resume continuity. The current compromise is strict about text and generated/playable span, and strict about total duration.

---

## localStorage keys

Current keys:

```text
abrPlayer:v3:position:${positionId}
abrPlayer:v3:bookmarks:${textId}
```

`v3` refers to the current ABR metadata/identity contract produced by the generation side. It is not dynamically based on the opened file's embedded `version` field; older ABR payloads are normalized into the current browser identity model.

Legacy keys are intentionally not read:

```text
fileId_${legacyFileId}
bookmarks_fileId_${legacyFileId}
```

This means old ABR files still load, but previously saved browser-player positions/bookmarks stored under the old `fileId` scheme start fresh under the new identity scheme.

---

## Hashing

New identity hashes use stable deep serialization before FNV-1a hashing, so nested object keys are sorted and nested object contents affect identity.

---

## Future metadata improvement

The browser currently infers playable range from timing data. If the app later needs exact export-range identity rather than playable-audio identity, ABR metadata should add explicit generated/export range fields, for example:

```json
{
  "generated_range": {
    "start_index": 0,
    "end_index": 123
  }
}
```

Until then, the inferred positive-duration range is the browser player's position-restore identity boundary.