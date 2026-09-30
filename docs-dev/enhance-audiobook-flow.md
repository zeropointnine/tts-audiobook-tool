# Enhance a pre-existing audiobook — UI flow, business rules, and judgement calls

This document describes the "Enhance a pre-existing audiobook" feature: its
dataflow, menu presentation, artifact lifecycle, and the deliberate judgement
calls behind behaviors that are not fully straightforward.

Relevant code:

- `tts_audiobook_tool/enhance/enhance_artifacts.py` — artifacts, state snapshot, caching
- `tts_audiobook_tool/enhance/enhance_menu.py` — menu presentation
- `tts_audiobook_tool/enhance/enhance_flow.py` — user-triggered operations
- Tests: `tests/test_enhance_artifacts.py`, `tests/test_enhance_menu.py`, `tests/test_enhance_flow.py`

## Overview

The feature combines the audio of a pre-existing audiobook with its original
source text to produce an `.abr.m4a` / `.abr.m4b` file playable by the
tts-audiobook-tool player/reader, just like TTS audiobooks created by the main
app.

## Dataflow: two input tracks converging on one output

The pipeline is not a linear checklist. It is two independent input tracks that
converge on the output step:

```
Enter audio ──▶ Transcribe ──┐
                             ├─▶ Create output
Enter text ────▶ Align ──────┘
```

- The **audio track** produces the transcription. Transcription (Whisper) runs
  on the waveform alone; the source text plays no part.
- The **text track** produces the alignment. Alignment consumes *both* the
  imported book and the transcription, so it cannot complete until the audio
  track has.
- **Create** consumes the book (title, section structure, output-suffix choice)
  plus the timed phrases, and writes the enhanced audiobook.

The menu items are ordered and visually grouped to match: (enter audio,
transcribe), blank line, (enter text, align), blank line, create. The blank
lines are static presentation (`MenuItem.blank_line_before`) applied on every
render.

## Artifacts

All artifacts are siblings of the selected audio file, named by its stem:

| Artifact | File | Produced by | Consumed by |
| --- | --- | --- | --- |
| Audio selection | *(user's file)* | Enter audio | Transcribe, Create |
| Book | `<stem>.abr.json` | Enter text | Align, Create |
| Transcription | `<stem>.transcription.bin` | Transcribe | Align |
| Timed phrases | `<stem>.timed_phrases.bin` | Align | Create |
| Output | `<stem>.abr.m4a` or `.abr.m4b` | Create | Review |

The book is a JSON file (`BOOK_FORMAT`); the transcription and timed-phrase
caches are pickles written via atomic replace. The output suffix depends on the
book's `text_source_kind`: `plain_text` → `.abr.m4a`, `epub` → `.abr.m4b`.

## Menu presentation rules

The menu is rendered from one defensive filesystem snapshot per redraw
(`make_enhance_state_cached`). Operations revalidate before doing work, so the
menu never acts on stale data — it can only *display* stale data at worst, and
the cache's stat signature makes even that a single-redraw window in practice.

Checkbox semantics per item:

| Item | Checked when |
| --- | --- |
| Enter source audiobook file path | audio file exists |
| Transcribe | transcription exists **and** parses as valid words |
| Enter text or EPUB file path | book artifact exists **and** parses as a valid `pre_existing`-audio book |
| Align source text with transcription | timed phrases exist, parse as valid, **and** the book is present |
| Create the output | output file exists **and** its embedded metadata reads back validly |

Status annotations:

- The audio item shows a terminal hyperlink to the file when it exists.
- The text item shows the flattened line count ("(currently: N lines)").
- The create item hyperlinks the expected output path when it exists.

Optional items (under an "Options" superlabel, only when relevant):

- **Review unmatched lines** — appears only when the output is valid; its
  counts describe the phrases embedded in the *output file*, not the current
  book, because the output may predate the latest alignment.
- **Clear** — appears when an audio path is selected.

Other behaviors:

- A stale audio selection (file no longer exists) is dropped and saved on menu
  entry.
- Re-entering text deletes the timed-phrases cache, because alignment is
  book-specific (see judgement calls).

## Flow rules

- **Enter audio**: normalizes and saves the selection; inspects the file's
  metadata and warns if it already carries tts-audiobook-tool metadata. If a
  parallel enhanced audiobook (`.abr.m4a`/`.abr.m4b`) already exists next to
  the selected audio — e.g. after Clear, which keeps the output — a
  non-blocking notice points at it; selection is never blocked.
- **Enter text**: validates `.txt`/`.epub`, imports the book, saves the book
  artifact, and clears any prior alignment cache. Overwriting an existing book
  asks for confirmation.
- **Transcribe**: requires only the audio selection. When a book is also
  present, it offers to chain into alignment + create when finished; with no
  book entered yet, that chain question is skipped (it needs the book for its
  output suffix, and alignment requires the book anyway).
- **Align**: requires the book and a valid transcription; can optionally chain
  straight into Create (`create_when_finished`). Interruption preserves any
  prior alignment cache untouched.
- **Create**: requires the book (title, sections, suffix) and a valid,
  non-empty alignment. Writes via staging file + atomic replace, with metadata
  read-back verification before replacing the destination. Replacing an
  existing output asks for confirmation.
- **Clear**: deletes only the three work files (book, transcription, timed
  phrases) — never the output — and clears the selection. Reselecting the same
  audio later rediscovers any retained work files.

## Judgement calls

These behaviors are deliberate deviations from what a naive reading of the
pipeline might suggest. They are presentation- or policy-level decisions, not
accidents.

### Completion means "present and readable", not "fresh"

An artifact counts as done if it exists and parses; there is no
mtime/fingerprint comparison against upstream inputs. Replacing an upstream
artifact does not delete or invalidate a finished one downstream — choosing the
step again is the explicit rebuild operation. Rationale: keeps persistence
independent from menu presentation and leaves room for a future freshness
policy; and it means a finished audiobook is never silently destroyed by
upstream edits. The checkbox for a step is therefore "this step produced a
usable artifact", not "this artifact reflects the current inputs".

### Exception: an alignment without its book is shown as incomplete

If the book artifact is deleted while the timed-phrases cache remains, Align
displays ⬜ even though the cache is present and valid. This is the one place
where the menu violates the pure "present and readable" rule, because the
orphaned cache is unusable: Create needs the book, and re-entering text wipes
the cache anyway. Showing ✅ would invite "just hit Create", which would block.
`create_output` mirrors this with a dedicated error message explaining the
stale state. This is presentation-only — the cache file itself is left in
place.

### "Enter text" is never checked based on the transcription

The transcription binary existing says nothing about the text step: Whisper
runs on the audio alone, so a transcription can exist with no text ever
entered. The text item's checkmark tracks only the book artifact.

### Re-entering text deletes the alignment cache

Alignment output is specific to the exact imported book. When new source text
is saved, the prior timed-phrases file is removed rather than kept as
"probably still fine". Cost: an unnecessary re-align after a trivial text
tweak. Benefit: no stale alignment is ever silently carried into an output
file.

### Transcribe runs without source text; its chain prompt does not

Transcription needs only audio, so it is not gated on a book — the audio
track can complete before any text is entered. The one book-dependent part is
the "align and create the `.abr.mX` file when finished?" chain question: its
suffix comes from the book, and alignment requires the book anyway, so the
question is simply skipped when no source text has been entered yet.

### Create without the book is not offered in degraded form

Create could theoretically proceed without the book (title from the audio stem,
empty sections, guessed suffix), but it deliberately blocks instead. A
structureless output file would be a silently worse deliverable: no chapters
for the reader UI, no real title, and the `.m4a`/`.m4b` choice would be a
guess.

### Review counts describe the output file, not current work state

"Review unmatched lines (N of M lines)" reads its counts from the metadata
embedded in the output file. After re-aligning but before re-creating, those
numbers describe the *old* output. This is intentional: review plays the
finished output, so its numbers must describe exactly that.

### The state cache is presentation-only

Deriving a full state unpickles the transcription and decodes the output's ABR
tag, which is too expensive per redraw. The cached snapshot is validated
against a stat signature (mtime_ns + size) of every source file; all writers
use atomic replace, which always bumps mtime, so a mutation cannot slip
through. Flows call the uncached variant after mutating files, so the cache
only ever affects labels — never business decisions.

### Stale audio selection is silently dropped

On menu entry, a selection whose audio file no longer exists is cleared
without prompting. The work files may still exist on disk and are simply
orphaned until the same audio is reselected.

### Audio with an existing parallel output is selectable, with a notice

Selecting source audio that already has a finished `.abr.m4a`/`.abr.m4b`
sibling is not blocked. That state is legitimate and self-produced: Clear
keeps the output, and reselecting afterward is how the user reaches "Review
unmatched lines", which reads the output file and needs no intermediates.
The checkbox state (output ✅, intermediates ⬜) truthfully reads as "done,
unless you want to redo it", and Create is confirm-gated before replacing.
Selection instead shows a non-blocking notice pointing at the existing file.
Because the expected suffix depends on the book's `text_source_kind` — not
knowable until text is entered — the notice checks both candidate names.
