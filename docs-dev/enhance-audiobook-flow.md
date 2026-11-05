# Enhance a pre-existing audiobook — UI flow, business rules, and judgement calls

This document describes the "Enhance a pre-existing audiobook" feature: its
dataflow, menu presentation, artifact lifecycle, and the deliberate judgement
calls behind behaviors that are not fully straightforward.

Relevant code:

- `tts_audiobook_tool/enhance/enhance_artifacts.py` — artifacts, state snapshot, caching
- `tts_audiobook_tool/enhance/enhance_menu.py` — menu presentation
- `tts_audiobook_tool/enhance/enhance_flow.py` — user-triggered operations
- `tts_audiobook_tool/enhance/enhance_alignment.py` — transcription windows, stitching, and the original greedy alignment
- `tts_audiobook_tool/enhance/enhance_anchored_alignment.py` — anchored global alignment (default)
- `tts_audiobook_tool/enhance/enhance_text.py` — book import/flattening; `align_book` selects the alignment algorithm
- Tests: `tests/enhance/test_enhance_artifacts.py`, `tests/enhance/test_enhance_menu.py`, `tests/enhance/test_enhance_flow.py`, `tests/enhance/test_enhance_alignment.py`, `tests/enhance/test_enhance_anchored_alignment.py`
- A/B scripts: `testx/enhance_alignment_compare.py`, `testx/enhance_stitch_compare.py`

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
| Timed phrases | `<stem>.timed_phrases.bin` | Align | Create, Review |
| Output | `<stem>.abr.m4a` or `.abr.m4b` | Create | Review (fallback) |

The book is a JSON file (`BOOK_FORMAT`); the transcription and timed-phrase
caches are pickles written via atomic replace. The output suffix depends on the
book's `text_source_kind`: `plain_text` → `.abr.m4a`, `epub` → `.abr.m4b`.

## Audio I/O

All decoding and encoding uses the app's pip-only audio stack (PyAV and
soundfile; see [audio I/O](<audio-io.md>)), never an FFmpeg executable:

- **Transcribe** streams the source through `AudioDecoder` into 16 kHz mono
  float32 windows of 30 s with 5 s overlap
  (`enhance_alignment._stream_audio_with_overlap`); the STT worker receives
  arrays only. The windows' words are stitched into one stream (see
  [Transcript stitching](#transcript-stitching)). The decoder uses the stream `best("audio")` picks — the default
  track when the file has several.
- **Review previews** decode at most one minute from the same track with
  `SoundFileUtil.load_range`, which seeks instead of decoding from the start.
- **Create** copies M4A/M4B sources and adds the ABR tag; other formats are
  re-encoded to AAC (`SoundFileUtil.transcode_to_aac_at` → `transcode_to_aac`),
  keeping container tags and chapters. Every decoder is closed before the
  staged file replaces the destination.

## Forced alignment

Alignment turns the flattened book phrases plus the transcript words into one
`TimedPhrase` per phrase (`enhance_text.align_book`). It is text-to-text:
Whisper's word timestamps supply the times, and the audio is not re-read.

### Feature flags

Both stages exist in an original and a new version, switchable in
`constants_config.py` for A/B testing:

| Flag | `True` (default) | `False` |
| --- | --- | --- |
| `ENHANCE_ANCHORED_ALIGNMENT` | anchored global alignment | original greedy alignment |
| `ENHANCE_OVERLAP_STITCHING` | overlap stitching | original timestamp-threshold stitching |

Stitching runs at transcription time and the transcription cache stores
stitched words, so `ENHANCE_OVERLAP_STITCHING` only affects transcriptions made
after it changes. The alignment flag takes effect on the next Align.

`testx/enhance_alignment_compare.py` runs both aligners on an audiobook's work
files; `testx/enhance_stitch_compare.py` transcribes a slice of an audiobook,
keeping the raw windows (saveable for reuse), and compares both stitchers.

### Output contract

Common to both aligners:

- The list is parallel to the flattened phrases.
- `(0.0, 0.0)` marks an unmatched line ("orphan"). The menu, review, and
  `count_misalignments` count only these.
- A span starts at the start of the phrase's first transcript word and ends where
  the next transcript word starts, so consecutive phrases tile the audio. The
  final phrase ends at its last word's end.
- Timed spans are ordered and do not overlap. A zero-length span (start equals
  end, non-zero) is not playable in the player, which skips it; it is not
  counted as unmatched.

### Transcript stitching

Transcription windows are 30 s long, each sharing 5 s with the previous
(`TRANSCRIBE_CHUNK_SECONDS`, `TRANSCRIBE_OVERLAP_SECONDS`). Each window's words
get absolute times; stitching joins the windows into one word stream.

- **Original** (`_stitch_transcripts`): drops the earlier window's last word,
  then appends the later window's words that start after the earlier window's
  second-to-last word. The two windows time the same word slightly differently,
  so this often duplicates a word ("the the sound"). It also keeps the earlier
  window's least reliable last seconds. About 260–280 seam duplicates per
  9-hour book, and occasional repeated phrases from text Whisper invents at
  window edges.
- **Overlap** (`_stitch_transcripts_by_overlap`): places each seam near the
  middle of the shared 5 s. It cuts at a word both windows transcribed (same
  text, start times within 1 s) closest to the middle, preferring one inside a
  run of shared words, and continues with the later window's copy. Without a
  shared word it cuts by time at the middle, and drops a later-window word that
  straddles the last kept word. When only one window has words in the shared
  span, that window's words are kept whole.

On two 10-minute slices of a real audiobook, overlap stitching cut seam
duplicates from 3 and 12 to 0 and 3 (the remaining 3 are in the text, eg
"when, when did you"). Leftover transcript tokens fell from 39 to 26 and from
12 to 1, with no loss of matched book text. Re-transcribing a whole 9-hour book
cut adjacent duplicate words from 476 to 255, and those at seams from 276 to
61, close to the rate expected by chance.

### Anchored alignment (default)

`enhance_anchored_alignment.align_phrases_detailed` aligns the whole book
against the whole transcript in one pass. Section boundaries play no part.

1. **Tokenize** both sides (`tokenize`). Text is casefolded and apostrophes are
   dropped ("don't" → `dont`). Hyphenated stutter prefixes collapse ("Wh-wh-what"
   → `what`); whole-word repeats such as "ha-ha" are kept. Other punctuation
   separates tokens. Runs of three or more of one letter collapse ("Ahhhh" →
   `ah`). Digits and English ordinals are spelled out *per token* with
   `num2words` in the book's language ("150" → `one hundred and fifty`), so each
   transcript token still maps to the Whisper word, and time, it came from.
2. **Anchor recursively.** In a window, word n-grams (sizes 3, 2, then 1) that
   occur exactly once on each side are candidates. The longest chain increasing on
   both sides is kept (LIS), and each gap between anchors is aligned the same way.
   Text present on only one side becomes a gap, eg narration missing from the
   EPUB, or EPUB pages that aren't narrated. A chain is used only if it covers at
   least `MIN_ANCHOR_DENSITY` (10%) of the window's shorter side. A sparser chain
   is coincidence between unrelated text, and the window is left unaligned.
3. **Align small windows** (at most `DP_MAX_CELLS` source × transcript tokens)
   with a word-level edit distance. Exact match costs 0. A fuzzy match costs 0.3:
   character similarity ≥ 0.75, or for English the same Double Metaphone code,
   with both tokens at least 3 letters. Two tokens may align with one when they
   join to it, exactly or fuzzily ("high school" ↔ "highschool", "Arc Ridill" ↔
   "Arcredil"), but not fuzzily when one part already matches alone.
   Substitution, insertion, and deletion cost 1. Ties, and a later pass for
   n-gram anchors, give a source word the *later* of two adjacent identical
   transcript copies, so a duplicated word never starts a phrase.
4. **Time phrases.** Each phrase gets one status:
   - **timed**: at least a third of its tokens matched exactly or fuzzily. A phrase
     of one or two tokens that only aligned by substitution is also timed,
     provided its nearest non-interjection neighbours are timed by the first
     rule (eg "Craeme," heard as "Kramer").
   - **attached**: interjection-only or non-verbal phrases that didn't truly
     match (see the judgement call below).
   - **unmatched**: everything else, `(0.0, 0.0)`.

   A phrase that starts with an unaligned stutter prefix emitted as its own
   Whisper word ("M-" before "Monica") starts at that prefix.

**Complexity.** Tokenizing and n-gram counting are linear per window, the
anchor chain is O(A log A), and the edit-distance pass only sees the small
leftover gaps. On a 12.3k-line, 75k-word book the passes touched about 14k–25k
cells in total, against 5.6 billion for a full DP; the largest gap was 3.4k
cells, against `DP_MAX_CELLS` = 20k. Runtime is about 0.7 s per 9-hour book.

Unrelated text never reaches the DP. For a matching book and transcript, the
top-level anchor chain covers about 95% of the tokens. For a mismatched pair
(eg the wrong volume's EPUB), every chain covers under 1%, and the density
check rejects it. Before that check, such a pair spent 147 s filling 11.9M
cells of junk; now it ends in about 0.2 s with every line unmatched. Text that
covers more or less than the audio still aligns normally, eg an omnibus EPUB
for one volume's audio, or audio of two volumes for one volume's text.

As a safety net, total DP work per run is capped at `DP_CELL_BUDGET_PER_TOKEN`
(10) cells per token of both sides, at least `DP_CELL_BUDGET_MIN`; real books
use about 0.2. Gaps beyond the budget are left unaligned, with a logged warning.

### Greedy alignment (original)

`enhance_alignment.align_phrases_with_state` goes phrase by phrase. For each
start position within `max_skip_words` (45, growing after misses up to 250) past
a cursor, and each length from 0.75× to 1.25×+2 the phrase's word count, it
scores the joined transcript text against the phrase with character-level
`difflib`. It takes the best score ≥ 0.7 and commits the cursor there. Per phrase
the cost is roughly S·n³ for window S and phrase length n, and it re-normalizes
every candidate word each time: about 4 minutes per 9-hour book.

A placement is never revisited, which causes its failure modes:

- A short or misheard line ("Hwah?!") latches onto a similar later line,
  skipping and orphaning the real ones.
- Audio-only narration longer than 250 words (eg a 560-word "characters so far"
  segment) can't be skipped, so everything after it is orphaned.
- A wrong placement that scores ≥ 0.7 isn't reported at all.

### Measured on two full audiobooks

| | Book A | Book B |
| --- | --- | --- |
| Lines / transcript words | 12,304 / 75,018 | 12,280 / 75,756 |
| Greedy: runtime, unmatched | ~240 s, 732 | ~220 s, 499 |
| Anchored: runtime, unmatched | ~0.7 s, 38 | ~0.8 s, 44 |
| Anchored: attached interjections | 104 | 99 |

Most remaining unmatched lines are one-word false starts that Whisper
transcribes once ("I— I think" → "I think"); the rest are genuinely missing
from the audio or are unnarrated EPUB pages. Where both aligners time a line,
about 98% start within 0.3 s of each other. Spot checks of lines placed 2 s or
more apart found the greedy placement wrong: derailments onto a similar later
line, or interjections pinned to an arbitrary word. Smaller differences are
mostly boundary words around stitching seams.

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

- **Review unmatched lines** — appears as soon as a valid, non-empty alignment
  and the original audio are available, even before Create. Its counts describe
  the saved alignment, and previews use the original audio. If that review source
  is unavailable, a valid output is used instead, with counts from its metadata.
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

### Review prefers current alignment, with output as a fallback

"Review unmatched lines (N of M lines)" uses the saved alignment and original
source audio whenever available. Both the post-alignment prompt and the menu
open the same review, so accepting the prompt works without Create and declining
it leaves review accessible later. Re-aligning also switches review to the new
alignment rather than an older output. With no usable alignment/source pair,
review falls back to the finished output and its embedded metadata. The menu
counts always describe the phrases that review will display.

### The state cache is presentation-only

Deriving a full state unpickles the transcription and decodes the output's ABR
tag, which is too expensive per redraw. The cached snapshot is validated
against a stat signature (mtime_ns + size) of every source file; all writers
use atomic replace, which always bumps mtime, so a mutation cannot slip
through. Flows call the uncached variant after mutating files, so the cache
only ever affects labels — never business decisions.

### Interjections are attached to their neighbors, not reported

Whisper usually leaves out "um", "Ahhh", "Heh-heh", and similar sounds, so
interjection-only lines would otherwise fill the review with noise. The
anchored aligner attaches a run of such lines (`is_interjection_only`: common
interjections, repeated syllables, elongated words like "Nooooo!") instead of
marking it unmatched. Lines with no words at all are treated the same way. The
run is placed in the pause between the previous line's last word and the next
line, split evenly, and the previous line's span then ends at its last word. If
the pause is shorter than 0.1 s, the run gets a zero-length span at the boundary.

Sharing the neighbour's span instead would break the player's assumption that
playable spans are ordered and non-overlapping. An interjection is timed normally
when Whisper did transcribe it. A run that also contains an ordinary unmatched
line stays unmatched as a whole, so the interjection is reviewed alongside it.
The greedy aligner has no such handling.

### Low-confidence lines are timed and not flagged

A line timed by the anchored aligner may have matched only a third of its
words, or, if very short, only by substitution between timed neighbours. Such
lines get their times without any marker or review entry. Their
`PhraseAlignment` counts are available to diagnostics only.

### False starts stay unmatched

When the text repeats a word that the narrator says once, or that Whisper
writes once ("I— I think"), the full line gets the word and the one-word
fragment stays unmatched. Most remaining unmatched lines are of this kind. They
are not attached like interjections, so they still appear in the review.

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
