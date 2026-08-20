# `generate_files()` scheduling and ordering

This document describes the ordering logic of [`GenerateUtil.generate_files()`](../tts_audiobook_tool/generate_util.py:66): how a set of queued lines is turned into a schedule of model calls, in what order those calls execute, and which ordering guarantees the schedule provides (and which it deliberately gives up).

Related documents:

- [`tts-validation-architecture.md`](tts-validation-architecture.md) — what happens *after* each call returns (transcription, word-error validation, retry criteria)
- [`tts-rolling-continuation-clear-architecture.md`](tts-rolling-continuation-clear-architecture.md) — rolling-continuation reset points, including the voice-sample-switch reset used by the loop described here
- [`dialog-segmentation.md`](dialog-segmentation.md) — how dialog lines get pre-assigned voice indices (`DIALOG_VOICE_INDEX`)
- [`project-spec-v2.md`](project-spec-v2.md) — the voice selection modes themselves

## The hard constraint

A single model call can use exactly **one voice sample**. In user-defined voice selection mode, a line's voice is fixed per line, so one call can never mix lines from different voices. Everything below is built around that constraint.

## Scheduling vocabulary

The queue consumed by `generate_files()` has three levels:

| term | definition |
|---|---|
| item | `(phrase_group_index, retry_count)` tuple |
| sub-batch | [`SubBatch`](../tts_audiobook_tool/generate_util.py:47): up to `batch_size` items that all use the *same* voice sample (`voice_selection_index`), or no explicit voice (`None`) so the mode logic inside [`GenerateUtil.generate()`](../tts_audiobook_tool/generate_util.py:609) resolves it. **One sub-batch = one model call.** |
| round | a list of sub-batches — a "batch of batches". A round is executed to completion before the next round starts. |
| queue | a FIFO list of rounds. |

For single-voice projects a round holds exactly one sub-batch, so the shape degenerates to the classic flat list of batches.

## Queue construction

`generate_files()` starts from `items`: the queued indices sorted in book order, each with retry count 0 ([generate_files()](../tts_audiobook_tool/generate_util.py:111)). It then picks one of two builders:

```
batch_size > 1  AND  voice_select_mode == USER_DEFINED  AND  voice samples exist?
├── yes → make_multi_voice_rounds(items, ..., voice_of_index, sort_by_words = not is_cjk)
└── no  → bucket_items(items, ...)            # only if batch_size > 1 and not CJK
          make_single_voice_rounds(items, batch_size)
```

`voice_of_index` comes from [`effective_voice_indices()`](../tts_audiobook_tool/generate_util.py:1028), which maps each queued line to the voice sample index the call will actually use: a per-line selection of `-1` (no explicit assignment) or a stale out-of-range value is clamped to the nearest valid sample index.

### The single-voice shape (`make_single_voice_rounds` + `bucket_items`)

- [`bucket_items()`](../tts_audiobook_tool/generate_util.py:987) reorders the flat list in **windows of `batch_size × 5`** items (`BATCH_ITERATIONS_PER_GROUP` = 5). Within each window, items are sorted by phrase-group word count — longest-first in one window, shortest-first in the next, alternating per window. The windowed alternation is what preserves a "semblance" of index monotonicity: always-sorting-longest-first would make long lines systematically drift to the front of every window; alternating cancels that drift over every pair of windows, so a line's *expected* position in the execution order stays centered on its book position.
- [`make_single_voice_rounds()`](../tts_audiobook_tool/generate_util.py:1045) then chunks the (reordered) list into sub-batches of `batch_size` with `voice_selection_index=None`. The voice for each call is resolved later by the mode logic in `generate()` (disabled mode → sample 1; auto-advance → rotation).
- CJK projects skip `bucket_items` entirely: no word-count sorting is meaningful for word-count-based packing, and the lines keep pure book order. `batch_size == 1` also skips it (there is nothing to pack).

### The multi-voice shape (`make_multi_voice_rounds`)

[`make_multi_voice_rounds()`](../tts_audiobook_tool/generate_util.py:1060) applies the same windowing principle per voice, widened to fit all voices:

1. **Window.** Items are walked in index order in windows of `batch_size × 5 × num_distinct_voices` items, where `num_distinct_voices` is the number of *distinct voice samples present among the queued items*. With two voices and batch size 2, a window spans 20 lines of book — twice as wide as a single-speaker window, so each voice keeps the same per-window share (5 × batch_size of its own lines) as the single-speaker path.
2. **Piles.** Within a window, items are grouped into per-voice piles in **first-appearance order** (which voice speaks first in that window, that pile comes first in the round). Piles therefore stay roughly in book order.
3. **Sort.** Each pile is sorted by word count with the same descending/ascending alternating pattern as `bucket_items` (skipped for CJK). This packs similar-length lines into the same call so the calls in a round finish at roughly the same time.
4. **Chunk.** Each pile is cut into sub-batches of `batch_size`, each stamped with its voice sample index. The round is the concatenation of all piles' sub-batches.

Because sub-batches execute sequentially within a round, **all of voice A's sub-batches run before any of voice B's** in a given round (subject to the pile order). That grouping is a consequence of the one-voice-per-call constraint, not an optimization: it cannot be removed, and the first-appearance pile order is the cheap book-order-favoring choice for it.

## Execution loop

The loop in [`generate_files()`](../tts_audiobook_tool/generate_util.py:178) is:

```
while True:
    if interrupted: break
    if current round exhausted:
        if pending_retries: start a retry round (make_retry_round)
        elif queue non-empty: current_round = queue.pop(0)
        else: break
    sub = current_round.pop(0)
    if sub switches voice sample vs previous sub: Tts.clear_continuation()
    print heading (includes [voice N] for explicit-voice sub-batches)
    results = generate_and_validate_batch(..., voice_selection_index=sub.voice)
    OOM in results → clear continuation, stop the run
    collect re-adds (lines that failed validation) → pending_retries
```

Notable behaviors:

- **Retries take head-of-queue priority.** Lines that fail validation are re-added with an incremented retry count and, when the current round is exhausted, are processed as their own dedicated round built by [`make_retry_round()`](../tts_audiobook_tool/generate_util.py:1125) — *before* any further items from the main queue. The retry round keeps the order in which the re-adds were produced (no word-count re-sorting) and, for multi-voice runs, groups re-adds by voice sample so each call is still voice-homogeneous. This is a deliberate responsiveness-over-book-order trade: a failed line is retried promptly even though that now executes out of book order.
- **Voice switches clear rolling continuation.** The rolling-continuation context is per voice sample, so a sub-batch using a different sample than the previous one resets it (defensive: rolling-continuation models force batch size 1 anyway — see the architecture doc).
- **FIFO over book-ordered windows.** Rounds come off the queue strictly in order, so all lines of window *N* are generated before any line of window *N+1*.

## Worked example

Two voices, 1000 queued lines, batch size 2:

- window = 2 × 5 × 2 = 20 lines → **50 rounds**
- a typical round: ~10 lines of voice A and ~10 of voice B (in a balanced dialogue) → 5 sub-batches per voice → ~10 sequential model calls, each generating 2 lines with one voice sample
- 1000 lines ÷ 2 lines per call = **500 model calls** total, grouped into 50 rounds

From each voice's perspective the schedule looks exactly like the old single-speaker batching: windows of 5 × batch_size of its own lines, word-count-sorted and chunked. The only differences are that the windows span twice as much of the book and the other voice's lines are interleaved into the same rounds.

## Ordering guarantees, at a glance

| level | out-of-orderness vs book order |
|---|---|
| between windows | **zero** — FIFO over book-ordered windows; no line is ever more than one window-width away from its book position |
| voice pile vs book order | not minimized — forced by the one-voice-per-call constraint; first-appearance pile order is the book-order-favoring choice |
| word-count sort within a pile | minimized *on average* — the alternating sort keeps each line's expected execution position centered on its book position (skipped for CJK) |
| retry rounds | intentionally out of order — head-of-queue priority so failed lines are retried promptly |

The design contract in one sentence: *worst case is bounded by the window width, average case stays centered on book position, and the one reordering the app cannot fix (per-voice grouping) is a hard constraint of the models.*

## Invariants worth preserving

- Single-voice projects (disabled or auto-advance mode, `batch_size == 1`, or no voice samples) take the `bucket_items` + `make_single_voice_rounds` path, which reproduces the pre-multi-voice flat queue byte-for-byte. Changes to the multi-voice path must not leak into that one.
- `make_multi_voice_rounds()` returns `[]` for empty input; with one distinct voice it is equivalent to the single-voice path (window collapses to `batch_size × 5`).
- `voice_of_index` is computed once up front from the *queued* indices, so a line's voice for the whole run — including retry rounds — is fixed at queue-build time.
- `generate()` raises a `ValueError` if handed a multi-item user-defined batch without an explicit `voice_selection_index` (a logic error, guarding against a future caller bypassing the sub-batch split).