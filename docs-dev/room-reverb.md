# Room Reverb

Last updated: 2026-10-13

## Overview

The project setting `room_reverb` ("Room reverb" in the Create menu, under Post-processing, after
"Room tone") places the narration in a room: convolution with a synthetic room impulse response
(IR), run as one continuous pass over the whole output, so tails decay naturally across segment
boundaries and pauses instead of resetting with each TTS segment. Durations and timing maps are
unchanged.

| `RoomReverb` | id | Room design | DRR (dB) | Label |
|---|---|---|---|---|
| `OFF` (default) | `off` | - | - | Off |
| `SMALL_DRR22` | `small_drr22` | small | 22 | Light |
| `SMALL_DRR17` | `small_drr17` | small | 17 | Moderate |
| `MEDIUM_DRR21` | `medium_drr21` | medium | 21 | Large room |

DRR is the direct-to-reverberant ratio: dry energy over reverb energy (the IR has unit energy, so
this is just the reverb gain), higher = drier. It isn't comparable across room designs: a bigger
room's later, longer tail is far more audible in pauses at the same DRR.

**Ids encode the sound, labels don't.** The id names the room design and DRR, so labels can change
freely ("Large room" is the `medium` design: mid-sized among the rooms tried, but the largest on
offer). A saved id must always sound the same: retuning a preset means a new member, with the old id
mapped to it on load (as `fill_room_tone` was mapped to `room_tone`); changing a room design means a
new design key. An unknown id loads as `OFF`.

## Signal chain

Per held segment in concat (`ConcatUtil.concatenate_sound_segments`, `flush_pending`):

    segment (resample, treble EQ, sentence-gap extension, gap limiting)
      -> pause or break effect appended (pause timing rules)
      -> room reverb (narration only)
      -> room tone noise bed
      -> ffmpeg stream -> loudness normalization (whole file) -> metadata

- **Narration only:** `apply(sound, send_end=narration_end)` feeds only the segment into the
  reverb; the appended pause or break effect passes through dry, with the earlier tail still
  decaying underneath. So break sound effects aren't reverberated.
- **Before room tone:** the noise bed isn't reverberated.
- **End:** the tail beyond the output's last sample is dropped; with `is_last`, the reverb (not the
  dry signal) fades over the last 50 ms. The output always ends in the final segment's pause, so
  this rarely matters.

**Realtime playback** applies the same setting in the same order: the segment, then its pause or
break effect (`send_end=0`), are fed in turn; `realtime_save` files include the reverb. Not applied
in the server or voice chat.

## Implementation (`sound/room_reverb.py`)

Plain convolution, written in-house (scipy `oaconvolve`), no new dependency. Considered and
rejected: pedalboard's `Reverb` (Freeverb: no pre-delay, no early reflection control, metallic at
small sizes); pyroomacoustics (useful for designing IRs offline, too heavy at runtime).

- `StreamingReverb`: dry + convolution, carrying the tail across calls, so chunked processing equals
  one long call sample for sample (tested). Cheap: the medium IR is ~0.9 s.
- `make_room_ir(RoomIrParams, sr)`: deterministic (fixed seed) unit-energy IR, reverb only (no
  direct sound): pre-delay; early reflections jittered through a window, each a short decaying noise
  burst rather than a discrete tap (discrete taps in mono read as hollow comb filtering); a late
  tail of exponentially decaying noise with a raised-cosine onset, whose band above 3 kHz decays
  faster; then reverb EQ (2nd-order high-pass, 4th-order low-pass) and unit-energy normalization.
- `ROOM_DESIGNS`: `small` (RT60 0.3 s, 10 ms pre-delay, early-reflection-weighted, 200 Hz-7 kHz)
  and `medium` (RT60 0.8 s, 15 ms pre-delay, denser reflections, louder tail, 250 Hz-6 kHz). IRs
  are cached per design and sample rate.
- `RoomReverbProcessor(sr, design, drr_db)`: one per export or realtime run; adds the `send_end`
  and `is_last` handling and a [-1, 1] clip safety net (the ffmpeg stream casts to int16 without
  clipping).

## Design notes (from listening tests)

Presets were picked by ear on a 110 s MOSS passage (high-quality Bluetooth headphones, noise
cancelling on), level-matched by loudness normalization, using
[`testx/room_reverb_ab.py`](../testx/room_reverb_ab.py).

- **Levels are presets, not one room scaled up.** The small room at DRR 22 and 17 was "tasteful and
  passable"; at 12 and 6 it turned "robotic, phasey". Reverb energy in the first ~50 ms fuses with
  the direct sound and is heard as timbre; in mono it adds comb-like spectral ripple that grows with
  level. Ripple of direct + first 50 ms (1/12-octave smoothed, 250 Hz-6 kHz, std dev): DRR 22
  0.28 dB, 17 0.51, 12 0.93 (phasey), 6 1.65. So more reverb comes from a bigger room, whose energy
  arrives later and more diffusely: the medium room at DRR 21 is 0.19 dB, while its tail just after
  a phrase is ~9 dB above the small room's at DRR 17.
- **Tried and not adopted:** medium at DRR 25; a hall (RT60 1.6 s) and a cathedral (2.8 s) at two
  levels each. Kept in the testx script for comparison.
- **Not tried:** recorded IRs (`--ir` in the testx script supports them); stereo, which would
  decorrelate the reflections and reduce the mono comb problem, but the pipeline is mono end to end.

## Tests

[`tests/test_room_reverb.py`](../tests/test_room_reverb.py) (IR, streaming continuity, DRR,
dry break effects, concat wiring, setting load/save); menu in
[`tests/test_concat_menu.py`](../tests/test_concat_menu.py); realtime in
[`tests/test_real_time_playback_app.py`](../tests/test_real_time_playback_app.py)
(`test_realtime_save_writes_exactly_what_was_played`).

## Open questions

- **Other voices and models:** levels were chosen on one passage with two voices of one model.
- **Interaction with room tone levels:** listened to with room tone Low only.
