# Room Reverb

Last updated: 2026-10-16

## Overview

The project setting `room_reverb` ("Room reverb" in the Create menu, under Post-processing, after
"Room tone") places the narration in a room: convolution with a synthetic room impulse response
(IR), run as one continuous pass over the whole output, so tails decay naturally across segment
boundaries and pauses instead of resetting with each TTS segment. Concat durations and timing maps
are unchanged. Realtime playback additionally emits the remaining room decay at run end.

| `RoomReverb` | id | Room design | DRR (dB) | Label |
|---|---|---|---|---|
| `OFF` (default) | `off` | - | - | Off |
| `SMALL_DRR22` | `small_drr22` | small | 22 | Small room, light |
| `SMALL_DRR17` | `small_drr17` | small | 17 | Small room, moderate |
| `MEDIUM_DRR21` | `medium_drr21` | medium | 21 | Larger room |

DRR is the direct-to-reverberant ratio: dry energy over reverb energy (the IR has unit energy, so
this is just the reverb gain), higher = drier. It isn't comparable across room designs: a bigger
room's later, longer tail is far more audible in pauses at the same DRR.

**Ids encode the sound, labels don't.** The id names the room design and DRR, so labels can change
freely ("Larger room" is the `medium` design: mid-sized among the rooms tried, but the largest on
offer; labels name the room first, then the level, since the larger room has less total reverb
energy than "Small room, moderate" yet sounds bigger). A saved id must always sound the same: retuning a preset means a new member, with the old id
mapped to it on load (as `fill_room_tone` was mapped to `room_tone`); changing a room design means a
new design key. An unknown id loads as `OFF`.

## Stereo effect

The project setting `room_reverb_stereo` ("Room reverb stereo effect", shown in the menu only while
Room reverb is on) spreads the reverb across left and right. Direct narration stays centered in
concat, realtime playback, and voice chat. Concat and realtime retain dormant voice-positioning
infrastructure (see below), but no app voice currently has a nonzero pan.

| `RoomReverbStereo` | id | Spread | Label |
|---|---|---|---|
| `MONO` (default) | `mono` | none | Centered (no spread) |
| `LATE_W100` | `late_w100` | late tail, full width | Wide |
| `ALL_W100` | `all_w100` | early reflections and late tail, full width | Wider |

Ids name the processing, like `RoomReverb`'s. With reverb on and anything but `MONO`, the export is
stereo: the reverb processor outputs `(n, 2)`, the noise bed adds the same bed to both channels
(so room tone stays centered), and the ffmpeg stream is opened with `-ac 2`. With reverb off the
spread setting does nothing. Stereo output is currently requested only by active stereo reverb,
regardless of the project's voice count. Loudness normalization measures stereo per BS.1770
(channels summed), so the target is met as before; chapters (M4B) and app metadata were checked on
real stereo FLAC and M4B exports, and the app's own loader downmixes to mono.

`make_stereo_irs` (mid/side per spread part, early/late split by a 6 ms crossfade at the end of the
early reflection window):

    L = c*A + s*B        R = c*A - s*B        c = cos(t), s = sin(t), t = width * pi/4

A is the mono IR's part, B the same part of a second rendering of the room with a different seed
(uncorrelated, energy-matched). Each channel gets about the mono preset's reverb energy, and the
spread part's interchannel correlation is ~0 at full width. Summed to mono, B cancels: the
fold-down is the mono preset with the spread parts 3 dB lower, never wetter or more colored. Wider
also reduces coloration on headphones, since the ears get different comb patterns from the early
reflections, which hearing partly averages out ("binaural decoloration"); Wide leaves the early
reflections (the source of the "phasey" quality) identical in both channels.

FLAC files are ~40-55% larger (the decorrelated reverb compresses poorly); AAC keeps its size at a
fixed bitrate, with fewer bits per channel.

## Dormant voice-positioning infrastructure (audiobook concat and realtime only)

Direct narration uses center-unity equal-power panning: `theta = (pan + 1) * pi/4`,
`L = sqrt(2)*cos(theta)`, `R = sqrt(2)*sin(theta)`. Center therefore preserves the existing
unit gain in both channels and the existing dry/reverb balance. The shared lookup currently returns
zero for every voice index: the app cannot activate positioning, and the former index-1 `+0.25`
mapping has been removed. DSP and both rendering integrations remain available for future use.
User-defined voice selections are clamped to available voices using the same rule as generation;
default `-1` selects voice zero.
Disabled voice selection stays centered. Auto-advance also stays centered: saved phrase metadata
does not identify the voice actually used in that mode, and guessing from render order would make
export and realtime disagree. Per-voice configuration/UI and persisted auto-advance identity are deferred.

Pan is applied on a parallel direct path, not to the reverb input or the completed room output.
Reverb always receives the same mono narration regardless of pan. Its wet-only signal is added to
the panned direct signal; preceding voices' tails therefore do not move at speaker transitions.
Appended break effects remain dry and centered; room tone remains centered. At extreme pans,
reflections can still be heard in the ear receiving no direct speech. Position-dependent room
responses/early reflections are deliberately not modeled.

The shared policy can request stereo if any available project voice has nonzero pan, or if stereo
reverb is active. Since all voice pans are currently zero, only active stereo reverb requests stereo.
This is independent of the selected output range; channels never switch per segment. Stereo output
does not itself turn on reverb spread. Actual mono
playback after device fallback retains original mono speech/reverb, rather than folding down pan
gains. Voice chat is excluded from the voice-pan policy.

The boosted-side gain can exceed unity (`+0.25`: approximately 0.786 left, 1.176 right). DSP sums
floating-point dry and wet signals before clipping safety checks; downstream export normalization
preserves channel balance. No per-channel normalization is applied.

## Signal chain

Per held segment in concat (`ConcatUtil.concatenate_sound_segments`, `flush_pending`):

    segment (resample, treble EQ, sentence-gap extension, gap limiting)
      -> pause or break effect appended (pause timing rules)
      -> direct voice pan + parallel room reverb (narration only)
      -> room tone noise bed
      -> ffmpeg stream -> loudness normalization (whole file) -> metadata

- **Narration only:** `apply(sound, send_end=narration_end)` feeds only the segment into the
  reverb; the appended pause or break effect passes through dry, with the earlier tail still
  decaying underneath. So break sound effects aren't reverberated.
- **Before room tone:** the noise bed isn't reverberated.
- **End:** the tail beyond the output's last sample is dropped; with `is_last`, the reverb (not the
  dry signal) fades over the last 50 ms. The output always ends in the final segment's pause, so
  this rarely matters.

**Realtime playback** applies the same settings in the same order: the segment, then its pause or
break effect (`send_end=0`), are fed in turn; `realtime_save` files include the reverb. With the
project requesting stereo from active reverb spread, the output stream (`SoundDeviceStream(channels=2)`, whose buffer then holds
`(n, 2)` frames; sample counts and positions stay in frames) and saved files are stereo. The stream
is opened before the reverb processor is created (`open_output_stream`): if the device can't open in
stereo, playback falls back to mono with a warning in the content area, and the reverb is the plain
mono preset.

At the end of a realtime run, `VoiceSpatialProcessor.drain_tail()` consumes the remaining finite
convolution tail and emits it after the last submitted audio. Speech is no longer marked with the
legacy `is_last` wet fade: only the final 50 ms of the drained tail are softened. Room tone continues
under that tail and fades at the actual output end. The final successful segment's saved realtime
file includes the emitted tail, including when later generation fails or the run stops early.
Ordinary segment boundaries do not drain/reset the room, so tails still overlap following speech.
No extra duration is added with reverb off or an empty room. Concat and voice chat retain their
existing endpoint behavior; realtime draining cannot recover ambience already cut off in stored
TTS source audio. An output-buffer underrun while waiting for generation can still interrupt audio;
this change finalizes completed runs rather than synthesizing audio in the device callback.

**Voice chat** (`conversation/`) applies the same settings; no room tone. The output stream is
opened once per session (`ConversationRuntime.initialize`, `open_chat_output_stream`): stereo with
the stereo effect on, falling back to mono with a warning in the transcript. The reverb processor
is made per turn (`_ResponseEngine.get_room_reverb`, lazily on the TTS worker thread), not per
session: an interrupted turn clears the buffered audio, so a carried-over tail would be stale.
Non-streaming TTS: each chunk with its pause appended is fed with `send_end` at the speech end, as
in concat; phrase alignment (STT) gets the dry mono sound. Streaming TTS
(`ConversationStreamingTts.generate_to_sound_stream`): each audio chunk is fed as it arrives, then
the trailing pause with `send_end=0`. Saved chat files include the reverb.

Not applied in the server.

## Diagnosing ambience cutoffs with reverb disabled

`room_reverb_stereo` alone does not add room sound while `room_reverb` is off. A roomy/reverberant
character can already be baked into generated mono audio (including its voice reference). Compare
stored source PCM against export PCM before attributing a cutoff to the room processor. In the
2026-10-16 jkharu example, the first line ended at 3.340 s in both source and concat; generation logs
recorded trimming 180 ms from the original 3.520 s result. Other stored lines ended loudly even
without a logged end trim. Export followed these existing endpoints with the configured pause/bed;
normalization preserved the waveform rather than gating it. Raw pre-trim audio was unavailable, so
whether the removed quiet audio contained useful decay could not be established. Natural-tail
draining cannot recover already removed source ambience, and should not synthesize reverb when off.

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
- **Stereo** (`--stereo` in the testx script): mono / late tail at half and full width / everything
  at full width, per preset; all four sounded good. Full width, late only (Wide) and everything
  (Wider) were adopted; half width was close to full.
- **Not tried:** recorded IRs (`--ir` in the testx script supports them).

## Tests

[`tests/test_room_reverb.py`](../tests/test_room_reverb.py) (IR, streaming continuity, DRR,
dry break effects, stereo IRs and mono fold-down, stereo concat wiring, setting load/save); menu in
[`tests/test_concat_menu.py`](../tests/test_concat_menu.py); realtime in
[`tests/test_real_time_playback_app.py`](../tests/test_real_time_playback_app.py)
(`test_realtime_save_writes_exactly_what_was_played`).

## Open questions

- **Other voices and models:** levels were chosen on one passage with two voices of one model.
- **Interaction with room tone levels:** listened to with room tone Low only.
- **Stereo room tone:** a decorrelated bed would sound more like a space than centered hiss; the
  bed is currently identical in both channels.
