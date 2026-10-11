# Room Tone (Noise Bed)

Last updated: 2026-10-12

## Overview

Generated speech is joined with inserted pauses (or break sound effects), and the pauses inside
segments are mostly near digital silence too. On good (especially noise-cancelling) headphones
the background then switches between dead silence and whatever noise rides on the vocalizations.

The project setting `room_tone` ("Room tone" in the Create menu) adds one constant, low-level
noise bed under the whole output: segments, inserted pauses and break effects alike. Durations and
timing maps are unchanged.

| `RoomTone` | Level (noise RMS, dBFS, before loudness normalization) |
|---|---|
| `OFF` (default) | none |
| `LOW` | -75 |
| `MEDIUM` | -70 |
| `HIGH` | -65 |

Loudness normalization applies one gain to the whole file afterwards, so the bed's final level
shifts by the same amount as the speech (several dB either way, depending on how loud the model's
output was).

Projects saved with the earlier bool `fill_room_tone` load as `HIGH` (`true`, the old fill level)
or `OFF` (`false`).

## Design: why constant, not matched

The bed deliberately ignores the audio it sits under. TTS output generally has no steady
background of its own: its noise is attached to the vocalizations (breath, unvoiced
consonants, decays) and gone in between. On one 500-segment fireredtts3 project, the quietest
25 ms of a typical in-segment pause sat around -91 dBFS, while the level within a pause swung by
about 27-30 dB (breath swells, often just before the next phrase). There is no steady floor to
match, and estimating one reads the breaths instead.

Two earlier approaches assumed such a floor and were removed:

- **Per-join room tone:** copied a noise patch from the neighbouring segments into each pause.
  It often found nothing usable, fell back to silence, or matched the quieter neighbour.
- **Adaptive noise floor layer:** measured each segment's floor, topped it up to -65 dBFS,
  capped pauses at -60 and eased between segments' levels. The measurement mostly read breath
  level (median "floor" -69 dBFS vs a real bed near -90), so segments with louder breaths got
  no layer under them and louder pauses beside them: uneven and hard to reason about.

A constant bed is predictable, partly masks the noise that comes and goes with each utterance,
and its level means exactly what the setting says.

## Noise character

`ContinuousNoise` (in `sound/noise_bed.py`): white Gaussian noise, a pink (1/f) filter, then a
band limit, normalized to unit RMS with a constant derived from the filter chain's
impulse-response energy (not per block, which would modulate the level).

- **Low-pass:** 4th-order Butterworth at 8 kHz (`NoiseBed.LOWPASS_HZ`). TTS speech rarely
  carries much above that, so noise "air" there would stand out against the voice. Relative to
  the 1 kHz octave, the 16 kHz octave is about -26 dB.
- **High-pass:** 2nd-order Butterworth at 80 Hz (`NoiseBed.HIGHPASS_HZ`), so pink noise's low
  end doesn't read as rumble.
- **Continuity:** all filters carry state between calls, and filter state is warmed up (1 s) at
  construction. `next(a)` then `next(b)` equals `next(a + b)` sample for sample (tested), so
  cutting the stream into segment-sized pieces leaves no seams.
- **Reproducible:** fixed seed (`NoiseBed.SEED`).

## Application

`NoiseBed(sr, level_db)`, one per export or realtime run, created at the first segment's sample
rate. `apply(sound, is_last=False)` adds the bed at a constant gain and returns a sound of the
same length:

- Fades in over the first `EDGE_FADE_S` (50 ms) of the output, however it is chunked.
- With `is_last`, fades out over the last 50 ms of that call.
- Clips to [-1, 1] as a safety net (the ffmpeg stream casts to int16 without clipping).

**Concat** (`ConcatUtil.concatenate_sound_segments`, param `room_tone_db`): each held segment
gets its pause or break effect appended, then one `apply` call; the final segment is `is_last`.

**Realtime playback** (`real_time_playback.start`): the segment, then its pause or break
effect, are fed in turn. A segment with nothing appended (end of range or interrupt) is
`is_last`. With `realtime_save` on, the saved file is what was played, so it includes the bed.
Not covered: digital silence the output stream plays when its buffer runs dry.

Not applied in the server or voice chat.

## Extended sentence gaps

Sentence gaps extended by "Minimum sentence silence inside segments" are filled with digital
silence (see [tts-validation-architecture.md](tts-validation-architecture.md)); with the bed on,
they get the bed underneath like everything else. An earlier fill synthesized room tone from a
noise patch sampled inside each gap (`RoomToneUtil`). It was removed for the same reason as the
adaptive layer: there was rarely steady noise to sample, so it fell back to silence or stretched
part of a breath.

## Tests

[`tests/test_noise_bed.py`](../tests/test_noise_bed.py) (generator continuity, level, band
limit, fades, concat wiring) and [`tests/test_room_tone_setting.py`](../tests/test_room_tone_setting.py)
(default, legacy migration, persistence).

## Open questions

- **Levels** were chosen by reasoning, not listening tests across models and voices.
- **ACX:** with a quiet model output, normalization can lift `HIGH` toward the -60 dBFS
  noise-floor limit. Worth checking on an export.
