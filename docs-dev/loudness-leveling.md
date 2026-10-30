# Loudness leveling

Last updated: 2026-10-30

Per-segment loudness leveling replaced the whole-file, two-pass ffmpeg `loudnorm` as the app's loudness
step. This doc records the problem, how it was diagnosed, the decisions taken, the method, and how it was
validated.

Code: [`sound/loudness_leveling.py`](../tts_audiobook_tool/sound/loudness_leveling.py)
(`LoudnessLeveling`, `LevelingStats`). Tests: [`tests/audio/test_loudness_leveling.py`](../tests/audio/test_loudness_leveling.py).

## The problem

A project can use several voice clones, and a book's generated segments are stitched together in order. Adjacent
segments, especially at voice changes, could differ noticeably in loudness, and the loudness normalization did not
even this out.

Two things combined:

1. **Each segment was matched by peak, not loudness.** After generation, each segment is peak normalized to
   -1 dBFS (`SoundPipeline.apply_generate_post_processing`). Perceived loudness depends on average level, and
   the peak-to-loudness ratio varies a lot: by voice (breathy/dark vs dense/bright, plosive spikes) and by
   segment within a voice (one sharp "p" makes a whole segment quieter). Equal peaks therefore meant unequal
   loudness.
2. **The loudness pass worked on the whole file.** `loudnorm` in linear mode applies one gain to the whole
   file, which keeps the differences between segments intact. Its dynamic mode is a slow (~3 s window) gain
   rider that only acts outside roughly target ± LRA/2 (±4.5 LU for ACX), so it left a 3-4 LU segment-to-segment
   jump alone, or corrected it late, which can be heard as pumping after voice changes.

Nothing in the chain measured loudness per segment.

## Diagnosis (measured)

A probe over a real 4-voice project (`ttlg`, 239 breeze2 segments, voices alternating quickly) measured each saved,
peak-normalized segment's integrated loudness (BS.1770, `pyloudnorm`):

| Voice | Segments | Median LUFS | p10 – p90 | Min – max |
|---|---|---|---|---|
| narra | 93 | -15.8 | -17.3 … -14.0 | -21.5 … -12.6 |
| alice | 51 | -15.6 | -16.7 … -12.9 | -18.5 … -12.2 |
| redqu | 63 | -15.3 | -16.9 … -13.1 | -17.5 … -12.4 |
| wtque | 32 | -16.5 | -18.9 … -13.4 | -21.3 … -12.9 |

- **Voice medians were close** (within ~1.2 LU). **Within-voice spread was much larger** (3-5.5 LU p10-p90, up
  to 9 LU min-max). Most of the variance came from peak normalization scattering individual segments, not from the
  reference clips.
- At voice changes, adjacent segments differed by a median of 1.3 LU, 3.9 LU at p90 and 7.9 LU at most.
- Typical loudness of a peak-normalized segment was about -16 LUFS (median -15.7).

## Decisions

| Question | Decision | Why |
|---|---|---|
| Where is the variance fixed? | At render time (concat and the playback paths), not at generation. | Saved segments stay as they are, so a re-concat applies to existing projects without regeneration, and render-time code knows the output channel count and the voice's pan. |
| Keep the post-generation peak normalization? | **Yes**, only as a predictable storage level. | It's a lossless linear gain; the leveling gain fully replaces it. STT, validation and saved files benefit from a consistent level. |
| Per voice or per segment? | **Per segment.** | The probe showed variance is mostly within voices; per-voice gain would leave it. Per-segment gain also fits the streaming render loop and the realtime paths. A single gain per segment keeps all dynamics *inside* the segment. |
| A "correction strength" setting? | **No.** Clamp and other limits are internal constants. | The clamp never engaged on the probe project; the variance removed is accidental, not performance. |
| Make leveling optional? | **No**, it is always on. | It fixes broken behavior. An opt-out would need a sensible un-leveled path first. |
| Loudness presets | **ACX standard** (default, -19 LUFS, -3 dBTP) and **Stronger** (-16 LUFS, -2.5 dBTP). "Disabled" removed. | ACX vs Stronger is a real listening-environment preference. "Disabled" just meant whatever peak normalization produced. With leveling now always on, projects saved with `"none"` adopt the Normal/default preset (`NORMALIZATION_LEGACY_IDS`), silently; they are not migrated to the louder preset. |
| Keep the whole-file `loudnorm`? | **Removed from export.** | Per-segment leveling to the preset target makes the file measure on target (validated below). It cost two full decode passes at 192 kHz internally, plus an intermediate FLAC and re-encode. `LoudnessNormalizationUtil` remains only for dev A/B scripts. |
| Short segments | **Measure from 0.4 s.** Below that, use the "typical loudness" gain. | In simulation, a "last good gain for this voice" fallback caused the *worst* jumps (up to 8.5 LU) on 0.56-0.99 s lines ("said Alice"). Measuring them directly worked. |
| Room tone and break effects | **Fixed offset relative to the target**, not leveled. | Both were authored against un-leveled speech, and the effects are intentionally quieter than speech (card flip -30 LUFS, page turn -24 LUFS). Leveling them to the speech target would make them as loud as speech. |

## Method

For each segment (mono), after all per-segment shaping (resample, upsampling, treble EQ, sentence-gap
extension, gap limiting) and before the pause, panning, reverb and room tone:

1. **Measure** integrated loudness (`LoudnessLeveling.measure_lufs`). Gating ignores silence. `None` if shorter
   than `MIN_MEASURE_SECONDS` (0.4 s) or silent.
2. **Gain** = target − measured, **clamped** to +`MAX_BOOST_DB` (6) / −`MAX_CUT_DB` (10). The clamp limits how
   far a segment is moved (a whispered or shouted line, or a glitch, isn't dragged all the way), not the final
   level. Unmeasurable segments get `reference_offset_db`, ie "assume it was at typical loudness".
3. **Limit** peaks with a lookahead limiter to the preset's ceiling (`ceiling_db_for`): `tp −
   TRUE_PEAK_MARGIN_DB` (0.5, covering inter-sample peaks and AAC overshoot), lowered further by the voice's **pan
   boost**. The center-unity pan law raises the louder channel by up to +3 dB (+1.6 dB at pan 0.3), so the
   ceiling before panning is lowered to keep the panned channel under the limit.

**Stereo target.** BS.1770 sums the channels' energy, so mono audio on both channels of a stereo file
measures 10·log₁₀2 = 3.01 dB louder. In stereo output, each mono segment is aimed 3.01 dB lower
(`target_lufs(specs, channels)`). The pan law keeps a panned voice's total energy equal to a centered one's, so
this holds for panned voices too. The first validation render, without this correction, measured -16.2 LUFS for
the ACX target.

**Reference offset.** Levels authored against un-leveled speech (room tone presets, break effect assets) are
shifted by `reference_offset_db = target_lufs − REFERENCE_SPEECH_LUFS (-16)`, eg -3 dB for ACX in mono and
-6 dB in stereo. That is what the old whole-file pass effectively did to them (it shifted everything by the
same amount as typical speech), so their level relative to the speech is unchanged.

**Mix limiter.** Room reverb adds its wet signal on top of the leveled, limited narration, and pushed peaks up to
+1.5 dB past the ceiling (only on the reverb voice). A final safety limiter (`limit_mix`) runs on each mixed chunk
(segment + pause/effect, after reverb and room tone, before ffmpeg) at the preset ceiling. It's usually a no-op.

**int16 safety.** `ConcatUtil.add_audio_to_ffmpeg_stream` now clips before the int16 cast. Out-of-range floats
don't saturate in that cast; they wrap around, which would come out as loud crackles.

### Limiter

`LoudnessLeveling.limit(data, sr, ceiling_db)`: channel-linked, fully vectorized:

1. Required gain per sample: `g = min(1, ceiling / |x|)` (max over channels).
2. Forward-looking minimum over the lookahead window (`LIMITER_LOOKAHEAD_S`, 5 ms), so the reduction is in place
   before the peak.
3. Release: the reduction in dB decays at `LIMITER_RELEASE_DB_PER_S` (30 dB/s) after the peak, computed in
   closed form as `cummax(r + rate·i) − rate·i`.
4. Backward moving average over the same window, which turns the attack into a ramp. Every value averaged at a
   peak's sample is already at or below that peak's required gain, so the ceiling is guaranteed (a final clip only
   absorbs float rounding).

Audio already under the ceiling is returned untouched.

## Where it runs

| Path | Leveling | Notes |
|---|---|---|
| Concat/export (`ConcatUtil.concatenate_sound_segments`, `loudness=`) | Per segment + mix limiter; room tone and effects offset | Output channels known exactly. The concat file is written in the final codec directly (no intermediate). A summary line is printed: segments, how many were peak-limited, mix-limited, clamped, too short to measure. |
| Realtime playback | Per segment (`prepare_generated_sound_for_playback(loudness=, pan=, output_channels=)`) + mix limiter; room tone and effects offset | Uses the *requested* channel count, since leveling runs before the device stream opens. A device fallback to mono would play 3 dB quieter. |
| Server (non-streaming), pronunciation preview | Per segment, mono target | No pan. |
| Voice chat (non-streaming) | Per segment + mix limiter | Uses the actual output stream's channel count (including mono fallback). Stereo output aims mono speech 3.01 dB lower. No pan; the final limiter runs after reverb/stereo expansion on speech and its trailing pause, before playback and saving. |
| Server and voice chat streaming | **Not leveled** | Plays raw model chunks as they arrive (no peak normalization either); a whole segment is never available to measure. Voice chat can still apply reverb. |

## Validation

The same `ttlg` project rendered through the new concat flow (stereo, two panned voices, one with
`small_drr17` reverb, room tone Low), measured with ffmpeg `ebur128`:

| | Integrated | True peak | LRA |
|---|---|---|---|
| Previous export (whole-file loudnorm, ACX) | -19.0 LUFS | -4.8 dBTP | 4.2 LU |
| **ACX**, FLAC | **-19.2 LUFS** | **-3.4 dBTP** | 2.8 LU |
| **ACX**, M4B via `make_file` (AAC 128k) | **-19.2 LUFS** | **-3.2 dBTP** | 2.8 LU |
| **Stronger**, FLAC | **-16.2 LUFS** | **-2.8 dBTP** | 3.0 LU |

Adjacent-segment loudness in the rendered output (|ΔLUFS| per segment, channels averaged):

| | Voice-change median | p90 | max |
|---|---|---|---|
| Before (saved segments, peak normalized) | 1.26 | 3.93 | 7.86 |
| After, ACX | 0.43 | 0.98 | 1.57 |

The remaining spread mostly comes from reverb tails and pauses inside each measured chunk, and from the limiter
on a few segments. ACX: 10 of 239 segments needed peak limiting (max 4.2 dB), 5 were touched by the mix limiter,
and the clamp engaged once. Stronger needs much more limiting (80 segments, max 6.7 dB), which is the cost of
being louder.

Speed: a full M4B export of the 13-minute project took 14 s end to end. One `loudnorm` pass alone took 13 s on the
same audio, and the old flow ran two plus a re-encode.

## Constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_BOOST_DB` / `MAX_CUT_DB` | 6 / 10 | Clamp on the leveling gain |
| `MIN_MEASURE_SECONDS` | 0.4 | pyloudnorm's minimum (one gating block) |
| `TRUE_PEAK_MARGIN_DB` | 0.5 | Sample-peak ceiling below the preset's dBTP limit |
| `REFERENCE_SPEECH_LUFS` | -16 | Typical loudness of a peak-normalized segment |
| `LIMITER_LOOKAHEAD_S` | 0.005 | Limiter attack / lookahead |
| `LIMITER_RELEASE_DB_PER_S` | 30 | Limiter release rate |

## Open questions

- **Expressive models.** Full per-segment correction erases deliberate between-line dynamics (a whispered line
  next to a shouted one). The clamp softens extremes. A partial-correction factor could be added if a model needs
  it.
- **Very short segments (< 0.4 s)** get the typical-loudness gain. None existed in the probe project.
- **Limiter character on Stronger.** It engages on most segments of a reverb-heavy stereo project; worth a
  listening check.
- **Realtime device fallback** to mono plays 3 dB quieter than intended (see table above).
- **Known issue: mix-limiter state resets at chunk boundaries.** Each `limit_mix` call starts a fresh
  release envelope. If a peak near the end of one chunk leaves attenuation active and the next chunk is
  below the ceiling, it returns unchanged, abruptly restoring unity gain. Continuous reverb across the
  boundary can therefore click. A synthetic probe confirmed the discontinuity; its frequency in real
  projects has not been measured. At 30 dB/s, 1.5 dB of reduction needs 50 ms to recover. Concat's included
  trailing pauses usually provide recovery time; realtime's separate speech, pause/effect and tail chunks
  are more exposed. The validation render's 5 mix-limited chunks out of 239 are not a count of audible
  clicks. Fix by preserving release state across mixed-output chunks, including under-ceiling chunks;
  seamless lookahead across boundaries also needs a small pending buffer and an end-of-stream flush.
