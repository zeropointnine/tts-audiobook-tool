# MOSS voice clone modes

This note analyzes the MOSS voice clone modes used by the audiobook tool, especially whether `MossVoiceCloneMode.CONTINUATION` is a forced adaptation for single-speaker audiobook generation or a justifiably interesting experiment.

These experiments are based on the `OpenMOSS-Team/MOSS-TTS-v1.5` model. A separate round of experimentation may be useful later with `OpenMOSS-Team/MOSS-TTS-Local-Transformer`, but the observations in this note should not be assumed to apply unchanged to that checkpoint.

## Summary

On its own terms, `MossVoiceCloneMode.CONTINUATION` is somewhat forced for single-speaker audiobook generation, but not stupidly forced. It is a defensible experiment with a clear hypothesis.

The forced part is that continuation mode treats the voice reference as if it were the already-spoken prefix of the same utterance or story, not merely a speaker/timbre sample. In `MossModel.generate()`, continuation mode builds a conversation where the user text is:

- reference transcript plus current prompt via `MossModel.build_continuation_text()`
- followed by assistant audio containing only the reference audio via `build_assistant_message()`
- then asks the model in `continuation` mode to generate the remainder

That is semantically different from ordinary voice cloning. Ordinary clone mode says: "speak this new text in the style/timbre of this reference." Continuation mode says: "this reference audio corresponds to the first part of the full text; now continue after it."

## Why static continuation is forced

For audiobooks, static continuation can be awkward because the reference clip is usually not actually the immediately preceding text. It may be an arbitrary speaker sample. So if the transcript/reference pair is unrelated to the current passage, continuation is conceptually lying to the model.

That makes transcript leakage, repeated phrasing, prosody carryover, or strange boundary behavior unsurprising. The local code comment about leakage in `MossVoiceCloneMode` fits this risk.

## Why it is still interesting

Static continuation is still justifiably interesting because audiobook generation has a real problem that plain voice cloning does not solve well: segment-to-segment continuity.

A single-speaker audiobook wants not just consistent timbre, but also stable cadence, loudness, affect, breathiness, pacing, and narrative line. Clone mode may preserve voice identity while still sounding like disconnected isolated samples. Continuation-style prompting is a plausible way to bias the model toward treating the new text as part of an ongoing read.

## Mode comparison

| Mode | Conceptual fit for audiobook | Risk | Novelty/usefulness |
|---|---:|---:|---:|
| `MossVoiceCloneMode.CLONE` | High | Low | Baseline / correct default |
| `MossVoiceCloneMode.CONTINUATION` | Medium-low if reference is arbitrary | Medium-high | Interesting experiment, but semantically impure |
| `MossVoiceCloneMode.ROLLING_CONTINUATION` | Medium-high if reset at boundaries | Medium | More justifiable for audiobooks than static continuation |

## Static continuation vs rolling continuation

The most defensible continuation variant for audiobooks is `MossVoiceCloneMode.ROLLING_CONTINUATION`, not static `MossVoiceCloneMode.CONTINUATION`.

In rolling mode, once generation starts, the previous generated segment really is the preceding spoken context. That makes the continuation story much less fake. The bootstrap path uses clone-style generation first, then subsequent segments use the actual generated previous text/audio as continuation context through `build_assistant_message()`.

This is conceptually cleaner than static continuation, because the model is no longer being told that an arbitrary reference clip is the immediate prefix of the requested passage.

## Verdict

- Static continuation is a clever but forced adaptation. It is useful as an A/B experiment, not something to present as the primary audiobook mode.
- Rolling continuation is genuinely interesting and more novel. It is a reasonable attempt to solve audiobook continuity using the model's continuation pathway.
- Clone mode should remain the default and safest production path.

If static continuation remains available, it is best understood as "reference-prefix continuation" rather than ordinary voice cloning. Its success depends heavily on whether the reference clip/transcript is stylistically and textually compatible with the target passage. If the reference is just an arbitrary voice sample, it is forced. If the reference is intentionally chosen as a narrative lead-in, such as from the same book, same narrator, same paragraph, or same scene style, then it becomes much more defensible and potentially useful.
