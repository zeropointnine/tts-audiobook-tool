# Section Reason Semantics

This note describes the current `Reason.SPACE_BREAK` behavior in the text segmentation,
audio concatenation, and browser-player display pipeline. The current implementation is
pragmatic and should be revisited with a more structured model of headings, sections,
layout, and prosody.

## Current pipeline

`tts_audiobook_tool/text_ops/phrase_segmenter.py` converts input text into `Phrase`
objects. A phrase receives a `Reason` describing why the segment ended:

- `Reason.SENTENCE` for ordinary sentence endings.
- `Reason.PARAGRAPH` for paragraph-like line breaks.
- `Reason.SPACE_BREAK` for text ending with three or more line breaks.

`tts_audiobook_tool/text_ops/phrase_grouper.py` groups phrases into `PhraseGroup`
objects. `PhraseGroup.last_reason` is derived from the final phrase in the group, so a
phrase-level `Reason.SPACE_BREAK` becomes the group-level section signal used downstream.

Reasons are preserved as segmentation produces them; pauses and browser display follow
the stored reasons directly. Break *sound effects* are decided at render time by the
rules below, so a `Reason.SPACE_BREAK` no longer implies that a sound effect will play.

The browser player currently receives timed text segments without explicit `Reason`
metadata. When break audio is present, `browser_player/book-text.js` infers
horizontal rule placement from displayed text ending with three or more line feeds. This
means section-like display is currently coupled to whitespace rather than an explicit
semantic marker in app metadata.

## Break sound effect rules (render time)

Break sound effects are governed by exactly two rules, implemented identically for
concatenation (`tts_audiobook_tool/concat_util.py`) and real-time playback
(`tts_audiobook_tool/real_time_playback.py`) by `BreakEffectTracker` in
`tts_audiobook_tool/sound/sound_pipeline.py`:

1. The first sound segment of a section never gets a SPACE_BREAK effect.
2. In a run of consecutive sound segments that would each get a SPACE_BREAK
   effect, only the first keeps it; the rest fall back to their configured
   silence pause.

Notes on the rules' semantics:

- A "sound segment" is a segment that actually emits audio (a present file during
  concat, a successful generation during realtime playback). Missing or failed
  segments do not advance tracker state.
- Rule 2 counts run membership on the *would-be* assignment, so a segment
  suppressed by rule 1 still anchors its run: its followers are suppressed as
  well. Any segment whose reason is not SPACE_BREAK resets the run.
- SECTION_BREAK effects are never suppressed by these rules (for example, a
  single-group EPUB section still gets its page-turn). The final segment of an
  output file or playback range never gets any break effect.
- When a SPACE_BREAK effect is suppressed, the segment still gets its configured
  `Reason.SPACE_BREAK` pause; only the sound effect is dropped.
- Section starts come from `ProjectBookUtil.get_section_start_indices`: Book
  sections for structurally sectioned books (eg EPUB), user-configured section
  markers for single-section books (eg plain text).

These rules replace the earlier mitigation layers: the render-time
`is_first_in_section` gate inside `SoundPipeline.should_append_break_sound_effect`
(removed), `PhraseSegmenter.downgrade_consecutive_space_breaks` (removed), and
`EpubExtractor.downgrade_leading_section_groups` (removed).

Consequences of removing the import-time downgrades:

- Consecutive SPACE_BREAK reasons keep their space-break pauses (previously the
  downgraded spots got shorter paragraph pauses).
- The browser player shows a horizontal rule at each blank-line run (previously
  consecutive ones were hidden by rewriting trailing whitespace).

## EPUB logical-section boundaries

EPUB import assembles readable spine content into logical sections defined by EPUB
navigation, and each logical section is segmented independently. The importer force-marks
the final phrase of each logical section as `Reason.SECTION_BREAK`. Ordinary transitions
between XHTML spine documents inside one logical section do not receive this reason.

Redundant leading space breaks at the start of the next logical section are no longer
downgraded at import time; rule 1 above suppresses their sound effects at render time
instead.

## Known architectural debt

The current system conflates several distinct concepts:

- source whitespace and layout,
- text segmentation boundaries,
- prosody/pause duration,
- generated section-break sound effects,
- browser horizontal-rule display,
- chapter/bookmark/file divider semantics.

The break effect rules are now a single render-time concern, but display and prosody
remain coupled to whitespace-derived reasons. A more structured approach should likely
separate these concerns. For example, future metadata could carry explicit layout
markers or structural annotations rather than asking the browser player to infer
horizontal rules from trailing line feeds. Similarly, section sound effects and long
pauses could be controlled by explicit semantic markers rather than being coupled
directly to whitespace-derived `Reason.SPACE_BREAK` values.
