# Section Reason Semantics

This note describes the current `Reason.SECTION` behavior in the text segmentation,
audio concatenation, and browser-player display pipeline. The current implementation is
pragmatic and should be revisited with a more structured model of headings, sections,
layout, and prosody.

## Current pipeline

`tts_audiobook_tool/text_ops/phrase_segmenter.py` converts input text into `Phrase`
objects. A phrase receives a `Reason` describing why the segment ended:

- `Reason.SENTENCE` for ordinary sentence endings.
- `Reason.PARAGRAPH` for paragraph-like line breaks.
- `Reason.SECTION` for text ending with three or more line breaks.

`tts_audiobook_tool/text_ops/phrase_grouper.py` groups phrases into `PhraseGroup`
objects. `PhraseGroup.last_reason` is derived from the final phrase in the group, so a
phrase-level `Reason.SECTION` becomes the group-level section signal used downstream.

During concat, `tts_audiobook_tool/concat_util.py` passes flattened phrases to the sound
pipeline. `tts_audiobook_tool/sound/sound_pipeline.py` uses the phrase reason to append
either silence or, when enabled, the section-break sound effect. Therefore,
`Reason.SECTION` is not just a descriptive label; it controls audible behavior.

The browser player currently receives timed text segments without explicit `Reason`
metadata. When section-break audio is present, `browser_player/book-text.js` infers
horizontal rule placement from displayed text ending with three or more line feeds. This
means section-like display is currently coupled to whitespace rather than an explicit
semantic marker in app metadata.

## Problem with consecutive section assignment

Some EPUB-to-text converters emit multiple blank lines around adjacent heading-like
blocks. For example:

```text
Chapter 1


The Beginning


Prose starts here. Etc.
```

Naively treating every segment ending with three or more line breaks as
`Reason.SECTION` gives both `Chapter 1` and `The Beginning` section-like prosody. That
overstates the intended structure: the chapter number may reasonably mark the section,
but the immediately following title should usually not trigger another section pause,
section-break sound effect, or horizontal rule.

## Tactical mitigation

The segmenter performs a final O(n) phrase-level pass after basic phrase segmentation and
ornamental-line merging. The pass preserves the first `Reason.SECTION` in an immediate
run, then downgrades subsequent immediate `Reason.SECTION` phrases to
`Reason.PARAGRAPH`.

This mitigation is guarded by the hardcoded `DOWNGRADE_CONSECUTIVE_SECTIONS` constant in
`tts_audiobook_tool/text_ops/phrase_segmenter.py`. It currently defaults to `True`.

When a phrase is downgraded, trailing section spacing is also reduced from section-like
spacing to paragraph spacing. This keeps browser horizontal-rule inference aligned with
the corrected semantic reason.

The rule is intentionally phrase-level rather than `PhraseGroup`-level because:

- `Reason` is assigned to `Phrase` objects.
- `PhraseGroup.last_reason` is derived state.
- Downstream grouping, concat, real-time playback, and metadata generation all depend on
  phrase reasons.

The rule only applies to immediate consecutive section reasons. A non-section phrase
between two sections resets the sequence, allowing later genuine section breaks to remain
`Reason.SECTION`.

## EPUB spine boundaries

EPUB import has an additional structural rule. Each retained EPUB spine document is
segmented independently, and the importer force-marks the final phrase of that document
as `Reason.SECTION`. This marks the boundary between EPUB HTML files and is not the same
as ordinary whitespace-derived section detection.

When a previous retained spine document has already ended with this forced section
boundary, section-like groups at the start of the next spine document are treated as
redundant heading/layout artifacts and downgraded to `Reason.PARAGRAPH`. The forced
boundary remains on the previous document's final phrase, while the new document's own
final phrase is still force-marked as `Reason.SECTION` after segmentation.

This EPUB-specific mitigation is guarded by the hardcoded
`DOWNGRADE_LEADING_SECTIONS_AFTER_EPUB_BOUNDARY` constant in
`tts_audiobook_tool/text_ops/epub_extractor.py`. It currently defaults to `True`.

## Known architectural debt

The current system conflates several distinct concepts:

- source whitespace and layout,
- text segmentation boundaries,
- prosody/pause duration,
- generated section-break sound effects,
- browser horizontal-rule display,
- chapter/bookmark/file divider semantics.

A more structured approach should likely separate these concerns. For example, future
metadata could carry explicit layout markers or structural annotations rather than asking
the browser player to infer horizontal rules from trailing line feeds. Similarly, section
sound effects and long pauses could be controlled by explicit semantic markers rather
than being coupled directly to whitespace-derived `Reason.SECTION` values.
