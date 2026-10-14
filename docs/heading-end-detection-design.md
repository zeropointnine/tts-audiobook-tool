# Dynamically detecting heading endings

## Status and purpose

This document describes the implemented refinement of [PR #45: Detect EPUB headings and pause longer after them](https://github.com/zeropointnine/tts-audiobook-tool/pull/45), including the policy and motivations behind it.

An opening heading already present in the spoken text receives a longer subsequent pause, without a scene-break sound effect. The reason is named `Reason.HEADING_END` to describe the ending boundary rather than imply a general text-content type.

The detection policy below is implemented by a shared global recomputation pass after EPUB segmentation and section assembly, and on confirmed text-editor save before atomic persistence.

## Problem

EPUB navigation supplies logical sections and their titles, such as a chapter title, “Dedication,” or “Acknowledgments.” The importer already assembles these logical sections; they need not correspond one-to-one with XHTML documents.

A section title may also appear at the beginning of its reading text. It may use an HTML heading element, a CSS-styled paragraph, or ordinary paragraph markup. It may differ from the navigation label: for example, navigation may say “Chapter 1: Priya” while the body says only “PRIYA.” Alternatively, the title may be entirely absent from the reading text.

Without special handling, these opening headings receive ordinary text-boundary treatment. Their identity is also not retained for special browser-reader presentation. This implementation addresses the audio boundary only; reader styling remains outside its scope.

## Existing representation and scope

The application does not currently retain a general heading/body/dialog content-type classification. Text segments are TTS prompt units; their ending `Reason` controls subsequent pause and related boundary behavior. Dialog detection informs voice assignments rather than introducing a retained dialog content type.

Most existing reasons can be recovered from text and segmentation rules. `SECTION_BREAK` additionally depends on section structure. An imported heading reason inferred solely from HTML would likewise require information that is no longer available in the editable text.

A general content-type system would more formally separate “what this text is” from “why this segment ends.” It could also support reader presentation. However, that broader model is not required for this narrowly scoped audio feature. Emotion or delivery tags are a separate concern and are not, by themselves, a reason to introduce a content taxonomy.

`HEADING_END` is a pragmatic semantic boundary within the existing ordered reason model. Its placement between `PARAGRAPH` and `SPACE_BREAK` expresses a boundary-strength policy, not a claim that headings naturally form another level of punctuation or layout.

## Policy

### 1. Classification is dynamic and automatic

`HEADING_END` is computed from the current section structure, section title, and current text segments. It is not a user-assigned classification and must not become sticky merely because it was assigned during import.

An edit can therefore make an opening paragraph qualify, cease to qualify, or continue to qualify. The classification must reflect the current inputs rather than preserve an obsolete import decision.

One shared, global recomputation pass runs after EPUB import has completed segmentation and logical-section assembly. The same pass runs on confirmed text-editor save, through `ProjectTextEditUtil.commit`, against a detached copy of the staged book before atomic persistence. This ensures the saved book reflects the final text and section structure rather than intermediate edits.

Before reclassifying, the pass clears stale `HEADING_END` assignments back to reasons derived from the current text: `SPACE_BREAK` for an ornamental ending or at least three trailing line feeds, `PARAGRAPH` for one or two trailing line feeds, and `SENTENCE` otherwise. It then evaluates each section's opening candidate and applies the precedence rules below. This is global recomputation, not preservation of old assignments or reconsideration of previously recognized headings alone.

**Motivation:** preserving HTML-derived heading identity through arbitrary edits would require rules for replacement, splitting, merging, and deletion. A reproducible classifier avoids maintaining that hidden semantic state. It also avoids adding a manual heading-boundary control.

The computed reason is serialized using the JSON code `"h"`, alongside other phrase reasons. It is derived state: the next import or confirmed editor-save pass revalidates it against the current inputs. Reclassification does not run on every load, keystroke, or playback request. Spot edits no longer preserve `HEADING_END` as immutable semantic metadata.

### 2. Only the opening paragraph of a logical section is eligible

Start at the first text segment of each logical section. Accumulate its opening paragraph, stopping at the first line break or the section's end. In the application's terminology, that line break is a paragraph boundary; it does not require an empty line. Section end also completes the candidate without a trailing newline, provided it is reached within the five-segment budget; `SECTION_BREAK` still takes precedence over a heading match.

Do not continue looking for a heading in later paragraphs if the first paragraph fails to match. Do not skip an opening epigraph or other prose to search for a more plausible heading farther down.

**Motivation:** position is strong evidence and limits false positives. Later mentions of a chapter title, repeated words, or internal headings should not accidentally acquire a heading pause. Logical section starts, rather than XHTML document starts, align the rule with the application's existing section model.

**Deliberate limitation:** headings preceded by other readable text, and titles spread across multiple paragraphs, are not recognized by this policy.

### 3. Inspect at most five text segments

Use a hardcoded scan limit of **five text segments per section**, stopping earlier at the opening paragraph's first line break or the section's end. Here, “text segment” means the application's user-visible TTS prompt unit (`PhraseGroup`), not an individual internal `Phrase`.

A candidate can span multiple segments, but only its ending segment can receive `HEADING_END`, subject to boundary precedence. Never compare a truncated prefix: if neither a line break nor section end completes the opening paragraph within the first five segments, do not classify it as a heading.

**Motivation:** this bounds the scan and keeps it focused on short section openings. It avoids complex or unbounded heading reconstruction and prevents the scan limit itself from manufacturing an apparent title match.

The limit intentionally depends on segmentation: the same unusually long opening paragraph can become ineligible under a strategy that splits it into more than five segments.

If the opening paragraph and later body text share one `PhraseGroup`, do not assign `HEADING_END` to that group. Its reason controls the pause after the entire prompt unit, which would place the heading pause after body text rather than after the heading. This is a pragmatic limitation for manually edited segmentation.

### 4. Compare the complete candidate against the section title

Pass the book's language code to the matcher. For all languages, first compare the entire opening paragraph against the full section title using conservative normalization:

- collapse whitespace;
- compare case-insensitively.

Require a nonempty candidate and a nonempty title. Do not use substring matching, fuzzy similarity, broad punctuation stripping, or a book-wide collection of known heading strings.

**Motivation:** exact normalized equivalence is understandable and reproducible. Matching against this section's own title avoids turning unrelated prose into a heading merely because the same text was a heading elsewhere in the book.

An absent body title is not inserted or spoken by this feature. A section with no usable title supplies no title-based match.

### 5. Allow a small English title-equivalence layer

If the full-title comparison fails and the supplied language code identifies English, allow a small, explicit set of English-language transformations for common differences between navigation titles and body headings. Regional English codes also qualify. Do not apply English chapter-prefix equivalences for non-English or absent language codes; basic case/whitespace normalization remains available for all languages.

The implemented equivalence removes a leading chapter label and number, with a recognized separator, from the metadata title. The entire remaining title is compared against the entire opening paragraph using the same normalization. Only the metadata title receives prefix removal; the body candidate is not rewritten.

Examples:

| Section title | Opening paragraph | Result |
| --- | --- | --- |
| `Acknowledgments` | `ACKNOWLEDGMENTS` | Match by normalization |
| `Chapter 1: Priya` | `PRIYA` | Match after removing chapter prefix |
| `Chapter 12 — Homecoming` | `Homecoming` | Match after removing chapter prefix |
| `Chapter 1: Priya` | `Priya entered the room.` | No match |
| `Chapter 1` | `An ordinary opening paragraph.` | No match |

Require a nonempty remainder. Do not infer that arbitrary short text is a heading merely because it starts a section.

**Motivation:** the body often omits a navigation label's “Chapter N” prefix. A few explicit transformations recover useful cases without escalating into general heading inference.

The initial implementation accepts a chapter number written in Arabic digits or Roman-numeral letters, followed by a colon, period, hyphen, en dash, or em dash. A separator and nonempty remainder are required. Spelled-out numbers, parts, prologues, translated labels, or other title conventions are not inferred.

### 6. Preserve explicit structural boundary precedence

A matching candidate replaces an ordinary paragraph ending or whitespace-derived `SPACE_BREAK` with `HEADING_END`. Extra blank lines alone must not prevent a recognized opening heading from receiving its heading pause.

Explicit ornamental separators and `SECTION_BREAK` retain precedence and are not replaced by `HEADING_END`. Recognition and precedence are distinct: a candidate can match while its explicit separator or section-ending boundary still governs audio treatment.

**Motivation:** whitespace around a heading is layout rather than an explicit scene divider. An ornament or section boundary carries independent structural meaning that title recognition must not erase.

## Audio behavior

For a qualifying heading-ending boundary:

- use a heading pause of **2.0 seconds** under Normal and **1.5 seconds** under Shorter;
- do not add a scene-break sound effect merely because this is a heading ending;
- clear TTS continuation at that ending, as for an ordinary paragraph boundary;
- do not combine subsequent body text across that boundary.

The feature does not introduce a different voice, emotional delivery, or model-specific prompt instruction for headings.

## Relationship to the original PR

The original PR supplied a useful initial implementation and examples. That implementation:

1. extracted heading strings from `<h1>`–`<h6>` elements;
2. added a document's first paragraph when it matched that document's title;
3. collected those strings into a book-wide set;
4. marked matching paragraph endings during segmentation;
5. preserved the imported heading reason during spot edits.

The refined implementation retains the audio intent but replaces that detection path and lifecycle:

| Original approach | Implemented policy |
| --- | --- |
| HTML headings and document-title matches | Current logical section title and current opening paragraph |
| Match against book-wide heading strings | Match against this section's title only |
| Any matching paragraph can qualify | Only the first paragraph is eligible |
| Imported reason preserved on editing | Classification recomputed from current inputs |
| No explicit candidate segment cap | At most five text segments |

HTML tags and CSS do not become an independent classification path under this policy: those inputs are unavailable after ordinary text editing. Consequently, an explicit HTML heading whose text does not match an accepted form of the section title will not qualify. Internal HTML headings likewise fall outside this opening-heading feature.

This narrower coverage is intentional. The aim is a simple rule that remains valid after import, not full preservation of EPUB typography.

## Concrete examples from local EPUB inspection

- **The Jasmine Throne:** navigation says “Chapter 1: Priya,” while the body begins with an `<h1>` containing “PRIYA.” Ordinary segmentation gives that paragraph a Normal pause of 1.2 seconds. The implemented chapter-prefix equivalence recognizes the heading without needing the original HTML.
- **Service Model:** navigation says “Acknowledgments,” while the body begins with “ACKNOWLEDGMENTS” inside a styled `<p>`. Normalized title matching recognizes it without CSS interpretation.
- **Service Model, “About the Author”:** the opening `<h1>` is followed by three line feeds, so ordinary segmentation produces `SPACE_BREAK`. The original PR detects the heading text but only replaces `PARAGRAPH` reasons, leaving this case unchanged. Under the settled policy, a matching title replaces this whitespace-derived `SPACE_BREAK` with `HEADING_END`; an explicit ornament or `SECTION_BREAK` would instead retain precedence.

These examples illustrate different markup and title conventions; they are not grounds for book-specific detection rules.

## Implementation locations

- [heading_end_detector.py](../tts_audiobook_tool/text_ops/heading_end_detector.py) implements `heading_title_matches` and the shared `recompute_heading_ends` pass. It changes reasons only, leaving text, grouping, and voice assignments unchanged.
- [epub_extractor.py](../tts_audiobook_tool/text_ops/epub_extractor.py) runs the pass over assembled, segmented logical sections after marking their section endings.
- [project_text_edit_util.py](../tts_audiobook_tool/project_support/project_text_edit_util.py) runs the pass before atomically saving the detached staged book, on the confirmed-save path used by [text_editor.py](../tts_audiobook_tool/textual/text_editor.py).
- [text_edit_session.py](../tts_audiobook_tool/text_ops/text_edit_session.py) no longer preserves the heading reason blindly during isolated spot edits.

## Validation coverage

Regression coverage is implemented in [test_heading_end_detector.py](../tests/test_heading_end_detector.py), [test_epub_extractor.py](../tests/test_epub_extractor.py), and [test_project_text_edit_util.py](../tests/test_project_text_edit_util.py). It covers:

- exact and case/whitespace-normalized opening-title matches;
- English chapter-prefix equivalences and the nonempty-remainder requirement, gated by English language codes including regional variants;
- basic case/whitespace normalization for non-English and absent language codes without English prefix transformations;
- no classification when the title is absent or does not match;
- no scanning beyond the first paragraph;
- no matching of repeated title text later in this or another section;
- candidate reconstruction across several segments;
- completion within the fifth segment versus exceeding the cap, including section end without a trailing newline;
- no heading assignment when the opening paragraph shares a prompt unit with later body text;
- edits that make a candidate gain, retain, or lose its match, clearing stale heading reasons to current text-derived reasons first;
- shared global recomputation after EPUB segmentation/section assembly and on confirmed editor save through `ProjectTextEditUtil.commit`, on the staged book before atomic persistence;
- classification at logical section starts, including navigation targets inside one XHTML document;
- replacement of whitespace-derived `SPACE_BREAK`, while explicit ornaments and `SECTION_BREAK` retain precedence.

Reader styling, a general content-type system, fuzzy title matching, arbitrary CSS interpretation, and automatic insertion of absent titles are outside this feature's scope.
