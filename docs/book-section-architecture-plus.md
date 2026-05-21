# Book Section Architecture

## Motivation

The project originally treated book text as one flat runtime list:

```python
Project.phrase_groups: list[PhraseGroup]
```

That flat model still fits core TTS/runtime behavior well because each `PhraseGroup`
maps naturally to:

- one generated sound segment,
- one flat line number in the UI,
- one regeneration/validation unit,
- one timing/text segment in exported playback metadata.

Over time, however, several structural concerns were layered onto that same flat view:

- `Phrase.reason`, especially `Reason.SPACE_BREAK`, influences segmentation/prosody,
- section-break audio can be triggered from `Reason.SPACE_BREAK`,
- the browser player can insert visible separators from text formatting heuristics,
- `Project.section_dividers` / `chapter_indices` mark flat indices where sections/files begin,
- EPUB import introduces spine-document boundaries that are structural rather than purely local text breaks.

That made `Reason.SPACE_BREAK` do too much. In practice it has been used to suggest some mix of:

1. a strong local text break,
2. a section-break sound effect trigger,
3. a visible reader/player separator,
4. an EPUB chapter/spine boundary,
5. a chapter/file export boundary.

The need for an explicit structural layer was real. The flat generation/runtime model was
still valuable, but the app needed a better source-of-truth representation for book text.

## Current State Summary

The codebase is now roughly halfway through the architecture described in the original
version of this document.

### Implemented

The structural text model now exists and is wired into project persistence:

- `BookSection` exists in `tts_audiobook_tool/app_types/__init__.py`
- `BookSegmentationSettings` exists
- `Book` exists with helper methods:
  - `book.phrase_groups()`
  - `book.section_start_indices()`
  - `book.section_ranges()`
- `Project.book` exists in `tts_audiobook_tool/project.py`
- `project_text.json` now persists the structural model as `book.v1`
- legacy `phrase_groups.v1` and bare-list text payloads are still supported on load
- legacy payloads are migrated forward to `book.v1`
- `Book.segmentation_settings` is the canonical persisted source for text-segmentation provenance
- `Project` keeps flat compatibility fields synchronized from `book`
- `Project` now exposes compatibility helpers:
  - `get_flat_phrase_groups()`
  - `get_section_start_indices()`
  - `get_section_ranges()`
- plain-text/manual text import creates a one-section `Book`
- EPUB import creates a multi-section `Book` via `Project.set_phrase_groups_chapters_and_save(...)`

### Not Implemented Yet

Several downstream consumers still operate directly on flat compatibility fields instead of
the newer section helpers:

- `ChapterInfo.make_chapter_infos`
- `sound/m4b_chapter_util.py`
- concat flows and section-marker UI flows
- browser/player section rendering metadata

EPUB import also still carries some pre-`Book` behavior:

- it still builds flat `phrase_groups` plus flat `section_dividers`,
- it still marks EPUB boundaries with `Reason.SPACE_BREAK`,
- it still downgrades some leading consecutive section groups after EPUB boundaries,
- the browser player still infers visible separators rather than consuming explicit section metadata.

So the project now has a real structural source model, but many operational consumers still
depend on the old flat compatibility layer.

## Current Model in Code

The important distinction now is:

```python
Project  # runtime/workspace state, still heavily flat-compatibility oriented
Book     # canonical persisted book text model
BookSection  # structural subdivision of the book
```

The current shared types are:

```python
@dataclass
class BookSection:
    phrase_groups: list[PhraseGroup]
    title: str = ""


class BookSegmentationSettings(NamedTuple):
    language_code: str = ""
    max_words_per_segment: int = 0
    strategy: SegmentationStrategy = SegmentationStrategy.NORMAL


@dataclass
class Book:
    sections: list[BookSection]
    title: str = ""
    text_source_kind: str = ""
    audio_source_kind: str = ""
    segmentation_settings: BookSegmentationSettings = BookSegmentationSettings()

    def phrase_groups(self) -> list[PhraseGroup]: ...
    def section_start_indices(self) -> list[int]: ...
    def section_ranges(self) -> list[tuple[int, int]]: ...
```

`Book.section_ranges()` currently returns half-open ranges:

```python
[(start_index, end_index), ...]
```

where `end_index` is exclusive.

`Project` still retains flat fields because many subsystems remain flat-indexed:

```python
Project.book
Project.phrase_groups
Project.section_dividers
```

At load/save time, the code keeps these views aligned.

## Project/Book Compatibility Layer

`Project.book` is now the persisted text source of truth, but `Project` still mirrors flat
compatibility data for the rest of the application.

Current sync behavior in `tts_audiobook_tool/project.py`:

```python
Project.phrase_groups == Project.book.phrase_groups()
Project.section_dividers == Project.book.section_start_indices()[1:]
```

And the applied segmentation fields are derived from `book.segmentation_settings` when a
book is present:

```python
Project.applied_language_code
Project.applied_max_words
Project.applied_strategy
```

Important nuance: those legacy `applied_*` fields still exist on `Project` for runtime and
compatibility purposes, but they are no longer persisted in `project.json` when the book-based
path is used. Tests now explicitly verify that save/load removes those stale fields from the
project settings file.

So the architectural transition is:

- canonical persisted provenance: **moved to `Book.segmentation_settings`**
- flat compatibility/runtime mirrors: **still present on `Project`**

## Storage Shape

The current persisted text format is now the structural one:

```json
{
  "format": "book.v1",
  "book": {
    "title": "Example Book",
    "text_source_kind": "epub",
    "audio_source_kind": "generated",
    "segmentation_settings": {
      "language_code": "en",
      "max_words_per_segment": 120,
      "strategy": "max_len"
    },
    "sections": [
      {
        "title": "Chapter 1",
        "phrase_groups": [
          [
            { "text": "One.", "reason": "s" }
          ]
        ]
      }
    ]
  }
}
```

The loader still accepts:

```json
{
  "format": "phrase_groups.v1",
  "phrase_groups": [...]
}
```

and also bare legacy phrase-group lists. Those are converted in memory to a single-section
or divider-derived `Book`, then migrated back out as `book.v1` on save.

This migration path is implemented in:

- `tts_audiobook_tool/app_types/book_serialization.py`
- `tts_audiobook_tool/project_support/project_util.py`
- `tts_audiobook_tool/project.py`

## Import Flows as Implemented Today

### Plain text / manual text

Plain text currently maps cleanly onto the new model. The project creates:

```python
Book(
    text_source_kind="plain_text" or "manual",
    audio_source_kind="generated",
    segmentation_settings=...,
    sections=[BookSection(phrase_groups=phrase_groups)],
)
```

This path is already aligned with the intended architecture.

### EPUB import

EPUB import is only partly transitioned.

What it does today:

1. Extracts chapter/spine text.
2. Produces flat `phrase_groups`.
3. Produces flat `section_dividers` at EPUB chapter boundaries.
4. Passes those into `Project.set_phrase_groups_chapters_and_save(...)`.
5. That method converts the flat data into a multi-section `Book` and also syncs the flat
   compatibility fields.

So the saved project ends up with a real `Book`, but the EPUB extractor itself still thinks
in terms of flat phrase groups and divider indices.

It also still contains old boundary-encoding behavior. In
`tts_audiobook_tool/text_ops/epub_extractor.py`:

- `section_dividers.append(len(phrase_groups))` is still used,
- `DOWNGRADE_LEADING_SECTIONS_AFTER_EPUB_BOUNDARY = True` still exists,
- the final phrase of a chapter can still be forced to `Reason.SPACE_BREAK`.

That means EPUB structural boundaries are currently represented **both** as:

- `BookSection` boundaries after project construction, and
- legacy `Reason.SPACE_BREAK` / divider-based compatibility signals during import.

### Enhance-existing-audiobook flow

The structural model was explicitly designed for this flow, but the current codebase state
should be described conservatively: the shared `Book` model is ready for it, and the type
docs already describe `Book` as alignment reference text as well as TTS input, but this
document should not claim a complete end-to-end architecture transition here unless that is
verified separately.

## Relationship to `Phrase.reason`

The original architectural goal still stands: `Reason.SPACE_BREAK` should eventually narrow to
mean a local text-derived section break inside a `BookSection`, not an EPUB document boundary.

That is **not** fully true yet.

Current status:

- plain-text/local segmentation still uses `Reason.SPACE_BREAK` for strong local breaks,
- EPUB import still uses `Reason.SPACE_BREAK` as part of boundary handling,
- compensating EPUB logic still exists to downgrade repeated section markers after a boundary.

So `Reason.SPACE_BREAK` remains somewhat overloaded in the current codebase, even though the app
now has the structural types needed to fix that cleanly.

## Relationship to `section_dividers`

`section_dividers` has not been removed. It is still an active runtime compatibility field.

This is deliberate and currently necessary because many important subsystems still work in
flat segment indices:

- generated sound segment lookup,
- chapter/file concat behavior,
- M4B chapter metadata generation,
- section-marker editing UI,
- older chapter/range helpers,
- some browser-player assumptions.

The architecture is therefore currently dual-view:

```python
Project.book              # canonical persisted structural text
Project.phrase_groups     # flat compatibility/runtime view
Project.section_dividers  # flat compatibility/runtime section starts
```

The new helper methods on `Project` exist specifically to ease migration away from direct
`section_dividers` access:

```python
project.get_flat_phrase_groups()
project.get_section_start_indices()
project.get_section_ranges()
```

However, not all consumers use them yet.

## What Has Moved Already

These parts of the original plan are now complete enough to rely on:

### 1. Structural types exist

`Book`, `BookSection`, and `BookSegmentationSettings` are in the shared app-types layer.

### 2. Serialization/deserialization exists

`tts_audiobook_tool/app_types/book_serialization.py` now handles:

- `book.v1` serialization,
- `book.v1` parsing,
- legacy `phrase_groups.v1` parsing,
- bare-list legacy parsing,
- conversion of legacy text payloads to a one-section `Book`.

### 3. `Project.book` is real

`Project` now owns a `book` field and synchronizes flat compatibility state from it.

### 4. Save/load migration is real

When loading older projects, the app can:

- load legacy external text payloads,
- convert them to `Book`,
- save them back as `book.v1`,
- remove stale persisted `applied_*` settings from `project.json`.

### 5. EPUB imports now end up as multi-section `Book`s

Even though the extractor is still flat-oriented internally, the final project state after
EPUB import includes structured `BookSection`s with section titles.

## What Still Has Not Moved

These parts of the earlier design remain pending.

### 1. Section-aware consumers still use flat indices directly

Examples verified in the codebase today:

- `app_types/chapter_info.py` still calls `make_file_line_ranges(project.section_dividers, len(project.phrase_groups))`
- `sound/m4b_chapter_util.py` still builds chapter metadata from `project.section_dividers`
- concat menus still use the flat divider model
- section-markers menu still edits `state.project.section_dividers` directly

### 2. EPUB import still encodes structural boundaries redundantly

The extractor still:

- emits flat section-divider indices,
- forces/sustains `Reason.SPACE_BREAK` semantics around chapter boundaries,
- retains downgrade logic for consecutive section markers after EPUB boundaries.

### 3. Browser-player export still lacks explicit structural section metadata

The player still relies on heuristic presentation behavior. In `browser_player/app.js`,
`has_section_break_audio` controls whether section dividers should be added, and in
`browser_player/book-text.js` the UI inserts `<hr>` markers from text formatting patterns
rather than from exported `BookSection` metadata.

However, the intended direction is now clearer than before: for ebook-imported projects,
`Book.sections` should become the explicit source for browser-player section rendering and
navigation. The missing work is no longer just “some future structural metadata export”, but
a concrete section payload that can drive both visible text delineation and section jumping.

### 4. Manual section-marker UX is still flat-overlay based

The current section-markers menu edits compatibility indices rather than explicitly splitting
or merging `BookSection` objects.

## Revised Migration Phases

Given the codebase as it exists now, the most accurate phased view is:

### Phase 1 — Structural model and persistence: mostly complete

Completed:

- structural types added,
- `Project.book` added,
- `book.v1` serialization added,
- legacy migration added,
- `Book.segmentation_settings` made canonical for persisted provenance,
- flat compatibility synchronization added,
- project helper methods for section access added.

### Phase 2 — Move section-aware consumers to helpers: partially complete

The helpers now exist, but major consumers still need to switch.

Priority candidates remain:

- `ChapterInfo.make_chapter_infos`
- `m4b_chapter_util.make_metadata`
- concat range/chapter code
- section-marker displays and edit flows
- any ABR export path that should reflect structural sections explicitly.

### Phase 3 — De-overload `Reason.SPACE_BREAK`: not complete

This is blocked less by data modeling now and more by remaining importer/consumer behavior.

The structural model is in place, but EPUB import still uses `Reason.SPACE_BREAK` to carry some
boundary meaning. That old behavior should be removed only after the consumers above stop
depending on the flat/boundary-encoded signals.

### Phase 4 — Export explicit section metadata to the browser player: not complete

No verified explicit `sections` payload is exported to the browser player yet. Structural
reader metadata remains a future step.

The more concrete ABR metadata versioning/payload proposal for that work is documented
separately in `docs/abr-section-metadata-plan.md`, so this document can stay focused on the
broader text-model and player-architecture direction.

The target shape of that step is now more specific:

- export explicit section metadata derived from `Book.sections`,
- include enough information for the player to map sections onto flat playback text/segment
  indices,
- render section boundaries clearly in the vertically scrolling text,
- add reader-style section navigation in the player,
- keep sections distinct from bookmarks rather than treating bookmarks as pseudo-sections.

This is still a future implementation phase, but it now has a clearer UX destination.

## Revised Player UX Direction

The new direction is to lean into the `BookSection` concept as an actual reader/player UX
concept, not just a persistence-layer abstraction.

For ebook-imported `Book` data:

- each `BookSection` should be available to the browser player as an explicit structural unit,
- the scrolling book text should show clear visual delineation between sections,
- the player should support jumping directly to sections through a dedicated navigation UI,
- that navigation should feel closer to a basic ebook reader than to an audio-only marker list.

This also clarifies the intended separation of concepts:

- **sections** are book-authored structural divisions imported from the ebook and stored in
  `Book.sections`,
- **bookmarks** are user-created saved positions in playback/reading progress.

Under this direction, bookmarks should no longer do double-duty as pseudo-section markers.
If the browser player sidebar currently resembles a bookmarks panel, the likely evolution is a
top-level tab bar that switches between:

- **Bookmarks**
- **Sections**

That would preserve the existing bookmark use case while giving structural sections their own
first-class navigation surface.

This direction is intentionally scoped for now to ebook-imported `Book` data, where section
information already exists structurally. The flat-text/manual-text import flow does not need to
be solved as part of this planning update.

## Recommended Near-Term Direction

The current direction still looks correct; it just needs to be described as a hybrid state
rather than a future proposal.

Recommended next steps from the codebase’s current midpoint:

1. Keep `Book` as the canonical persisted text model.
2. Keep flat sound-segment indexing and flat runtime compatibility fields for now.
3. Migrate section-aware consumers from direct `section_dividers` access to:

   ```python
   project.get_section_ranges()
   project.get_section_start_indices()
   project.get_flat_phrase_groups()
   ```

4. Use the player/reader UX direction above as the product rationale for explicit structural
   section export: ebook-imported sections should become visible reading boundaries and a
   dedicated navigation surface, not just hidden metadata.
5. Once those consumers no longer require EPUB boundary encoding, remove EPUB-specific
   `Reason.SPACE_BREAK` boundary forcing and downgrade logic.
6. Export explicit section metadata to ABR/browser-player consumers so structural boundaries
   do not need to be inferred from whitespace or audio-only cues.

More specifically for the browser player, the likely near-term target is:

- export a `sections` payload derived from `Book.sections`,
- include section titles plus flat start/end index information,
- render clear boundaries in the scrolling text,
- provide section navigation UI separate from bookmarks.

The more implementation-shaped ABR metadata considerations for that export step—such as the
metadata version bump, the exact top-level `sections` payload shape, and subdivision-aware
index remapping—are handled in `docs/abr-section-metadata-plan.md`.

That still does **not** require solving flat/manual text import right now. The immediate
architectural direction can remain focused on ebook-imported projects, which already contain
the relevant structural information.

## Bottom Line

The project now has the missing structural text layer.

- `Book`/`BookSection`/`BookSegmentationSettings` are real,
- `Project.book` is real,
- `book.v1` persistence is real,
- legacy text migration is real,
- plain-text and EPUB imports now land in the structural model.

What remains unfinished is the second half of the migration:

- consumers still need to stop depending directly on flat divider fields,
- EPUB import still needs to stop encoding structural boundaries with `Reason.SPACE_BREAK`,
- browser/player metadata still needs explicit section export.

What is newly clarified is the UX destination for that remaining work: structural book sections
are not just migration plumbing, they are the natural basis for reader/player section display and
navigation in ebook-imported projects. That resolves the open-ended question of how to treat the
remaining section-related TODOs, even though those items are still intentionally deferred.

That is the current architecture and direction: **canonical structural storage with a still-active
flat compatibility runtime layer, now paired with a clearer section-first player UX target.**
