# Book Section Architecture

## Motivation

The current project text model is intentionally simple:

```python
Project.phrase_groups: list[PhraseGroup]
```

That flat list works well for TTS generation because each `PhraseGroup` maps to one
generated sound segment and one flat line number in the UI. Over time, however, several
additional concepts have been layered onto that same flat representation:

- `Phrase.reason`, especially `Reason.SECTION`, controls local segmentation/prosody.
- `Reason.SECTION` can trigger section-break sound effects during concat.
- The browser player infers horizontal rules from section-like trailing line feeds.
- `Project.section_dividers` / `chapter_indices` mark flat phrase-group indices where
  chapters or files should begin.
- EPUB import introduces source-level spine document boundaries.

This has made `Reason.SECTION` overloaded. It can currently mean:

1. the source text had strong whitespace or an ornamental break,
2. the audio should receive a longer pause or section-break sound effect,
3. the browser should display section-like separation,
4. an EPUB HTML/spine document ended,
5. a chapter/file/bookmark boundary may be nearby.

The EPUB import path made the problem more visible. EPUBs already have structural chunks
via spine documents, while plain-text segmentation only has whitespace heuristics. Trying
to encode both through one flat `Reason.SECTION` signal leads to brittle compensating
logic such as suppressing consecutive section reasons after import.

The app needs an explicit structural layer while preserving the flat phrase-group model
that generation, validation, sound segment storage, and much of the UI already rely on.

## Current Model

The current persisted project text has two related pieces:

```python
phrase_groups: list[PhraseGroup]
section_dividers: list[int]
```

`phrase_groups` is the operational list. Flat indices are used throughout the app:

- generated sound segment file lookup,
- regeneration and validation,
- concat ranges,
- real-time playback,
- text preview,
- bookmarks,
- browser-player text segment metadata.

`section_dividers` is an overlay of flat start indices. The first section starts
implicitly at index `0`; each entry marks the start of a later section/chapter/file range.

This means the current model is effectively:

```python
FlatBook = list[PhraseGroup]
SectionStarts = list[int]
```

That is usable, but it treats structure as a derived index overlay rather than a real
document concept.

## Proposed Model

Introduce an explicit `BookSection` concept:

```python
@dataclass
class BookSection:
    phrase_groups: list[PhraseGroup]
    title: str = ""
    source_id: str = ""
    source_kind: str = ""
```

Then the book becomes:

```python
BookSections = list[BookSection]
```

Examples:

```python
# Plain-text import
[
    BookSection(
        source_kind="plain_text",
        phrase_groups=all_phrase_groups,
    )
]

# EPUB import
[
    BookSection(
        title="Chapter 1",
        source_id="Text/chapter001.xhtml",
        source_kind="epub_spine",
        phrase_groups=chapter_1_phrase_groups,
    ),
    BookSection(
        title="Chapter 2",
        source_id="Text/chapter002.xhtml",
        source_kind="epub_spine",
        phrase_groups=chapter_2_phrase_groups,
    ),
]
```

The central idea: an EPUB spine boundary should be represented by the boundary between
two `BookSection` instances, not by mutating the final phrase inside a flat list to
`Reason.SECTION`.

## Why Not Store Bare `list[list[PhraseGroup]]`?

A raw nested list captures the basic shape:

```python
list[list[PhraseGroup]]
```

But it underfits the likely next requirements. Sections will need metadata almost
immediately:

- EPUB href / spine source ID,
- chapter title,
- source kind (`plain_text`, `epub_spine`, `epub_metadata_title`, `manual`, etc.),
- possible future reader/rendering hints,
- possible mapping/debug information from import.

A named `BookSection` object keeps the initial implementation small while leaving room
for these fields without changing the shape again.

## Relationship to `Phrase.reason`

`Phrase.reason` should remain, but its meaning should narrow.

Recommended meaning:

- `Reason.SENTENCE`: local sentence boundary.
- `Reason.PARAGRAPH`: local paragraph boundary.
- `Reason.SECTION`: local scene/section break inferred from text inside a section, such
  as strong blank-line spacing or ornamental separators.

It should not mean “an EPUB spine document ended.” That is structural information and
belongs to `BookSection` boundaries.

This still allows ordinary text imports to use whitespace heuristics within their single
section. A plain text document can contain local `Reason.SECTION` markers without having
multiple structural `BookSection`s.

## Relationship to `section_dividers`

Do not remove `section_dividers` immediately.

Many parts of the app currently operate on flat phrase-group indices. Generated sound
segments are also flat-indexed. Replacing that everywhere at once would be a high-risk
migration.

Instead, treat `section_dividers` as a derived compatibility view:

```python
phrase_groups = BookSection.flatten(book_sections)
section_dividers = BookSection.to_flat_start_indices(book_sections)[1:]
```

The first section starts implicitly at `0`, so `section_dividers` continues to omit it.

In the short term:

```python
Project.book_sections  # structural source, new
Project.phrase_groups  # flat operational list, retained
Project.section_dividers  # flat compatibility/cache, retained
```

The invariant should be:

```python
Project.phrase_groups == flatten(Project.book_sections)
Project.section_dividers == starts(Project.book_sections)[1:]
```

For legacy projects with only `phrase_groups`, load them as one section:

```python
book_sections = [BookSection(source_kind="legacy_flat", phrase_groups=phrase_groups)]
```

## Storage Shape

A future `project_text.json` format could become:

```json
{
  "format": "book_sections.v1",
  "sections": [
    {
      "title": "Chapter 1",
      "source_id": "Text/chapter001.xhtml",
      "source_kind": "epub_spine",
      "phrase_groups": [
        [
          { "text": "Chapter 1\n\n", "reason": "p" }
        ],
        [
          { "text": "Opening prose.", "reason": "s" }
        ]
      ]
    }
  ]
}
```

The loader should continue supporting the current format:

```json
{
  "format": "phrase_groups.v1",
  "phrase_groups": [...]
}
```

and convert it to a single `BookSection` in memory.

## Migration Strategy

### Phase 1: Add structural model while keeping flat runtime

Add:

- `BookSection` type,
- serializer/deserializer helpers,
- flatten/start-index helper functions,
- optional `Project.book_sections` field.

Keep `Project.phrase_groups` as the operational list used by generation, concat,
validation, and existing UI.

Plain-text import should create one section. EPUB import should create one section per
retained spine document, then derive the flat list and flat divider indices.

This phase introduces structure without requiring every consumer to change immediately.

### Phase 2: Move section-aware consumers to helpers

Update code that is semantically section-aware to ask the project for ranges instead of
directly manipulating `section_dividers`:

```python
project.get_section_ranges() -> list[tuple[int, int]]
project.get_section_start_indices() -> list[int]
project.get_flat_phrase_groups() -> list[PhraseGroup]
```

Likely consumers:

- `ChapterInfo.make_chapter_infos`,
- `ConcatUtil` chapter/file ranges,
- `m4b_chapter_util.make_metadata`,
- Section markers menu display/edit flow,
- browser/app metadata export.

The implementation can still return flat ranges because sound segment storage remains
flat-indexed.

### Phase 3: De-overload `Reason.SECTION`

Once structural sections exist, EPUB import should stop forcing the final phrase of each
spine document to `Reason.SECTION` just to encode the spine boundary.

`Reason.SECTION` should only represent local text-derived scene/section breaks inside a
`BookSection`.

At that point, section-break audio policy can be made explicit:

- play effects for local `Reason.SECTION`,
- play effects for structural `BookSection` boundaries,
- play effects for both,
- or configure them separately.

This decision should not be hardcoded into the meaning of one enum value.

### Phase 4: Export explicit section metadata to the browser player

The browser player currently infers horizontal rules from trailing line feeds when
section-break audio is present. A structured model should eventually export explicit
section metadata:

```json
{
  "text_segments": [...],
  "sections": [
    { "title": "Chapter 1", "start_segment_index": 0 },
    { "title": "Chapter 2", "start_segment_index": 17 }
  ]
}
```

Then the browser can render structural boundaries directly instead of guessing from
whitespace.

## Open Questions

### What should section-break sound effects mean?

Currently they are tied to `Reason.SECTION`. With structural sections, there are at least
two possible triggers:

1. local scene/section breaks inside a section,
2. structural boundaries between `BookSection`s.

These may deserve separate settings.

### How should manually entered section markers interact with `BookSection`s?

The existing Section Markers menu edits flat start indices. In a structured model, this
could either:

- continue editing the compatibility `section_dividers` overlay,
- split/merge `BookSection`s directly,
- or maintain a separate user-defined export/chapter marker layer.

The safest first step is to keep manual markers as a flat compatibility overlay and avoid
rewriting source-derived `BookSection`s until the UX is clearer.

### How should injected EPUB metadata titles be represented?

The current EPUB import can inject the book title as a pseudo-chapter for readability but
does not create a section divider before the first real spine chapter.

With `BookSection`, this could be represented explicitly:

```python
BookSection(source_kind="epub_metadata_title", title=book_title, ...)
```

Then downstream consumers can decide whether it behaves like a chapter, a prelude, or a
display-only block.

## Recommended Near-Term Direction

Do not rewrite the whole app around nested phrase groups in one step.

Recommended sequence:

1. Add `BookSection` and helpers.
2. Make EPUB import produce `book_sections` plus derived `phrase_groups` and
   `section_dividers`.
3. Keep flat generated sound segment indices.
4. Move section-aware features to section helper methods gradually.
5. Remove EPUB boundary encoding from `Reason.SECTION` once section consumers understand
   `BookSection` boundaries.

This gives the project the missing structural layer while preserving the existing flat
runtime model that most of the application depends on.
