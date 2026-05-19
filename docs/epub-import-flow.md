# EPUB Import Architecture

## Purpose

The app now supports importing an EPUB directly into an audiobook project through the existing **Text** workflow.

The implemented import path is intentionally conservative: it converts EPUB spine content into polished plain text, segments that text with the same project settings used by ordinary text import, derives chapter dividers from EPUB document boundaries, and commits the result through the existing project persistence model.

This keeps EPUB import aligned with the app's pre-existing design constraints:

- the project text model remains `PhraseGroup`-based,
- raw source text remains available as `project_text_raw.txt`,
- segmented text remains serialized as `project_text.json`,
- chapter boundaries continue to use `Project.section_dividers`, serialized as `chapter_indices` in `project.json`,
- generated sound segments are invalidated when source text is replaced,
- preview/confirmation behavior matches the existing text import flow,
- richer ebook/reader formatting is not forced into the current audiobook-generation model.

The original EPUB is also copied into the project as `project_text.epub` so the imported source remains available for reference and for possible future structured-reader workflows.

---

## Current Import Path: Plain-Text EPUB Import

### Architectural Role

The current EPUB pathway is an import adapter into the app's existing text pipeline, not a new project representation.

EPUB-specific parsing, spine traversal, HTML flattening, warning collection, and chapter-boundary calculation are isolated in `tts_audiobook_tool/epub_extractor.py`. The app menu layer in `tts_audiobook_tool/text_menu.py` only coordinates user interaction, preview, confirmation, source-file copy, and project commit.

This boundary keeps ebook handling from leaking into the rest of the application. Downstream audiobook features continue to consume the same structures they already understand:

- `PhraseGroup` instances for generation and playback,
- `Reason.SECTION` markers for phrase-level section endings,
- `section_dividers` / `chapter_indices` for chapter-aware concat and metadata flows,
- `project_text_raw.txt` as the readable raw text reference.

### Implemented Pipeline

```text
EPUB file selected in Text menu
  ↓
EpubExtractor.import_epub(...)
  ↓
EbookLib reads the EPUB package
  ↓
Spine XHTML documents are loaded in book order
  ↓
EpubSourceChapter objects represent source spine documents
  ↓
BeautifulSoupEpubChapterTextExtractor flattens XHTML to clean plain text
  ↓
EpubTextChapter objects retain per-spine-document text chunks
  ↓
Each chapter is segmented independently with PhraseGrouper
  ↓
The last phrase in each chapter is marked Reason.SECTION
  ↓
Chapter-start phrase indices become Project.section_dividers
  ↓
Existing AppUtil.print_project_text(...) preview is shown
  ↓
On confirmation, old sound segments are deleted
  ↓
Project text, chapter dividers, raw text, metadata, and EPUB copy are saved
```

---

## Source and Result Structures

The importer uses small dataclasses as import-pipeline structures. They are deliberately not stored directly in the project model.

```python
@dataclass
class EpubSourceChapter:
    title: str
    href: str
    media_type: str
    html: str


@dataclass
class EpubTextChapter:
    title: str
    href: str
    text: str


@dataclass
class EpubTextExtractionResult:
    text: str
    warnings: list[str]
    significant_warnings: list[str]


@dataclass
class EpubImportResult:
    phrase_groups: list[PhraseGroup]
    raw_text: str
    section_dividers: list[int]
    chapters: list[EpubTextChapter]
    warnings: list[str]
    significant_warnings: list[str]
```

This shape gives the import path enough structure to preserve chapter boundaries during conversion and segmentation while still committing the final result into the established project format.

---

## Extraction Boundary

HTML-to-text conversion is behind an extractor protocol:

```python
class EpubChapterTextExtractor(Protocol):
    def extract_text(self, chapter: EpubSourceChapter) -> EpubTextExtractionResult:
        ...
```

The default implementation is `BeautifulSoupEpubChapterTextExtractor`.

It is intentionally more deliberate than raw `BeautifulSoup.get_text()`:

- removes non-reading tags such as `script`, `style`, `nav`, `img`, `svg`, and form/media elements,
- treats paragraphs, headings, lists, blockquotes, tables, and other block elements as paragraph-like text blocks,
- preserves `<br>` line breaks where appropriate,
- renders `<hr>` as a scene-break-like `* * *`,
- keeps list items readable,
- normalizes whitespace and blank lines,
- warns when a spine document appears to contain multiple major headings,
- warns when no readable body text is found.

The extractor does **not** preserve visual formatting such as italics, bold, underline, CSS layout, inline links, images, or footnote structure. That omission is a design choice: the current import path feeds audiobook generation, whose source-of-truth text model is plain text segmented into phrase groups.

---

## Chapter Boundary Strategy

EPUB spine documents are the authoritative chapter chunks for the current importer.

For each readable spine document:

1. EbookLib resolves the spine item.
2. Navigation documents are skipped.
3. The XHTML content is decoded.
4. A title is inferred from `h1`, `h2`, `title`, or the fallback href stem.
5. The source document becomes an `EpubSourceChapter`.
6. The extractor converts it to an `EpubTextChapter`.

The importer does not split within a spine document based on `h1` or `h2`. If multiple major headings are detected, that condition is surfaced as a warning instead of changing the chapter model automatically.

### Mapping to `section_dividers`

Chapters are segmented independently and concatenated afterward. This avoids fragile text-offset mapping after segmentation and ensures segments do not cross EPUB spine boundaries.

Conceptually:

```python
phrase_groups = []
section_dividers = []
raw_text_parts = []

for chapter in text_chapters:
    chapter_text = chapter.text.strip()
    if not chapter_text:
        continue

    if phrase_groups:
        section_dividers.append(len(phrase_groups))

    chapter_phrase_groups = PhraseGrouper.text_to_groups(
        chapter_text,
        max_words=max_words,
        strategy=segmentation_strategy,
        pysbd_lang=language_code,
    )
    EpubExtractor.mark_last_phrase_as_section(chapter_phrase_groups)

    phrase_groups.extend(chapter_phrase_groups)
    raw_text_parts.append(chapter_text)

raw_text = "\n\n".join(raw_text_parts)
```

The first chapter starts at phrase group `0`, so it does not need a divider entry. Each subsequent chapter adds a divider at the current phrase-group count before its groups are appended.

`mark_last_phrase_as_section(...)` also marks the last phrase of each chapter as `Reason.SECTION` and normalizes its trailing line breaks, matching the app's existing section-ending expectations.

This EPUB boundary marker is structural: it represents the end of a retained EPUB spine
document, not merely whitespace found in the source text. To avoid duplicated section
behavior, the importer also downgrades section-like groups at the start of a subsequent
spine document when a previous retained document already ended with a forced
`Reason.SECTION`. This handles common heading patterns where the next XHTML file begins
with chapter-title text separated by multiple blank lines.

---

## Text Menu Integration

The user-facing import entry point is the existing text menu:

```python
MenuItem("Import from EPUB file", on_set_text, data="epub")
```

The EPUB branch in `on_set_text(...)` mirrors the existing text-file and manual import flow:

1. If generated sound segments exist, the user is warned that replacing text will delete them.
2. The app asks for an `.epub` path.
3. `EpubExtractor.import_epub(...)` imports, extracts, segments, and returns an `EpubImportResult`.
4. Significant warnings are shown as import-time FYIs.
5. The generated phrase groups are previewed with `AppUtil.print_project_text(...)`.
6. The user confirms or cancels.
7. On confirmation, old generated sound segments are deleted.
8. The original EPUB is copied to the project as `project_text.epub`.
9. Phrase groups, chapter dividers, applied segmentation settings, and raw text are committed atomically.
10. Real-time line range state is reset the same way as existing text replacement.
11. The app reports the imported chapter and divider counts.

This preserves the app's established UX contract: text replacement is previewed, confirmed, and treated as invalidating generated audio.

---

## Project Persistence Integration

The current importer writes the same core project files as regular text import, plus a copy of the EPUB source.

On confirmed import:

- `[project root]/project_text_raw.txt` stores the flattened raw text,
- `[project root]/project_text.json` stores segmented phrase groups,
- `[project root]/project.json` stores project metadata, including `chapter_indices`,
- `[project root]/project_text.epub` stores the original EPUB copy.

`PROJECT_TEXT_EPUB_FILE_NAME = "project_text.epub"` is defined in `tts_audiobook_tool/constants.py`.

### Atomic commit method

Regular text replacement uses `Project.set_phrase_groups_and_save(...)`, which clears `section_dividers` because a generic flat text import has no reliable chapter structure.

EPUB import uses a separate method:

```python
def set_phrase_groups_chapters_and_save(
        self,
        phrase_groups: list[PhraseGroup],
        section_dividers: list[int],
        strategy: SegmentationStrategy,
        max_words: int,
        language_code: str,
        raw_text: str
) -> None:
    ...
```

This method mirrors the standard text commit behavior while preserving the chapter dividers computed from the EPUB structure:

- sets `phrase_groups`,
- records `applied_strategy`, `applied_max_words`, and `applied_language_code`,
- sets `section_dividers`,
- clears `generate_range_string`,
- clears `realtime_line_range`,
- saves project metadata and phrase groups through the existing save flow,
- saves `project_text_raw.txt`.

Keeping this as a separate method makes the distinction explicit: ordinary text import resets chapter dividers, while EPUB import has a valid structural source for them.

---

## Warning and Error Handling

The importer distinguishes general warnings from significant warnings.

General warnings are logged through the app logging utility (`L.w`) for diagnostic visibility. Significant warnings are also presented to the user before the import preview continues.

Examples include:

- EPUB spine item missing,
- no readable spine documents found,
- EPUB could not be read,
- multiple major headings detected in a single spine document,
- readable text missing from a document that does not look like front/back matter,
- the import produced no text segments.

This keeps messy EPUB behavior visible without requiring new project-model fields for import diagnostics.

---

## Design Constraints Satisfied

The implementation fits the existing app architecture by observing these constraints:

### No new reader/document model in the current import path

The current app's audiobook generation flow works from phrase groups. EPUB import therefore terminates in phrase groups and raw plain text rather than introducing an HTML or ebook document model.

### Existing segmentation settings remain authoritative

EPUB import uses:

- `state.project.max_words`,
- `state.project.segmentation_strategy`,
- `state.project.language_code`.

This makes EPUB import behave like every other text source from the perspective of TTS prompt generation.

### Chapter structure is projected into existing fields

EPUB chapter boundaries become `section_dividers` / `chapter_indices`; they are not stored as a parallel chapter system.

### Existing generated-audio invalidation rules apply

Replacing text from EPUB deletes old generated sound segments, matching text-file and manual import behavior.

### Future structure is enabled but not required

The original EPUB copy and the extractor boundary leave room for richer representations later without burdening the current audiobook pipeline.

---

## Current Quality Target

The plain-text output is intended to be clean enough for immediate audiobook segmentation without routine manual cleanup.

The practical benchmark is Calibre-like default text export quality for typical EPUBs: not byte-for-byte parity, but readable text with stable paragraph spacing, sensible handling of obvious headings and scene breaks, and minimal extraction artifacts.

---

## Future Architecture: Structured Reader Formats

The implemented importer handles the audiobook-generation requirement first. More structured formats remain on the table for the ereader/browser-reader side, where formatting and document structure matter more than they do for TTS inference.

Two future directions are described below as **Phase 2a** and **Phase 2b**. They are not necessarily dependent on each other:

- **Phase 2a** can add a project-side and ereader-side structured format that coexists with the current plain-text phrase-group pipeline.
- **Phase 2b** can improve formatting-preserving conversion machinery, potentially feeding Phase 2a or another reader representation.

Either can be explored first, depending on whether the priority is the persisted reader contract or the fidelity of EPUB-to-structured-content conversion.

---

## Phase 2a: Structured Project/Ereader Representation

Phase 2a would define a more structured format for imported book text, including a corresponding representation on the ereader/browser-player side.

This would not replace the current plain-text audiobook path. Instead, EPUB import could produce two related outputs:

1. the existing plain-text `PhraseGroup`/`chapter_indices` data needed for TTS generation,
2. a structured reader document used for display, navigation, and formatting-aware reading.

Possible structured-reader content could include:

- stable chapter IDs,
- chapter titles,
- paragraph blocks,
- heading levels,
- scene breaks,
- inline emphasis spans,
- links or footnote markers if useful,
- mapping hints between reader blocks and phrase-group indices.

This would let the ereader side display richer book structure without requiring the TTS pipeline to understand EPUB or HTML directly.

### Possible storage shape

A Phase 2a document might be stored separately from the core phrase-group file, for example:

```text
project_reader.json
```

Conceptual shape:

```json
{
  "format": "reader_document.v1",
  "source": {
    "type": "epub",
    "file_name": "project_text.epub"
  },
  "chapters": [
    {
      "id": "chapter-1",
      "title": "Chapter 1",
      "href": "chapter1.xhtml",
      "start_phrase_group_index": 0,
      "blocks": [
        { "type": "heading", "level": 1, "text": "Chapter 1" },
        { "type": "paragraph", "spans": [{ "text": "Opening text." }] },
        { "type": "scene_break" }
      ]
    }
  ]
}
```

The exact schema should be designed around browser-player rendering requirements, not around EPUB internals. EPUB would be one import source for that schema, not the schema itself.

### Relationship to the current importer

The current import architecture already helps Phase 2a because it keeps per-chapter intermediate structures before flattening fully into project text.

Phase 2a could extend the extractor result from "text plus warnings" to "plain text plus structured reader blocks plus warnings," or it could add a parallel reader-document extractor. Either approach can reuse:

- EPUB path validation,
- EbookLib package/spine traversal,
- source chapter loading,
- warning collection,
- source EPUB copy handling,
- chapter-to-phrase-group mapping.

The important architectural rule is that structured reader data should remain an additional reader-facing artifact, not a forced replacement for `PhraseGroup`-based TTS input.

---

## Phase 2b: Formatting-Preserving Conversion

Phase 2b is the future target for preserving enough source formatting to produce a richer reader representation.

Formatting likely worth preserving:

- italics,
- boldface,
- underline,
- headings, especially `h1` and `h2`,
- paragraph structure,
- scene breaks, if possible.

Possible Phase 2b direction:

```text
EPUB file
  ↓
EbookLib
  ↓
Extract XHTML spine documents
  ↓
Inscriptis or a custom structured extractor
  ↓
Convert XHTML to text/content with annotations
  ↓
Custom serializer
  ↓
Micro-HTML / SimpleBook HTML or reader_document.v1 blocks
```

Inscriptis remains a plausible tool here because it is built for HTML-to-text rendering and annotations. A custom extractor may also be appropriate if the app needs stricter control over the subset of supported reader markup.

Phase 2b should build on the current importer boundaries where practical, especially the separation between EPUB orchestration and extraction/serialization implementation. It does not have to wait for Phase 2a if the first goal is to evaluate conversion fidelity, and Phase 2a does not have to wait for Phase 2b if the first goal is to define the ereader storage/rendering contract.
