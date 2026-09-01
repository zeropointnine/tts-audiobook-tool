# Navigation-Defined EPUB Sectioning

## Purpose

EPUB import preserves two different kinds of order and structure:

- the **spine** defines the publication's physical reading order, and
- the **navigation table** defines the logical sections presented to the reader.

These are related but are not interchangeable. A spine XHTML document is a transport and layout unit; it is not automatically a chapter boundary. One logical section may span several spine documents, and one spine document may contain several logical sections.

The importer therefore reads content in spine order while deriving audiobook section boundaries and titles from EPUB navigation targets. The resulting logical sections are then passed to the existing plain-text segmentation and project persistence pipeline.

## Navigation Source Selection

EbookLib reads the EPUB package and exposes its navigation structure. The importer prefers an EPUB 3 navigation document (`nav.xhtml` with a table of contents). When EPUB 3 navigation is absent, EbookLib falls back to the EPUB 2 NCX table of contents.

The selected table is flattened in navigation order while retaining each entry's label and target. Nested entries are valid boundaries; nesting describes the publication's hierarchy but does not create a second hierarchy in the current project model.

The importer does not use the visible navigation document itself as reading content. It uses the document's links as metadata and continues to obtain readable content from retained spine items.

## Spine Order and Logical Sections

The spine remains authoritative for content order. Navigation data does not reorder XHTML documents or make out-of-spine resources readable.

Navigation targets identify logical starts within that ordered content stream:

- a target without a fragment starts at the beginning of its spine document,
- a target with a fragment starts at the matching anchor within its spine document,
- the next usable target ends the current logical section, and
- the end of the retained spine content ends the final logical section.

Consequently:

- several consecutive spine documents can contribute to one logical section,
- several targets in one spine document can split that document into several logical sections, and
- crossing a spine document boundary does not by itself end a logical section.

The importer never infers additional logical boundaries from `h1`, `h2`, file names, or ordinary HTML spacing when usable navigation boundaries exist. Headings remain part of the extracted text.

## Canonical Targets

Navigation links and spine item names can spell the same resource differently. The importer resolves every internal navigation link relative to its owning navigation resource and converts it to a canonical target before matching it to spine content.

A canonical target consists of:

1. a normalized package-relative document path, and
2. an optional fragment identifier kept separately from that path.

Canonicalization handles relative path components, slash normalization, URL escaping, and fragment separation. Query components do not participate in spine-document identity. Fragment-free and fragment-bearing links to the same document therefore share one canonical document path but remain distinct section starts.

The canonical form is used consistently for:

- matching a navigation entry to a retained spine item,
- detecting repeated targets,
- ordering targets against the spine,
- locating fragment anchors, and
- reporting unresolved or non-reading targets.

Canonicalization is an identity operation, not permission to guess. A target that still cannot be matched unambiguously is not silently redirected to a similarly named file.

## Fragment-Aware Text Slices

Text extraction retains enough source position information to associate readable text with anchors in document order. An anchor can be supplied by the forms supported by EPUB XHTML, including element identifiers used by navigation fragments.

For a fragment-bearing target, the logical slice begins at the target anchor and continues to the next usable target in that document or to the end of the document. A fragment-free target begins at the document start. If the next logical target is in a later spine item, the current section also consumes the intervening readable spine content.

This slicing occurs before phrase segmentation. It avoids reconstructing section boundaries from character offsets after whitespace normalization or sentence grouping.

Readable source content is consumed at most once. Adjacent slices neither overlap nor duplicate text. Readable text before the first in-document fragment remains in spine order: it belongs to the preceding active logical section when one exists; otherwise it is retained as leading content according to the section assembly rules rather than being discarded merely because the first navigation link uses a fragment.

## Logical Section Assembly

The importer walks retained spine content in reading order and applies usable canonical navigation targets as boundary events. A logical section contains:

- the navigation label used as its title,
- the canonical target at which it starts,
- one or more fragment-aware text slices, and
- the concatenated readable text from those slices.

Assembly is independent of spine file boundaries. When a section reaches the end of one document without encountering another target, it continues through later retained spine documents. When another usable target occurs in the same document, the current section ends at that anchor and the next section starts there. Text merged into one section this way is joined with two blank lines so each merged boundary remains a visible, section-like spacing inside the assembled text.

Only non-empty logical sections reach segmentation. Navigation labels provide metadata; they are not injected into the text unless the corresponding heading is already present in the XHTML reading content.

### Image-only pending targets

A valid navigation target can point to a cover, ornament, plate, or chapter-opening document whose slice contains no readable text. Such an image-only start is kept as a **pending target** rather than emitted as an empty section.

If readable content follows before another usable navigation target, that content starts the pending logical section. This supports EPUBs that place a chapter image or title artwork in one spine document and the chapter text in the next. If a later usable target arrives first, it supersedes the still-empty pending target; the importer does not create an empty marker or duplicate the later text. The omitted empty target is diagnosable through warnings when it materially affects navigation mapping.

## Non-reading Targets

A navigation table can contain links that are not part of the audiobook reading stream. Examples include:

- external URLs,
- resources that are not in the spine,
- the EPUB navigation document itself,
- targets in documents removed as publication metadata or table-of-contents content,
- unsupported media resources, and
- missing documents or unresolved fragments.

These targets do not reorder content, create empty output sections, or claim unrelated text. The importer ignores them as boundaries and records an appropriate warning when the condition can indicate malformed or incomplete navigation.

A link to a retained spine document is not non-reading merely because its target slice is image-only; it follows the pending-target behavior above.

## Fallback to Spine Sections

Navigation-defined sectioning is used only when the chosen EPUB 3 nav or EPUB 2 NCX produces usable boundaries in the retained reading stream. If navigation is missing, empty, yields no usable reading targets, or contains an unresolved fragment that could otherwise assign prose to the wrong section, the importer falls back to one section per retained readable spine document. An EPUB package that EbookLib cannot read remains an import error rather than entering this fallback.

Fallback preserves importability but is explicitly visible: the importer emits a significant warning that navigation-defined sections were unavailable and spine-document sections were used. Spine boundaries are therefore a compatibility fallback, not the normal authoritative chapter model.

The fallback uses the same filtering, extraction, segmentation, and result rules as navigation-defined import. Empty retained documents do not produce empty sections.

## Segmentation and Result Invariants

Logical section assembly completes before `PhraseGrouper` runs. Each non-empty logical section is segmented independently with the project's selected maximum word count, segmentation strategy, language, and dialog-segmentation setting.

The importer maintains these invariants:

- phrase groups remain in spine reading order,
- no phrase group crosses a logical section boundary,
- every retained readable text slice appears once in `raw_text`,
- logical sections with no readable text produce no phrase groups or divider,
- the final phrase of every emitted section is marked as a section break,
- `section_start_indices` contains the phrase-group start of every emitted section after the first,
- section start indices are strictly increasing, unique, and within the phrase-group result,
- a non-empty import has `len(section_start_indices) + 1` emitted sections,
- `raw_text` joins emitted logical-section text in the same order as phrase groups, and
- section/chapter result metadata describes logical navigation sections, not incidental XHTML file boundaries.

Source-derived leading section-like spacing is normalized at an already established logical boundary so it does not create duplicate section behavior. Segmentation never determines navigation boundaries retroactively.

## Warnings and Edge Cases

Warnings make degraded or ambiguous imports visible without preventing recovery when readable content remains. Conditions include:

- neither navigation source yields usable targets and spine-section fallback is used,
- a spine target document is missing or cannot be decoded,
- a fragment does not resolve to an anchor,
- navigation targets conflict with spine or in-document reading order,
- a pending image-only target is superseded or reaches the end of the book without text,
- a retained slice has no readable body text,
- inline XHTML spacing requires substantial repair, and
- extraction or segmentation produces no text groups.

Malformed entries are handled locally when that cannot change text ownership. The importer preserves spine order, skips unsafe boundary guesses, and reports the degradation. An unresolved fragment triggers full spine-section fallback because dropping only that target could silently attach its prose to the preceding navigation section.

Warnings are collected in the import result and logged. Significant warnings are also shown before confirmation so the user can review the preview or cancel the import.
