# Display Formatting Layer

## Purpose

EPUB text carries limited formatting worth showing in the browser player: headings,
bold, italics and underline. The app's text model is plain text segmented into
`Phrase`/`PhraseGroup` objects, and that model drives generation, validation,
editing and alignment.

The formatting layer adds that formatting **without disturbing the existing data
flow**:

- one optional property, `Phrase.fmt`, holds it;
- the Python app carries `fmt` along but never parses it to make decisions;
- nothing in TTS prompting, validation, audio identity, segmentation or dialog
  attribution reads it;
- the browser player is its only real consumer, via ABR metadata.

## Representation: shadow markup

`fmt` is the phrase's own text with a few tags:

```json
{"text": "It was very cold. ", "reason": "s", "fmt": "It was <i>very</i> cold. "}
```

| Tag | Meaning | EPUB sources |
|---|---|---|
| `<h>` | heading | `h1`–`h6` |
| `<b>` | bold | `b`, `strong`, inline `font-weight` ≥ 600 / `bold` |
| `<i>` | italic | `i`, `em`, `cite`, inline `font-style: italic`/`oblique` |
| `<u>` | underline | `u`, inline `text-decoration(-line): underline` |

Literal `&`, `<` and `>` are escaped as `&amp;`, `&lt;` and `&gt;`. Heading levels are
deliberately not distinguished.

### Rules

1. **Sparse.** `fmt` is absent (`None`, not serialized) unless at least one character
   is styled.
2. **Self-contained per phrase.** Tags are properly nested and every tag opened in a
   phrase closes in that phrase. Formatting that crosses a phrase boundary is split:
   it closes at the end of one phrase and reopens in the next. No formatting state
   carries from one phrase to another.
3. **Self-validating.** `fmt` is valid only when its stripped text equals `text`
   exactly (including whitespace). Malformed or stale markup is ignored by every
   consumer — never an error, never a load failure.
4. **Visible characters carry style.** A whitespace run takes the styles shared by
   the visible characters on both sides of it within the phrase; leading and trailing
   whitespace is never styled.
5. **Canonical form.** Producers render markup through `text_format.render`, which
   nests tags in the fixed order `h`, `b`, `i`, `u` and keeps shared outer tags open
   across runs.

Combining upward is trivial: the markup of a phrase group (or any run of phrases) is
derived by joining the phrases' styles (`text_format.join`), so nothing stores
group-level formatting.

### Why this shape

- **Self-contained markup** makes each phrase independently checkable, survives
  splitting groups at phrase boundaries for free, and keeps one stale phrase from
  corrupting its neighbors. Producing it costs nothing extra: each phrase is rendered
  from its own styled characters.
- **Shadow text instead of offsets** avoids a separate staleness check and avoids
  Python code points versus JavaScript UTF-16 index mismatches.
- **Phrase level, not group level**: ABR leaves are phrases (subdivided and no-audio
  groups) or whole groups, and joining phrase markup is safe, while slicing group
  markup would need markup-aware cutting at several consumers.

## Producing formatting

### EPUB import: attach after segmentation

Formatting is not threaded through the segmenter, grouper or dialog segmenter.
Instead:

1. `BeautifulSoupEpubChapterTextExtractor.append_node_text` tracks a style mask
   down the element tree and records the mask of each emitted text piece by its index
   in the append-only piece list (the same mechanism used for navigation anchors).
2. The extractor builds a `StyledText` (text plus a parallel mask list) for each
   extraction result and navigation slice; section assembly joins them alongside the
   plain text. `EpubTextChapter.styled` is `None` for unstyled sections.
3. After `PhraseGrouper.text_to_groups()` segments a section, `attach_formatting`
   (`text_ops/format_attach.py`) projects the section's styles onto the phrases and
   renders each phrase's `fmt`.

Projection aligns **words** (whitespace-separated runs of visible characters) of the
source and of the concatenated phrase text (`text_format.align_masks`). Segmentation
only rearranges whitespace, so the word sequences are normally identical (fast path),
and phrase boundaries inside a word (eg, after an em dash) are handled because the
phrase text is aligned as one concatenated string. When the word sequences differ,
`difflib` aligns them at word level; unmatched words stay unstyled. Import logs, but
never surfaces or fails on, an inexact match. Raw text (`project_text_raw.txt`) never
contains markup.

### Re-apply formatting from EPUB

Text menu → **Re-apply formatting from EPUB** (shown for EPUB projects). It re-extracts
the project's `project_text.epub` (or a user-chosen EPUB when the copy is missing),
pairs EPUB sections with project sections, and runs the same attach step against the
**current** phrases. Only `fmt` changes, so generated audio, markers, edits and dialog
assignments are untouched; re-exporting the audiobook carries the result into ABR
files. This is also the backfill path for projects imported before this feature.

Pairing (`epub_format_reapply.pair_sections`) is positional only when section counts
and titles all match; otherwise sections pair in order by identical title. Unpaired
sections keep their existing formatting. Within paired sections, formatting is
recomputed: words that no longer match the EPUB (edited lines) end up unstyled.

## Changing text: business rule

Changing a phrase's text drops its formatting:

- the `Phrase.text` setter clears `fmt` when the value changes;
- the text editor re-creates the edited group's phrases, which start without `fmt`;
- deleting groups or splitting a group at a phrase boundary keeps the existing
  `Phrase` objects, so their formatting survives.

So editing a line loses that one group's formatting. The remedy is the re-apply
action, which restores formatting for every word that still matches the EPUB.

A possible later refinement is to carry formatting across editor edits (character
diff of old versus new group text, then the same attach step); the attach function is
deliberately generic to allow that.

## Persistence and dormancy

- `project_text.json`: `"fmt"` is written on a phrase only when set; on load, `fmt`
  that is not valid for its text is dropped silently.
- `Phrase.__eq__` ignores `fmt`.
- `fmt` is not part of the segment hash (index + text), so it never invalidates
  generated audio. It is not part of any prompt.
- Older app versions read only `text`/`reason` and drop `fmt` when saving.

## ABR metadata

Each `text_segments` leaf may carry an optional `"fmt"` string (see
[ABR Metadata Spec](abr-metadata-spec.md)). Concat produces it as follows:

| Leaf kind | `fmt` source |
|---|---|
| whole group (subdivision off, or missing/invalid sidecar) | `PhraseGroup.as_flattened_phrase()` joins the phrases' styles |
| subdivided group (sidecar leaves) | the corresponding current project phrase, only when the sidecar leaf count equals the phrase count and the leaf text equals the phrase text |
| no-audio group (zero-timed phrase leaves) | the phrase's `fmt` |

Sidecars (`*.json` STT info) never store `fmt`; formatting always comes from the
current project text at export. EPUB enhancement (`type: "conversion"`) carries the
imported phrases' `fmt` into its leaves the same way. The Python ABR reader preserves
valid `fmt` on round trips and drops invalid values.

## Browser player

`browser_player/text-format.js` mirrors the Python parser. `MetadataUtil` passes a
string `fmt` through unchanged; `BookText` renders it:

- `TextFormat.masksFor(fmt, text)` returns per-character masks only when the markup is
  valid for the leaf text (otherwise plain rendering);
- the masks of the leaf's displayed content are sliced out after the existing
  leading/trailing-whitespace split, and styled runs become child `<span>`s
  (`fmtBold`, `fmtItalic`, `fmtUnderline`, `fmtHeading`) inside the segment span;
- whitespace between child leaves of a group (outside both leaf spans) takes the
  styles shared by the adjacent visible characters, so eg underlines continue across
  phrase boundaries;
- no `innerHTML` is used.

Formatting never crosses a segment span, which HTML nesting would not allow anyway.
Headings render as inline bold, larger text inside the existing `<pre>` layout.

## Limitations

- Stylesheet (class-based) formatting, which many EPUBs use, is not interpreted; only
  semantic tags and inline `style` attributes are.
- Plain-text and manually entered projects have no formatting source.
- Headings are display-only; they do not influence `HEADING_END` pauses.

## Future: prompt-only annotation layers

A similar layer could hold non-verbal TTS instructions (eg, emotion tags). It should
reuse the same grammar, parser and attach machinery, but as a **separate** field
(eg, `cues`), because its rules are opposite on the axis that matters most: it
changes the prompt, so it must be part of audio identity, be stripped before
transcript validation, and be rendered into model-specific syntax. Keeping display
formatting in its own field preserves the guarantee that `fmt` never affects audio.

## Implementation map

| File | Responsibility |
|---|---|
| `tts_audiobook_tool/text_ops/text_format.py` | Markup grammar, parse/render/join, `StyledText`, word alignment |
| `tts_audiobook_tool/text_ops/format_attach.py` | Attaching styles to segmented phrases |
| `tts_audiobook_tool/text_ops/epub_extractor.py` | Style capture during HTML flattening; attach at import |
| `tts_audiobook_tool/text_ops/epub_format_reapply.py` | Re-apply action: section pairing and attach |
| `tts_audiobook_tool/menus/text_menu.py` | Re-apply menu item |
| `tts_audiobook_tool/app_types/phrase.py` | `Phrase.fmt`, persistence, flattened join |
| `tts_audiobook_tool/app_types/timed_phrase.py`, `concat_util.py`, `app_metadata.py` | ABR leaf `fmt` |
| `browser_player/text-format.js`, `book-text.js`, `main.css` | Player parsing and rendering |
