# Text Segmentation System

## Purpose

Text segmentation converts imported book text into the prompt-sized units sent to a TTS model while retaining enough structure for pauses, navigation, highlighting, alignment, and optional dialog voices.

The system has three principal layers:

1. `PhraseSegmenter` discovers sentence, phrase, paragraph, and length boundaries and produces `Phrase` objects.
2. `PhraseGrouper` combines phrases into `PhraseGroup` objects according to the selected segmentation strategy. A phrase group is the atomic TTS prompt.
3. When enabled, `DialogSegmenter` subdivides those groups at accepted dialog boundaries and assigns the dialog voice.

The strict text-preservation rule applies to sentence discovery:

```python
"".join(sentence_strings) == source
```

Sentence segmentation slices its original input without rewriting it. The higher-level `PhraseGrouper.text_to_groups()` intentionally performs additional normalization: it converts CRLF/CR line endings to LF, removes spaces and tabs immediately before line feeds, and may canonicalize repeated structural breaks. Group text therefore corresponds to the normalized segmentation input rather than necessarily being byte-for-byte identical to the imported source.

## High-level pipeline

`PhraseGrouper.text_to_groups()` performs these operations in order:

```text
source text
  → normalize line endings and line-ending whitespace
  → pySBD and quote-aware sentence segmentation
  → sentence-internal phrase segmentation
  → classify phrase-ending reasons
  → split phrases that exceed max_words
  → normalize ornamental separators and repeated space breaks
  → group phrases according to the selected strategy
  → enforce max_words at group boundaries
  → optional dialog segmentation and voice assignment
  → final ornamental-group normalization
  → PhraseGroup list used as TTS prompts
```

This ordering is significant. Dialog segmentation is a downstream refinement; it does not replace grammatical sentence segmentation and does not recombine boundaries created by earlier stages.

## Sentence segmentation

The public entry point is:

```python
PhraseSegmenter.string_to_sentence_strings(source, pysbd_lang)
```

### Language selection

The requested language code is validated through pySBD. An unsupported code silently falls back to English, preserving the previous application behavior.

### Full-source pySBD pass

The complete source is passed to pySBD with:

```python
clean=False
char_span=True
```

pySBD is used to discover boundary offsets, not to supply final sentence text. This distinction matters because pySBD may:

- omit leading whitespace from a returned span;
- normalize a line feed while processing;
- emit overlapping or gapped spans around intermittent punctuation or ellipses.

The system derives boundaries from the starts of logical spans after the first span. Punctuation/whitespace-only spans are folded backward before those boundaries are selected. Using the next logical start means any pySBD gap belongs to the preceding sentence and every original character remains accounted for.

### Source-offset reconstruction

Boundaries from all sentence-detection passes are collected, normalized, sorted, and deduplicated. The final strings are produced by slicing the original `source` once.

Consequently:

- no quote mark or punctuation is recreated manually;
- repeated text cannot confuse substring lookup;
- leading and trailing whitespace is retained;
- gaps or overlaps in pySBD metadata cannot duplicate or discard source characters.

A punctuation-only result after slicing is merged into its predecessor when one exists.

## Quote-aware sentence segmentation

pySBD deliberately protects punctuation inside balanced quotations. This is useful for many short quotations, but it can hide genuine sentence boundaries in multi-sentence dialog or quoted passages.

The application therefore adds quote-interior sentence boundaries in a controlled second sentence-detection pass.

### Shared quote-span detection

`tts_audiobook_tool/text_ops/quote_spans.py` identifies balanced spans as half-open source ranges:

```python
QuoteSpan(start, end, depth)
```

The range `[start, end)` includes both quote delimiters. The helper only describes quote structure; it does not decide whether a quotation represents spoken dialog.

Sentence segmentation recognizes these families:

- straight double quotes: `"..."`
- curly double quotes: `“...”`
- straight single quotes: `'...'`
- curly single quotes: `‘...’`
- guillemets: `«...»`
- single guillemets: `‹...›`
- corner quotes: `「...」`
- white corner quotes: `『...』`

Straight and curly glyphs within the same family may form mixed pairs. This supports imperfectly normalized ebook text such as `“Hello."`.

Directional glyphs have fixed opening or closing roles. Symmetric straight glyphs use surrounding context. Escaped straight quotes are ignored, and an apostrophe between word characters is never treated as a straight single-quote delimiter. Unmatched or ambiguous delimiters do not raise errors and do not create a quote span.

Nested spans are returned in source order with their depth. Sentence segmentation may inspect every level; dialog segmentation applies its own narrower policy.

### Interior pySBD passes

For every balanced quote span, the quote delimiters are excluded and pySBD is run over the interior using the same language as the full-source pass. Interior-relative boundaries are translated back to absolute source offsets and added to the full-source boundaries.

For example:

```text
He said, "One. Two." Then left.
```

becomes sentence strings equivalent to:

```python
[
    'He said, "One. ',
    'Two." ',
    'Then left.',
]
```

This works even though the complete pySBD segment does not begin with a quote.

### Closing quotes and attributions

The final end of a quote interior is not automatically added as an interior boundary. The full-source pass and closing-quote rules decide whether following text is a continuation.

Thus:

```text
"One? Two!" she said.
```

becomes:

```python
['"One? ', 'Two!" she said.']
```

The attribution remains part of the final grammatical sentence. Later phrase and dialog processing can separate `Two!"` from `she said.` while assigning the low-pause `PHRASE_QUOTE_END` reason.

When pySBD places a boundary immediately before a closing delimiter, the boundary is moved after the closing quote, attached punctuation, and boundary whitespace. A lowercase continuation remains attached. A capitalized continuation or another opening quote may begin a new sentence.

These rules also provide deterministic ownership for adjacent quotations:

```text
"One." "Two."
```

becomes:

```python
['"One." ', '"Two."']
```

## Sentence-internal phrase segmentation

Each sentence string is split into smaller prosodic phrases. Breaks are currently introduced:

- before an opening parenthesis;
- after a double quote followed by a space;
- after a curly closing double quote followed by a space;
- after a comma, semicolon, or colon followed by a space;
- after an en dash or em dash;
- after a closing parenthesis;
- after an eligible run of exactly two normal dashes.

A `--` run is eligible only when vocalizable content appears on both sides, allowing optional surrounding whitespace. Single dashes, runs of three or more dashes, edge-bounded dashes, and punctuation-bounded runs are retained.

Leading whitespace created by a split is moved to the preceding phrase. Whitespace/punctuation-only pieces are also folded backward. This maintains the project convention that boundary whitespace belongs to the preceding segment.

## Phrase reasons

Every `Phrase` records why its text ends at that boundary. Reasons are ordered from weaker to stronger:

| Reason | Meaning |
|---|---|
| `UNDEFINED` | Backward-compatible or unknown boundary |
| `WORD` | Arbitrary split needed to satisfy a word limit |
| `PHRASE_QUOTE_END` | Closing quote followed by an attribution; almost no pause |
| `PHRASE` | Sentence-internal prosodic boundary |
| `SENTENCE` | Grammatical sentence boundary |
| `PARAGRAPH` | Paragraph boundary |
| `SPACE_BREAK` | Blank-line/scene-style structural break |
| `SECTION_BREAK` | End of an imported section |

For the last phrase in each sentence string, trailing line-feed count determines whether its reason is `SENTENCE`, `PARAGRAPH`, or `SPACE_BREAK`. Earlier pieces receive `PHRASE`.

A phrase exceeding `max_words` is divided into approximately balanced word slices. Intermediate slices receive `WORD`; the final slice retains the original reason.

## Structural cleanup

### Ornamental lines

Non-vocalizable paragraph or space-break phrases, such as dinkuses and decorative divider lines, are merged with neighboring vocalizable text. A merged ornament promotes the preceding boundary to `SPACE_BREAK`. Leading ornaments are attached to the first vocalizable phrase.

A final analogous cleanup occurs after dialog segmentation because splitting at quote edges can expose an ornamental-only group again.

### Consecutive space breaks

Immediate repeated `SPACE_BREAK` reasons are downgraded after the first one. This avoids overstating structure around adjacent headings emitted with multiple blank lines.

## Phrase grouping strategies

After phrase creation, `PhraseGrouper` creates the TTS prompt groups.

### `SENTENCE`

Ends a group at every sentence-or-stronger reason. Internal sentences in a multi-sentence quotation therefore remain distinct groups unless another later operation subdivides them further.

### `SENTENCE_PLUS`

Starts from sentence groups, then merges very short sentences with a neighbor when the result remains within `max_words` and does not cross a paragraph boundary. This reduces TTS artifacts from very short prompts.

### `MULTI_SENTENCE`

Uses the same merging mechanism with a deliberately high shortness threshold, combining multiple sentences up to `max_words`.

### `MAX_LEN`

Initially groups through paragraph boundaries rather than sentence boundaries, then applies the maximum-word constraint.

All strategies receive a final max-word grouping pass. An already indivisible phrase that exceeds the limit remains as one non-empty group.

## Interaction with dialog segmentation

Dialog segmentation is optional and runs after normal grouping. Its detailed behavior is documented in [dialog-segmentation.md](dialog-segmentation.md).

The responsibilities are deliberately separate:

| Concern | Sentence/phrase segmentation | Dialog segmentation |
|---|---:|---:|
| Find grammatical sentences inside quotes | Yes | No |
| Decide whether a quote is probable dialog | No | Yes |
| Split narration from accepted dialog at quote edges | No | Yes |
| Assign dialog voice sample 2 | No | Yes |
| Detect attribution continuations | Only enough to avoid a premature final sentence boundary | Yes |
| Apply `PHRASE_QUOTE_END` | No | Yes |
| Recombine existing groups | Depending on grouping strategy | Never |

For:

```text
"One? Two!" she said.
```

using the `SENTENCE` strategy with dialog segmentation enabled, the final groups are conceptually:

```text
"One?       dialog voice; SENTENCE
Two!"       dialog voice; PHRASE_QUOTE_END
she said.   narration voice; SENTENCE
```

For `SENTENCE_PLUS`, `MULTI_SENTENCE`, or `MAX_LEN`, the first two quoted sentence phrases may share a group before the dialog pass because those strategies intentionally create larger TTS prompts. The dialog pass still prevents accepted dialog and narration from sharing a final group.

### Dialog classification independence

Quote pairing is shared, but quote interpretation is not. Dialog segmentation currently accepts straight and curly double quotes only. It uses textual signals such as capitalization, length, paragraph position, introductory punctuation, a leading dialog dash, and question/exclamation punctuation.

Whether a quote crosses existing phrase groups is not evidence that it is dialog. This prevents the new quote-interior sentence boundaries—or a different grouping strategy—from changing dialog classification accidentally.

Dialog pairing remains physical-line scoped, while sentence segmentation can inspect a broader set of balanced quote styles. Expanding dialog voice assignment to guillemets, single quotes, or other language-specific conventions is a separate policy decision.

## Important invariants

Changes to segmentation should preserve these properties:

- Sentence strings concatenate to their exact input.
- Final phrase groups preserve content and ordering, subject only to the documented line-ending, line-whitespace, and repeated-break normalization.
- Boundary whitespace belongs to the preceding unit.
- Quote marks and punctuation are retained exactly once and in source order.
- Dialog segmentation may subdivide but never merge pre-existing groups.
- Grouping strategy affects prompt size, not whether a quote is structurally balanced.
- Dialog classification does not depend on incidental first-pass group count.
- Unsupported or malformed quote syntax degrades safely to ordinary pySBD behavior.
- No segmentation stage emits an avoidable empty prompt.

## Implementation map

| File | Responsibility |
|---|---|
| `tts_audiobook_tool/text_ops/quote_spans.py` | Shared balanced quote-span detection |
| `tts_audiobook_tool/text_ops/phrase_segmenter.py` | Sentence offsets, quote-interior boundaries, phrases, reasons, and structural cleanup |
| `tts_audiobook_tool/text_ops/phrase_grouper.py` | Strategy-based prompt grouping and max-word enforcement |
| `tts_audiobook_tool/text_ops/dialog_segmenter.py` | Optional dialog qualification, quote-edge splitting, voice assignment, and attribution reason |
| `tts_audiobook_tool/app_types/phrase.py` | `Phrase`, `PhraseGroup`, and ordered `Reason` definitions |
| `tests/test_quote_spans.py` | Quote-pairing behavior |
| `tests/test_phrase_segmenter.py` | Sentence and phrase boundary behavior and source preservation |
| `tests/test_phrase_grouper.py` | Grouping strategies and end-to-end dialog composition |
| `tests/test_dialog_segmenter.py` | Dialog qualification, splitting, reasons, and voice assignment |

## Limitations and extension points

The system remains heuristic rather than a full parser.

- pySBD determines language-specific grammatical boundaries and inherits its abbreviation and punctuation tradeoffs.
- Straight quote marks are inherently ambiguous.
- Capitalization is not a universal sentence-start signal.
- Multi-paragraph literary dialog conventions vary by language and publisher.
- A quotation can contain multiple sentences without representing spoken dialog.
- The phrase delimiter list is intentionally small and may not capture ideal prosody in every language.

The shared structural quote detector should be extended independently from dialog policy. Supporting a new delimiter for sentence discovery must not automatically assign that style a dialog voice.

## Old vs. new

### Old sentence handling

Previously, pySBD first segmented the entire source into strings. Because pySBD protects punctuation inside quotes, the application applied a special second pass only when a complete pySBD result appeared to start and end with a double quote.

The check did not establish a matched pair. It inspected the first and last non-whitespace characters, removed them, segmented the middle, and reattached those edge characters.

That handled a standalone quotation:

```text
"One. Two."
```

but failed when the quote was embedded:

```text
He said, "One. Two." Then left.
```

Old result:

```python
['He said, "One. Two." ', 'Then left.']
```

Behavior also depended on pySBD's coarse grouping. In:

```text
"One. Two," she said. "Three? Four!"
```

only the second quote was reprocessed because it happened to occupy a complete coarse segment:

```python
[
    '"One. Two," she said. ',
    '"Three? ',
    'Four!"',
]
```

Adjacent quotes could receive inconsistent delimiter ownership:

```python
# Input:  '"One." "Two."'
# Old:
['"One.', '" "Two."']
```

The old string-return path could also omit characters that pySBD excluded from its spans, notably leading whitespace.

Dialog qualification additionally treated crossing an existing group boundary as evidence that a lowercase quotation was dialog. Improving sentence segmentation would have created more such boundaries and therefore changed voice classification as an unintended side effect.

### New sentence handling

The new system identifies balanced quote spans anywhere in the source and runs pySBD over each interior. All detected boundaries are represented as absolute offsets, and final sentence strings are sliced directly from the original source.

The embedded example now becomes:

```python
[
    'He said, "One. ',
    'Two." ',
    'Then left.',
]
```

Both quotes in the attribution example are treated consistently:

```python
[
    '"One. ',
    'Two," she said. ',
    '"Three? ',
    'Four!"',
]
```

Adjacent quote ownership is deterministic:

```python
['"One." ', '"Two."']
```

Nested and additional literary quote styles can contribute sentence boundaries without automatically becoming dialog. Exact source reconstruction is enforced by design rather than relying on pySBD's returned strings.

Finally, dialog classification no longer uses group crossing as evidence. Sentence discovery, grouping strategy, and dialog voice qualification can therefore evolve without silently redefining one another.