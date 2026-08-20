# Dialog Segmentation

## Purpose

Dialog segmentation helps keep dialog and non-dialog text in separate text segments.

The main use case is voice selection. A project may use one voice for narration and another for dialog, so a segment such as:

```text
"Hello," the man said.
```

should be separable into:

```text
"Hello,"
the man said.
```

The same separation also supports accurate real-time text highlighting. Segment text must continue to correspond exactly to the source, including quote marks, punctuation, and structural whitespace.

This feature identifies useful dialog boundaries and preassigns every detected dialog segment to voice sample 2. It does not determine the speaker or attempt a complete grammatical interpretation of the passage; users can edit the resulting line-by-line voice selections.

---

## Relationship to normal segmentation

Dialog segmentation is an optional second pass after the normal sentence, phrase, paragraph, and length-based segmentation rules have run.

The second pass may subdivide an existing segment when it contains both quoted dialog and surrounding text. It must not merge segments that were already separated by the normal pass.

Detected dialog may therefore span multiple existing segments. The opening segment, interior segments, and closing segment remain separate; only segments that contain a dialog boundary alongside other text need to be divided. All detected dialog segments receive the same voice-sample-2 preassignment.

For example, these existing segments:

```text
Before "This spans 
several existing 
segments." After.
```

become:

```text
Before 
"This spans 
several existing 
segments." 
After.
```

The three pieces of dialog are not recombined.

---

## Recognized quote marks

The recognized trigger characters are double quote marks:

- straight double quote: `"`
- curly opening double quote: `“`
- curly closing double quote: `”`

Straight and curly glyphs may form a pair, including mixed pairs in imperfectly normalized source text.

The following are outside the current scope:

- single quote marks
- guillemets such as `«` and `»`
- other language-specific or unconventional quote glyphs

---

## Pairing rules

Quote marks are considered within a paragraph and processed in reading order.

- An opening quote mark must have a corresponding closing quote mark.
- Unmatched quote marks do not create boundaries.
- Pairing may span existing phrase groups or text segments.
- Pairing must not cross a paragraph boundary.
- A paragraph may contain multiple independent quote pairs.
- Double-quote nesting is not given a special semantic interpretation.

The paragraph limit is intentionally strict. It prevents one malformed or unmatched quote mark from changing the interpretation of large amounts of later text.

---

## Dialog qualification

A paired quote is not automatically treated as dialog. It must contain verbalizable content and then satisfy either the capitalization rule or one of the structural dialog signals below. The rules deliberately favor accepting probable dialog; the narrow rejection case is a short lowercase inline quote without any dialog signal.

### Verbalizable content

The quoted content must contain text that the application considers verbalizable. Punctuation and whitespace alone are not sufficient.

This is ignored:

```text
I was all like: "?!!"
```

### Capitalized first meaningful letter

After leading whitespace and nonverbal punctuation are ignored, a quote is accepted as dialog when its first meaningful letter is uppercase:

```text
He paused. "...Well, yes."
```

### Lowercase dialog signals

A quote whose first meaningful content is lowercase, numeric, or caseless is also accepted as dialog when any of the following is true:

- the quoted content contains more than three verbalizable words;
- the quote spans an existing phrase-group boundary;
- the opening quote is at the beginning of a paragraph, ignoring leading whitespace;
- the preceding non-whitespace character is a comma, colon, em dash, or en dash;
- the quoted content begins with an em dash or en dash, ignoring leading whitespace;
- the quoted content contains a question mark or exclamation mark.

Examples accepted by these rules include:

```text
thought Alice to herself, “after such a fall as this ...”
“yes,” she replied.
(aside) “—yes...” Then he stopped.
He shouted “go!”
```

A quote is ignored as probable non-dialog only when it lacks all of those signals and has no more than three verbalizable words. This continues to ignore short inline emphasis, labels, and scare quotes such as:

```text
That was "too much" for me.
He chose “red” over blue.
The answer was “yes”.
```

These are structural heuristics, not grammatical or semantic analysis. They intentionally prefer occasional false-positive dialog assignments over silently leaving likely dialog mixed with narration.

---

## Boundary and punctuation ownership

An accepted opening quote mark begins the quoted portion and stays with it.

An accepted closing quote mark ends the quoted portion and also stays with it. Punctuation immediately following the closing quote mark belongs to the quoted segment as well:

```text
Did he say "Hello"?
```

is separated conceptually as:

```text
Did he say 
"Hello"?
```

This ownership rule serves two purposes:

- highlighting includes punctuation that visually belongs to the quote;
- punctuation is not left dangling in a separate or following phrase group.

Boundary whitespace follows the project's general text-segmentation convention and remains associated with the preceding segment. All characters must be preserved when the resulting segments are concatenated.

---

## Boundary reason semantics

Dialog boundaries are phrase-level boundaries. When the second pass divides an existing phrase:

- a fragment that ends at a newly introduced dialog boundary receives `Reason.PHRASE`;
- the final fragment retains the original phrase's terminal reason, such as `SENTENCE`, `PARAGRAPH`, or `SPACE_BREAK`;
- if a dialog boundary already coincides with an existing phrase boundary, the existing reason is preserved;
- narration pieces retain the selected voice index of the segment from which they were derived;
- detected dialog pieces override that inherited value with voice index `1` (voice sample 2).

For example, an original sentence-level phrase:

```text
"Hello," the man said.   [SENTENCE]
```

is represented conceptually as:

```text
"Hello,"                 [PHRASE]
the man said.            [SENTENCE]
```

Likewise:

```text
I told them, "What is up?"   [SENTENCE]
```

becomes:

```text
I told them,             [PHRASE]
"What is up?"            [SENTENCE]
```

There is no dedicated `Reason.DIALOG`. Using `Reason.PHRASE` reflects that the quote mark creates an intra-sentence boundary while allowing the original structural reason to remain attached to the text that reaches the original phrase ending.

This also means that `Reason` alone does not classify a segment as dialog. Dialog status must be inferred from quote context or represented separately if explicit segment roles are added in the future.

---

## Voice preassignment

The import-time dialog-segmentation path assigns zero-based `voice_index = 1`—voice sample 2 in the UI—to every resulting segment wholly inside an accepted dialog span. This includes dialog that already occupies an entire segment and interior dialog segments spanning several existing groups.

Narration retains its inherited voice index. With the normal default of `-1`, narration resolves to voice sample 1 under user-defined voice selection.

The preassignment is stored even when a second voice sample has not been configured. It takes effect only when the project uses User-defined voice selection; dialog segmentation does not change the project's voice-selection mode. Multi-item batching honors user-defined line assignments: each generation round is split into per-voice sub-batches (groups of batches), so every line is generated with its own voice sample.

This is a narration-versus-dialog default, not speaker detection. All detected dialog receives voice sample 2 until the user edits individual line selections.

---

## Required invariants

Dialog segmentation must preserve the following properties:

- Concatenating the resulting segment text reproduces the original text exactly.
- Quote marks and punctuation are never removed or rewritten.
- Existing segment boundaries are never eliminated.
- Paragraph boundaries are never crossed while searching for a pair.
- Ignored or unmatched quote pairs leave segmentation unchanged.
- Existing per-segment properties remain associated with narration pieces subdivided from that segment.
- Accepted quoted pieces receive voice index `1` without mutating reused source groups.
- When the option is disabled, normal segmentation and voice-assignment behavior are unchanged.

---

## Representative examples

### Dialog followed by attribution

```text
"Hello," the man said.
```

Result:

```text
"Hello,"
the man said.
```

### Narration followed by dialog

```text
I told them, "What is up?"
```

Result:

```text
I told them,
"What is up?"
```

### Entire segment is quoted

```text
"This is me talking in a free-standing paragraph. This quote may span more than one phrase group, but yea."
```

No additional boundary is needed when the existing segment already contains only dialog.

### Multiple dialog passages

```text
He said "Hello." Then she said “Goodbye.” Finally.
```

Result:

```text
He said
"Hello."
Then she said
“Goodbye.”
Finally.
```

The examples above omit boundary whitespace from the visual presentation. The underlying text retains it according to the normal segmentation convention.

---

## Tradeoffs in the current rules

### Heuristic recognition

The lowercase dialog signals reduce false negatives for dialog continuations, brief replies, and writing systems or content without an uppercase opening. Because the rules intentionally favor likely dialog, long quoted labels or lowercase quoted passages near dialog-like punctuation may occasionally be assigned the dialog voice.

Conversely, capitalization alone cannot prove that text is dialog. Capitalized labels, titles, interface terms, and quoted names may still be treated as dialog. Short lowercase inline quotes remain unassigned only when they have no structural dialog signal.

### Paragraph-local pairing

Stopping at every paragraph boundary limits the effect of malformed source text and makes unmatched-quote behavior predictable. The tradeoff is that conventional multi-paragraph dialog is not yet recognized as one continuing quote context.

### No recombination

Preserving normal first-pass boundaries protects maximum-length, sentence, and phrase-group decisions. A long dialog passage may consequently remain divided among several dialog segments rather than becoming one segment.

### Fixed automatic voice

Every detected dialog passage is preassigned to voice sample 2, regardless of speaker. This provides a useful two-voice narration/dialog default, but books with multiple speaking characters still require manual line-by-line voice editing.

### Punctuation after the closing quote

Assigning immediately adjacent punctuation to the quote is useful for highlighting and prevents dangling punctuation. In some grammatical interpretations—particularly where punctuation falls outside the quote—that punctuation belongs to the containing sentence rather than the quoted speech. Visual and segmentation consistency take priority here.

### Limited glyph set

Supporting only common straight and curly double quotes keeps pairing behavior predictable, but does not cover every language or publishing convention.

---

## Possible refinements

### Multi-paragraph dialog

Quotes that span paragraphs are a likely future extension. Supporting them safely requires more than simply removing the paragraph limit. Published multi-paragraph dialog commonly places an opening quote mark at the start of each paragraph while omitting a closing mark until the final paragraph.

A future rule set could carry quote state across paragraph boundaries while considering:

- repeated opening marks at paragraph starts;
- section and chapter boundaries as hard stops;
- limits on how far an unmatched quote may propagate;
- malformed source text and OCR errors;
- whether each paragraph should remain independently classifiable for voice assignment.

### Language-aware qualification

The capital-letter heuristic could be adapted for scripts without letter case or made dependent on the project's language. Any replacement should remain conservative and understandable to users.

### Stronger dialog classification

Additional grammatical hints could distinguish spoken dialog from titles, labels, scare quotes, and quoted terminology. More aggressive classification would improve some cases but would also be less predictable and introduce new false positives.

### Additional quote conventions

Single quotes, guillemets, and language-specific quote marks could be added as explicit, configurable conventions rather than being folded into the default rules.

### Explicit segment roles

A future text model could record whether a segment is dialog, narration, or unknown. Explicit roles would make voice assignment easier and would preserve quote context for interior segments that do not themselves contain an opening or closing mark.

### Preview and diagnostics

A segmentation preview could show which pairs were accepted, ignored, or left unmatched. This would help users understand conservative decisions and identify malformed source text before importing or regenerating an audiobook.
