# Special Codes Refactor Architecture

## Status

Proposed architecture and migration plan.

## Summary

The project currently recognizes, strips, preserves, measures, and emits terminal control sequences in several independent places. These implementations have different grammars and policies, and some are incorrect at sequence or stream boundaries.

This document proposes a shared terminal-sequence layer built around an incremental ECMA-48 tokenizer. The tokenizer will provide common recognition while explicit sink policies determine whether sequences are preserved, interpreted, rejected, or removed.

The objective is not to apply one regular expression everywhere. Terminal input, captured worker output, Rich rendering, plain-text persistence, keyboard protocols, and TTS text normalization have different requirements. They should share parsing primitives without losing those distinctions.

## Terminology

This document uses the following terms:

- **C0 controls:** ASCII control characters such as NUL, BEL, BS, TAB, LF, CR, and ESC.
- **C1 controls:** The 8-bit control range, including the single-byte CSI form U+009B.
- **ESC sequence:** A sequence introduced by ESC (U+001B).
- **CSI:** Control Sequence Introducer, normally `ESC [` followed by parameters, intermediates, and a final byte.
- **OSC:** Operating System Command, introduced by `ESC ]` and terminated by BEL or ST.
- **DCS-family string:** DCS, SOS, PM, and APC strings introduced by `ESC P`, `ESC X`, `ESC ^`, and `ESC _`, normally terminated by ST.
- **SGR:** CSI `m` styling sequences.
- **OSC 8:** Terminal hyperlinks.
- **ST:** String Terminator, normally `ESC \\`.
- **ANSI text:** Used informally in existing code for strings containing terminal styling. This document prefers the more precise term **terminal-formatted text**.

## Goals

1. Use one implementation to recognize terminal sequences in complete strings and incrementally delivered streams.
2. Make the policy at every trust boundary explicit.
3. Prevent terminal control payloads or fragments from leaking through incomplete sanitizers.
4. Give console and Textual worker output equivalent safety policies.
5. Preserve supported styling and hyperlinks where required.
6. Make transcript behavior deterministic at chunk boundaries and end-of-stream.
7. Calculate visible widths consistently for terminal-formatted Unicode text.
8. Eliminate cross-stream parser-state corruption between stdout and stderr.
9. Centralize safe OSC 8 construction and terminal-control emission where practical.
10. Provide a shared, table-driven test corpus for all consumers.

## Non-goals

1. Implementing a complete VT terminal emulator.
2. Making keyboard escape-sequence decoding use the same output-sanitization policy.
3. Treating TTS/source Unicode normalization as terminal sanitization.
4. Removing Rich or Textual's ANSI-to-style conversion.
5. Eliminating intentionally standalone ANSI emission in `launch.py`.
6. Preserving every cursor-control behavior emitted by arbitrary third-party programs.

## Current architecture

### Terminal emission

Most app-owned formatting constants and builders live in:

- `tts_audiobook_tool/system_support/ansi.py`
- `tts_audiobook_tool/constants.py`

Some literal sequences remain in:

- `tts_audiobook_tool/ask_advanced.py`
- `tts_audiobook_tool/conversation/conversation_internals.py`
- `tts_audiobook_tool/concat_util.py`
- developer scripts under `testx/`
- the intentionally standalone `launch.py`

Emission is already more centralized than parsing, but the remaining literals make it harder to establish a complete supported-sequence vocabulary.

### General stripping

`tts_audiobook_tool/text_util.py` defines `strip_ansi_codes()` using two regular expressions. It removes complete OSC sequences first, followed by common 7-bit CSI and two-byte ESC sequences.

It is used for several different purposes:

- deriving plain labels and breadcrumbs;
- calculating divider lengths;
- producing plain generation transcripts;
- simplifying status text;
- test assertions.

These callers implicitly require different policies even though they use one ambiguously named function.

### Input filtering

`tts_audiobook_tool/ask.py` defines a second terminal-sequence expression and applies it to completed line input. `tts_audiobook_tool/ask_advanced.py` separately rejects ASCII controls and DEL in prefilled input before selecting an advanced editor.

Raw hotkey handling in `ask.py` is a keyboard protocol and intentionally interprets escape sequences instead of stripping them.

### Worker stream normalization

`ConsoleLineAssembler` in `tts_audiobook_tool/textual/worker_app.py` performs partial stream-aware terminal processing:

- buffers selected incomplete CSI and OSC sequences;
- converts CRLF to LF;
- treats CR as replacement of the current live line;
- treats CSI `G` as line home;
- removes CSI `K`;
- removes non-hyperlink OSC;
- preserves OSC 8 for Rich;
- produces committed lines plus one replaceable live line.

`GenerationTranscript` in `tts_audiobook_tool/textual/generation_app.py` imports the assembler's private pending-control helper, strips sequences with `text_util.strip_ansi_codes()`, and turns carriage returns into newlines so progress history remains visible.

### Rich rendering

Many Textual components use `Text.from_ansi()`. This correctly converts supported SGR into Rich styles and can preserve hyperlinks after appropriate preprocessing. It is a renderer, not a complete terminal sanitizer. For example, unsupported OSC payload can become visible text.

`HangingIndentText.from_ansi_prefix()` demonstrates the preferred trust split: parse a trusted formatting prefix, then append content literally.

### Display width

Width is currently calculated through several mechanisms:

- `Ui.ANSI_RE` and `Ui.display_width()` in the conversation UI;
- `_display_width()` in the Windows advanced input editor;
- `Text.cell_len` and `rich.cells.cell_len` in Textual code;
- `len(strip_ansi_codes(...))` for dividers and labels.

These implementations disagree for some terminal sequences, CJK text, combining characters, emoji, and tabs.

## Audit findings

### Broken OSC and string-control filtering in line input

The alternation in `tts_audiobook_tool/ask.py` places the generic two-byte ESC branch before OSC and DCS-family branches. Consequently, `ESC ]`, `ESC P`, `ESC X`, `ESC ^`, and `ESC _` are consumed as complete two-byte sequences before the longer alternatives can match.

The introducer and terminator may be removed while the payload remains. Examples include terminal-title parameters, OSC 8 URI parameters, and DCS payload text.

This is the highest-priority correctness defect.

### General stripping is not general terminal sanitization

`text_util.strip_ansi_codes()`:

- recompiles its regular expressions on every call;
- handles complete OSC and common 7-bit CSI;
- does not remove DCS/SOS/PM/APC payloads;
- does not recognize 8-bit C1 forms;
- does not define a policy for BEL, BS, DEL, CR, or other controls;
- can expose payload from malformed or incomplete sequences.

The name obscures these limitations.

### Conversation width uses a narrower grammar

`Ui.ANSI_RE` recognizes only `ESC [` followed by digits/semicolons and an ASCII letter. It misses:

- OSC hyperlinks;
- private CSI parameters such as `?` and `>`;
- colon-separated SGR parameters;
- CSI intermediates;
- valid non-letter final bytes;
- C1 forms.

Missed printable parameters and OSC payloads distort wrapping calculations. That can cause incorrect cursor clearing, stale rows, or overwritten output.

### Streaming recognition is incomplete

The worker stream's pending-control logic handles only a restricted trailing CSI grammar and OSC. It does not robustly buffer:

- CSI containing colon parameters;
- CSI containing all private parameter bytes;
- CSI intermediates such as spaces;
- DCS/SOS/PM/APC strings;
- C1 forms.

Valid sequences split at unsupported positions can leak fragments into visible output or transcripts.

### End-of-stream behavior is unspecified

`ConsoleLineAssembler.finish()` and `GenerationTranscript.close()` silently discard pending incomplete control sequences. This may be a reasonable fail-closed policy, but it is neither shared nor explicitly tested.

The architecture must specify handling for incomplete terminal data at EOF.

### Cursor behavior is a deliberate but undocumented approximation

The assembler treats CR as clearing its accumulated live line. A terminal normally moves the cursor to column zero and overwrites cells without erasing untouched suffix cells. Other cursor movement and erase operations are not modeled.

The project does not need a full terminal emulator, but the supported approximation should be explicit and tested. Unsupported motion should be consistently dropped or normalized rather than accidentally delegated to Rich.

### stdout and stderr share parser state

One worker assembler processes both streams. A partial sequence emitted by stdout can be completed with text from the next stderr event, or vice versa. The transcript has an analogous shared pending state.

Incremental recognition state must be maintained per `(operation_id, stream)`. Streams may be merged only after each has been tokenized or normalized independently.

### Console mode bypasses normalization

Console-mode generation writes `ConsoleOutput.text` directly to the real stdout or stderr before feeding the assembler. Textual output removes at least non-hyperlink OSC before rendering, while console output can pass arbitrary worker-generated control sequences to the user's terminal.

This permits inconsistent behavior and potentially dangerous controls such as OSC 52 clipboard operations. Worker libraries may usually be trusted, but raw output should still be subject to an explicit allowlist.

### Rich is being used as an implicit sanitizer

`Text.from_ansi()` is appropriate for trusted styling but is not a full control-sequence policy. Unsupported controls can be removed differently from the project's plain-text and stream paths, and some payloads can remain visible.

All captured or untrusted terminal output should be normalized before it reaches Rich.

### OSC 8 construction permits sequence injection

`text_util.make_terminal_hyperlink()` interpolates the link target and display text directly into an OSC 8 sequence. BEL, ESC, or ST in either value can terminate the sequence and inject additional terminal controls.

The target must be validated or safely encoded. Display text should be appended as literal content where possible, or sanitized according to the output sink.

### Input policies are inconsistent

Advanced-input prefill rejects ASCII controls and DEL, but not C1 controls or Unicode formatting/bidirectional controls. The Windows editor permits typed TAB and generally accepts characters at or above U+0020. Post-read filtering in `ask.py` removes only recognized terminal sequences and does not define a broader C0/C1 policy.

Input sanitation must distinguish:

- prompt strings produced by the application;
- editable prefill content;
- completed line input;
- raw hotkey input.

### TTS normalization is not terminal sanitization

`normalize_text_general()` removes selected Unicode categories such as `Cf`, but not `Cc`. ESC and other controls can therefore survive source normalization.

This should not be merged blindly with terminal rendering policy. Instead, content ingress should have an explicit control-character policy before text enters terminal, persistence, or TTS paths.

### Width calculation is inconsistent

`len(strip_ansi_codes(...))` is not a cell-width calculation. It is wrong for wide characters and combining sequences. The two custom width implementations also differ from Rich's cell measurement and incompletely model emoji.

A single cell-width primitive should be used after producing the sink's visible plain text.

## Architectural principles

### Share recognition, not policy

A shared tokenizer should identify terminal constructs without deciding their final meaning. Consumers apply named policies appropriate to their sink.

This avoids both current extremes:

- duplicated regexes with inconsistent grammars;
- one universal stripping function that cannot express rendering, transcript, input, and keyboard requirements.

### Parse streams before merging them

stdout and stderr maintain independent tokenizer and normalization state. Normalized events can retain stream identity and then be merged in event order for display or persistence.

### Allowlist terminal behavior

Captured output should preserve only behavior the application intentionally supports:

- printable text;
- LF and a documented CR policy;
- selected SGR styles;
- validated OSC 8 hyperlinks where supported;
- the small set of cursor/erase controls required for progress rendering.

All other OSC commands, DCS-family strings, and unsupported cursor operations should be dropped or converted to documented plain behavior.

### Fail closed on malformed controls

A malformed or unterminated control sequence must not cause its parameter or command payload to be emitted as ordinary text by default. Incremental buffers must have size limits so an unterminated sequence cannot consume unbounded memory.

At EOF, a pending sequence should be discarded and optionally reported through diagnostics. It must not be replayed as raw terminal data.

### Keep trusted formatting separate from content

App-generated formatting prefixes may be parsed as terminal styling. User-, book-, model-, filesystem-, or subprocess-derived display content should be appended literally or sanitized before formatting.

### Use precise names

Names such as `strip_ansi_codes()` conceal important policy. New APIs should name their sink or result, such as `plain_terminal_text()`, `sanitize_terminal_input()`, or `normalize_captured_stream()`.

## Proposed module structure

Introduce:

```text
tts_audiobook_tool/system_support/terminal_sequences.py
```

The existing `system_support/ansi.py` remains responsible for emitting app-owned terminal controls. It may import safe builders from the new module, or the two modules may later be combined if their responsibilities remain clear.

### Tokenizer

The core should be an incremental tokenizer/state machine supporting:

- printable text;
- C0 controls;
- optional C1 equivalents;
- two-byte and multi-byte ESC sequences;
- CSI parameters in the `0x30-0x3f` range;
- CSI intermediates in the `0x20-0x2f` range;
- CSI final bytes in the `0x40-0x7e` range;
- OSC terminated by BEL or ST;
- DCS/SOS/PM/APC strings terminated by ST;
- chunk boundaries at every byte;
- explicit EOF handling;
- bounded buffering for string controls.

The ordering must recognize OSC and DCS-family introducers before treating them as standalone ESC sequences.

A conceptual token model is:

```python
@dataclass(frozen=True)
class TerminalToken:
    kind: TerminalTokenKind
    raw: str
    command: str = ""
    parameters: str = ""
    payload: str = ""
```

The exact public representation may be narrower. Avoid exposing internals unless multiple consumers genuinely need token-level access.

### Complete-string API

A convenience API should run the same tokenizer to completion:

```python
def plain_terminal_text(
    text: str,
    *,
    control_policy: PlainControlPolicy = PlainControlPolicy.TEXT,
) -> str:
    """Return visible text with terminal sequences removed."""
```

The control policy should explicitly define treatment of TAB, LF, CR, BS, BEL, DEL, C1, and Unicode format characters. Different defaults may be appropriate for labels and multiline persisted logs, so avoid hidden behavior.

### Input API

```python
def sanitize_terminal_input(
    text: str,
    *,
    allow_tab: bool = False,
    allow_newline: bool = False,
) -> str:
    """Remove terminal sequences and disallowed controls from textual input."""
```

An accompanying predicate may support prefill fallback decisions:

```python
def contains_unsafe_terminal_input(text: str, ...) -> bool: ...
```

Raw hotkey readers remain separate because their ESC sequences are commands, not text.

### Captured-stream API

```python
class CapturedTerminalStream:
    def feed(self, chunk: str) -> list[TerminalEvent]: ...
    def finish(self) -> list[TerminalEvent]: ...
```

A stream instance belongs to exactly one logical source stream. It should produce normalized events rather than raw controls.

Possible event types include:

- printable terminal-formatted text with allowlisted SGR/OSC 8;
- newline;
- carriage return or line-home;
- erase-line;
- ignored-control diagnostic.

The UI line model and transcript model consume the same events but apply different policies.

### UI policy

The Textual worker policy should:

- preserve allowlisted SGR;
- preserve validated OSC 8 metadata;
- drop all other OSC and DCS-family commands;
- interpret LF, CR/line-home, and supported erase operations;
- drop unsupported cursor controls rather than rely on Rich to decide;
- append resulting text to Rich only after normalization.

The current simplified CR replacement policy may remain initially for compatibility, but it must be named and tested. A later implementation may track cursor columns if suffix-preserving overwrite behavior is important.

### Transcript policy

The transcript policy should:

- remove all terminal controls and styling;
- preserve printable hyperlink labels but not OSC metadata;
- turn each CR progress update into a separate line, retaining current behavior;
- define behavior for BS, TAB, form feed, and other controls;
- discard incomplete control sequences at EOF;
- operate per source stream before events are merged.

The distinction between UI replacement and transcript history is intentional and should be represented as policy, not duplicated parsing code.

### Console-output policy

Console mode must no longer write raw worker chunks before normalization. It should emit a safe reconstruction from normalized tokens:

- allowlisted SGR only;
- validated OSC 8 only if hyperlinks are desired in console mode;
- LF, CR, and supported erase/home controls;
- no arbitrary OSC, DCS-family, clipboard, title, device, or cursor commands.

If reconstruction complexity is undesirable, the initial console policy can emit plain normalized text while retaining CR/LF progress behavior.

### Width API

```python
def terminal_cell_width(text: str, *, tab_size: int = 8) -> int:
    """Return visible terminal cells after shared sequence normalization."""
```

Implementation should use Rich's existing cell-width support, such as `rich.cells.cell_len`, after removing non-display sequences. Multiline row counting should build on the same primitive.

Tabs require a starting column and tab size; the API may need a lower-level helper for contextual tab expansion.

### OSC 8 builder

Provide one safe builder:

```python
def make_terminal_hyperlink(
    target: str,
    display: str,
    *,
    is_file: bool = False,
) -> str:
    ...
```

Requirements:

- reject or encode BEL, ESC, ST, and disallowed controls in the target;
- construct file URIs correctly rather than blindly prefixing arbitrary text;
- do not parse terminal sequences embedded in display content;
- document allowed URI schemes or explicitly state that schemes are unrestricted;
- provide a Rich-native path where possible so Textual code need not round-trip through OSC.

## Stream ownership and event flow

The desired worker flow is:

```text
ConsoleOutput(operation_id, stream, chunk)
                |
                v
per-(operation_id, stream) tokenizer/normalizer
                |
                v
normalized terminal events retaining stream identity
          /            |             \
         v             v              v
 Textual UI policy  transcript policy  console policy
```

Parser state must not be shared across stdout and stderr. Event ordering is preserved by merging only normalized results in the order their source `ConsoleOutput` events were received.

If styled state is retained across chunks, it also belongs to the individual stream. Merging two independently styled streams may require each normalized event to carry complete style state or to insert resets at stream boundaries.

## Compatibility behavior to preserve

The refactor should initially preserve these tested behaviors:

1. Partial worker lines remain buffered until newline.
2. The UI exposes committed lines plus one replaceable live line.
3. CR progress updates replace the UI live line.
4. Transcripts preserve CR progress updates as separate lines.
5. CSI split across worker chunks does not leak.
6. Non-display OSC 0/2 is removed for BEL and ST termination.
7. OSC 8 hyperlinks survive into Rich and remain clickable after hard wrapping.
8. SGR becomes Rich style spans rather than visible bytes.
9. ANSI prefixes do not affect hanging-indent offsets.
10. Unsupported terminals receive an ESC-free startup error.
11. Conversation rendering retains its existing exact-width no-phantom-row behavior.

Any intentional behavior change should be isolated and documented separately from parser consolidation.

## Testing strategy

### Shared corpus

Create a table-driven corpus under `tests/`, for example:

```text
tests/terminal_sequence_cases.py
```

Each case should carry:

- source text or chunks;
- recognized sequence type;
- expected plain text;
- expected Rich/display result;
- expected transcript result;
- expected input-sanitization result;
- expected cell width;
- whether the sequence is safe for reconstructed console output.

The same corpus should drive complete-string stripping, streaming normalization, transcript handling, Rich rendering, width calculation, and input validation.

### Sequence coverage

Include:

- basic, 256-color, and truecolor SGR;
- semicolon and colon SGR syntax;
- CSI cursor and erase commands;
- private CSI parameters;
- CSI intermediates;
- OSC 0 and OSC 2;
- OSC 8 open and close sequences;
- BEL- and ST-terminated OSC;
- DCS, SOS, PM, and APC;
- two-byte ESC sequences, including legacy save/restore forms;
- optional 8-bit C1 forms;
- C0 controls, DEL, and selected Unicode formatting controls;
- malformed and unterminated sequences.

### Chunk coverage

For every supported sequence, test:

- one complete chunk;
- a split before the introducer;
- a split after ESC;
- a split in parameters or payload;
- a split between ESC and `\\` in ST;
- every-byte split permutations;
- consecutive sequences;
- EOF while pending;
- CR, LF, and CRLF split permutations;
- stdout/stderr interleaving with independent pending states.

A central invariant should hold where policies are equivalent:

```text
normalize(one complete string) == normalize(the same string split at every boundary)
```

### Unicode and width coverage

Include:

- ASCII;
- combining marks;
- East Asian wide and full-width characters;
- ZWJ emoji;
- variation selectors;
- flags and regional indicators;
- skin-tone modifiers;
- tabs at different starting columns;
- controls embedded between visible characters;
- OSC 8 targets whose URI length must not affect width.

### Security cases

Include:

- OSC 52 clipboard commands;
- OSC title commands;
- BEL, ESC, and ST embedded in hyperlink targets;
- malformed OSC 8 close sequences;
- adjacent and nested hyperlink controls;
- oversized unterminated OSC and DCS payloads;
- terminal controls embedded in user input and source-derived display content.

## Migration plan

### Phase 0: Characterize current behavior

Before replacing implementations:

1. Add direct tests for `_strip_terminal_escape_sequences()`.
2. Add dedicated tests for `text_util.strip_ansi_codes()`.
3. Add width tests containing SGR, OSC 8, private CSI, and Unicode.
4. Add EOF and every-byte stream-split tests.
5. Add stdout/stderr interleaving tests.
6. Add OSC 8 injection tests.

These tests should distinguish behavior that must be preserved from defects that should be fixed.

### Phase 1: Immediate correctness fixes

1. Fix the `ask.py` alternation ordering so OSC and DCS-family strings are recognized before generic ESC sequences.
2. Precompile any retained transitional regular expressions at module scope.
3. Validate or reject control characters in OSC 8 targets.
4. Document the temporary limitations until the tokenizer lands.

This phase should be small and independently reviewable.

### Phase 2: Introduce shared recognition

1. Add `system_support/terminal_sequences.py`.
2. Implement complete and incremental tokenization with bounded buffers.
3. Add the shared test corpus.
4. Implement `plain_terminal_text()` on the tokenizer.
5. Keep `text_util.strip_ansi_codes()` as a deprecated compatibility wrapper.

No UI behavior needs to change in this phase.

### Phase 3: Migrate complete-string consumers

Migrate:

- breadcrumbs and labels;
- memory/status strings;
- menu and segment formatting helpers;
- tests currently using `strip_ansi_codes()` as an assertion helper.

Replace formatting-oriented `len(strip_ansi_codes(...))` with the shared cell-width API.

### Phase 4: Migrate worker streaming

1. Replace `_split_pending_control()` and the assembler's local sequence regexes with per-stream tokenizer instances.
2. Preserve current committed/live-line behavior through an explicit UI policy.
3. Feed transcripts from the same parsed events using a transcript policy.
4. Define and test EOF behavior.
5. Keep stdout and stderr parser states separate.
6. Remove transcript imports of private worker UI helpers.

### Phase 5: Normalize console output

1. Stop writing raw `ConsoleOutput.text` directly to terminal streams.
2. Reconstruct safe terminal output from normalized events or emit plain output.
3. Allow only supported SGR, validated OSC 8, and documented line controls.
4. Verify parity between console and Textual paths.

### Phase 6: Unify width calculation

1. Replace `Ui.ANSI_RE` with shared normalization.
2. Replace `Ui.display_width()` with Rich cell-width facilities where compatible.
3. Reuse the width helper in the Windows advanced editor.
4. Replace remaining formatting-oriented `len()` calculations.
5. Verify conversation cursor-clearing invariants.

### Phase 7: Consolidate emission and developer scripts

1. Move reusable literal controls into `Ansi` or named builders.
2. Replace simplified `ANSI_RE` copies under `testx/` where package imports are appropriate.
3. Retain standalone definitions in `launch.py` unless its deployment model changes.
4. Remove deprecated wrappers after all application call sites migrate.

## Proposed initial decisions

Unless implementation work discovers a compatibility constraint, use these defaults:

1. **Tokenizer:** internal state machine, not a third-party terminal emulator.
2. **Malformed sequences:** fail closed; do not emit payload as ordinary text.
3. **Buffer limit:** bounded, with discarded oversize controls optionally recorded for diagnostics.
4. **C1 support:** recognize and remove C1 terminal controls even if the app does not emit them.
5. **OSC policy:** allow OSC 8 only where hyperlink support is explicitly requested; drop all other OSC.
6. **DCS-family policy:** drop completely in all current sinks.
7. **Console worker output:** allowlist safe controls rather than pass raw chunks through.
8. **Transcript CR policy:** preserve each progress update as a separate line.
9. **UI CR policy:** preserve current full-line replacement initially; consider cursor-column emulation separately.
10. **Width:** use Rich cell widths after shared terminal normalization.
11. **Stream state:** independent per operation and stdout/stderr stream.
12. **Keyboard escape decoding:** remains a separate subsystem.

## Open questions

1. Should typed C1 and bidirectional format controls be rejected, stripped, or retained in ordinary user input?
2. Should OSC 8 be permitted in console-mode captured worker output, or only in app-generated output?
3. Which URI schemes should the OSC 8 builder allow?
4. Should transcript output preserve TAB or expand it using a configured tab size?
5. Should unsupported cursor operations be silently dropped or replaced with a visible diagnostic marker in developer mode?
6. Is current full-line CR replacement sufficient, or do important model libraries rely on suffix-preserving terminal overwrite semantics?
7. Should stdout and stderr be visually distinguished after independent normalization?
8. Should source/book ingestion reject all `Cc` characters except approved whitespace before TTS normalization?
9. Is a diagnostic count of discarded or malformed controls useful in worker logs?

## Acceptance criteria

The refactor is complete when:

1. No application code defines an independent terminal-stripping regex.
2. Complete-string and incremental recognition use the same shared implementation.
3. stdout and stderr maintain independent parser state.
4. Textual display, transcripts, and console output consume normalized events through named policies.
5. Raw worker chunks are never written directly to the terminal.
6. OSC 8 targets cannot inject BEL, ESC, ST, or additional controls.
7. Width calculations use one shared visible-text/cell-width path.
8. Existing documented UI and transcript behavior remains covered by tests.
9. Every supported sequence passes split-at-every-byte tests.
10. Malformed, incomplete, and oversized controls have explicit tested behavior.
11. `strip_ansi_codes()` and private `_split_pending_control()` are removed or retained only as deprecated wrappers with no application call sites.
12. The distinction between keyboard protocols, terminal controls, Unicode normalization, and display styling is documented in code-level APIs.
