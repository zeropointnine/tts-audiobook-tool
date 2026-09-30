# Browser Player Highlight Color Taxonomy

This document describes the highlight/background states used in the browser
player's book text view, the CSS variables that implement them, and the
dimensions that affect their concrete colors.

The relevant CSS lives in `browser_player/base.css` (variable definitions)
and `browser_player/main.css` (rules that apply the variables).

## Overview

Highlight colors answer the question: *what is the background tint behind a
piece of book text, and why?* Two kinds of elements get tinted:

- **Segments** — the atomic `span.textSegment` leaves of the book text.
- **Segment groups** — the generated-phrase wrappers (`span.segmentGroup`)
  that cluster several sub-segments into a sub-list.

A given element is in exactly one highlight state at a time. The state
depends on the element's *role* (segment vs. group), its *status* (currently
active vs. being hovered), and whether it *bears audio*.

## State table

| State | Applies to | Audio? | Trigger | CSS variable |
|---|---|---|---|---|
| **Active** | Segment | has audio | The segment currently being played | `--color-segment-highlight` |
| **Active (paused)** | Segment | has audio | The active segment while the player is paused | `--color-segment-hover` |
| **Hover** | Segment | has audio | Pointer hovering a playable segment that is not active | `--color-segment-hover` |
| **Hover, no audio** | Segment | no audio | Pointer hovering a segment with no associated audio | `--color-segment-noaudio-hover` |
| **Group active** | Segment group | group has audio | The group containing the currently active segment | `--color-segment-group-current` |
| **Group hover** | Segment group | group has audio | Pointer hovering a group whose segments bear audio | `--color-segment-group-hover` |
| **Group hover, no audio** | Segment group | no audio | Pointer hovering a group with no associated audio | `--color-segment-group-noaudio-hover` |

Notes:

- "Active" and "hover" are mutually exclusive in appearance: the active
  segment keeps its active color even while hovered.
- A group's audio status is *any child bears audio* — computed once in JS
  (`book-text.js`) when the group span is built and expressed as a
  `hasAudio` / `noAudio` class on the group span.
- The active (playing) segment always sits inside its active group, so the
  two "current" tints stack: the group's tint shows around the gaps between
  child segments (separators), while the segment's tint covers the segment
  text itself.

## Color derivation

All audio-bearing highlight colors derive from a single per-theme
**reference color** (`--color-segment-highlight`). The other variables are
produced from it with `color-mix()`, holding the hue constant and varying
only opacity. This guarantees every audio-bearing state is visibly the same
*color* and differs only in strength.

| Variable | Derivation (relative to reference color) | Relative strength |
|---|---|---|
| `--color-segment-highlight` | Reference (defined directly per theme) | Strongest |
| `--color-segment-hover` | Reference at moderate opacity | Medium |
| `--color-segment-group-current` | Reference at light opacity | Light |
| `--color-segment-group-hover` | Reference at light opacity | Light |

The no-audio hover states do **not** use the reference color. They use the
theme's dimmest text tone (`--color-text-dimmer`) as their base, signaling
"inert text" rather than "playable text":

| Variable | Derivation | Character |
|---|---|---|
| `--color-segment-noaudio-hover` | Dim tone at moderate opacity | Neutral, cool |
| `--color-segment-group-noaudio-hover` | Dim tone at light opacity | Neutral, faint |

## Dimensions

Concrete colors depend on two orthogonal settings, both applied as
attributes on the root element:

### Colorway (`data-segment-colors`)

Selects the *hue* of the reference color (e.g. the default orange, or blue,
or red). Colorway only redefines the reference color; every derived
variable automatically follows because they are `color-mix()` formulas
over it.

### Light / dark theme (`data-theme`)

Selects light or dark rendering. The theme redefines the reference color
(and the text tones used by the no-audio hovers) with values suited to the
background; the derived variables follow automatically.

| Dimension | What it changes | Where overridden |
|---|---|---|
| Colorway | Reference color hue | `data-segment-colors` blocks, per theme |
| Theme | Reference color value + text tone bases | `data-theme` blocks |

In short: **colorway and theme only define base colors; every highlight
state is a formula over those bases.** Adding a new colorway requires
exactly one new reference-color definition per theme, nothing else.

## Behavioral rules in CSS

| Rule (abridged) | Effect |
|---|---|
| `span.highlight` | Active segment gets the reference color |
| `[data-player-status="pause"] span.highlight` | Active segment drops to the hover color while paused |
| `.segmentGroup.groupHighlight` | Active group gets the group-current color |
| `.segmentGroup.hasAudio:hover` | Group hover color (pointer devices only) |
| `.segmentGroup.noAudio:hover` | Group no-audio hover color |
| `.textSegment.hasAudio:hover` | Segment hover color |
| `.textSegment.noAudio:hover` | Segment no-audio hover color |
| `.segmentGroup > .textSegment.highlight:hover` | Active segment keeps the reference color even when hovered |

Hover rules are wrapped in `@media (pointer: fine)` so touch devices never
produce a "stuck hover" appearance.

## Interaction diagram

```
                    audio-bearing? ── no ──> dim-tone hover (segment/group)
                           │ yes
                           ▼
             group: group-current tint surrounds the
             active segment's stronger tint
                           │
              segment active? ── yes ──> reference color
              (playing)          paused ─> hover color
                           │ no
                           ▼
                  hovered? ── yes ──> hover color
                           └─ no ──> no tint
```
