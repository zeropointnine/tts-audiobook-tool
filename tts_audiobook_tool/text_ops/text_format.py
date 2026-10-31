"""
Display-formatting markup ("fmt") carried alongside phrase text.

The Python app treats a phrase's `fmt` as opaque: it is produced at EPUB import
(or by re-applying formatting from the EPUB), joined upward when phrases are
flattened, and written into ABR metadata for the browser player. Nothing in
generation, validation or prompting reads it.

Rules (see docs-dev/text-formatting-layer.md):
- `fmt` is the phrase's own text plus a few tags: <h> <b> <i> <u>.
  Literal `&`, `<` and `>` are escaped as `&amp;`, `&lt;` and `&gt;`.
- Markup is self-contained per phrase: tags are properly nested and every tag
  opened in a phrase is closed in that phrase.
- `fmt` is valid only when its stripped text equals the phrase text exactly.
  Invalid or stale markup is ignored, never an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BOLD = 1
ITALIC = 2
UNDERLINE = 4
HEADING = 8
ALL_STYLES = BOLD | ITALIC | UNDERLINE | HEADING

# Canonical tag order, outermost first
_TAGS: tuple[tuple[str, int], ...] = (
    ("h", HEADING),
    ("b", BOLD),
    ("i", ITALIC),
    ("u", UNDERLINE),
)
_STYLE_BY_TAG = {tag: style for tag, style in _TAGS}
_ENTITIES = {"amp": "&", "lt": "<", "gt": ">"}
_TOKEN_RE = re.compile(r"<(/?)([a-z]*)>|&(amp|lt|gt);|[<>&]")
_WORD_RE = re.compile(r"\S+")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse(fmt: str) -> tuple[str, list[int]] | None:
    """
    Returns the plain text of `fmt` and a parallel per-character list of style
    masks, or None when the markup is malformed (unknown tag, bad nesting,
    unclosed tag, unescaped `&`, `<` or `>`).
    """
    parts: list[str] = []
    masks: list[int] = []
    stack: list[int] = []
    current = 0
    pos = 0
    for match in _TOKEN_RE.finditer(fmt):
        literal = fmt[pos:match.start()]
        parts.append(literal)
        masks.extend([current] * len(literal))
        pos = match.end()

        tag = match.group(2)
        entity = match.group(3)
        if tag is not None:
            style = _STYLE_BY_TAG.get(tag)
            if style is None:
                return None
            if match.group(1):
                if not stack or stack[-1] != style:
                    return None
                stack.pop()
                current &= ~style
            else:
                if current & style:
                    return None
                stack.append(style)
                current |= style
        elif entity is not None:
            parts.append(_ENTITIES[entity])
            masks.append(current)
        else:
            return None

    literal = fmt[pos:]
    parts.append(literal)
    masks.extend([current] * len(literal))
    if stack:
        return None
    return "".join(parts), masks


def masks_for(fmt: str | None, text: str) -> list[int] | None:
    """Per-character style masks of `fmt`, when it is valid for `text`."""
    if fmt is None:
        return None
    parsed = parse(fmt)
    if parsed is None or parsed[0] != text:
        return None
    return parsed[1]


def is_valid_for(fmt: str | None, text: str) -> bool:
    return masks_for(fmt, text) is not None


def render(text: str, masks: list[int]) -> str | None:
    """
    Serializes text plus per-character style masks into canonical markup.
    Returns None when no character is styled.

    Only visible (non-whitespace) characters carry style. A whitespace run takes
    the styles shared by the visible characters on both sides of it within
    `text`; leading and trailing whitespace is never styled.
    """
    if len(masks) != len(text):
        raise ValueError("masks must be parallel to text")

    effective = _effective_masks(text, masks)
    if not any(effective):
        return None

    out: list[str] = []
    stack: list[tuple[str, int]] = []  # open tags, outermost first
    run_start = 0
    for index in range(1, len(text) + 1):
        if index < len(text) and effective[index] == effective[run_start]:
            continue
        mask = effective[run_start]
        # Close from the innermost tag down to the outermost one this run drops
        first_dropped = next(
            (position for position, (_, style) in enumerate(stack) if not mask & style),
            len(stack),
        )
        while len(stack) > first_dropped:
            out.append(f"</{stack.pop()[0]}>")
        open_styles = sum(style for _, style in stack)
        for tag, style in _TAGS:
            if mask & style and not open_styles & style:
                out.append(f"<{tag}>")
                stack.append((tag, style))
        out.append(escape(text[run_start:index]))
        run_start = index
    while stack:
        out.append(f"</{stack.pop()[0]}>")
    return "".join(out)


def join(parts: list[tuple[str, str | None]]) -> str | None:
    """
    Markup for the concatenation of several (text, fmt) parts, such as the
    phrases of one phrase group. Parts without valid markup count as unstyled.
    Whitespace between parts is styled by the same rule as `render`.
    Returns None when no part is styled.
    """
    if all(fmt is None for _, fmt in parts):
        return None
    texts: list[str] = []
    masks: list[int] = []
    any_styles = False
    for text, fmt in parts:
        part_masks = masks_for(fmt, text)
        if part_masks is None:
            part_masks = [0] * len(text)
        else:
            any_styles = any_styles or any(part_masks)
        texts.append(text)
        masks.extend(part_masks)
    if not any_styles:
        return None
    return render("".join(texts), masks)


def _effective_masks(text: str, masks: list[int]) -> list[int]:
    result = [0] * len(text)
    previous_visible_mask: int | None = None
    pending_whitespace: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            pending_whitespace.append(index)
            continue
        mask = masks[index] & ALL_STYLES
        if previous_visible_mask is not None:
            shared = previous_visible_mask & mask
            for whitespace_index in pending_whitespace:
                result[whitespace_index] = shared
        pending_whitespace = []
        result[index] = mask
        previous_visible_mask = mask
    return result


# ---


@dataclass
class StyledText:
    """
    Source text with a parallel per-character style mask list, as produced by a
    formatting-aware extractor. Whitespace masks are not meaningful.
    """
    text: str
    masks: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.masks:
            self.masks = [0] * len(self.text)
        if len(self.masks) != len(self.text):
            raise ValueError("masks must be parallel to text")

    @staticmethod
    def plain(text: str) -> StyledText:
        return StyledText(text, [0] * len(text))

    @staticmethod
    def join(parts: list[StyledText], separator: str) -> StyledText:
        texts: list[str] = []
        masks: list[int] = []
        for index, part in enumerate(parts):
            if index > 0:
                texts.append(separator)
                masks.extend([0] * len(separator))
            texts.append(part.text)
            masks.extend(part.masks)
        return StyledText("".join(texts), masks)

    @property
    def has_styles(self) -> bool:
        return any(self.masks)


@dataclass
class MaskAlignment:
    masks: list[int]
    num_words: int
    num_unmatched_words: int

    @property
    def is_exact(self) -> bool:
        return self.num_unmatched_words == 0


def align_masks(source: StyledText, target: str) -> MaskAlignment:
    """
    Projects the styles of `source` onto `target`, a text that should contain the
    same words (whitespace-separated runs of visible characters), possibly with
    different whitespace or with local edits.

    Identical word sequences take the fast path. Otherwise words are aligned with
    difflib, and target words with no identical counterpart stay unstyled.
    """
    target_masks = [0] * len(target)
    target_words = [(m.group(), m.start()) for m in _WORD_RE.finditer(target)]
    if not source.has_styles:
        return MaskAlignment(target_masks, len(target_words), 0)

    source_words = [(m.group(), m.start()) for m in _WORD_RE.finditer(source.text)]

    def copy_word(source_index: int, target_index: int) -> None:
        _, source_start = source_words[source_index]
        word, target_start = target_words[target_index]
        target_masks[target_start:target_start + len(word)] = (
            source.masks[source_start:source_start + len(word)]
        )

    source_strings = [word for word, _ in source_words]
    target_strings = [word for word, _ in target_words]
    if source_strings == target_strings:
        for index in range(len(target_words)):
            copy_word(index, index)
        return MaskAlignment(target_masks, len(target_words), 0)

    import difflib
    matcher = difflib.SequenceMatcher(None, source_strings, target_strings, autojunk=False)
    num_matched = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            copy_word(block.a + offset, block.b + offset)
        num_matched += block.size
    return MaskAlignment(target_masks, len(target_words), len(target_words) - num_matched)
