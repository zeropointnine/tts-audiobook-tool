from __future__ import annotations

from dataclasses import dataclass
import unicodedata


DOUBLE_QUOTE_STYLES = frozenset({"double"})
SENTENCE_QUOTE_STYLES = frozenset({
    "double",
    "single",
    "guillemet",
    "single_guillemet",
    "corner",
    "white_corner",
})


@dataclass(frozen=True)
class QuoteSpan:
    """A balanced quotation range, using an exclusive end offset."""

    start: int
    end: int
    depth: int


@dataclass(frozen=True)
class _QuoteStyle:
    opening: frozenset[str]
    closing: frozenset[str]
    symmetric: frozenset[str] = frozenset()


_QUOTE_STYLES = {
    # Straight and curly glyphs are deliberately compatible within each family.
    # Imperfectly normalized ebooks commonly mix them within one quote pair.
    "double": _QuoteStyle(frozenset({'"', "“"}), frozenset({'"', "”"}), frozenset({'"'})),
    "single": _QuoteStyle(frozenset({"'", "‘"}), frozenset({"'", "’"}), frozenset({"'"})),
    "guillemet": _QuoteStyle(frozenset({"«"}), frozenset({"»"})),
    "single_guillemet": _QuoteStyle(frozenset({"‹"}), frozenset({"›"})),
    "corner": _QuoteStyle(frozenset({"「"}), frozenset({"」"})),
    "white_corner": _QuoteStyle(frozenset({"『"}), frozenset({"』"})),
}

_OPENING_CONTEXT_PUNCTUATION = frozenset("([{<:;,—–-")


def find_quote_spans(
    text: str,
    *,
    styles: frozenset[str] = SENTENCE_QUOTE_STYLES,
    paragraph_scoped: bool = False,
) -> list[QuoteSpan]:
    """Find balanced quotation spans without interpreting them as dialog.

    Ambiguous straight quote glyphs are classified from their surrounding text.
    Directional glyphs have fixed roles. Malformed or unmatched delimiters are
    ignored, and pairing can optionally be reset at every physical line.
    """

    enabled = tuple(
        (name, _QUOTE_STYLES[name])
        for name in styles
        if name in _QUOTE_STYLES
    )
    if not enabled or not text:
        return []

    spans: list[QuoteSpan] = []
    # Each item is (style name, source index, nesting depth).
    stack: list[tuple[str, int, int]] = []

    for index, char in enumerate(text):
        if paragraph_scoped and char in "\r\n":
            stack.clear()
            continue
        if _is_escaped(text, index):
            continue

        candidates = [
            (name, style)
            for name, style in enabled
            if char in style.opening or char in style.closing
        ]
        if not candidates:
            continue

        # Apostrophes inside words are never straight single-quote delimiters.
        if char == "'" and _is_word_apostrophe(text, index):
            continue

        name, style = candidates[0]
        is_symmetric = char in style.symmetric
        can_close = char in style.closing and (
            not is_symmetric or _looks_like_closing_quote(text, index)
        )

        matching_stack_index = _last_opening_for_style(stack, name)
        if can_close and matching_stack_index is not None:
            _, opening_index, depth = stack.pop(matching_stack_index)
            spans.append(QuoteSpan(opening_index, index + 1, depth))
            continue

        can_open = char in style.opening and (
            not is_symmetric or _looks_like_opening_quote(text, index)
        )
        if can_open:
            stack.append((name, index, len(stack)))

    return sorted(spans, key=lambda span: (span.start, -span.end))


def _last_opening_for_style(
    stack: list[tuple[str, int, int]],
    style_name: str,
) -> int | None:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index][0] == style_name:
            return index
    return None


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slash_count += 1
        index -= 1
    return slash_count % 2 == 1


def _is_word_apostrophe(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isalnum()
        and text[index + 1].isalnum()
    )


def _looks_like_opening_quote(text: str, index: int) -> bool:
    next_index = _next_non_whitespace_index(text, index + 1)
    if next_index is None:
        return False
    if index == 0 or text[index - 1].isspace():
        return True
    return text[index - 1] in _OPENING_CONTEXT_PUNCTUATION


def _looks_like_closing_quote(text: str, index: int) -> bool:
    previous_index = _previous_non_whitespace_index(text, index - 1)
    if previous_index is None:
        return False
    if index == len(text) - 1 or text[index + 1].isspace():
        return True
    return unicodedata.category(text[index + 1]).startswith("P")


def _next_non_whitespace_index(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        if not text[index].isspace():
            return index
    return None


def _previous_non_whitespace_index(text: str, start: int) -> int | None:
    for index in range(start, -1, -1):
        if not text[index].isspace():
            return index
    return None
