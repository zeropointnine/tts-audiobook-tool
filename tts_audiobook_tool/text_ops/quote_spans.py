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


def find_paragraph_chained_quote_spans(
        text: str,
        *,
        styles: frozenset[str] = SENTENCE_QUOTE_STYLES,
) -> list[QuoteSpan]:
    """Find multi-paragraph quotation spans following the publishing convention.

    Conventional multi-paragraph dialog opens each paragraph of the speech
    with a new opening quote mark and omits the closing mark until the final
    paragraph. A paragraph whose leading opening quote has no closing quote
    in that paragraph opens (or continues) a chained context; the chain ends
    at the first paragraph that contains a closing quote, and is abandoned
    when a paragraph does not begin with an opening quote mark.

    A speech paragraph may also hold complete quote pairs earlier in the
    paragraph and still end with an unmatched opening quote (the speech
    continuing into the next paragraph). Such a trailing unmatched opener
    opens a chained context as well.
    """

    enabled = tuple(
        (name, _QUOTE_STYLES[name])
        for name in styles
        if name in _QUOTE_STYLES
    )
    if not enabled or not text:
        return []

    spans: list[QuoteSpan] = []
    pending_opening_index: int | None = None
    paragraph_start = 0

    def next_paragraph_start(current: int) -> int:
        index = current
        while index < len(text) and text[index] not in "\r\n":
            index += 1
        while index < len(text) and text[index] in "\r\n":
            index += 1
        return index

    while paragraph_start < len(text):
        paragraph_end = next_paragraph_start(paragraph_start)
        leading_quote = _leading_opening_quote_index(
            text,
            paragraph_start,
            paragraph_end,
            enabled,
        )

        if leading_quote is not None:
            closing_index = _closing_quote_index(
                text,
                leading_quote + 1,
                paragraph_end,
                enabled,
            )
            if closing_index is not None:
                if pending_opening_index is not None:
                    spans.append(QuoteSpan(
                        pending_opening_index,
                        closing_index + 1,
                        0,
                    ))
                # A self-contained pair is the paragraph-scoped pass's job;
                # either way, no chained context remains.
                pending_opening_index = None
            elif pending_opening_index is None:
                pending_opening_index = leading_quote

            if pending_opening_index is None:
                # The leading pair closed, yet the paragraph may end with a
                # further unmatched opening quote whose speech continues in
                # the next paragraph.
                trailing_opening = _first_unmatched_opening(
                    text,
                    paragraph_start,
                    paragraph_end,
                    enabled,
                )
                if trailing_opening is not None:
                    pending_opening_index = trailing_opening

        else:
            # A paragraph that does not begin with an opening quote mark
            # (narration, a new speaker, a heading) ends the speech.
            pending_opening_index = None

        paragraph_start = paragraph_end

    return spans


def _first_unmatched_opening(
        text: str,
        paragraph_start: int,
        paragraph_end: int,
        enabled: tuple[tuple[str, _QuoteStyle], ...],
) -> int | None:
    """Index of the first opening quote mark left unmatched in the paragraph.

    Pairing mirrors find_quote_spans within paragraph bounds; glyphs that
    close in-paragraph pairs are popped, and the earliest surviving opening
    (the outermost unmatched one) is returned.
    """

    stack: list[tuple[str, int]] = []
    for index in range(paragraph_start, paragraph_end):
        if _is_escaped(text, index):
            continue
        char = text[index]
        for name, style in enabled:
            if char not in style.opening and char not in style.closing:
                continue
            is_symmetric = char in style.symmetric
            can_close = char in style.closing and (
                not is_symmetric or _looks_like_closing_quote(text, index)
            )
            if can_close:
                for stack_index in range(len(stack) - 1, -1, -1):
                    if stack[stack_index][0] == name:
                        stack.pop(stack_index)
                        break
                break
            can_open = char in style.opening and (
                not is_symmetric or _looks_like_opening_quote(text, index)
            )
            if can_open:
                stack.append((name, index))
                break
    return stack[0][1] if stack else None


def _leading_opening_quote_index(
        text: str,
        paragraph_start: int,
        paragraph_end: int,
        enabled: tuple[tuple[str, _QuoteStyle], ...],
) -> int | None:
    """Index of an opening quote mark leading the paragraph, if any."""

    index = paragraph_start
    while index < paragraph_end and text[index].isspace():
        index += 1
    if index >= paragraph_end:
        return None

    char = text[index]
    for _, style in enabled:
        if char in style.opening and (
            char not in style.symmetric
            or _looks_like_opening_quote(text, index)
        ):
            return index
    return None


def _closing_quote_index(
        text: str,
        search_start: int,
        paragraph_end: int,
        enabled: tuple[tuple[str, _QuoteStyle], ...],
) -> int | None:
    """Index of the first closable quote mark in the paragraph, if any."""

    for index in range(search_start, paragraph_end):
        if _is_escaped(text, index):
            continue
        char = text[index]
        for _, style in enabled:
            if char in style.closing and (
                char not in style.symmetric
                or _looks_like_closing_quote(text, index)
            ):
                return index
    return None


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
