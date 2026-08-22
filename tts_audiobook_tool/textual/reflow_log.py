"""A log widget with dynamic word wrapping that reflows on resize.

`RichLog` stores its log as already-rendered `Strip` objects, so in its
`wrap=True` mode wrap positions are frozen at the width the line was
written at, and in its `wrap=False` mode lines are never wrapped at all.

`ReflowLog` stores the logical (unwrapped) lines instead, and
precomputes per line, at write time, the width-independent word-chunk
data that rich's `divide_line` wrap algorithm consumes. Re-deriving the
wrap positions at any width is then a pure integer walk over those
arrays (`_fast_breaks`), a verbatim replica of the decision structure of
rich's greedy walk, so the result is bit-for-bit identical to rich's
`Text.wrap` (verified against the full `Text.__rich_console__` pipeline
by `testx/reflow_log_diff.py`). A resize only recomputes per-line row
counts over precomputed data (and never re-renders stored content), so
its cost is a few milliseconds for a 50,000-line log instead of the
seconds a full re-wrap would take, and rows stay pixel-identical to
`RichLog(wrap=True)` output at every width.

The log is capped at `max_lines` *logical* lines (input lines, not
wrapped rows), so long lines can no longer consume history depth.
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from typing import TYPE_CHECKING

from rich._loop import loop_last
from rich._wrap import re_word
from rich.cells import cell_len, chop_cells
from rich.text import Text

from textual.cache import LRUCache
from textual.geometry import Size
from textual.reactive import var
from textual.scroll_view import ScrollView
from textual.strip import Strip

if TYPE_CHECKING:
    from typing_extensions import Self


# `re_word` and its helpers live in rich's private `_wrap` module: this is
# the exact object the `Text.wrap` pipeline uses, and the identity of our
# reflow with rich's output depends on matching it object-for-object.
_WORD_RE = re_word


class _Line:
    """One logical (unwrapped) line plus its width-independent wrap data.

    The four arrays describe the word chunks of the line (in the
    ``\\s*\\S+\\s*`` sense used by rich's ``divide_line``): ``starts`` and
    ``ends`` are character offsets into ``text.plain``, while
    ``full_cells`` and ``rstrip_cells`` hold the chunk's cell width with
    and without its trailing whitespace.
    """

    __slots__ = (
        "text",
        "starts",
        "ends",
        "full_cells",
        "rstrip_cells",
        "breaks_by_width",
    )

    def __init__(self, text: Text) -> None:
        self.text = text
        starts: array = array("I")
        ends: array = array("I")
        full_cells: array = array("I")
        rstrip_cells: array = array("I")
        plain = text.plain
        for match in _WORD_RE.finditer(plain):
            word = match.group(0)
            starts.append(match.start())
            ends.append(match.end())
            full_cells.append(cell_len(word))
            rstrip_cells.append(cell_len(word.rstrip()))
        self.starts = starts
        self.ends = ends
        self.full_cells = full_cells
        self.rstrip_cells = rstrip_cells
        self.breaks_by_width: dict[int, tuple[int, ...]] = {}

    def breaks(self, width: int) -> tuple[int, ...]:
        """The character offsets this line wraps at at ``width`` cells."""
        breaks = self.breaks_by_width.get(width)
        if breaks is None:
            breaks = _fast_breaks(self.text.plain, self, width)
            if len(self.breaks_by_width) >= 8:
                # Only a few widths are live at any time (the current
                # layout width plus whatever resizes are in flight); keep
                # the per-line cache tiny.
                self.breaks_by_width.clear()
            self.breaks_by_width[width] = breaks
        return breaks

    def row_count(self, width: int) -> int:
        """Number of wrapped rows this line occupies at ``width`` cells."""
        return len(self.breaks(width)) + 1

    def row_text(self, width: int, row: int) -> Text:
        """The ``row``-th wrapped row of this line at ``width`` cells.

        This is exactly what rich's ``Text.wrap`` materializes for the
        row: a ``Text.divide`` over the row's offsets. (The
        ``rstrip_end``/``justify``/``truncate`` steps of rich's post-wrap
        pipeline are all no-ops for a default ``Text``.)
        """
        text_length = len(self.text.plain)
        bounds = (0, *self.breaks(width), text_length)
        return self.text.divide((bounds[row], bounds[row + 1]))[1]


def _fast_breaks(plain: str, line: _Line, width: int) -> tuple[int, ...]:
    """Break offsets for ``plain`` at ``width`` cells, identical to rich's
    ``divide_line``.

    Same decision structure as ``rich._wrap.divide_line`` (fold mode, the
    one ``Text.wrap`` uses for a default ``Text``), with the regex
    matches and per-chunk cell lengths precomputed at write time.
    """
    starts = line.starts
    ends = line.ends
    full_cells = line.full_cells
    rstrip_cells = line.rstrip_cells
    breaks: list[int] = []
    append = breaks.append
    cell_offset = 0
    for i in range(len(starts)):
        start = starts[i]
        word_length = rstrip_cells[i]
        if width - cell_offset >= word_length:
            cell_offset += full_cells[i]
        elif word_length > width:
            word = plain[start:ends[i]]
            for last, fragment in loop_last(chop_cells(word, width=width)):
                if start:
                    append(start)
                if last:
                    cell_offset = cell_len(fragment)
                else:
                    start += len(fragment)
        elif cell_offset and start:
            append(start)
            cell_offset = full_cells[i]
    return tuple(breaks)


class ReflowLog(ScrollView, can_focus=True):
    """A scrollable text log with dynamic word wrapping.

    Content is stored as logical lines and rewrapped whenever the
    available width changes, so the log always fills (and never
    overflows) its content region.

    Args:
        max_lines: Maximum number of logical lines in the log, or
            `None` for no maximum.
        markup: Apply Rich console markup to strings written to the log.
        auto_scroll: Scroll to the end after each write, unless the
            write overrides it via `scroll_end`.
        name: The name of the log.
        id: The ID of the log in the DOM.
        classes: The CSS classes of the log.
        disabled: Whether the log is disabled or not.
    """

    DEFAULT_CSS = """
    ReflowLog {
        background: $surface;
        color: $foreground;
        overflow-y: scroll;
        overflow-x: hidden;
        &:focus {
            background-tint: $foreground 5%;
        }
    }
    """

    max_lines: var[int | None] = var(None)
    """Maximum number of logical lines in the log, or `None` for no maximum."""
    auto_scroll: var[bool] = var(True)
    """Scroll to the end after each write."""
    markup: var[bool] = var(False)
    """Apply Rich console markup to strings written to the log."""

    def __init__(
        self,
        *,
        max_lines: int | None = None,
        markup: bool = False,
        auto_scroll: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.max_lines = max_lines
        self.markup = markup
        self.auto_scroll = auto_scroll
        self._lines: list[_Line] = []
        self._row_count: array = array("I")
        """Row count of each stored line at `_layout_width`."""
        self._prefix: array = array("I", [0])
        """Prefix sums of `_row_count`: the row at which each stored line
        begins, with the first entry 0. Trimming lines from the front
        never shifts the row positions of the survivors (wrapping is
        per-line), so scroll coordinates are stable across trims without
        any offset bookkeeping."""
        self._layout_width = 0
        """The content-region width the current layout was computed at."""
        self._size_known = False
        self._generation = 0
        """Incremented whenever stored lines change; part of the strip
        cache key so trimmed/reused line objects can never alias."""
        self._strip_cache: LRUCache[tuple[int, int, int, int], Strip] = LRUCache(1024)

    # -- storage ---------------------------------------------------------

    def write(
        self,
        content: str | Text,
        *,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> Self:
        """Write a string or `Text` to the end of the log.

        Content is stored as logical lines (tabs expanded, ANSI-style
        spans preserved) and wrapped at the current content-region
        width; it is rewrapped transparently whenever the width changes.

        Args:
            content: The text to append. Strings are treated as literal
                text unless `markup` is enabled.
            scroll_end: Scroll to the end after writing, or `None` to
                use `auto_scroll`.
            animate: Animate the scroll to the end.

        Returns:
            The `ReflowLog` instance.
        """
        if isinstance(content, str):
            text = Text.from_markup(content) if self.markup else Text(content)
        elif isinstance(content, Text):
            text = content
        else:
            raise TypeError(
                "ReflowLog.write accepts str or Text content, "
                f"not {type(content).__name__}"
            )

        # Split into logical lines exactly the way rich's wrap pipeline
        # does (same `Text.split` call, so the per-line Text objects are
        # bit-for-bit identical to what `Text.wrap` would consume), and
        # expand tabs per line, as `Text.wrap` does.
        tab_size = (
            self.app.console.tab_size if text.tab_size is None else text.tab_size
        )
        new_lines: list[_Line] = []
        for line_text in text.split("\n", allow_blank=True):
            if "\t" in line_text.plain:
                line_text.expand_tabs(tab_size)
            new_lines.append(_Line(line_text))

        lines = self._lines
        pre_len = len(lines)
        lines.extend(new_lines)

        # How many of the just-written lines the cap removed (the trim
        # takes from the front, so it drops old lines first and only then
        # the earliest of the new ones).
        trimmed_new = self._trim_overflow(pre_len)

        if self._size_known:
            width = max(self.scrollable_content_region.width, 1)
            if width != self._layout_width:
                # The region changed without a reflow (e.g. a scrollbar
                # appeared/disappeared); re-derive the whole layout, which
                # includes the new lines.
                self._reflow(width)
            else:
                # Only the surviving new lines need row counts, at the
                # live width.
                total = self._prefix[-1]
                for line in new_lines[trimmed_new:]:
                    count = line.row_count(width)
                    self._row_count.append(count)
                    total += count
                    self._prefix.append(total)
                self.virtual_size = Size(self._layout_width, total)

        scroll = self.auto_scroll if scroll_end is None else scroll_end
        if scroll:
            self.scroll_end(animate=animate, immediate=False, x_axis=False)
        return self

    def clear(self) -> Self:
        """Clear the text log.

        Returns:
            The `ReflowLog` instance.
        """
        self._lines.clear()
        self._row_count = array("I")
        self._prefix = array("I", [0])
        self._invalidate_strips()
        if self._size_known:
            self._layout_width = max(self.scrollable_content_region.width, 1)
            self.virtual_size = Size(self._layout_width, 0)
        else:
            self._layout_width = 0
            self.virtual_size = Size(0, 0)
        self.refresh()
        return self

    def _invalidate_strips(self) -> None:
        """Invalidate cached strips because stored line objects changed.

        Bumping the generation keeps a recycled address from aliasing a
        cached strip built for the previous occupant of that address.
        """
        self._generation += 1
        self._strip_cache.clear()

    def _trim_overflow(self, pre_len: int) -> int:
        """Drop stored lines past `max_lines`, oldest first.

        `pre_len` is the number of stored lines before the most recent
        append; the return is how many of the appended lines the cap
        removed (0 when there was no overflow). Surviving lines keep
        their row positions (wrapping is per-line), so scroll
        coordinates stay stable and the prefix sums only need
        rebuilding over the survivors.
        """
        lines = self._lines
        if self.max_lines is None or len(lines) <= self.max_lines:
            return 0
        overflow = len(lines) - self.max_lines
        del lines[:overflow]
        # Trim the matching layout bookkeeping and rebuild prefix sums.
        if self._row_count:
            del self._row_count[:overflow]
            self._prefix = array("I", [0])
            total = 0
            for count in self._row_count:
                total += count
                self._prefix.append(total)
        self._invalidate_strips()
        return max(0, overflow - pre_len)

    def _sync_layout_tail(self, drop_tail_counts: int = 0) -> None:
        """Resync layout bookkeeping after a tail mutation of `_lines`.

        The caller must have changed only the tail of `_lines`
        (replaced/appended lines there) and already run `_trim_overflow`
        if lines were dropped from the front. When `drop_tail_counts` is
        nonzero, that many stored lines were removed from the *end* of
        the list; their row counts are the last entries of the
        bookkeeping, because a front trim can only remove leading
        entries.

        While the size is unknown the arrays stay empty and the first
        `_reflow` rebuilds them from `_lines`. If the content-region
        width changed since the last layout, a full `_reflow` is used
        instead (it also restores the viewport). Otherwise only the
        appended lines get row counts; leading row positions are left
        untouched, so scrolling stays stable.
        """
        if not self._size_known:
            return
        width = max(self.scrollable_content_region.width, 1)
        if width != self._layout_width:
            self._reflow(width)
            return
        if drop_tail_counts:
            if len(self._row_count) >= drop_tail_counts:
                del self._row_count[-drop_tail_counts:]
            else:
                # Defensive: the bookkeeping is shorter than the dropped
                # count (should not happen); realign from the start.
                self._row_count = array("I")
            self._prefix = self._prefix[: len(self._row_count) + 1]
        lines = self._lines
        if len(self._row_count) > len(lines):
            # Defensive: the stored list shrank below the bookkeeping;
            # realign so the extension below cannot double-count rows.
            del self._row_count[len(lines):]
            self._prefix = self._prefix[: len(self._row_count) + 1]
        total = self._prefix[-1]
        row_append = self._row_count.append
        prefix_append = self._prefix.append
        for line in lines[len(self._row_count):]:
            count = line.row_count(width)
            row_append(count)
            total += count
            prefix_append(total)
        self.virtual_size = Size(self._layout_width, total)

    # -- layout ----------------------------------------------------------

    def _reflow(self, width: int) -> None:
        """Re-derive the wrapped layout at ``width`` cells.

        Recomputes per-line row counts (a fast walk over precomputed
        chunk data, with per-line caching) and the prefix sums, updates
        the virtual size, and restores the viewport: content stays
        pinned to the end if it was at the end, otherwise the logical
        line that was at the top of the viewport stays at the top.
        """
        width = max(int(width), 1)
        if width == self._layout_width:
            return

        lines = self._lines
        anchor: tuple[int, int] | None = None
        follow_end = False
        if self._size_known and lines:
            old_top = int(self.scroll_offset.y)
            old_total = self._prefix[-1]
            if old_top >= self.max_scroll_y and old_total > 0:
                follow_end = True
            elif 0 <= old_top < old_total:
                line_index = bisect_right(self._prefix, old_top) - 1
                if line_index < len(lines):
                    anchor = (line_index, old_top - self._prefix[line_index])

        self._row_count = array("I")
        self._prefix = array("I", [0])
        total = 0
        prefix_append = self._prefix.append
        row_append = self._row_count.append
        for line in lines:
            count = line.row_count(width)
            row_append(count)
            total += count
            prefix_append(total)
        self._layout_width = width
        self.virtual_size = Size(width, total)

        if self._size_known and lines:
            if follow_end:
                self.scroll_end(animate=False, immediate=True, x_axis=False)
            elif anchor is not None:
                self.scroll_to(
                    y=self._prefix[anchor[0]] + anchor[1],
                    animate=False,
                )
        self.refresh()

    def _size_updated(
        self, size: Size, virtual_size: Size, container_size: Size, layout: bool = True
    ) -> bool:
        changed = super()._size_updated(size, virtual_size, container_size, layout)
        if size.width and not self._size_known:
            # The size is known for the first time; compute an initial
            # layout from whatever has been written so far.
            self._size_known = True
        if self._size_known:
            # The content-region width can change without a resize event
            # (e.g. the vertical scrollbar appearing), so reflow here
            # rather than in a Resize handler.
            width = max(self.scrollable_content_region.width, 1)
            if width != self._layout_width:
                self._reflow(width)
        return changed

    def get_content_width(self, container: Size, viewport: Size) -> int:
        if self._size_known:
            return self.virtual_size.width
        return container.width

    def notify_style_update(self) -> None:
        super().notify_style_update()
        # Materialized rows embed styles resolved through the console,
        # which change with the theme.
        self._strip_cache.clear()

    # -- rendering -------------------------------------------------------

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        width = max(self.scrollable_content_region.width, 1)
        prefix = self._prefix
        total = prefix[-1]
        absolute = scroll_y + y
        line_index = -1
        strip: Strip | None = None
        if 0 < total and 0 <= absolute < total:
            line_index = bisect_right(prefix, absolute) - 1
            row = absolute - prefix[line_index]
            line = self._lines[line_index]
            key = (self._generation, id(line), width, row)
            strip = self._strip_cache.get(key)
            if strip is None:
                strip = Strip(line.row_text(width, row).render(self.app.console))
                self._strip_cache[key] = strip
        if strip is None:
            strip = Strip.blank(width)
        strip = strip.crop_extend(scroll_x, scroll_x + width, self.rich_style)
        return self._style_row(strip, line_index).apply_style(self.rich_style)

    def _style_row(self, strip: Strip, line_index: int) -> Strip:
        """Style one rendered row; overridden to highlight find matches."""
        return strip
