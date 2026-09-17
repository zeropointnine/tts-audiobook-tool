"""Content area shared by the worker-driven TUI apps (generation and
realtime playback).

Both apps render the same layout — a scrollable, reflowing log of the
worker's console output — so they share `WorkerLog` and
`WorkerLogContentArea` from here instead of keeping per-app copies.

The log's document always ends with the *current line*: in-progress
output (a dynamic progress bar, a transcribing status, ...) lives on
the document's last line and updates in place, exactly as it would in
a conventional console. When the line is committed (the worker emits a
newline) it becomes a normal history line and a fresh empty current
line begins below it, so a commit advances the document by one line in
the normal document flow — the log's area never resizes, and so can
never flicker.
"""

from __future__ import annotations

from bisect import bisect_right

from typing_extensions import Self

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.strip import Strip

from .reflow_log import ReflowLog, _Line
from .textual_shared import STYLE_DIM

# Cap on retained history, in logical (pre-wrap) lines.
#
# A console line can wrap to many rows (multi-megabyte library log
# lines are possible), so a row-based cap would make history depth
# unpredictable; a logical-line cap keeps it deterministic while
# per-line wrapping keeps any single line from consuming unbounded
# memory.
VISIBLE_HISTORY_LINES = 50_000


class _SeparatorLine(_Line):
    """One semantic rule row in the otherwise text-based log document.

    The original finite dash line is retained in ``text`` for logical-line
    bookkeeping and find support, but the visible stroke is materialized at
    the log's live content width.  Consequently the rule follows resizes
    without becoming a child widget of its own.
    """

    __slots__ = ()

    def row_count(self, width: int) -> int:
        return 1

    def row_text(self, width: int, row: int) -> Text:
        # ReflowLog calls Text.render() directly. A Text object's base style is
        # applied by Rich's outer console-rendering pipeline, so it would be
        # lost on this path; an explicit span survives direct rendering.
        stroke = ("- " * ((width + 1) // 2))[:width]
        text = Text(stroke)
        text.stylize(STYLE_DIM)
        return text


class WorkerLog(ReflowLog):
    """Read-only ANSI log with dynamic word wrapping, conventional
    navigation, explicit tail following, and an in-flow current line.

    The document always ends with a current line. In-progress output
    (progress bars, status lines) replaces it in place; committing it
    (a newline) or presenting app lines turns it into history and
    starts a fresh current line below it. Because the current line is
    part of the document, a dynamic line can never change the log's
    area size: it only ever changes the document's last line, and a
    commit grows the document by one line in the normal document flow
    (a one-row scroll when following the tail).

    Lines are word-wrapped at the log's live content-region width and
    rewrapped whenever that width changes, so content always fills (and
    never overflows) the available space. History is capped at
    `VISIBLE_HISTORY_LINES` logical lines, so long lines cannot consume
    the history depth.
    """

    BINDINGS = [
        Binding("up", "history_up", "Up", show=False),
        Binding("down", "history_down", "Down", show=False),
        Binding("pageup", "history_page_up", "PageUp", show=False),
        Binding("pagedown", "history_page_down", "PageDown", show=False),
        Binding("home", "history_home", "Home", show=False),
        Binding("end", "history_end", "End", show=False),
    ]

    def __init__(
        self,
        *,
        output_filters: list[str] | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            max_lines=VISIBLE_HISTORY_LINES,
            markup=False,
            auto_scroll=False,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.follow_tail = True
        self.output_filters = tuple(output_filters or ())
        # The document starts with its (empty) current line, like a
        # freshly cleared terminal.
        self._lines.append(_Line(Text("")))
        # Logical line index of the active find match, or None when find is
        # not showing a result. ``_style_row`` reverse-videos that line.
        self.highlight_line_index: int | None = None
        # Structured progress arrives immediately before a batch heading. It
        # arms conversion of that heading's finite console divider into a
        # semantic rule row; unrelated dashed output remains ordinary text.
        self._separator_expected = False

    # -- separators ---------------------------------------------------------

    def expect_separator(self) -> None:
        """Promote the next printed all-dash console line to a full-width rule.

        Used for batch-heading dividers and for the run-end summary block,
        both of which already print a finite dash rule in the console stream.
        Blank console lines may precede the divider (realtime playback emits
        one deliberately). The rule is omitted when nothing visible has been
        written to the log yet, so a heading can begin the scrolling area
        instead of being separated from nothing.

        The decision is made when the divider line actually arrives, so
        output printed between the structured batch event and the heading
        (for example a warning) still counts as prior output.
        """
        self._separator_expected = True

    def append_separator(self) -> None:
        """Append a full-width rule to the document tail right now.

        Unlike `expect_separator`, this inserts a rule that the console
        stream does not already provide (for example the realtime
        await-continue boundary). The current line is committed first,
        exactly like an app line, and the call is a no-op when the log
        would still be empty afterwards, so a rule can never be the first
        thing in the scrolling area.
        """
        current = self._lines[-1] if self._lines else None
        current_text = current.text.plain if current is not None else ""
        keep_current = bool(current_text) and not self._is_filtered_console_line(
            current_text
        )
        has_visible_output = any(
            line.text.plain.strip() for line in self._lines[:-1]
        )
        if not has_visible_output and not keep_current:
            return
        # Inserting a rule now supersedes any pending promotion: a stale
        # expectation must not later swallow an unrelated dash line.
        self._separator_expected = False
        self._mutate_tail(
            keep_current=keep_current,
            new_tail=[_SeparatorLine(Text("")), ""],
        )

    @staticmethod
    def _is_separator_source(line: _Line) -> bool:
        plain = line.text.plain
        return len(plain) >= 3 and all(character == "-" for character in plain)

    # -- find support -------------------------------------------------------

    @property
    def line_count(self) -> int:
        """Number of logical lines in the document."""
        return len(self._lines)

    def line_texts(self) -> list[str]:
        """Plain text of each logical line, in document order."""
        return [line.text.plain for line in self._lines]

    def line_index_at_scroll_y(self, y: int) -> int | None:
        """The logical line whose wrapped rows begin at or before row ``y``."""
        if not self._lines:
            return None
        return bisect_right(self._prefix, y) - 1

    def line_scroll_y(self, line_index: int) -> int:
        """The top row at which a logical line begins."""
        return self._prefix[line_index]

    def _style_row(self, strip: Strip, line_index: int) -> Strip:
        """Reverse-video the active find match line."""
        if line_index == self.highlight_line_index:
            return strip.apply_style(Style(reverse=True))
        return strip

    # -- current-line protocol --------------------------------------------

    def _make_line(self, text: str) -> _Line:
        """Build a stored line from a plain (ANSI-containing) string."""
        line_text = Text.from_ansi(text)
        if "\t" in line_text.plain:
            tab_size = (
                self.app.console.tab_size
                if line_text.tab_size is None
                else line_text.tab_size
            )
            line_text.expand_tabs(tab_size)
        return _Line(line_text)

    def _is_filtered_console_line(self, text: str) -> bool:
        """Whether a worker console line should be omitted from history."""
        return any(substring in text for substring in self.output_filters)

    def feed_console(self, completed: list[str], live: str) -> None:
        """Apply one console chunk to the document.

        `completed` are the lines the chunk committed (each terminated
        by a newline) and `live` is the assembler's current line
        afterwards. Together they are exactly the document tail a
        terminal would hold after the same bytes, except completed lines
        matching an output filter are omitted from history. `live` is
        always the new current line, even when it matches a filter. A
        chunk without a newline only updates the current line in place.
        """
        if not completed:
            current = self._lines[-1].text.plain if self._lines else ""
            if current == live:
                # No change: the current line already holds the live
                # text (the common case for repeated flushes).
                return
        retained: list[str | _Line] = []
        # Only committed history counts as prior output: the live tail line
        # is dropped by the tail mutation below, so it can never precede the
        # batch heading.
        has_visible_output = any(
            line.text.plain.strip() for line in self._lines[:-1]
        )
        for text in completed:
            if self._is_filtered_console_line(text):
                continue
            line = self._make_line(text)
            if self._separator_expected:
                if self._is_separator_source(line):
                    self._separator_expected = False
                    if has_visible_output:
                        retained.append(_SeparatorLine(line.text))
                    # The divider is dropped when the batch heading is the
                    # first visible content in the scrolling area.
                    continue
                plain = line.text.plain.strip()
                if plain and plain.startswith(("Processing line ", "Processing lines ")):
                    # Defensive: the heading arrived without its expected
                    # divider, so a later unrelated rule must not convert.
                    self._separator_expected = False
            if line.text.plain.strip():
                has_visible_output = True
            retained.append(line)
        self._mutate_tail(keep_current=False, new_tail=[*retained, live])

    def append_application_lines(self, lines: list[str]) -> None:
        """Present lines produced by the app (notices, terminal summary).

        A current line that still holds content (e.g. a bar interrupted
        by the notice) becomes committed history first unless it matches
        an output filter; an empty current line is only a placeholder and
        is dropped. App-generated lines are never filtered. A fresh empty
        current line ends up last.
        """
        current = self._lines[-1] if self._lines else None
        current_text = current.text.plain if current is not None else ""
        keep_current = bool(current_text) and not self._is_filtered_console_line(
            current_text
        )
        self._mutate_tail(keep_current=keep_current, new_tail=[*lines, ""])

    def finalize(self) -> None:
        """Commit the current line at session end.

        The in-progress line (a bar, a status) becomes history unless it
        matches an output filter, and a fresh empty current line begins
        below it — exactly like a newline commit. The document mirrors
        the assembler's current
        line at all times (the one exception, an app line interrupting
        a bar, commits the bar to history at that moment), so the
        assembler's remaining line adds nothing here.
        """
        self._separator_expected = False
        current = self._lines[-1] if self._lines else None
        current_text = current.text.plain if current is not None else ""
        if current_text:
            self._mutate_tail(
                keep_current=not self._is_filtered_console_line(current_text),
                new_tail=[""],
            )

    def clear(self) -> Self:
        """Clear the document, leaving a fresh empty current line."""
        self._separator_expected = False
        super().clear()
        self._lines.append(_Line(Text("")))
        if self._size_known:
            self._sync_layout_tail()
        self.refresh()
        return self

    # -- tail mutation ------------------------------------------------------

    def _mutate_tail(
        self, *, keep_current: bool, new_tail: list[str | _Line]
    ) -> None:
        """Replace the document tail and resync the layout bookkeeping.

        The current line is dropped or committed to history per
        `keep_current`; the `new_tail` lines are appended, and the last
        of them becomes the new current line. Callers may pass an already
        constructed semantic line alongside ordinary strings.
        """
        lines = self._lines
        dropped = 0
        if not keep_current and lines:
            lines.pop()
            dropped = 1
        pre_len = len(lines)
        lines.extend(
            item if isinstance(item, _Line) else self._make_line(item)
            for item in new_tail
        )
        self._trim_overflow(pre_len)
        # The tail objects are new, and the dropped object may get its
        # address reused; the generation bump keeps a reused id from
        # aliasing a cached strip (the same rule `write`'s trim
        # follows).
        self._invalidate_strips()
        self._sync_layout_tail(drop_tail_counts=dropped)
        if self.follow_tail:
            self.scroll_end(animate=False, immediate=True, x_axis=False)
        self.refresh()

    # -- navigation ---------------------------------------------------------

    def action_history_home(self) -> None:
        self.follow_tail = False
        self.scroll_home(animate=False, x_axis=False)

    def action_history_up(self) -> None:
        self.follow_tail = False
        self.scroll_up(animate=False)

    def action_history_down(self) -> None:
        self._resume_if_at_end()
        self.scroll_down(animate=False)

    def action_history_page_up(self) -> None:
        self.follow_tail = False
        self.scroll_page_up(animate=False)

    def action_history_page_down(self) -> None:
        self._resume_if_at_end()
        self.scroll_page_down(animate=False)

    def action_history_end(self) -> None:
        self._resume_if_at_end()
        self.scroll_end(animate=False, x_axis=False)

    def _resume_if_at_end(self) -> None:
        """Resume following the tail if the user navigated all the way
        to it."""
        if self.scroll_offset.y >= self.max_scroll_y:
            self.follow_tail = True

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Manual scrolling up detaches from the tail; scrolling back to
        the very end reattaches."""
        super().watch_scroll_y(old_value, new_value)
        if self.follow_tail and new_value < self.max_scroll_y:
            self.follow_tail = False
        elif not self.follow_tail and new_value >= self.max_scroll_y:
            self.follow_tail = True

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """A wheel click detaches from the tail so wheel scrolling is
        not immediately snapped back."""
        if event.delta.y != 0:
            self.follow_tail = False
            self._resume_if_at_end()

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """A wheel scroll down detaches from the tail; the base does the
        actual scrolling, and scrolling back to the very end resumes
        following the tail."""
        self.follow_tail = False
        super()._on_mouse_scroll_down(event)
        self._resume_if_at_end()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Scrolling back to the very end resumes following the tail."""
        super()._on_mouse_scroll_up(event)
        self._resume_if_at_end()


class WorkerLogContentArea(Vertical):
    """Scrollable worker-output log area.

    The whole area is the log: its document ends with the current
    line, which holds in-progress worker output (model inference
    progress, transcribing status, ...) and updates in place — see
    `WorkerLog` for the model. There is no separate chrome row for
    in-progress output, so a dynamic line can never resize the area,
    and a committed line enters the document flow with a normal
    one-row scroll.
    """

    def __init__(
        self,
        *,
        output_filters: list[str] | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.output_filters = list(output_filters or ())

    DEFAULT_CSS = """
    WorkerLogContentArea {
        height: 1fr;
        padding: 0 1;
    }

    .worker-log {
        height: 1fr;
        width: 100%;
        background: ansi_default;
        color: ansi_default;
        scrollbar-size-vertical: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield WorkerLog(
            output_filters=self.output_filters,
            classes="worker-log",
        )

    @property
    def worker_log(self) -> WorkerLog:
        return self.query_one(".worker-log", WorkerLog)

    def expect_separator(self) -> None:
        """Arm semantic rendering of the next printed console divider."""
        self.worker_log.expect_separator()

    def append_separator(self) -> None:
        """Append a full-width rule to the log's document tail."""
        self.worker_log.append_separator()

    def feed(self, completed: list[str], live: str) -> None:
        """Feed one console chunk into the document's current line."""
        self.worker_log.feed_console(completed, live)

    def append_lines(self, lines: list[str]) -> None:
        """Present app-generated lines, committing the current line
        first."""
        self.worker_log.append_application_lines(lines)

    def finalize(self) -> None:
        """Commit the current line at session end."""
        self.worker_log.finalize()
