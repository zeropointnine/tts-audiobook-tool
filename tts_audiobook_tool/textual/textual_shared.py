from collections.abc import Callable
import os
import sys
from typing import ClassVar

from rich.console import Console, ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding, BindingType
from textual.strip import Strip
from textual.widgets import OptionList
from textual.widgets.option_list import Option


STYLE_ACCENT = "#ffaa44"
STYLE_ERROR = "#ff0000"
STYLE_DIM = "#888888"
STYLE_OK = "#00ff00"
STYLE_DEFAULT = "default"


TEXTUAL_SHARED_CSS = """\
$col-accent: #ffaa44;
$col-error: #ff0000;
$col-dim: #888888;
$col-ok: #00ff00;
$col-default: ansi_default;

/* Used by the test suite to verify that this shared stylesheet was loaded. */
#textual-shared-css-test {
    color: #123456;
}

Screen {
    layout: vertical;
}

Screen:ansi.-screen-suspended {
    text-style: none !important;
}

#header {
    overflow: hidden;
}

#header > .header-line {
    height: 1;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}

#header-divider {
    height: 1;
    margin: 0;
    color: $col-dim;
}
"""


CONTENT_TEXTUAL_APP_CSS = """\
#line-list {
    height: 1fr;
    max-height: 100%;
    border: none;
    padding: 0;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}

#line-list:focus {
    border: none;
    background-tint: transparent;
}

#line-list > .option-list--option-highlighted,
#line-list:focus > .option-list--option-highlighted {
    text-style: reverse;
}

#status-bar {
    height: 1;
    layout: horizontal;
}

#playing-status {
    width: 1fr;
    height: 1;
    color: $col-dim;
    text-style: italic;
    content-align: left middle;
}

#selection-status {
    width: 1fr;
    height: 1;
    color: $col-dim;
    text-style: italic;
    content-align: right middle;
}

#find-bar {
    display: none;
    height: 1;
    layout: horizontal;
}

#find-label {
    width: auto;
    height: 1;
    color: $col-accent;
    text-style: italic;
}

#find-input {
    width: 1fr;
    height: 1;
    border: none;
    padding: 0;
    background: ansi_default;
    background-tint: transparent;
}

#find-result {
    width: 12;
    height: 1;
    color: $col-dim;
    text-style: italic;
    content-align: right middle;
}

#find-input:focus {
    border: none;
    color: $col-default;
    background: ansi_default;
    background-tint: transparent;
}
"""


class HangingIndentText:
    """Render ANSI-styled text with a fixed prefix and capped hanging indent."""

    def __init__(
        self,
        text: Text,
        content_start: int,
        max_lines: int = 3,
        style: str = "",
    ) -> None:
        self.text = text
        self.content_start = content_start
        self.max_lines = max(max_lines, 1)
        self.style = style

    @classmethod
    def from_ansi(
        cls,
        ansi_text: str,
        content_start: int,
        max_lines: int = 3,
        style: str = "",
    ) -> "HangingIndentText":
        """Create a renderable using an offset into the ANSI source string."""
        plain_content_start = len(Text.from_ansi(ansi_text[:content_start]).plain)
        return cls(
            Text.from_ansi(ansi_text),
            plain_content_start,
            max_lines,
            style,
        )

    @property
    def spans(self):
        """Expose source spans for callers that inspect the logical row styling."""
        return self.text.spans

    def __str__(self) -> str:
        return self.text.plain

    def __rich_measure__(
        self, _console: Console, options: ConsoleOptions
    ) -> Measurement:
        maximum = min(self.text.cell_len, options.max_width)
        return Measurement(min(maximum, 1), maximum)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        prefix, content = self.text.divide([self.content_start])
        prefix_width = prefix.cell_len
        content_width = max(options.max_width - prefix_width, 1)
        content_lines = content.wrap(
            console,
            content_width,
            overflow="fold",
            no_wrap=False,
        ) or [Text()]
        was_truncated = len(content_lines) > self.max_lines
        visible_lines = content_lines[: self.max_lines]
        for visible_line in visible_lines:
            visible_line.rstrip()
        if was_truncated:
            last_line = visible_lines[-1]
            last_line.truncate(max(content_width - 1, 0), overflow="crop")
            last_line.append("…")

        rendered = Text()
        continuation_indent = " " * prefix_width
        for line_index, content_line in enumerate(visible_lines):
            if line_index:
                rendered.append("\n")
                line = Text(continuation_indent)
            else:
                line = prefix.copy()
            line.append(content_line)
            rendered.append(line)
        if self.style:
            rendered.stylize(self.style)
        yield rendered


class NonWrappingOptionList(OptionList):
    BINDINGS: ClassVar[list[BindingType]] = [
        *OptionList.BINDINGS,
        Binding("shift+up", "extend_cursor_up", show=False),
        Binding("shift+down", "extend_cursor_down", show=False),
        Binding("shift+pageup", "extend_page_up", show=False),
        Binding("shift+pagedown", "extend_page_down", show=False),
        Binding("shift+home", "extend_first", show=False),
        Binding("shift+end", "extend_last", show=False),
    ]

    def __init__(
        self,
        *content: Option,
        collapse_selection: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        self.extend_selection = False
        self.collapse_selection = collapse_selection
        self.inactive_selection_indices: set[int] = set()
        self.inactive_selection_style = Style.parse(f"{STYLE_DIM} reverse")
        super().__init__(*content, **kwargs)

    def set_inactive_selection_indices(self, indices: set[int]) -> None:
        """Update inactive selections without replacing prompts or layout caches."""
        if indices == self.inactive_selection_indices:
            return
        self.inactive_selection_indices = set(indices)
        self.refresh()

    def render_line(self, y: int) -> Strip:
        """Apply full-width styling to inactive selected visual rows."""
        line_number = self.scroll_offset.y + y
        try:
            option_index, _line_offset = self._lines[line_number]
        except IndexError:
            return super().render_line(y)
        strip = super().render_line(y)
        if option_index in self.inactive_selection_indices:
            return Strip(
                Segment.apply_style(strip, post_style=self.inactive_selection_style),
                strip.cell_length,
            )
        return strip

    def prepare_navigation(self, extend_selection: bool) -> None:
        """Record navigation mode and collapse selection for unshifted movement."""
        self.extend_selection = extend_selection
        if not extend_selection and self.collapse_selection is not None:
            self.collapse_selection()

    def move_cursor(self, direction: int) -> None:
        """Move to the next enabled option in one direction without wrapping."""
        if self.highlighted is None:
            return
        stop = -1 if direction < 0 else len(self.options)
        for index in range(self.highlighted + direction, stop, direction):
            if not self.options[index].disabled:
                self.highlighted = index
                return

    def action_cursor_up(self) -> None:
        """Move to the previous enabled option without wrapping at the top."""
        self.prepare_navigation(False)
        self.move_cursor(-1)

    def action_cursor_down(self) -> None:
        """Move to the next enabled option without wrapping at the bottom."""
        self.prepare_navigation(False)
        self.move_cursor(1)

    def action_page_up(self) -> None:
        self.prepare_navigation(False)
        super().action_page_up()

    def action_page_down(self) -> None:
        self.prepare_navigation(False)
        super().action_page_down()

    def action_first(self) -> None:
        self.prepare_navigation(False)
        super().action_first()

    def action_last(self) -> None:
        self.prepare_navigation(False)
        super().action_last()

    def action_extend_cursor_up(self) -> None:
        self.prepare_navigation(True)
        self.move_cursor(-1)

    def action_extend_cursor_down(self) -> None:
        self.prepare_navigation(True)
        self.move_cursor(1)

    def action_extend_page_up(self) -> None:
        self.prepare_navigation(True)
        super().action_page_up()

    def action_extend_page_down(self) -> None:
        self.prepare_navigation(True)
        super().action_page_down()

    def action_extend_first(self) -> None:
        self.prepare_navigation(True)
        super().action_first()

    def action_extend_last(self) -> None:
        self.prepare_navigation(True)
        super().action_last()

    async def _on_click(self, event: events.Click) -> None:
        """Extend from the selection anchor on Shift+click; otherwise collapse."""
        self.prepare_navigation(event.shift)
        await super()._on_click(event)

def can_textual() -> bool:
    """Coarse test to see if terminal can support a full-screen interface."""

    from tts_audiobook_tool.ask import can_hotkey

    if not can_hotkey:
        return False

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False

    if os.name == "nt":
        return True

    term = os.environ.get("TERM", "")
    return term not in {"", "dumb"}
