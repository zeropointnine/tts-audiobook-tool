from collections.abc import Callable, Iterable
from typing import Any, ClassVar, cast

from rich.console import Console, ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding, BindingType
from textual.css.styles import RulesMap
from textual.strip import Strip
from textual.visual import VisualType
from textual.widgets import OptionList
from textual.widgets.option_list import Option


STYLE_ACCENT = "#ffaa44"
STYLE_ERROR = "#ff0000"
STYLE_DIM = "#888888"
STYLE_OK = "#00ff00"
STYLE_DEFAULT = "default"


OptionReconcileItem = tuple[str, Any, bool]
"""An option ID, optional replacement prompt, and whether its height may change."""


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
#content-shell {
    height: 1fr;
    layout: horizontal;
}

#content-main {
    width: 1fr;
    height: 100%;
}

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

#empty-state {
    display: none;
    height: 1fr;
    color: $col-dim;
    text-style: italic;
    padding: 1 1;
    content-align: left top;
}

#side-panel-divider {
    width: 1;
    height: 100%;
    margin: 0;
    color: $col-dim;
}

#side-panel {
    width: 35%;
    min-width: 20;
    max-width: 40;
    height: 100%;
    overflow: hidden;
}

#line-list > .option-list--option-highlighted,
#line-list:focus > .option-list--option-highlighted {
    text-style: reverse;
}

#status-bar {
    height: 1;
    layout: horizontal;
}

#status-line {
    width: 1fr;
    height: 1;
    text-style: italic;
}

#status-line.status-pinned {
    color: $col-dim;
    content-align: right middle;
}

#status-line.status-selected {
    color: $col-default;
    content-align: right middle;
}

#status-line.status-toast {
    color: $col-accent;
    content-align: left middle;
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

    @classmethod
    def from_ansi_prefix(
        cls,
        prefix_ansi: str,
        content: str,
        max_lines: int = 3,
        style: str = "",
    ) -> "HangingIndentText":
        """Create a renderable without parsing plain row content as ANSI."""
        text = Text.from_ansi(prefix_ansi)
        content_start = len(text.plain)
        text.append(content)
        return cls(text, content_start, max_lines, style)

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
        visible_lines = content_lines[: self.max_lines]
        # Trailing blank lines produced by final newlines do not count as
        # truncated content, so skip the ellipsis when every hidden line is
        # empty (e.g., structural rows rendered with a blank spacer line).
        was_truncated = any(
            line.plain.strip() for line in content_lines[self.max_lines :]
        )
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
        multi_select_enabled: bool = True,
        **kwargs,
    ) -> None:
        self.extend_selection = False
        self.collapse_selection = collapse_selection
        self.multi_select_enabled = multi_select_enabled
        self.inactive_selection_indices: set[int] = set()
        self.inactive_selection_style = Style.parse(f"{STYLE_DIM} reverse")
        self.defer_option_cache_clear = False
        super().__init__(*content, **kwargs)

    def _clear_caches(self) -> None:
        """Allow a prompt update batch to invalidate caches only once."""
        if not self.defer_option_cache_clear:
            super()._clear_caches()

    def replace_option_prompts(
        self,
        prompts: Iterable[tuple[int, VisualType]],
        *,
        reflow: bool = True,
    ) -> None:
        """Replace several prompts with one redraw and optional layout reflow."""
        prompt_list = list(prompts)
        if not prompt_list:
            return
        changed_options = {self.options[index] for index, _prompt in prompt_list}
        self.defer_option_cache_clear = True
        try:
            for index, prompt in prompt_list:
                self.replace_option_prompt_at_index(index, prompt)
        finally:
            self.defer_option_cache_clear = False
        for cache_key in list(self._option_render_cache.keys()):
            if cache_key[0] in changed_options:
                self._option_render_cache.discard(cache_key)
        if reflow:
            self._line_cache.clear()
        self.refresh()

    def reconcile_options(self, items: Iterable[OptionReconcileItem]) -> None:
        """Reconcile options by stable ID while retaining unaffected render caches.

        A ``None`` prompt retains the existing option unchanged. A non-``None``
        prompt replaces that option's content, or creates it when the ID is new.
        The final flag marks prompts whose wrapped height may have changed.
        """
        item_list = list(items)
        item_ids = [item_id for item_id, _prompt, _reflow in item_list]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Reconciled option IDs must be unique")

        old_options_by_id = {
            option.id: option for option in self.options if option.id is not None
        }
        old_indices = {
            option: index for index, option in enumerate(self.options)
        }
        old_heights = {
            option: self._line_cache.heights[index]
            for index, option in enumerate(self.options)
            if index in self._line_cache.heights
        }
        changed_options: set[Option] = set()
        reflow_options: set[Option] = set()
        reconciled_options: list[Option] = []

        for item_id, prompt, reflow in item_list:
            option = old_options_by_id.get(item_id)
            if option is None:
                if prompt is None:
                    raise ValueError(f"New option {item_id!r} requires a prompt")
                option = Option(prompt, id=item_id)
                changed_options.add(option)
                reflow = True
            elif prompt is not None:
                option._set_prompt(prompt)
                changed_options.add(option)
            if reflow:
                reflow_options.add(option)
            reconciled_options.append(option)

        retained_options = set(reconciled_options)
        new_indices = {
            option: index for index, option in enumerate(reconciled_options)
        }
        for cache_key in list(self._option_render_cache.keys()):
            cached_option = cache_key[0]
            if (
                cached_option not in retained_options
                or cached_option in changed_options
                or new_indices.get(cached_option) != old_indices.get(cached_option)
            ):
                self._option_render_cache.discard(cache_key)

        self._options[:] = reconciled_options
        self._id_to_option = {
            option.id: option
            for option in reconciled_options
            if option.id is not None
        }
        self._option_to_index = new_indices
        self._mouse_hovering_over = None

        line_cache = self._line_cache
        line_cache.clear()
        if self.scrollable_content_region:
            padding = self.get_component_styles("option-list--option").padding
            rules = cast(RulesMap, self.styles)
            width = (
                self.scrollable_content_region.width - self._get_left_gutter_width()
            )
            for index, option in enumerate(reconciled_options):
                line_cache.index_to_line[index] = len(line_cache.lines)
                line_count = old_heights.get(option)
                if line_count is None or option in reflow_options:
                    line_count = (
                        self._get_visual(option).get_height(
                            rules, width - padding.width
                        )
                        + option._divider
                    )
                line_cache.heights[index] = line_count
                line_cache.lines.extend(
                    (index, line_offset) for line_offset in range(line_count)
                )

        self._update_lines()
        self.refresh(layout=self.styles.auto_dimensions)

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
        self.extend_selection = extend_selection and self.multi_select_enabled
        if not self.extend_selection and self.collapse_selection is not None:
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
    """Coarse test to see if the terminal can support a full-screen interface."""
    from tts_audiobook_tool.system_support.terminal import can_use_full_screen_terminal

    return can_use_full_screen_terminal()
