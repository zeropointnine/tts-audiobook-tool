from collections.abc import Callable
import os
from pathlib import Path
import sys
from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.widgets import OptionList
from textual.widgets.option_list import Option


STYLE_ACCENT = "#ffaa44"
STYLE_ERROR = "#ff0000"
STYLE_DIM = "#888888"
STYLE_OK = "#00ff00"
STYLE_DEFAULT = "default"


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
        super().__init__(*content, **kwargs)

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


def load_css(*filenames: str) -> str:
    """Load multiple TCSS files into one variable-substitution scope."""
    textual_dir = Path(__file__).parent
    return "\n".join(
        (textual_dir / filename).read_text(encoding="utf-8")
        for filename in filenames
    )

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
