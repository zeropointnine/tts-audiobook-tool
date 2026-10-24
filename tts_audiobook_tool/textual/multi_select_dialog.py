from collections.abc import Sequence
from typing import ClassVar, Literal

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static

from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM

# Dialog result requesting every line tagged as dialog
DIALOG_LINES: Literal["dialog"] = "dialog"
# Dialog result requesting the manual line-selection dialog
MANUAL_ENTRY: Literal["manual"] = "manual"

MultiSelectResult = int | Literal["dialog", "manual"] | None

LABEL_DIALOG_LINES = "All lines tagged as dialog"
LABEL_MANUAL_ENTRY = "Enter line selections manually"

# Rows the dialog uses around its list: border (2), padding (2), and the
# title plus its trailing blank line (2)
DIALOG_CHROME_HEIGHT = 6


class MultiSelectDialog(ModalScreen[MultiSelectResult]):
    """List the voice samples (with line counts), an optional dialog-lines
    item, and a manual-entry item.

    Dismisses with the chosen zero-based voice index, DIALOG_LINES,
    MANUAL_ENTRY, or None when cancelled.
    """

    AUTO_FOCUS = "#multi-select-list"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("space", "select_highlighted", show=False),
        *(
            Binding(str(number), f"choose_voice({number - 1})", show=False)
            for number in range(1, 10)
        ),
    ]

    CSS = """
    MultiSelectDialog {
        align: center middle;
        background: transparent;
    }

    #multi-select-dialog {
        width: 62;
        height: auto;
        max-height: 100%;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #multi-select-title {
        height: auto;
    }

    #multi-select-list {
        height: auto;
        border: none;
        padding: 0;
        color: ansi_default;
        background: ansi_default;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        scrollbar-size-vertical: 1;
    }

    #multi-select-list:focus {
        border: none;
        background-tint: transparent;
    }

    /* Like the editor's #line-list, keep the theme's cursor colors: Textual
       renders reverse by swapping concrete colors, so reversing default on
       default would show no highlight at all. */
    #multi-select-list > .option-list--option-highlighted,
    #multi-select-list:focus > .option-list--option-highlighted {
        text-style: reverse;
    }

    #multi-select-list > .option-list--option-hover {
        background: ansi_default;
    }
    """

    def __init__(
        self,
        voice_line_counts: Sequence[int],
        voice_names: Sequence[str],
        dialog_line_count: int | None = None,
    ) -> None:
        """dialog_line_count of None omits the dialog-lines item (eg when the
        text was segmented without dialog detection)."""
        super().__init__()
        self.voice_line_counts = list(voice_line_counts)
        self.voice_names = list(voice_names)
        self.dialog_line_count = dialog_line_count

    @staticmethod
    def line_count_text(count: int) -> str:
        line_noun = "line" if count == 1 else "lines"
        return f"{COL_DIM}({count} {line_noun})"

    def voice_option_text(self, voice_index: int) -> Text:
        number = voice_index + 1
        name = (
            self.voice_names[voice_index].strip()
            if voice_index < len(self.voice_names)
            else ""
        )
        name_text = f" {name}" if name else ""
        count = self.voice_line_counts[voice_index]
        return Text.from_ansi(
            f"[{number}] {COL_ACCENT}Voice {number}{name_text} "
            f"{self.line_count_text(count)}"
        )

    def trailing_results(self) -> list[MultiSelectResult]:
        """Results of the items listed after the voices, in display order."""
        results: list[MultiSelectResult] = []
        if self.dialog_line_count is not None:
            results.append(DIALOG_LINES)
        results.append(MANUAL_ENTRY)
        return results

    def compose(self) -> ComposeResult:
        options = [
            self.voice_option_text(voice_index)
            for voice_index in range(len(self.voice_line_counts))
        ]
        if self.dialog_line_count is not None:
            options.append(
                Text.from_ansi(
                    f"{LABEL_DIALOG_LINES} "
                    f"{self.line_count_text(self.dialog_line_count)}"
                )
            )
        options.append(Text(LABEL_MANUAL_ENTRY))
        yield Vertical(
            Static(
                Text.from_ansi(f"{COL_ACCENT}Multi-select\n"),
                id="multi-select-title",
                markup=False,
            ),
            OptionList(*options, id="multi-select-list", compact=True),
            id="multi-select-dialog",
        )

    def on_mount(self) -> None:
        self.fit_list_height()
        self.query_one("#multi-select-list", OptionList).highlighted = 0

    def on_resize(self, _event: events.Resize) -> None:
        self.fit_list_height()

    def fit_list_height(self) -> None:
        """Cap the list to the screen space left by the dialog's other rows.

        The dialog grows with its content; once it would exceed the screen,
        the list scrolls instead. A relative max-height does not constrain the
        list inside the auto-height dialog (it overflows and is clipped), so
        the cap is set in absolute rows.
        """
        option_list = self.query_one("#multi-select-list", OptionList)
        option_list.styles.max_height = max(1, self.size.height - DIALOG_CHROME_HEIGHT)

    def choose(self, option_index: int | None) -> None:
        """Dismiss with the result represented by one list item."""
        if option_index is None:
            return
        voice_count = len(self.voice_line_counts)
        if option_index < voice_count:
            self.dismiss(option_index)
            return
        trailing_results = self.trailing_results()
        trailing_index = option_index - voice_count
        if trailing_index < len(trailing_results):
            self.dismiss(trailing_results[trailing_index])

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        # Keep the message from bubbling to the host editor's own list handler
        event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.choose(event.option_index)

    def action_select_highlighted(self) -> None:
        self.choose(self.query_one("#multi-select-list", OptionList).highlighted)

    def action_choose_voice(self, voice_index: int) -> None:
        if voice_index < len(self.voice_line_counts):
            self.choose(voice_index)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Close only when the modal backdrop, rather than the dialog, is clicked."""
        dialog = self.query_one("#multi-select-dialog", Vertical)
        clicked_widget = event.widget
        if clicked_widget is None or (
            clicked_widget is not dialog and dialog not in clicked_widget.ancestors
        ):
            self.dismiss(None)
