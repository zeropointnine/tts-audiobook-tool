from collections.abc import Sequence
from typing import ClassVar, Literal

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM, COL_DIM_ITALICS

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

# Voices reachable by number key ([1] to [9]); later voices are unlabeled
VOICE_HOTKEY_COUNT = 9
# Width of a "[n] " hotkey label, which unlabeled voices pad to for alignment
VOICE_HOTKEY_LABEL_WIDTH = len(f"[{VOICE_HOTKEY_COUNT}] ")


class MultiSelectDialog(ModalScreen[MultiSelectResult]):
    """List the voice samples (with line counts), an optional dialog-lines
    item, and a manual-entry item.

    A blank row separates the voice items from the trailing items, and a dim
    italic subheading can optionally introduce the voice items. The blank row
    is a disabled option, which the list's keyboard navigation and selection
    skip over.

    Dismisses with the chosen zero-based voice index, DIALOG_LINES,
    MANUAL_ENTRY, or None when cancelled.
    """

    AUTO_FOCUS = "#multi-select-list"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("space", "select_highlighted", show=False),
        *(
            Binding(str(number), f"choose_voice({number - 1})", show=False)
            for number in range(1, VOICE_HOTKEY_COUNT + 1)
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

    #multi-select-title,
    #multi-select-subheading {
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
        *,
        visible_voice_line_counts: Sequence[int] | None = None,
        visible_dialog_line_count: int | None = None,
        voices_subheading: str = "",
    ) -> None:
        """dialog_line_count of None omits the dialog-lines item (eg when the
        text was segmented without dialog detection).

        The visible counts are given when the host editor's list is filtered;
        each item then shows "(visible/total lines)" instead of "(total lines)".

        voices_subheading is shown above the voice items. It, like the blank
        row before the trailing items, applies only when there are voice items.
        """
        super().__init__()
        self.voice_line_counts = list(voice_line_counts)
        self.voice_names = list(voice_names)
        self.dialog_line_count = dialog_line_count
        self.visible_voice_line_counts = (
            list(visible_voice_line_counts)
            if visible_voice_line_counts is not None
            else None
        )
        self.visible_dialog_line_count = visible_dialog_line_count
        has_voices = bool(self.voice_line_counts)
        self.voices_subheading = voices_subheading if has_voices else ""
        # The result of each list option, in display order; None marks the
        # (disabled, never chosen) blank separator row
        self.option_results: list[MultiSelectResult] = []

    @staticmethod
    def line_count_text(count: int, visible_count: int | None = None) -> str:
        line_noun = "line" if count == 1 else "lines"
        visible_text = f"{visible_count}/" if visible_count is not None else ""
        return f"{COL_DIM}({visible_text}{count} {line_noun})"

    def voice_option_text(self, voice_index: int) -> Text:
        """Only voices with a number key get a "[n]" label; later ones are
        padded to the same width so the voice labels stay aligned."""
        number = voice_index + 1
        hotkey_text = (
            f"[{number}] "
            if number <= VOICE_HOTKEY_COUNT
            else " " * VOICE_HOTKEY_LABEL_WIDTH
        )
        name = (
            self.voice_names[voice_index].strip()
            if voice_index < len(self.voice_names)
            else ""
        )
        name_text = f" {name}" if name else ""
        count = self.voice_line_counts[voice_index]
        visible_count = (
            self.visible_voice_line_counts[voice_index]
            if self.visible_voice_line_counts is not None
            else None
        )
        return Text.from_ansi(
            f"{hotkey_text}{COL_ACCENT}Voice {number}{name_text} "
            f"{self.line_count_text(count, visible_count)}"
        )

    def trailing_results(self) -> list[MultiSelectResult]:
        """Results of the items listed after the voices, in display order."""
        results: list[MultiSelectResult] = []
        if self.dialog_line_count is not None:
            results.append(DIALOG_LINES)
        results.append(MANUAL_ENTRY)
        return results

    def compose(self) -> ComposeResult:
        options: list[Option] = []
        self.option_results = []
        for voice_index in range(len(self.voice_line_counts)):
            options.append(Option(self.voice_option_text(voice_index)))
            self.option_results.append(voice_index)
        if self.voice_line_counts:
            options.append(Option("", disabled=True))
            self.option_results.append(None)
        for result in self.trailing_results():
            options.append(Option(self.trailing_option_text(result)))
            self.option_results.append(result)

        widgets: list[Static | OptionList] = [
            Static(
                Text.from_ansi(f"{COL_ACCENT}Multi-select\n"),
                id="multi-select-title",
                markup=False,
            )
        ]
        if self.voices_subheading:
            widgets.append(
                Static(
                    Text.from_ansi(f"{COL_DIM_ITALICS}{self.voices_subheading}"),
                    id="multi-select-subheading",
                    markup=False,
                )
            )
        widgets.append(OptionList(*options, id="multi-select-list", compact=True))
        yield Vertical(*widgets, id="multi-select-dialog")

    def trailing_option_text(self, result: MultiSelectResult) -> Text:
        if result == DIALOG_LINES:
            assert self.dialog_line_count is not None
            return Text.from_ansi(
                f"{LABEL_DIALOG_LINES} "
                f"{self.line_count_text(self.dialog_line_count, self.visible_dialog_line_count)}"
            )
        return Text(LABEL_MANUAL_ENTRY)

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
        chrome_height = DIALOG_CHROME_HEIGHT + (1 if self.voices_subheading else 0)
        option_list.styles.max_height = max(1, self.size.height - chrome_height)

    def choose(self, option_index: int | None) -> None:
        """Dismiss with the result represented by one list item."""
        if option_index is None or not 0 <= option_index < len(self.option_results):
            return
        result = self.option_results[option_index]
        if result is not None:
            self.dismiss(result)

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
