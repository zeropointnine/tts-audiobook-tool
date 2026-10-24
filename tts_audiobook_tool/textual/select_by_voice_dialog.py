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

# Dialog result requesting the manual line-selection dialog
MANUAL_ENTRY: Literal["manual"] = "manual"

SelectByVoiceResult = int | Literal["manual"] | None

# Rows the dialog uses around its list: border (2), padding (2), and the
# title plus its trailing blank line (2)
DIALOG_CHROME_HEIGHT = 6


class SelectByVoiceDialog(ModalScreen[SelectByVoiceResult]):
    """List the voice samples (with line counts) plus a manual-entry item.

    Dismisses with the chosen zero-based voice index, MANUAL_ENTRY, or None
    when cancelled.
    """

    AUTO_FOCUS = "#select-by-voice-list"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("space", "select_highlighted", show=False),
        *(
            Binding(str(number), f"choose_voice({number - 1})", show=False)
            for number in range(1, 10)
        ),
    ]

    CSS = """
    SelectByVoiceDialog {
        align: center middle;
        background: transparent;
    }

    #select-by-voice-dialog {
        width: 62;
        height: auto;
        max-height: 100%;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #select-by-voice-title {
        height: auto;
    }

    #select-by-voice-list {
        height: auto;
        border: none;
        padding: 0;
        color: ansi_default;
        background: ansi_default;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        scrollbar-size-vertical: 1;
    }

    #select-by-voice-list:focus {
        border: none;
        background-tint: transparent;
    }

    /* Like the editor's #line-list, keep the theme's cursor colors: Textual
       renders reverse by swapping concrete colors, so reversing default on
       default would show no highlight at all. */
    #select-by-voice-list > .option-list--option-highlighted,
    #select-by-voice-list:focus > .option-list--option-highlighted {
        text-style: reverse;
    }

    #select-by-voice-list > .option-list--option-hover {
        background: ansi_default;
    }
    """

    def __init__(
        self, voice_line_counts: Sequence[int], voice_names: Sequence[str]
    ) -> None:
        super().__init__()
        self.voice_line_counts = list(voice_line_counts)
        self.voice_names = list(voice_names)

    def voice_option_text(self, voice_index: int) -> Text:
        number = voice_index + 1
        name = (
            self.voice_names[voice_index].strip()
            if voice_index < len(self.voice_names)
            else ""
        )
        name_text = f" {name}" if name else ""
        count = self.voice_line_counts[voice_index]
        line_noun = "line" if count == 1 else "lines"
        return Text.from_ansi(
            f"[{number}] {COL_ACCENT}Voice {number}{name_text} "
            f"{COL_DIM}({count} {line_noun})"
        )

    def compose(self) -> ComposeResult:
        options = [
            self.voice_option_text(voice_index)
            for voice_index in range(len(self.voice_line_counts))
        ]
        options.append(Text("Enter manually..."))
        yield Vertical(
            Static(
                Text.from_ansi(f"{COL_ACCENT}Select by voice\n"),
                id="select-by-voice-title",
                markup=False,
            ),
            OptionList(*options, id="select-by-voice-list", compact=True),
            id="select-by-voice-dialog",
        )

    def on_mount(self) -> None:
        self.fit_list_height()
        self.query_one("#select-by-voice-list", OptionList).highlighted = 0

    def on_resize(self, _event: events.Resize) -> None:
        self.fit_list_height()

    def fit_list_height(self) -> None:
        """Cap the list to the screen space left by the dialog's other rows.

        The dialog grows with its content; once it would exceed the screen,
        the list scrolls instead. A relative max-height does not constrain the
        list inside the auto-height dialog (it overflows and is clipped), so
        the cap is set in absolute rows.
        """
        option_list = self.query_one("#select-by-voice-list", OptionList)
        option_list.styles.max_height = max(1, self.size.height - DIALOG_CHROME_HEIGHT)

    def choose(self, option_index: int | None) -> None:
        """Dismiss with the result represented by one list item."""
        if option_index is None:
            return
        if option_index >= len(self.voice_line_counts):
            self.dismiss(MANUAL_ENTRY)
        else:
            self.dismiss(option_index)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        # Keep the message from bubbling to the host editor's own list handler
        event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.choose(event.option_index)

    def action_select_highlighted(self) -> None:
        self.choose(self.query_one("#select-by-voice-list", OptionList).highlighted)

    def action_choose_voice(self, voice_index: int) -> None:
        if voice_index < len(self.voice_line_counts):
            self.choose(voice_index)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Close only when the modal backdrop, rather than the dialog, is clicked."""
        dialog = self.query_one("#select-by-voice-dialog", Vertical)
        clicked_widget = event.widget
        if clicked_widget is None or (
            clicked_widget is not dialog and dialog not in clicked_widget.ancestors
        ):
            self.dismiss(None)
