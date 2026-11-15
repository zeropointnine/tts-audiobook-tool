from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Rule, Static
from textual.widgets.option_list import Option

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.text_ops import language_util
from tts_audiobook_tool.textual.textual_shared import STYLE_DIM

UNCOMMON_WORDS_TITLE = "Uncommon words"
NONE_FOUND_LABEL = "None found"
# Only the most frequent words are listed
MAX_UNCOMMON_WORDS = 100

# "%1" is replaced with the normalized project language code. Word-wraps to
# the dialog width.
UNCOMMON_WORDS_DESC = (
    "Source text words not found in the app's %1 common words"
    " dictionary. These may have a higher likelihood of being mispronounced"
    " by the TTS model."
)

# Width of the dialog box (border included); keep in sync with the CSS
DIALOG_WIDTH = 80
# Horizontal screen margin around the box (2 per side)
DIALOG_MARGIN_WIDTH = 4
# Columns inside the box around its content: border (2) and padding (4)
DIALOG_CHROME_WIDTH = 6
# Rows the dialog uses around its list, not counting the wrapped description:
# screen margin (2), border (2), padding (2), title (1), hints (1) and the
# divider (1)
DIALOG_CHROME_HEIGHT = 9


@dataclass(frozen=True)
class UncommonWordItem:
    """One uncommon word: its lowercase key, occurrence count, and casings seen."""

    key: str
    count: int
    variants: tuple[str, ...]

    @property
    def word(self) -> str:
        """The first casing seen in the text, used for pronunciation previews."""
        return self.variants[0] if self.variants else self.key


def collect_uncommon_words(project: Project) -> list[UncommonWordItem]:
    """List the project text's uncommon words in descending order of frequency."""
    # Raw tokens still include their surrounding whitespace and punctuation;
    # get_uncommon_words() peels them.
    raw_words: list[str] = []
    for group in project.phrase_groups:
        for phrase in group.phrases:
            raw_words.extend(phrase.words)
    return [
        UncommonWordItem(key, count, tuple(variants))
        for key, count, variants in app_text.get_uncommon_words(raw_words)
    ]


def make_uncommon_words_description(language_code: str) -> str:
    return UNCOMMON_WORDS_DESC.replace(
        "%1", language_util.normalize_language_code(language_code)
    )


class UncommonWordsDialog(ModalScreen[None]):
    """Read-only, scrollable list of the project's uncommon words.

    [Q] previews the highlighted word's pronunciation and [ENTER] adds a
    substitution for it, both through injected callbacks; the dialog stays
    open while their modals run over it.
    """

    AUTO_FOCUS = "#uncommon-words-list"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding("q", "preview", show=False),
    ]

    CSS = """
    UncommonWordsDialog {
        align: center middle;
        background: transparent;
    }

    #uncommon-words-dialog {
        width: 80;
        max-width: 100%;
        height: auto;
        max-height: 100%;
        margin: 1 2;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #uncommon-words-title,
    #uncommon-words-desc,
    #uncommon-words-empty {
        height: auto;
    }

    /* Only the description wraps; DIALOG_CHROME_HEIGHT counts one hint row. */
    #uncommon-words-hints {
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #uncommon-words-divider {
        height: 1;
        margin: 0;
        color: #888888;
    }

    #uncommon-words-list {
        height: auto;
        border: none;
        padding: 0;
        color: ansi_default;
        background: ansi_default;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        scrollbar-size-vertical: 1;
    }

    #uncommon-words-list:focus {
        border: none;
        background-tint: transparent;
    }

    /* Keep the theme's cursor colors (see MultiSelectDialog). */
    #uncommon-words-list > .option-list--option-highlighted,
    #uncommon-words-list:focus > .option-list--option-highlighted {
        text-style: reverse;
    }

    #uncommon-words-list > .option-list--option-hover {
        background: ansi_default;
    }
    """

    def __init__(
        self,
        items: Sequence[UncommonWordItem],
        language_code: str,
        preview_word: Callable[[str], None],
        add_word: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.is_clamped = len(items) > MAX_UNCOMMON_WORDS
        self.items = list(items[:MAX_UNCOMMON_WORDS])
        self.description = make_uncommon_words_description(language_code)
        self.preview_word = preview_word
        self.add_word = add_word

    def compose(self) -> ComposeResult:
        children: list[Static | Rule | OptionList] = [
            Static(
                Text.from_ansi(self.make_title()),
                id="uncommon-words-title",
                markup=False,
            ),
            Static(
                Text.from_ansi(self.description),
                id="uncommon-words-desc",
                markup=False,
            ),
            Static(
                Text.from_ansi(
                    f"{COL_DIM}- [{COL_ACCENT}Q{COL_DIM}] Preview pronunciation"
                    f"  - [{COL_ACCENT}ENTER{COL_DIM}] Add substitution word"
                ),
                id="uncommon-words-hints",
                markup=False,
            ),
            Rule(id="uncommon-words-divider"),
        ]
        if self.items:
            children.append(
                OptionList(
                    *(Option(row) for row in self.make_rows()),
                    id="uncommon-words-list",
                    compact=True,
                )
            )
        else:
            children.append(
                Static(
                    Text.from_ansi(f"{COL_DIM}{NONE_FOUND_LABEL}"),
                    id="uncommon-words-empty",
                    markup=False,
                )
            )
        yield Vertical(*children, id="uncommon-words-dialog")

    def make_title(self) -> str:
        title = f"{COL_ACCENT}{UNCOMMON_WORDS_TITLE}"
        if self.is_clamped:
            title += f" {COL_DIM}(first {MAX_UNCOMMON_WORDS} items)"
        return title

    def make_rows(self) -> list[Text]:
        """Format each item as a right-aligned dim count plus its casings."""
        count_width = max(len(str(item.count)) for item in self.items)
        rows: list[Text] = []
        for item in self.items:
            row = Text(str(item.count).rjust(count_width), style=STYLE_DIM)
            row.append("  ")
            row.append(" | ".join(item.variants))
            rows.append(row)
        return rows

    @property
    def option_list(self) -> OptionList | None:
        lists = self.query("#uncommon-words-list").results(OptionList)
        return next(iter(lists), None)

    def on_mount(self) -> None:
        self.fit_list_height()
        option_list = self.option_list
        if option_list is not None:
            option_list.highlighted = 0

    def on_resize(self, _event: events.Resize) -> None:
        self.fit_list_height()

    def fit_list_height(self) -> None:
        option_list = self.option_list
        if option_list is not None:
            chrome = DIALOG_CHROME_HEIGHT + self.description_line_count()
            option_list.styles.max_height = max(1, self.size.height - chrome)

    def description_line_count(self) -> int:
        """Rows the description wraps to at the dialog's current width."""
        box_width = min(DIALOG_WIDTH, self.size.width - DIALOG_MARGIN_WIDTH)
        content_width = box_width - DIALOG_CHROME_WIDTH
        if content_width <= 0:
            return 1
        lines = Text(self.description).wrap(self.app.console, content_width)
        return max(1, len(lines))

    @property
    def highlighted_item(self) -> UncommonWordItem | None:
        option_list = self.option_list
        if option_list is None or option_list.highlighted is None:
            return None
        index = option_list.highlighted
        return self.items[index] if 0 <= index < len(self.items) else None

    # Keep list events from reaching the editor's own row handlers.
    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        event.stop()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """ENTER (or a click) on a row adds a substitution for its word."""
        event.stop()
        if 0 <= event.option_index < len(self.items):
            self.add_word(self.items[event.option_index].word)

    def action_preview(self) -> None:
        item = self.highlighted_item
        if item is not None:
            self.preview_word(item.word)

    def action_close(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Close only when the modal backdrop, rather than the dialog, is clicked."""
        dialog = self.query_one("#uncommon-words-dialog", Vertical)
        if event.widget is None or (
            event.widget is not dialog and dialog not in event.widget.ancestors
        ):
            self.dismiss(None)
