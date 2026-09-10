from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from rich.console import Console, ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import OptionList, Rule, Static

from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.textual.content_textual_app import (
    ContentTextualApp,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.textual_shared import STYLE_DIM, STYLE_HIGHLIGHT
from tts_audiobook_tool.textual.word_substitutions_dialog import (
    WordSubstitutionEdit,
    WordSubstitutionsDialog,
)
from tts_audiobook_tool.util import make_error_string

NO_ITEMS_LABEL = "No items currently"
ADD_ITEM_LABEL = "Add item"


@dataclass(frozen=True)
class WordSubstitutionItem:
    """One staged original/replacement pair shown as an editable row."""

    original: str
    substitution: str
    ordinal: int


@dataclass(frozen=True)
class WordSubstitutionSentinel:
    """A non-editable row such as "No items currently" or "Add item"."""

    label: str
    is_add: bool


WordSubstitutionListItem = WordSubstitutionItem | WordSubstitutionSentinel


def fit_cell(
    value: str,
    width: int,
    *,
    align: str = "left",
    ellipsis: bool = False,
) -> str:
    """Truncate or pad one table cell to an exact display-cell width.

    Long values are hard cropped by default, which suits the ordinal column,
    where a cropped digit would read as a different number.

    `ellipsis=True` truncates with a trailing "…" and keeps the cell's final
    column blank, so a truncated value stays a full character clear of whatever
    is drawn after the cell. Values that already fit are padded, not truncated,
    so keeping the margin costs no visible text.
    """
    if width <= 0:
        return ""
    text = Text(value)
    if text.cell_len > width:
        if ellipsis:
            # max(..., 1): rich's truncate(0, overflow="ellipsis") is a no-op,
            # and a one-cell column has no room for a margin anyway.
            text.truncate(max(width - 1, 1), overflow="ellipsis")
        else:
            text.truncate(width, overflow="crop")
        value = text.plain
    padding = width - Text(value).cell_len
    if padding <= 0:
        return value
    if align == "right":
        return (" " * padding) + value
    return value + (" " * padding)


class WordSubstitutionTableRow:
    """Render one three-column table row sized to the given width."""

    ORDINAL_WIDTH = 3
    ORDINAL_GAP = 2
    COLUMN_GAP = 1

    def __init__(
        self,
        ordinal: str,
        original: str,
        substitution: str,
        style: str = "",
    ) -> None:
        self.ordinal = ordinal
        self.original = original
        self.substitution = substitution
        self.style = style

    def render_line(self, width: int) -> Text:
        """Lay out the ordinal, original, and substitution columns.

        The word cells ellipsize with a one-column margin inside the cell, so a
        truncated value never touches the column gap or the row's end.
        """
        width = max(width, 0)
        remaining = max(
            width - self.ORDINAL_WIDTH - self.ORDINAL_GAP - self.COLUMN_GAP, 0
        )
        original_width = remaining // 2
        substitution_width = remaining - original_width
        text = Text()
        ordinal_cell = fit_cell(self.ordinal, self.ORDINAL_WIDTH, align="right")
        if self.ordinal:
            text.append(Text.from_ansi(f"{COL_DIM}{ordinal_cell}"))
        else:
            text.append(ordinal_cell)
        text.append(" " * self.ORDINAL_GAP)
        text.append(fit_cell(self.original, original_width, ellipsis=True))
        text.append(" " * self.COLUMN_GAP)
        text.append(fit_cell(self.substitution, substitution_width, ellipsis=True))
        if text.cell_len > width:
            text.truncate(width, overflow="crop")
        if self.style:
            text.stylize(self.style)
        return text

    def __rich_measure__(
        self, _console: Console, options: ConsoleOptions
    ) -> Measurement:
        maximum = options.max_width
        return Measurement(min(maximum, 1), maximum)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield self.render_line(options.max_width)


class WordSubstitutionsApp(ContentTextualApp[EditorSaved | EditorSaveFailed]):
    """Edit the project's inference-time word substitutions as a table."""

    CSS = ContentTextualApp.CSS + """
    #table-header {
        width: 1fr;
        height: 1;
        padding: 0 2 0 0;
        text-wrap: nowrap;
    }

    #table-header-divider {
        height: 1;
        margin: 0;
        color: $col-dim;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        Binding("x", "delete_item", show=False),
    ]

    def __init__(self, project: Project) -> None:
        self.original = dict(project.word_substitutions)
        self.staged = dict(project.word_substitutions)
        self.list_items: list[WordSubstitutionListItem] = self.make_list_items()
        header_lines = [
            f"{COL_ACCENT}Edit word substitutions",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]"
            f"  - [CTRL-F] Find text",
            f"{COL_DIM}- [{COL_ACCENT}ENTER{COL_DIM}] Edit or add item"
            f"  - [{COL_ACCENT}X{COL_DIM}] Delete item"
            f"  - [{COL_ACCENT}ESC{COL_DIM}] Finish",
        ]
        super().__init__(
            project,
            header_lines,
            phrase_indices=list(range(len(self.list_items))),
            empty_state_text=NO_ITEMS_LABEL,
            multi_select_enabled=False,
        )

    def make_list_items(self) -> list[WordSubstitutionListItem]:
        """Order staged items by key and append the sentinel rows."""
        ordered = sorted(self.staged.items(), key=lambda item: item[0].casefold())
        rows: list[WordSubstitutionListItem] = [
            WordSubstitutionItem(original, substitution, ordinal)
            for ordinal, (original, substitution) in enumerate(ordered, start=1)
        ]
        if not rows:
            rows.append(WordSubstitutionSentinel(NO_ITEMS_LABEL, is_add=False))
        rows.append(WordSubstitutionSentinel(ADD_ITEM_LABEL, is_add=True))
        return rows

    @property
    def has_changes(self) -> bool:
        return self.staged != self.original

    def compose_content_top(self) -> ComposeResult:
        """Pin the table header and its rule above the scrolling row list."""
        yield Static(self.table_header_row(), id="table-header", markup=False)
        yield Rule(id="table-header-divider")

    @staticmethod
    def table_header_row() -> WordSubstitutionTableRow:
        return WordSubstitutionTableRow(
            "", "Original word:", "Substitution word:", style=STYLE_DIM
        )

    def format_line(self, index: int) -> WordSubstitutionTableRow:
        """Format one visible row, styling sentinels and find matches."""
        item = self.list_items[self.phrase_indices[index]]
        find_style = f"{STYLE_DIM} reverse" if index == self.find_match_index else ""
        if isinstance(item, WordSubstitutionSentinel):
            sentinel_style = STYLE_HIGHLIGHT if item.is_add else STYLE_DIM
            return WordSubstitutionTableRow(
                "", item.label, "", style=find_style or sentinel_style
            )
        return WordSubstitutionTableRow(
            str(item.ordinal),
            item.original,
            item.substitution,
            style=find_style,
        )

    def content_line_index(self, item_index: int) -> int | None:
        """Map editable item rows to their backing index; sentinels are structural."""
        item = self.list_items[item_index]
        if isinstance(item, WordSubstitutionSentinel):
            return None
        return item_index

    def find_text_strings(self, item_index: int) -> Sequence[str]:
        item = self.list_items[item_index]
        if isinstance(item, WordSubstitutionSentinel):
            return [item.label]
        return [str(item.ordinal), item.original, item.substitution]

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Open the add or edit dialog for the activated row."""
        item = self.list_items[self.phrase_indices[event.option_index]]
        existing_keys = list(self.staged.keys())
        if isinstance(item, WordSubstitutionSentinel):
            if not item.is_add:
                return
            dialog = WordSubstitutionsDialog.for_add(existing_keys)
        else:
            dialog = WordSubstitutionsDialog.for_edit(
                item.original, item.substitution, existing_keys
            )
        self.push_screen(dialog, self.handle_dialog_result)

    def handle_dialog_result(self, result: WordSubstitutionEdit | None) -> None:
        """Stage a committed edit and rebuild the rows."""
        if result is None:
            return
        if result.previous_original is not None:
            self.staged.pop(result.previous_original, None)
        self.staged[result.original] = result.substitution
        self.rebuild_rows(result.original)

    def action_delete_item(self) -> None:
        """Delete the highlighted substitution; sentinel rows are not deletable."""
        if self.find_active or self.selected_index is None:
            return
        item = self.list_items[self.phrase_indices[self.selected_index]]
        if not isinstance(item, WordSubstitutionItem):
            return
        del self.staged[item.original]
        self.rebuild_rows_at(self.selected_index)
        self.show_status_toast(left="Deleted item")

    def rebuild_rows_at(self, selected_index: int) -> None:
        """Re-derive rows, keeping the highlight near the given visible position."""
        self.list_items = self.make_list_items()
        selected_index = min(selected_index, len(self.list_items) - 1)
        self.replace_phrase_indices(range(len(self.list_items)), selected_index)

    def rebuild_rows(self, select_original: str) -> None:
        """Re-derive rows and keep the selection on the edited key."""
        self.list_items = self.make_list_items()
        selected_index = 0
        for index, item in enumerate(self.list_items):
            if (
                isinstance(item, WordSubstitutionItem)
                and item.original == select_original
            ):
                selected_index = index
                break
        self.replace_phrase_indices(range(len(self.list_items)), selected_index)

    def commit_changes_and_exit(self) -> None:
        """Persist staged substitutions, rolling memory back on failure."""
        self.project.word_substitutions = dict(self.staged)
        try:
            error = self.project.save()
        except Exception as exception:
            error = make_error_string(exception)
        if error:
            self.project.word_substitutions = dict(self.original)
            self.exit(EditorSaveFailed(f"Save failed: {error}"))
            return
        self.exit(EditorSaved())
