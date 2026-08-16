from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import OptionList, Static

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_GRAY
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual.section_markers_dialog import (
    SectionMarkersDialog,
    ClearSectionMarkers,
)
from tts_audiobook_tool.textual.content_textual_app import (
    ContentTextualApp,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    STYLE_DIM,
)
from tts_audiobook_tool.util import make_noun


@dataclass(frozen=True)
class SectionMarkersPhraseGroupItem:
    """A project phrase displayed in the section-markers editor."""

    phrase_index: int


@dataclass(frozen=True)
class SectionMarkersSectionItem:
    """A structural book section displayed above its phrase rows."""

    ordinal: int
    section_count: int
    title: str
    line_count: int

    @property
    def display_text(self) -> str:
        title_text = f": {self.title}" if self.title else ""
        line_noun = "line" if self.line_count == 1 else "lines"
        return (
            f"Section {self.ordinal}/{self.section_count}{title_text} "
            f"({self.line_count} {line_noun})"
        )


SectionMarkersListItem = SectionMarkersSectionItem | SectionMarkersPhraseGroupItem


class SectionMarkersEditor(ContentTextualApp[EditorSaved | EditorSaveFailed]):
    """Editor for staging section markers, committed on exit confirmation."""

    # The base "m" binding opens manual multi-selection, which this editor
    # does not use; it is replaced by the section-markers dialog below.
    BINDINGS: ClassVar[list[BindingType]] = [
        binding
        for binding in ContentTextualApp.BINDINGS
        if not (isinstance(binding, Binding) and binding.key == "m")
    ] + [
        Binding("m", "open_markers_dialog", "More options", show=False),
        Binding("space", "toggle_marker", "Toggle marker", show=False),
        Binding("]", "next_marker", "Next marker", show=False),
        Binding("[", "previous_marker", "Previous marker", show=False),
    ]

    CSS = f"""
{ContentTextualApp.CSS}

#markers-panel {{
    height: 100%;
    padding: 1 0 0 1;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}}
"""

    def __init__(self, project: Project) -> None:
        self.list_items: list[SectionMarkersListItem] = []
        self.original_markers: set[int] = set()
        self.staged_markers: set[int] = set()
        header_lines = [
            f"{COL_ACCENT}Edit {app_text.get_section_marker_label(project, is_title_case=False)}",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END], [L/R BRACKET] previous/next item  - [CTRL-F] Find text",
            f"{COL_DIM}- Press [{COL_ACCENT}SPACE{COL_DIM}] to toggle a {app_text.get_section_marker_label(project, is_title_case=False, is_singular=True)} on the highlighted line",
            f"{COL_DIM}- [M] More options (add manually, by regex, by blank lines, clear)",
            f"{COL_DIM}- Press [ESC] to finish",
        ]
        super().__init__(
            project,
            header_lines,
            empty_state_text="No text lines",
            loading_state_text="...",
            multi_select_enabled=False,
            side_panel_enabled=True,
        )

    def initialize_content(self) -> range:
        """Build section and phrase rows after the loading view is mounted."""
        self.list_items = self.make_list_items()
        self.original_markers = self.project.markers
        self.staged_markers = self.project.markers
        return range(len(self.list_items))

    def on_content_loaded(self) -> None:
        """Refresh the side panel now that the staged markers are populated."""
        self.update_markers_panel()

    def action_open_markers_dialog(self) -> None:
        """Open the dialog for choosing section marker options."""
        if not self.content_initialized:
            return
        self.push_screen(
            SectionMarkersDialog(self.project, len(self.staged_markers)),
            self.handle_section_markers_dialog,
        )

    def handle_section_markers_dialog(
        self, result: list[int] | ClearSectionMarkers | None
    ) -> None:
        """Merge or clear staged markers based on the dialog result."""
        if result is None:
            return
        if isinstance(result, ClearSectionMarkers):
            marked = set(self.staged_markers)
            cleared = len(marked)
            self.staged_markers = set()
            for item_index in range(len(self.phrase_indices)):
                if self.content_line_index(item_index) in marked:
                    self.refresh_line(item_index, reflow=False)
            self.update_markers_panel()
            label = app_text.get_section_marker_label(
                self.project,
                is_title_case=False,
                is_singular=cleared == 1,
            )
            self.set_toast_text(f"Cleared all {cleared} {label}")
            return
        added = set(result) - self.staged_markers
        if not added:
            return
        self.staged_markers |= added
        for item_index in range(len(self.phrase_indices)):
            if self.content_line_index(item_index) in set(added):
                self.refresh_line(item_index, reflow=False)
        self.update_markers_panel()
        added_count = len(added)
        label = app_text.get_section_marker_label(
            self.project,
            is_title_case=False,
            is_singular=added_count == 1,
        )
        self.set_toast_text(f"Added {added_count} {label}")

    @property
    def has_changes(self) -> bool:
        return (
            self.content_initialized
            and self.staged_markers != self.original_markers
        )

    def make_list_items(self) -> list[SectionMarkersListItem]:
        """Project non-empty book sections and their phrase groups into rows."""
        sections = self.project.book.sections
        show_sections = len(sections) > 1
        section_count = len(sections)
        next_phrase_index = 0
        list_items: list[SectionMarkersListItem] = []
        for ordinal, section in enumerate(sections, start=1):
            line_count = len(section.phrase_groups)
            if line_count == 0:
                continue
            if show_sections:
                list_items.append(
                    SectionMarkersSectionItem(
                        ordinal=ordinal,
                        section_count=section_count,
                        title=section.title,
                        line_count=line_count,
                    )
                )
            list_items.extend(
                SectionMarkersPhraseGroupItem(phrase_index)
                for phrase_index in range(
                    next_phrase_index,
                    next_phrase_index + line_count,
                )
            )
            next_phrase_index += line_count
        return list_items

    def format_line(self, index: int) -> HangingIndentText:
        """Format structural headings and numbered project phrase rows."""
        item = self.list_items[self.phrase_indices[index]]
        if isinstance(item, SectionMarkersSectionItem):
            return self.format_section_list_item(item.display_text, index)

        style = f"{STYLE_DIM} reverse" if index == self.find_match_index else ""
        is_marked = item.phrase_index in self.staged_markers
        number_color = COL_ACCENT if is_marked else COL_DIM
        asterisk = "*" if is_marked else " "
        prefix_ansi = f"{number_color}{item.phrase_index + 1:05d}{asterisk} {COL_DEFAULT}"
        phrase_group = self.project.phrase_groups[item.phrase_index]
        return HangingIndentText.from_ansi_prefix(
            f"{prefix_ansi}{Ansi.RESET}",
            phrase_group.presentable_text,
            max_lines=3,
            style=style,
        )

    def action_toggle_marker(self) -> None:
        """Toggle the section marker on the highlighted phrase row."""
        if not self.content_initialized or self.selected_index is None:
            return
        highlighted_phrase_index = self.highlighted_content_line_index()
        if highlighted_phrase_index is None:
            return
        if (
            highlighted_phrase_index == 0
            and highlighted_phrase_index not in self.staged_markers
        ):
            self.set_toast_text("Adding first line is not allowed")
            return
        if highlighted_phrase_index in self.staged_markers:
            self.staged_markers.remove(highlighted_phrase_index)
            toast = f"Removed line {highlighted_phrase_index + 1}"
        else:
            self.staged_markers.add(highlighted_phrase_index)
            toast = f"Added line {highlighted_phrase_index + 1}"
        self.refresh_line(self.selected_index, reflow=False)
        self.update_markers_panel()
        self.set_toast_text(toast)

    def update_markers_panel(self) -> None:
        """Re-render the read-only side panel from the staged markers."""
        if self.is_running:
            self.query_one("#markers-panel", Static).update(
                self.markers_panel_text()
            )

    def find_text(self, phrase_index: int) -> str:
        """Search phrase text and complete generated section heading text."""
        item = self.list_items[phrase_index]
        if isinstance(item, SectionMarkersSectionItem):
            return item.display_text
        return self.project.phrase_groups[item.phrase_index].presentable_text

    def action_next_marker(self) -> None:
        """Move the highlight to the nearest staged marker below the current row."""
        self.jump_to_marker(direction=1)

    def action_previous_marker(self) -> None:
        """Move the highlight to the nearest staged marker above the current row."""
        self.jump_to_marker(direction=-1)

    def jump_to_marker(self, direction: int) -> None:
        """Highlight the nearest staged marker in the given direction, wrapping around."""
        if not self.content_initialized or not self.staged_markers:
            return
        marker_row_set = set(self.marker_row_indices())
        if not marker_row_set:
            return
        row_count = len(self.phrase_indices)
        start = self.selected_index if self.selected_index is not None else 0
        for offset in range(1, row_count + 1):
            index = (start + direction * offset) % row_count
            if index in marker_row_set:
                self.query_one("#line-list", OptionList).highlighted = index
                return

    def marker_row_indices(self) -> list[int]:
        """Return the visible rows holding staged markers, in row order."""
        marker_set = set(self.staged_markers)
        return [
            index
            for index in range(len(self.phrase_indices))
            if (line_index := self.content_line_index(self.phrase_indices[index]))
            is not None
            and line_index in marker_set
        ]

    def content_line_index(self, item_index: int) -> int | None:
        """Map phrase rows to project lines while excluding section headings."""
        item = self.list_items[item_index]
        if isinstance(item, SectionMarkersSectionItem):
            return None
        return item.phrase_index

    def compose_side_panel(self) -> ComposeResult:
        """Show the current section markers as read-only text."""
        yield Static(self.markers_panel_text(), id="markers-panel", markup=False)

    def markers_panel_text(self) -> Text:
        """Build the read-only panel text from the staged section markers."""
        if not self.content_initialized:
            return Text(self.loading_state_text or "", style=STYLE_DIM)
        markers = sorted(self.staged_markers)
        items_noun = make_noun("item", "items", len(markers))
        label = app_text.get_section_marker_label(
            self.project, is_title_case=False
        )
        text = Text()
        text.append(f"Current {label} ({len(markers)} {items_noun})")
        text.append("\n\n")
        if not markers:
            text.append("None", style=STYLE_DIM)
            return text

        panels = self.query("#markers-panel") if self.is_running else []
        panel_height = panels[0].size.height if panels else 0
        item_capacity = panel_height - 2
        has_overflow = panel_height > 0 and len(markers) > item_capacity
        displayed_markers = (
            markers
            if not has_overflow
            else markers[: max(item_capacity - 1, 0)]
        )

        for index in displayed_markers:
            text.append(f"{index + 1:05d}  ")
            text.append(
                self.project.phrase_groups[index].presentable_text, style=STYLE_DIM
            )
            text.append("\n")
        if has_overflow:
            overflow_text = f"+{len(markers) - len(displayed_markers)} more items"
            padding = max(panels[0].size.width - 1 - len(overflow_text), 0)
            text.append_text(
                Text.from_ansi(
                    f"{' ' * padding}{COL_GRAY}{overflow_text}"
                    f"{Ansi.RESET}"
                )
            )
        return text

    def commit_changes_and_exit(self) -> None:
        """Apply staged markers and persist them, rolling memory back on failure."""
        if not self.content_initialized:
            label = app_text.get_section_marker_label(self.project)
            self.exit(EditorSaveFailed(f"{label} editor was not initialized"))
            return

        self.project.markers = self.staged_markers

        error = self.project.save()
        if error:
            self.project.markers = self.original_markers
            self.exit(EditorSaveFailed(f"Save failed: {error}"))
            return

        self.exit(EditorSaved())
