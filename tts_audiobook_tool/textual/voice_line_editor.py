from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from textual.binding import Binding, BindingType

from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.speaker_assign import line_speakers
from tts_audiobook_tool.textual.content_textual_app import (
    ContentTextualApp,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.multi_select_dialog import (
    DIALOG_LINES,
    MANUAL_ENTRY,
    MultiSelectDialog,
    MultiSelectResult,
)
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    STYLE_DIM,
    dialog_marker,
    has_dialog_detection,
)
from tts_audiobook_tool.textual.voice_select_dialog import (
    VoiceSelectDialog,
    make_voice_select_title,
)
from tts_audiobook_tool.util import make_error_string

# The editor title intentionally matches the voice menu item label that opens it
TITLE_EDIT_VOICE_SELECTIONS = "Edit voice/line selections"

# Voice column: number, truncated/padded voice name, then a dialog marker
# (the marker column only when the text was segmented with dialog detection)
VOICE_NAME_WIDTH = 5

# Character column: the line's attributed speaker id (from Auto-assign
# character voices), truncated; only shown when any line has one
CHARACTER_ID_MAX_WIDTH = 6


@dataclass(frozen=True)
class VoiceLinePhraseGroupItem:
    """A voice-assignable phrase group at its flat index in the book."""

    phrase_index: int


@dataclass(frozen=True)
class VoiceLineSectionItem:
    """A non-empty book section displayed among voice-assignable rows."""

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


VoiceLineListItem = VoiceLineSectionItem | VoiceLinePhraseGroupItem


class VoiceLineEditorTextualApp(ContentTextualApp[EditorSaved | EditorSaveFailed]):
    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        *(
            Binding(str(number), f"assign_voice({number - 1})", show=False)
            for number in range(1, 10)
        ),
        # Shares the "m" key with the base manual-selection binding, which
        # check_action disables, so this one runs instead
        Binding("m", "show_multi_select", show=False),
        Binding("v", "show_voice_select", show=False),
    ]

    def __init__(
        self,
        project: Project,
        voice_sample_count: int | None = None,
    ) -> None:
        self.original_voice_indices: list[int] = []
        self.staged_voice_indices: list[int] = []
        self.list_items: list[VoiceLineListItem] = []
        self.show_dialog_markers = False
        # Attributed speaker label by phrase index, and the character column's
        # width (0 hides the column)
        self.line_speakers: dict[int, str] = {}
        self.character_width = 0
        self.voice_values = ProjectVoiceUtil.get_voice_values(project, project.get_tts_model_type())
        if voice_sample_count is None:
            voice_sample_count = len(self.voice_values)
        self.voice_sample_count = voice_sample_count
        self.voice_names = [voice.name.strip() for voice in project.voice_references]
        highest_voice_key = min(max(self.voice_sample_count, 1), 9)
        header_lines = [
            f"{COL_ACCENT}{TITLE_EDIT_VOICE_SELECTIONS}",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]  - [CTRL-F] Find text",
            f"{COL_DIM}- Select multiple lines: [SHIFT] + navigation keys  - [CTRL-A] Select all  - [M] Multi-select",
            f"{COL_DIM}- Use keys [{COL_ACCENT}1{COL_DIM}] to [{COL_ACCENT}{highest_voice_key}{COL_DIM}] to set voice sample for selected lines  - [{COL_ACCENT}V{COL_DIM}] Voice list dialog",
            f"{COL_DIM}- Press [ESC] to finish",
        ]
        super().__init__(
            project,
            header_lines,
            empty_state_text="No text lines",
            loading_state_text="...",
        )

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        """Unbind the base "m" manual-selection key, leaving "m" to multi-select.

        Manual entry is offered from the multi-select dialog instead, which
        calls the action method directly (bypassing this check). Textual merges
        BINDINGS across base classes, so the binding cannot be filtered out.
        """
        if action == "show_manual_selection":
            return False
        return super().check_action(action, parameters)

    def initialize_content(self) -> range:
        """Snapshot voice assignments before installing their formatted rows."""
        self.original_voice_indices = [
            phrase_group.voice_index for phrase_group in self.project.phrase_groups
        ]
        self.staged_voice_indices = list(self.original_voice_indices)
        self.show_dialog_markers = self.has_dialog_detection()
        self.line_speakers = self.load_line_speakers()
        self.character_width = min(
            max((len(label) for label in self.line_speakers.values()), default=0),
            CHARACTER_ID_MAX_WIDTH,
        )
        self.list_items = self.make_list_items()
        return range(len(self.list_items))

    def has_dialog_detection(self) -> bool:
        """Whether the dialog-marker column is shown (see has_dialog_detection)."""
        book = getattr(self.project, "book", None)
        return has_dialog_detection(
            book is not None and book.segmentation_settings.dialog_segmentation,
            (phrase_group.is_dialog for phrase_group in self.project.phrase_groups),
        )

    def load_line_speakers(self) -> dict[int, str]:
        """Saved speaker attributions by phrase index (read-only, for display)."""
        book = getattr(self.project, "book", None)
        project_dir = getattr(self.project, "dir_path", "")
        if book is None or not project_dir:
            return {}
        return line_speakers.load_line_speakers(project_dir, book)

    def make_list_items(self) -> list[VoiceLineListItem]:
        """Project non-empty book sections and phrase groups into visible rows."""
        book = getattr(self.project, "book", None)
        if book is None:
            return [
                VoiceLinePhraseGroupItem(phrase_index)
                for phrase_index in range(len(self.project.phrase_groups))
            ]

        sections = book.sections
        show_sections = len(sections) > 1
        section_count = len(sections)
        next_phrase_index = 0
        list_items: list[VoiceLineListItem] = []
        for ordinal, section in enumerate(sections, start=1):
            line_count = len(section.phrase_groups)
            if line_count == 0:
                continue
            if show_sections:
                list_items.append(
                    VoiceLineSectionItem(
                        ordinal=ordinal,
                        section_count=section_count,
                        title=section.title,
                        line_count=line_count,
                    )
                )
            list_items.extend(
                VoiceLinePhraseGroupItem(phrase_index)
                for phrase_index in range(
                    next_phrase_index,
                    next_phrase_index + line_count,
                )
            )
            next_phrase_index += line_count
        return list_items

    @property
    def has_changes(self) -> bool:
        return (
            self.content_initialized
            and self.staged_voice_indices != self.original_voice_indices
        )

    def voice_label(self, voice_index: int) -> str:
        """Build the row's voice label, kept shared between display and search.

        In-range labels are fixed width so rows stay aligned: the number is
        padded to the digit count of the highest voice number, followed (only
        when any voice is named) by the voice name's first VOICE_NAME_WIDTH
        characters, blank when unnamed.
        """
        voice_number = voice_index + 1
        # Keep showing the stored number, but flag stale selections after voices change.
        if voice_index >= self.voice_sample_count:
            return f"#{voice_number} *OUT OF RANGE*"
        number_width = len(str(max(self.voice_sample_count, 1)))
        number_text = f"#{voice_number:<{number_width}}"
        if not any(self.voice_names):
            return number_text
        name_index = voice_number - 1
        name = self.voice_names[name_index] if name_index < len(self.voice_names) else ""
        return f"{number_text} {name[:VOICE_NAME_WIDTH]:<{VOICE_NAME_WIDTH}}"

    def format_line(self, index: int) -> HangingIndentText:
        """Format one row, styling selected rows except for the active row."""
        list_item = self.list_items[self.phrase_indices[index]]
        if isinstance(list_item, VoiceLineSectionItem):
            return self.format_section_list_item(list_item.display_text, index)

        is_find_match = index == self.find_match_index
        style = f"{STYLE_DIM} reverse" if is_find_match else ""
        phrase_index = list_item.phrase_index
        phrase_group = self.project.phrase_groups[phrase_index]
        if self.show_dialog_markers:
            dialog_column = f"{dialog_marker(phrase_group.is_dialog)}  "
        else:
            dialog_column = ""
        if self.character_width:
            character_id = self.line_speakers.get(phrase_index, "")[: self.character_width]
            # Padded inside the brackets, like the voice label; blank (no
            # brackets) for unattributed lines
            character_column = (
                f"[{character_id:<{self.character_width}}] "
                if character_id
                else " " * (self.character_width + 3)
            )
        else:
            character_column = ""
        prefix_ansi = (
            f"{COL_DIM}{self.format_line_number(phrase_index + 1)} "
            f"{character_column}"
            f"{COL_DIM}[{COL_ACCENT}{self.voice_label(self.staged_voice_indices[phrase_index])}{COL_DIM}] "
            f"{COL_DEFAULT}{dialog_column}"
        )
        return HangingIndentText.from_ansi(
            ansi_text=f"{prefix_ansi}{COL_DEFAULT}{phrase_group.presentable_text_lf_ansi}",
            content_start=len(prefix_ansi),
            max_lines=3,
            style=style,
        )

    def find_text_strings(self, item_index: int) -> Sequence[str]:
        """Search the row's number, full character id, voice label, and phrase
        text as separate fields."""
        list_item = self.list_items[item_index]
        if isinstance(list_item, VoiceLineSectionItem):
            return [list_item.display_text]
        phrase_index = list_item.phrase_index
        return [
            self.format_line_number(phrase_index + 1),
            *([self.line_speakers[phrase_index]] if phrase_index in self.line_speakers else []),
            self.voice_label(self.staged_voice_indices[phrase_index]).rstrip(),
            self.project.phrase_groups[phrase_index].presentable_text,
        ]

    def content_line_index(self, item_index: int) -> int | None:
        """Map phrase rows by project line number, excluding section rows."""
        item = self.list_items[item_index]
        if isinstance(item, VoiceLineSectionItem):
            return None
        return item.phrase_index

    def dialog_phrase_indices(self) -> set[int]:
        return {
            phrase_index
            for phrase_index, phrase_group in enumerate(self.project.phrase_groups)
            if phrase_group.is_dialog
        }

    def action_show_multi_select(self) -> None:
        """Show the multi-select dialog: select by voice, all dialog lines
        (only with dialog detection), or manual entry."""
        if (
            not self.multi_select_enabled
            or not self.content_initialized
            or self.find_active
            or self.manual_selection_line_count() <= 0
        ):
            return
        voice_line_counts = [
            self.staged_voice_indices.count(voice_index)
            for voice_index in range(self.voice_sample_count)
        ]
        dialog_line_count = (
            len(self.dialog_phrase_indices()) if self.show_dialog_markers else None
        )
        self.push_screen(
            MultiSelectDialog(voice_line_counts, self.voice_names, dialog_line_count),
            self.handle_multi_select,
        )

    def handle_multi_select(self, result: MultiSelectResult) -> None:
        """Select the chosen voice's or dialog lines, or open manual selection."""
        if result is None:
            return
        if result == MANUAL_ENTRY:
            self.action_show_manual_selection()
            return
        if result == DIALOG_LINES:
            self.handle_manual_selection(self.dialog_phrase_indices())
            return
        self.handle_manual_selection(
            {
                phrase_index
                for phrase_index, voice_index in enumerate(self.staged_voice_indices)
                if voice_index == result
            }
        )

    def action_show_voice_select(self) -> None:
        """Show the voice list dialog, which assigns the entered voice to the
        selected lines (and reaches voices beyond the 1-9 number keys)."""
        if (
            not self.content_initialized
            or self.find_active
            or self.voice_sample_count <= 0
        ):
            return
        phrase_indices = sorted(self.selected_content_line_indices())
        if not phrase_indices:
            return
        current_voice_index = (
            self.staged_voice_indices[phrase_indices[0]]
            if len(phrase_indices) == 1
            else None
        )
        voice_names = [
            self.voice_names[index] if index < len(self.voice_names) else ""
            for index in range(self.voice_sample_count)
        ]
        self.push_screen(
            VoiceSelectDialog(
                make_voice_select_title([index + 1 for index in phrase_indices]),
                voice_names,
                current_voice_index,
            ),
            self.handle_voice_select,
        )

    def handle_voice_select(self, voice_index: int | None) -> None:
        if voice_index is not None:
            self.action_assign_voice(voice_index)

    def action_assign_voice(self, voice_index: int) -> None:
        """Assign an available voice sample to all selected phrase groups."""
        if not self.content_initialized or voice_index >= self.voice_sample_count:
            return
        if not self.selected_content_line_indices():
            return

        def assign_voice(_visible_index: int, phrase_index: int) -> bool:
            if self.staged_voice_indices[phrase_index] == voice_index:
                return False
            self.staged_voice_indices[phrase_index] = voice_index
            return True

        selected_have_out_of_range_voice = any(
            self.staged_voice_indices[list_item.phrase_index] >= self.voice_sample_count
            for index in self.selected_indices
            if isinstance(
                list_item := self.list_items[self.phrase_indices[index]],
                VoiceLinePhraseGroupItem,
            )
        )
        self.mutate_selected_items(
            assign_voice,
            reflow=selected_have_out_of_range_voice,
        )

    def commit_changes_and_exit(self) -> None:
        """Apply staged values and persist them, rolling memory back on failure."""
        if not self.content_initialized:
            self.exit(EditorSaveFailed("Voice editor was not initialized"))
            return
        phrase_groups = self.project.phrase_groups
        if len(phrase_groups) != len(self.staged_voice_indices):
            self.exit(
                EditorSaveFailed("Save failed: project text changed while editing")
            )
            return

        for phrase_group, voice_index in zip(
            phrase_groups, self.staged_voice_indices, strict=True
        ):
            phrase_group.voice_index = voice_index

        try:
            error = ProjectTextIOUtil.save_book(self.project)
        except Exception as exception:
            error = make_error_string(exception)
        if error:
            for phrase_group, voice_index in zip(
                phrase_groups, self.original_voice_indices, strict=True
            ):
                phrase_group.voice_index = voice_index
            result = EditorSaveFailed(f"Save failed: {error}")
        else:
            result = EditorSaved()
        self.exit(result)

    def save_changes_and_exit(self) -> None:
        """Backward-compatible name for committing the staged voice values."""
        self.commit_changes_and_exit()
