from dataclasses import dataclass
from typing import ClassVar

from textual.binding import Binding, BindingType

from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.textual.content_textual_app import (
    ContentTextualApp,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    STYLE_DIM,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import make_error_string


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
    ]

    def __init__(
        self,
        project: Project,
        voice_sample_count: int | None = None,
    ) -> None:
        self.original_voice_indices: list[int] = []
        self.staged_voice_indices: list[int] = []
        self.list_items: list[VoiceLineListItem] = []
        self.voice_values = ProjectVoiceUtil.get_voice_values(project, Tts.get_type())
        if voice_sample_count is None:
            voice_sample_count = len(self.voice_values)
        self.voice_sample_count = voice_sample_count
        highest_voice_key = min(max(self.voice_sample_count, 1), 9)
        header_lines = [
            f"{COL_ACCENT}Edit voice selections",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]  - [CTRL-F] Find text",
            f"{COL_DIM}- Select multiple lines: [SHIFT] + navigation keys  - [CTRL-A] Select all  - [M] Enter manually",
            f"{COL_DIM}- Use number keys [{COL_ACCENT}1{COL_DIM}] to [{COL_ACCENT}{highest_voice_key}{COL_DIM}] to set voice sample for selected text line/s",
            f"{COL_DIM}- Press [ESC] to finish",
        ]
        super().__init__(
            project,
            header_lines,
            empty_state_text="No text lines",
            loading_state_text="...",
        )

    def initialize_content(self) -> range:
        """Snapshot voice assignments before installing their formatted rows."""
        self.original_voice_indices = [
            phrase_group.voice_index for phrase_group in self.project.phrase_groups
        ]
        self.staged_voice_indices = list(self.original_voice_indices)
        self.list_items = self.make_list_items()
        return range(len(self.list_items))

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

    def format_line(self, index: int) -> HangingIndentText:
        """Format one row, styling selected rows except for the active row."""
        list_item = self.list_items[self.phrase_indices[index]]
        if isinstance(list_item, VoiceLineSectionItem):
            return self.format_section_list_item(list_item.display_text, index)

        is_find_match = index == self.find_match_index
        style = f"{STYLE_DIM} reverse" if is_find_match else ""
        phrase_index = list_item.phrase_index
        phrase_group = self.project.phrase_groups[phrase_index]
        voice_index = self.staged_voice_indices[phrase_index]
        voice_number = max(voice_index + 1, 1)
        # Keep showing the stored number, but flag stale selections after voices change.
        if voice_index >= self.voice_sample_count:
            voice_status = " *OUT OF RANGE*"
        else:
            voice_status = ""
        prefix_ansi = (
            f"{COL_DIM}[{phrase_index + 1:05d}] "
            f"{COL_ACCENT}[Voice {voice_number}{voice_status}]{COL_DIM} "
        )
        return HangingIndentText.from_ansi(
            ansi_text=f"{prefix_ansi}{COL_DEFAULT}{phrase_group.presentable_text}",
            content_start=len(prefix_ansi),
            max_lines=3,
            style=style,
        )

    def find_text(self, phrase_index: int) -> str:
        """Search phrase text and the complete generated section heading text."""
        list_item = self.list_items[phrase_index]
        if isinstance(list_item, VoiceLineSectionItem):
            return list_item.display_text
        return self.project.phrase_groups[list_item.phrase_index].presentable_text

    def content_line_index(self, item_index: int) -> int | None:
        """Map phrase rows by project line number, excluding section rows."""
        item = self.list_items[item_index]
        if isinstance(item, VoiceLineSectionItem):
            return None
        return item.phrase_index

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
