from dataclasses import dataclass
from typing import ClassVar

from textual.binding import Binding, BindingType
from textual.css.errors import StylesheetError

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.textual.content_textual_app import ContentTextualApp
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    NonWrappingOptionList,
    STYLE_DIM,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import make_error_string, print_feedback


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


class VoiceLineEditorTextualApp(ContentTextualApp):
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
        self.did_save_changes = False
        self.save_error = ""
        self.voice_values = ProjectVoiceUtil.get_voice_values(project, Tts.get_type())
        if voice_sample_count is None:
            voice_sample_count = len(self.voice_values)
        self.voice_sample_count = voice_sample_count
        highest_voice_key = min(max(self.voice_sample_count, 1), 9)
        header_lines = [
            f"{COL_ACCENT}Edit voice selections",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]",
            f"{COL_DIM}- Select multiple lines by holding [SHIFT] + navigation keys",
            f"{COL_DIM}- Use number keys [{COL_ACCENT}1{COL_DIM}] to [{COL_ACCENT}{highest_voice_key}{COL_DIM}] to set voice sample for selected text line/s",
            f"{COL_DIM}- Press [ESC] to finish  - [CTRL-F] Find text"
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

    @property
    def selection_status_text(self) -> str:
        """Count selected phrase rows while excluding section headings."""
        selection_count = sum(
            isinstance(
                self.list_items[self.phrase_indices[index]],
                VoiceLinePhraseGroupItem,
            )
            for index in self.selected_indices
        )
        return f"{selection_count} lines selected" if selection_count >= 2 else ""

    def format_line(self, index: int) -> HangingIndentText:
        """Format one row, styling selected rows except for the active row."""
        list_item = self.list_items[self.phrase_indices[index]]
        is_find_match = index == self.find_match_index
        style = f"{STYLE_DIM} reverse" if is_find_match else ""
        if isinstance(list_item, VoiceLineSectionItem):
            return HangingIndentText.from_ansi(
                # Rich's line-height measurement doesn't count a final empty
                # line, so two trailing newlines are needed to render one.
                f"\n{COL_ACCENT}{list_item.display_text}\n\n",
                content_start=0,
                max_lines=3,
                style=style,
            )

        phrase_index = list_item.phrase_index
        phrase_group = self.project.phrase_groups[phrase_index]
        voice_index = self.staged_voice_indices[phrase_index]
        voice_number = max(voice_index + 1, 1)
        # Keep showing the stored number, but flag stale selections after voices change.
        voice_status = (
            " *OUT OF RANGE*" if voice_index >= self.voice_sample_count else ""
        )
        prefix_text = (
            f"[{phrase_index + 1:05d}] "
            f"[Voice sample {voice_number}{voice_status}] "
        )
        prefix_ansi = f"{COL_DIM}{prefix_text}{COL_DEFAULT}"
        return HangingIndentText.from_ansi(
            ansi_text=f"{prefix_ansi}{phrase_group.presentable_text}",
            content_start=len(prefix_ansi),
            max_lines=3,
            style=style,
        )

    def update_inactive_selection_style(self) -> None:
        """Style selected phrase rows while leaving section headings unchanged."""
        inactive_indices = self.selected_indices - (
            {self.selected_index} if self.selected_index is not None else set()
        )
        selectable_inactive_indices = {
            index
            for index in inactive_indices
            if isinstance(
                self.list_items[self.phrase_indices[index]],
                VoiceLinePhraseGroupItem,
            )
        }
        option_lists = self.query("#line-list")
        if option_lists:
            option_lists.first(NonWrappingOptionList).set_inactive_selection_indices(
                selectable_inactive_indices
            )

    def find_text(self, phrase_index: int) -> str:
        """Search phrase text and the complete generated section heading text."""
        list_item = self.list_items[phrase_index]
        if isinstance(list_item, VoiceLineSectionItem):
            return list_item.display_text
        return self.project.phrase_groups[list_item.phrase_index].presentable_text

    def action_assign_voice(self, voice_index: int) -> None:
        """Assign an available voice sample to all selected phrase groups."""
        if not self.content_initialized or voice_index >= self.voice_sample_count:
            return
        if self.selected_index is None or isinstance(
            self.list_items[self.phrase_indices[self.selected_index]],
            VoiceLineSectionItem,
        ):
            return

        def assign_voice(_visible_index: int, item_index: int) -> bool:
            list_item = self.list_items[item_index]
            if isinstance(list_item, VoiceLineSectionItem):
                return False
            phrase_index = list_item.phrase_index
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
            self.exit()
            return
        phrase_groups = self.project.phrase_groups
        if len(phrase_groups) != len(self.staged_voice_indices):
            self.save_error = "Save failed: project text changed while editing"
            self.exit()
            return

        for phrase_group, voice_index in zip(
            phrase_groups, self.staged_voice_indices, strict=True
        ):
            phrase_group.voice_index = voice_index

        try:
            error = ProjectTextIOUtil.save_phrase_groups(self.project)
        except Exception as exception:
            error = make_error_string(exception)
        if error:
            for phrase_group, voice_index in zip(
                phrase_groups, self.original_voice_indices, strict=True
            ):
                phrase_group.voice_index = voice_index
            self.save_error = f"Save failed: {error}"
        else:
            self.did_save_changes = True
        self.exit()

    def save_changes_and_exit(self) -> None:
        """Backward-compatible name for committing the staged voice values."""
        self.commit_changes_and_exit()

    @classmethod
    def start(cls, project: Project) -> None:
        """Run an editor for a project and report its save result."""
        if not cls.check_terminal_support():
            return
        app = cls(project)
        app.run(inline=False)
        if isinstance(app._exception, StylesheetError):
            print_feedback("Couldn't load textual css", is_error=True)
        elif app.save_error:
            print_feedback(app.save_error, is_error=True)
        elif app.did_save_changes:
            print_feedback("Saved changes", long_pause=True)
