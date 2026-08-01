from dataclasses import dataclass
from typing import ClassVar

from textual.binding import Binding, BindingType
from textual.css.errors import StylesheetError

from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_edit_util import (
    ProjectTextEditUtil,
)
from tts_audiobook_tool.text_ops.text_edit_session import (
    PhraseGroupSplitPoint,
    TextEditMutationResult,
    TextEditSession,
)
from tts_audiobook_tool.textual.content_textual_app import ContentTextualApp
from tts_audiobook_tool.textual.phrase_group_split_dialog import (
    PhraseGroupSplitDialog,
)
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    NonWrappingOptionList,
    STYLE_DIM,
)
from tts_audiobook_tool.util import print_feedback


@dataclass
class TextEditorPhraseGroupItem:
    """A staged phrase group and its one-based ordinal in the whole book."""

    phrase_group: PhraseGroup
    ordinal: int
    item_id: int = -1

    @property
    def searchable_text(self) -> str:
        return self.phrase_group.presentable_text


@dataclass
class TextEditorSectionItem:
    """A staged section which owns its editable phrase-group items."""

    ordinal: int
    section_count: int
    title: str
    phrase_group_items: list[TextEditorPhraseGroupItem]
    item_id: int = -1

    @property
    def display_text(self) -> str:
        title_text = f": {self.title}" if self.title else ""
        line_count = len(self.phrase_group_items)
        line_noun = "line" if line_count == 1 else "lines"
        return (
            f"Section {self.ordinal}/{self.section_count}{title_text} "
            f"({line_count} {line_noun})"
        )

    @property
    def searchable_text(self) -> str:
        return self.title


TextEditorListItem = TextEditorSectionItem | TextEditorPhraseGroupItem


class TextEditor(ContentTextualApp):

    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        Binding("x", "delete_phrase_groups", show=False),
        Binding("s", "split_phrase_group", show=False),
    ]

    def __init__(
        self,
        project: Project
    ) -> None:
        self.project = project
        self.save_error = ""
        self.did_save_changes = False
        self.edit_session_or_none: TextEditSession | None = None
        self.section_items: list[TextEditorSectionItem] = []
        self.list_items: list[TextEditorListItem] = []
        header_lines = [
            f"{COL_ACCENT}View/edit text",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]",
            f"{COL_DIM}- Select multiple lines by holding [SHIFT] + navigation keys",
            f"{COL_DIM}- Press [{COL_ACCENT}X{COL_DIM}] to delete selected lines   [S] Split line",
            f"{COL_DIM}- Press [ESC] to finish   - [CTRL-F] Find text"
        ]
        super().__init__(
            project,
            header_lines,
            empty_state_text="No text lines",
            loading_state_text="...",
        )

    def initialize_content(self) -> range:
        """Construct the complete editable row model after first draw."""
        if self.edit_session_or_none is not None:
            return range(len(self.list_items))
        edit_session = TextEditSession(self.project.book)
        section_items = self.make_section_items_from_session(edit_session)
        list_items = self.make_list_items(section_items)

        self.edit_session_or_none = edit_session
        self.section_items = section_items
        self.list_items = list_items
        return range(len(list_items))

    @property
    def edit_session(self) -> TextEditSession:
        """Return the loaded edit model to code which requires initialization."""
        edit_session = self.edit_session_or_none
        if edit_session is None:
            raise RuntimeError("Text editor has not finished loading")
        return edit_session

    @property
    def has_changes(self) -> bool:
        edit_session = self.edit_session_or_none
        return edit_session is not None and edit_session.has_changes

    @property
    def selection_status_text(self) -> str:
        """Count only selected phrase rows, excluding section headings."""
        selection_count = sum(
            isinstance(
                self.list_items[self.phrase_indices[index]],
                TextEditorPhraseGroupItem,
            )
            for index in self.selected_indices
        )
        return f"{selection_count} lines selected" if selection_count >= 2 else ""

    @staticmethod
    def make_section_items(project: Project) -> list[TextEditorSectionItem]:
        """Build a detached hierarchy which can later support staged text edits."""
        return TextEditor.make_section_items_from_session(TextEditSession(project.book))

    @staticmethod
    def make_section_items_from_session(
        edit_session: TextEditSession,
    ) -> list[TextEditorSectionItem]:
        """Project canonical staged hierarchy into newly numbered UI items."""
        section_count = len(edit_session.sections)
        next_phrase_ordinal = 1
        section_items: list[TextEditorSectionItem] = []
        for section_ordinal, section in enumerate(edit_session.sections, start=1):
            phrase_group_items: list[TextEditorPhraseGroupItem] = []
            for phrase_group in section.phrase_groups:
                phrase_group_items.append(
                    TextEditorPhraseGroupItem(
                        phrase_group=phrase_group.phrase_group,
                        ordinal=next_phrase_ordinal,
                        item_id=phrase_group.item_id,
                    )
                )
                next_phrase_ordinal += 1
            section_items.append(
                TextEditorSectionItem(
                    ordinal=section_ordinal,
                    section_count=section_count,
                    title=section.title,
                    phrase_group_items=phrase_group_items,
                    item_id=section.item_id,
                )
            )
        return section_items

    @staticmethod
    def make_list_items(
        section_items: list[TextEditorSectionItem],
    ) -> list[TextEditorListItem]:
        """Project staged sections into the ordered rows shown by the editor."""
        if not any(section.phrase_group_items for section in section_items):
            return []
        show_sections = len(section_items) > 1
        list_items: list[TextEditorListItem] = []
        for section_item in section_items:
            if show_sections:
                list_items.append(section_item)
            list_items.extend(section_item.phrase_group_items)
        return list_items

    def format_line(self, index: int) -> HangingIndentText:
        """Format one row, styling selected rows except for the active row."""
        item_index = self.phrase_indices[index]
        list_item = self.list_items[item_index]
        is_find_match = index == self.find_match_index
        style = f"{STYLE_DIM} reverse" if is_find_match else ""

        if isinstance(list_item, TextEditorSectionItem):
            return HangingIndentText.from_ansi(
                # Rich's line-height measurement doesn't count a final empty
                # line, so two trailing newlines are needed to render one.
                f"\n{COL_ACCENT}{list_item.display_text}\n\n",
                content_start=0,
                max_lines=3,
                style=style,
            )

        prefix_text = f"{list_item.ordinal:05d}  "
        ansi_text = f"{COL_DIM}{prefix_text}{COL_DEFAULT}{list_item.phrase_group.presentable_text}"
        return HangingIndentText.from_ansi(
            ansi_text=ansi_text,
            content_start=len(prefix_text),
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
            if isinstance(self.list_items[index], TextEditorPhraseGroupItem)
        }
        option_lists = self.query("#line-list")
        if option_lists:
            option_lists.first(NonWrappingOptionList).set_inactive_selection_indices(
                selectable_inactive_indices
            )

    def find_text(self, phrase_index: int) -> str:
        """Return searchable text from the staged row rather than the Project."""
        return self.list_items[phrase_index].searchable_text

    def apply_mutation_result(self, result: TextEditMutationResult) -> None:
        """Rebuild all derived rows and focus the mutation's surviving target."""
        edit_session = self.edit_session_or_none
        if not result.changed or edit_session is None:
            return
        self.section_items = self.make_section_items_from_session(edit_session)
        self.list_items = self.make_list_items(self.section_items)
        selected_visible_index = next(
            (
                index
                for index, item in enumerate(self.list_items)
                if (
                    isinstance(item, TextEditorPhraseGroupItem)
                    and item.item_id == result.focus_item_id
                )
            ),
            0 if self.list_items else None,
        )
        self.replace_phrase_indices(
            range(len(self.list_items)),
            selected_visible_index,
        )

    def action_delete_phrase_groups(self) -> None:
        """Delete phrase rows, or one section when it is the sole selected row."""
        edit_session = self.edit_session_or_none
        if self.find_active or edit_session is None:
            return
        selected_items = [
            self.list_items[self.phrase_indices[index]]
            for index in sorted(self.selected_indices)
        ]
        if len(selected_items) == 1 and isinstance(
            selected_items[0], TextEditorSectionItem
        ):
            result = edit_session.delete_section(selected_items[0].item_id)
        else:
            item_ids = {
                item.item_id
                for item in selected_items
                if isinstance(item, TextEditorPhraseGroupItem)
            }
            result = edit_session.delete_phrase_groups(item_ids)
        self.apply_mutation_result(result)
        if result.deleted_count:
            line_noun = "line" if result.deleted_count == 1 else "lines"
            self.show_transient_status(
                f"{result.deleted_count} {line_noun} deleted"
            )

    def action_split_phrase_group(self) -> None:
        """Open a boundary chooser for exactly one selected phrase-group row."""
        if self.find_active or self.edit_session_or_none is None:
            return
        phrase_items = [
            item
            for index in sorted(self.selected_indices)
            if isinstance(
                item := self.list_items[self.phrase_indices[index]],
                TextEditorPhraseGroupItem,
            )
        ]
        if len(phrase_items) != 1 or len(phrase_items[0].phrase_group.phrases) < 2:
            return
        item = phrase_items[0]
        self.push_screen(
            PhraseGroupSplitDialog(item.phrase_group),
            lambda split_point: self.handle_split_point(item.item_id, split_point),
        )

    def handle_split_point(
        self,
        item_id: int,
        split_point: PhraseGroupSplitPoint | None,
    ) -> None:
        """Apply a chosen split point when the chooser was not cancelled."""
        if split_point is None:
            return
        edit_session = self.edit_session_or_none
        if edit_session is None:
            return
        self.apply_mutation_result(
            edit_session.split_phrase_group(item_id, split_point)
        )

    def make_confirmation_dialog(self) -> SaveChangesDialog:
        """Warn about generated audio invalidated by the staged text edit."""
        edit_session = self.edit_session_or_none
        if edit_session is None:
            return SaveChangesDialog()
        first_index = edit_session.earliest_affected_original_index
        if first_index is None:
            return SaveChangesDialog()
        segment_count = len(
            self.project.sound_segments.snapshot_paths_from_index(first_index)
        )
        if segment_count == 0:
            return SaveChangesDialog()
        segment_word = "segment" if segment_count == 1 else "segments"
        warning_text = (
            f"Saving these changes requires deleting {segment_count} generated sound "
            f"{segment_word} from line {first_index + 1} onward."
        )
        return SaveChangesDialog(warning_text=warning_text)

    def commit_changes_and_exit(self) -> None:
        """Commit the detached Book and remove generated audio invalidated by it."""
        edit_session = self.edit_session_or_none
        if edit_session is None:
            self.exit()
            return
        error = ProjectTextEditUtil.commit(
            project=self.project,
            staged_book=edit_session.to_book(),
            original_snapshot=edit_session.original_snapshot,
            earliest_affected_original_index=(
                edit_session.earliest_affected_original_index
            ),
        )
        if error:
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
