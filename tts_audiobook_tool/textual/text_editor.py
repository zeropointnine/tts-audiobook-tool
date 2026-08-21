from dataclasses import dataclass
from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.widgets import TextArea as BaseTextArea

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_edit_util import (
    ProjectTextEditUtil,
)
from tts_audiobook_tool.text_ops.text_edit_session import (
    PhraseGroupSplitPoint,
    TextEditMutationResult,
    TextEditSession,
)
from tts_audiobook_tool.textual.content_textual_app import (
    ContentTextualApp,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.phrase_group_split_dialog import (
    PhraseGroupSplitDialog,
)
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    NonWrappingOptionList,
    OptionReconcileItem,
    STYLE_DIM,
)


SHOW_NEWLINE_CHARS = True

# Headers reused between normal mode and manual line editing mode.
_HEADER_LINES_DEFAULT = [
    f"{COL_ACCENT}View/edit text",
    f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]  - [CTRL-F] Find text",
    f"{COL_DIM}- Select multiple lines: [SHIFT] + navigation keys  - [CTRL-A] Select all  - [M] Enter manually",
    f"{COL_DIM}- Press [{COL_ACCENT}X{COL_DIM}] to delete selected lines   [S] Split line   [E] Edit line",
    f"{COL_DIM}- Press [ESC] to finish",
]
_HEADER_LINES_EDITING = [
    f"{COL_ACCENT}View/edit text",
    f"{COL_DIM}- Press [{COL_ACCENT}CTRL+ENTER{COL_DIM}] to confirm   [{COL_ACCENT}ESC{COL_DIM}] to cancel",
]


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
        return self.display_text


TextEditorListItem = TextEditorSectionItem | TextEditorPhraseGroupItem


class TextEditor(ContentTextualApp[EditorSaved | EditorSaveFailed]):

    CSS = ContentTextualApp.CSS + """
    TextArea .text-area--selection {
        color: $text;
        background: $accent 60%;
    }

    #edit-area {
        border: round #ffaa44;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        Binding("x", "delete_phrase_groups", show=False),
        Binding("s", "split_phrase_group", show=False),
        Binding("e", "edit_current_line", show=False),
        Binding("escape", "cancel_edit", show=False, priority=True),
        Binding("ctrl+enter", "confirm_edit", show=False, priority=True),
    ]

    def __init__(self, project: Project) -> None:
        self.project = project
        self.edit_session_or_none: TextEditSession | None = None
        self.section_items: list[TextEditorSectionItem] = []
        self.list_items: list[TextEditorListItem] = []

        # State of manual line editing (in-place with TextArea).
        self.editor_widget: BaseTextArea | None = None
        self.is_editing = False
        self.editing_index: int | None = None
        self._edit_original_text: str | None = None

        super().__init__(
            project,
            list(_HEADER_LINES_DEFAULT),
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

    @staticmethod
    def presentable_phrase_group_ansi(phrase_group: PhraseGroup) -> str:
        """Return presentable text with line feeds shown as dim literal tokens."""
        text = "".join(f"{phrase.text} " for phrase in phrase_group.phrases)
        text = text.replace("\r", " ")
        presentable_lines = [
            app_text.massage_post_normalize(line) for line in text.split("\n")
        ]
        newline_token = f"{COL_DIM}↵\N{NO-BREAK SPACE}{COL_DEFAULT}"
        return newline_token.join(presentable_lines)

    def format_line(self, index: int) -> HangingIndentText:
        """Format one row, styling selected/found/edited rows except headings."""
        item_index = self.phrase_indices[index]
        list_item = self.list_items[item_index]

        if isinstance(list_item, TextEditorSectionItem):
            return self.format_section_list_item(list_item.display_text, index)
        else:
            is_find_match = index == self.find_match_index
            is_being_edited = self.is_editing and index == self.editing_index
            if is_find_match:
                style = f"{STYLE_DIM} reverse"
            elif is_being_edited:
                style = "reverse"
            else:
                style = ""
            prefix_text = f"{list_item.ordinal:05d}  "
            presentable_text = (
                self.presentable_phrase_group_ansi(list_item.phrase_group)
                if SHOW_NEWLINE_CHARS
                else list_item.phrase_group.presentable_text
            )
            ansi_text = f"{COL_DIM}{prefix_text}{COL_DEFAULT}{presentable_text}"
            return HangingIndentText.from_ansi(
                ansi_text=ansi_text,
                content_start=len(prefix_text),
                max_lines=3,
                style=style,
            )

    def find_text(self, phrase_index: int) -> str:
        """Return searchable text from the staged row rather than the Project."""
        return self.list_items[phrase_index].searchable_text

    def option_id(self, index: int) -> str:
        """Use staged identities so options can survive structural mutations."""
        return self.stable_option_id(self.list_items[self.phrase_indices[index]])

    def content_line_index(self, item_index: int) -> int | None:
        """Map staged phrase rows by current ordinal, excluding section rows."""
        item = self.list_items[item_index]
        if isinstance(item, TextEditorSectionItem):
            return None
        return item.ordinal - 1

    def manual_selection_line_count(self) -> int:
        """Return the number of currently staged phrase-group lines."""
        return sum(
            isinstance(item, TextEditorPhraseGroupItem) for item in self.list_items
        )

    def apply_mutation_result(self, result: TextEditMutationResult) -> None:
        """Rebuild all derived rows and focus the mutation's surviving target."""
        edit_session = self.edit_session_or_none
        if not result.changed or edit_session is None:
            return
        old_items_by_id = {
            self.stable_option_id(item): item for item in self.list_items
        }
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
        # Reconciliation formats against the new projection before the shared
        # replacement helper installs its final selection state.
        self.phrase_indices = list(range(len(self.list_items)))
        self.replace_phrase_indices(
            self.phrase_indices,
            selected_visible_index,
            self.make_mutation_reconcile_items(old_items_by_id),
        )

    @staticmethod
    def stable_option_id(item: TextEditorListItem) -> str:
        """Return an option identity which survives ordinal and position changes."""
        item_kind = (
            "section" if isinstance(item, TextEditorSectionItem) else "phrase"
        )
        return f"text-{item_kind}-{item.item_id}"

    def make_mutation_reconcile_items(
        self,
        old_items_by_id: dict[str, TextEditorListItem],
    ) -> list[OptionReconcileItem]:
        """Describe changed prompts without reformatting unaffected editor rows."""
        items: list[OptionReconcileItem] = []
        for index, item in enumerate(self.list_items):
            option_id = self.stable_option_id(item)
            old_item = old_items_by_id.get(option_id)
            prompt_changed = old_item is None
            reflow = old_item is None
            if isinstance(item, TextEditorSectionItem):
                if not isinstance(old_item, TextEditorSectionItem):
                    prompt_changed = True
                    reflow = True
                elif item.display_text != old_item.display_text:
                    prompt_changed = True
                    reflow = True
            elif not isinstance(old_item, TextEditorPhraseGroupItem):
                prompt_changed = True
                reflow = True
            else:
                if item.phrase_group is not old_item.phrase_group:
                    prompt_changed = True
                    reflow = True
                elif item.ordinal != old_item.ordinal:
                    prompt_changed = True

            items.append(
                (
                    option_id,
                    self.format_line(index) if prompt_changed else None,
                    reflow,
                )
            )
        return items

    # ---------------------------------------------------
    # Manual line editing (in-place, with TextArea) 
    # ---------------------------------------------------

    def action_edit_current_line(self) -> None:
        """Open a TextArea for editing the currently selected line."""
        if self.is_editing:
            return
        if self.selected_index is None:
            return
        if self.find_active:
            return

        item = self.list_items[self.phrase_indices[self.selected_index]]

        # Block editing of SectionItem
        if isinstance(item, TextEditorSectionItem):
            return

        item_id = item.item_id
        phrase_group = self.edit_session.get_phrase_group(item_id)
        if phrase_group is None:
            return

        original_text = phrase_group.phrase_group.text

        # Normalize \r\n / \r to \n before handing text to the TextArea widget.
        # Pure \n round-trips through TextArea unchanged, but a *mixed*
        # \r\n/\n string gets coalesced by the widget to all \r\n - so we
        # normalize here and keep this same normalized value as the baseline
        # for the change-detection comparison in action_confirm_edit, instead
        # of re-reading the raw (un-normalized) phrase_group.text there.
        original_text = app_text.normalize_line_terminators(original_text)
        self._edit_original_text = original_text

        # Remove the previous widget before creating the new one (avoids DuplicateIds)
        if self.editor_widget:
            self.editor_widget.remove()
            self.editor_widget = None

        self.editor_widget = BaseTextArea(
            id="edit-area",
            text=original_text,
            theme="vscode_dark",
        )
        self.mount(self.editor_widget)

        # Enter editing mode. Keep editing_index/selected_index untouched so
        # the line being edited remains the one visually selected/highlighted.
        self.is_editing = True
        self.editing_index = self.selected_index

        # Refresh the line to apply the "reverse" style to the edited line
        self.refresh_line(self.editing_index)

        # Disable the OptionList so it can't respond to clicks or keyboard
        # navigation while editing is in progress.
        option_list = self.query_one("#line-list", NonWrappingOptionList)
        option_list.disabled = True
        option_list.highlighted = self.selected_index

        # Focus the TextArea last, to make sure it (and not the OptionList)
        # ends up holding keyboard focus.
        self.editor_widget.focus()

        # Defer the scroll until after layout has been recalculated with the
        # newly mounted TextArea taking up space, so the visible area used for
        # the scroll calculation is accurate.
        self.call_after_refresh(option_list.scroll_to_highlight, top=True)

        self.update_header(list(_HEADER_LINES_EDITING))

    def action_confirm_edit(self) -> None:
        """Confirm the text edit by applying the changes to the edit session."""
        if self.editor_widget is None or self.editing_index is None:
            return

        try:
            self.query_one("#edit-area")
        except Exception:
            return

        edited_text = self.editor_widget.text

        item = self.list_items[self.phrase_indices[self.editing_index]]
        if isinstance(item, TextEditorSectionItem):
            return

        item_id = item.item_id
        phrase_group = self.edit_session.get_phrase_group(item_id)
        if phrase_group is None:
            return

        # Use the same normalized baseline stored when the editor was opened,
        # instead of re-reading phrase_group.phrase_group.text raw - keeps
        # both sides of the comparison consistent with what the TextArea
        # widget can actually preserve unchanged.
        if self._edit_original_text is None:
            return
        original_text = self._edit_original_text

        if edited_text == original_text:
            # Text did not change, remove the widget and finish
            self._end_edit()
            return

        result = self.edit_session.update_phrase_group_text(
            item_id=item_id,
            new_text=edited_text,
        )

        # Apply the mutation to update list_items and refresh the UI
        self.apply_mutation_result(result)

        if self.editor_widget:
            self.editor_widget.remove()
            self.editor_widget = None

        self._end_edit()
        self.refresh()

    def action_cancel_edit(self) -> None:
        """Cancel the text edit by discarding the changes."""
        if self.editor_widget:
            self.editor_widget.remove()
            self.editor_widget = None
        self._end_edit()

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        """Disable app-level bindings that would otherwise shadow in-editor keys.

        With priority=True, these bindings would intercept keys unconditionally:
        - "escape" -> "cancel_edit" would shadow the inherited "escape" ->
        "quit_editor" binding from ContentTextualApp, preventing the TUI from
        closing when not editing.
        - "ctrl+a" -> "select_all" would prevent Ctrl+A from reaching the
        focused TextArea. Note: Textual's TextArea binds Ctrl+A to
        cursor_line_start by default (readline-style), not select-all, so we
        explicitly call the TextArea's own select_all() here to get
        VS Code-style "select all" behavior instead.
        Also, "ctrl+a" behaves weirdly here. To make it work as commonly intended
        ("selec all") is necessary to "ctrl+a" and then "ctrl+c" (only tested on Windows).

        Returning False tells Textual to skip the binding, letting the key fall
        through to the next candidate (e.g. the TextArea's own bindings).
        """
        if action == "cancel_edit" and not self.is_editing:
            return False
        if action == "select_all" and self.is_editing:
            if self.editor_widget:
                self.editor_widget.select_all()
            return False
        return True

    def _end_edit(self) -> None:
        """End editing the current line."""
        if self.editor_widget:
            self.editor_widget.remove()
            self.editor_widget = None

        previously_editing_index = self.editing_index
        self.is_editing = False
        self.editing_index = None
        self._edit_original_text = None

        # Refresh the line to remove the "reverse" style
        if previously_editing_index is not None:
            self.refresh_line(previously_editing_index)

        # Re-enable the OptionList and restore focus/highlight to it.
        try:
            option_list = self.query_one("#line-list", NonWrappingOptionList)
            option_list.disabled = False
            if self.selected_index is not None:
                option_list.highlighted = self.selected_index
            self.set_focus(option_list)
        except Exception:
            pass

        self.update_header(list(_HEADER_LINES_DEFAULT))

    def on_key(self, event: events.Key) -> None:
        """Prevent TextArea from consuming CTRL+ENTER/ESC/CTRL+F when in edit mode."""
        if self.is_editing:
            if event.key in ("ctrl+return", "ctrl+enter", "ctrl+j", "escape", "ctrl+f"):
                event.prevent_default()
                if event.key == "escape" and self.editor_widget:
                    self.action_cancel_edit()
                elif event.key in ("ctrl+return", "ctrl+enter", "ctrl+j") and self.editor_widget:
                    self.action_confirm_edit()

    # ---------------------------------------------------

    def action_delete_phrase_groups(self) -> None:
        """Delete phrase rows, or one section when it is the sole selected row."""
        edit_session = self.edit_session_or_none
        if self.is_editing or self.find_active or edit_session is None:
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
            self.set_toast_text(f"{result.deleted_count} {line_noun} deleted")

    def action_split_phrase_group(self) -> None:
        """Open a boundary chooser for exactly one selected phrase-group row."""
        if self.is_editing or self.find_active or self.edit_session_or_none is None:
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
        """Warn about generated audio and markers invalidated by the staged edit."""
        edit_session = self.edit_session_or_none
        if edit_session is None:
            return SaveChangesDialog()
        first_index = edit_session.earliest_affected_original_index
        if first_index is None:
            return SaveChangesDialog()
        segment_count = len(
            self.project.sound_segments.snapshot_paths_from_index(first_index)
        )
        marker_count = len(
            [marker for marker in self.project.markers if marker >= first_index]
        )
        if segment_count == 0 and marker_count == 0:
            return SaveChangesDialog()
        deletion_parts: list[str] = []
        if segment_count > 0:
            segment_word = "segment" if segment_count == 1 else "segments"
            deletion_parts.append(
                f"{segment_count} generated sound {segment_word}"
            )
        if marker_count > 0:
            marker_label = app_text.get_section_marker_label(
                self.project,
                is_title_case=False,
                is_singular=marker_count == 1,
            )
            deletion_parts.append(f"{marker_count} {marker_label}")
        warning_text = (
            f"Saving these changes requires deleting "
            f"{' and '.join(deletion_parts)} "
            f"from line {first_index + 1} onward."
        )
        return SaveChangesDialog(
            [
                "Save changes before exiting?",
                "",
                f"{COL_ERROR}{warning_text}",
            ]
        )

    def commit_changes_and_exit(self) -> None:
        """Commit the detached Book and remove generated audio invalidated by it."""
        edit_session = self.edit_session_or_none
        if edit_session is None:
            self.exit(EditorSaveFailed("Text editor was not initialized"))
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
            result = EditorSaveFailed(f"Save failed: {error}")
        else:
            result = EditorSaved()
        self.exit(result)

    def save_changes_and_exit(self) -> None:
        """Backward-compatible name for committing the staged voice values."""
        self.commit_changes_and_exit()