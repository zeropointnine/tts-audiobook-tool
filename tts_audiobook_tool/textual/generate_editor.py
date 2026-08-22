from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from textual.binding import Binding, BindingType
from tts_audiobook_tool import readiness, text_util
from tts_audiobook_tool.app_types import SoundSegment
from tts_audiobook_tool.constants import (
    COL_ACCENT,
    COL_DEFAULT,
    COL_DIM,
    COL_ERROR,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.segment_transcript_util import (
    SegmentTranscriptUtil,
)
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.textual.alert_dialog import AlertDialog
from tts_audiobook_tool.textual.content_textual_app import (
    ContentTextualApp,
    EditorClosed,
    EditorSaveFailed,
)
from tts_audiobook_tool.textual.filter_dialog import FilterDialog
from tts_audiobook_tool.textual.save_changes_dialog import (
    ExitDecision,
    SaveChangesDialog,
)
from tts_audiobook_tool.textual.segment_info_dialog import (
    SegmentInfoAction,
    SegmentInfoDialog,
)
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    OptionReconcileItem,
    STYLE_DIM,
)
from tts_audiobook_tool.system_support.ansi import Ansi


class FilterType(tuple[str, str], Enum):
    ALL = "Show all lines", "all"
    UNGENERATED = "Show ungenerated lines", "ungenerated"
    GENERATED = "Show generated lines", "generated"
    GENERATED_WITH_ERRORS = (
        "Show generated lines with any errors",
        "generated w/ errors",
    )
    FAILED = "Show generated lines with errors, flagged as Failed", "generated/failed"

    @property
    def menu_label(self) -> str:
        return self.value[0]

    @property
    def value_label(self) -> str:
        return self.value[1]


@dataclass(frozen=True)
class PhraseSegmentStatus:
    """Best generated segment and its derived state for one project phrase."""

    project: Project
    phrase_index: int
    best_segment: SoundSegment | None

    @property
    def is_generated(self) -> bool:
        return self.best_segment is not None

    @property
    def has_errors(self) -> bool:
        return self.best_segment is not None and self.best_segment.num_errors > 0

    @property
    def is_failed(self) -> bool:
        return self.best_segment is not None and (
            self.project.sound_segments.is_segment_failed(
                self.phrase_index, self.best_segment
            )
        )


@dataclass(frozen=True)
class GeneratePhraseGroupItem:
    """An actionable project phrase displayed by the generation editor."""

    phrase_index: int


@dataclass(frozen=True)
class GenerateSectionItem:
    """A structural section heading for the phrases visible under a filter."""

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


GenerateListItem = GenerateSectionItem | GeneratePhraseGroupItem


@dataclass(frozen=True)
class QuickGenerationRequested:
    """One phrase should be regenerated before reopening the editor."""

    phrase_index: int
    save_error: str = ""


GenerateEditorResult = QuickGenerationRequested | EditorSaveFailed


class GenerateEditor(ContentTextualApp[GenerateEditorResult]):
    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        Binding("space", "toggle_queued", show=False),
        Binding("p", "play_sound", show=False),
        Binding("q", "quick_generate", show=False),
        Binding("i", "show_info", show=False),
        Binding("f", "show_filter", show=False),
        Binding("x", "delete_generated", show=False),
    ]

    def __init__(
        self,
        state: State,
        quick_gen_index: int | None = None,
    ) -> None:

        self.state = state
        self.quick_gen_restore_phrase_index = quick_gen_index
        self.all_phrase_indices: list[int] = []
        self.list_items: list[GenerateListItem] = []
        self.original_queued_indices: set[int] = set()
        self.staged_queued_indices: set[int] = set()
        self.generated_indices: set[int] = set()
        self.generated_with_errors_indices: set[int] = set()
        self.failed_indices: set[int] = set()
        self.ungenerated_indices: set[int] = set()
        self.queued_ungenerated_count = 0
        self.phrase_segment_statuses: dict[int, PhraseSegmentStatus] = {}
        self.presentable_texts: dict[int, str] = {}
        self.filter_type = FilterType.ALL
        self.playing_sound_id = ""
        self.playing_sound_path = ""
        self.playing_phrase_index: int | None = None

        super().__init__(
            state.project,
            self.make_editor_header_lines(),
            empty_state_text="No lines",
            loading_state_text="...",
        )

    def make_filter_status_ansi(self) -> str:
        """Return the active-filter suffix displayed in the editor header."""
        if self.filter_type == FilterType.ALL:
            return ""
        return f" (currently: {COL_ACCENT}{self.filter_type.value_label}{COL_DIM})"

    def make_editor_header_lines(self) -> list[str]:
        """Build header copy that reflects the active line filter."""

        filter_status = self.make_filter_status_ansi()
        return [
            f"{COL_ACCENT}Generate - Select lines / review sound segments",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]  - [CTRL-F] Find text",
            f"{COL_DIM}- Select multiple lines: [SHIFT] + navigation keys  - [CTRL-A] Select all  - [M] Enter manually",
            f"{COL_DIM}- Press [{COL_ACCENT}SPACE{COL_DIM}] to queue or unqueue lines  - [F] Filter lines{filter_status}",
            f"{COL_DIM}- Sound segments: [P] Play  [X] Delete sound  [Q] Quick gen  [I] Info",
            f"{COL_DIM}- Press [ESC] to close",
        ]

    def initialize_content(self) -> list[int]:
        """Show every phrase group and load queued flags from the project range."""
        self.all_phrase_indices = list(range(len(self.project.phrase_groups)))
        queued_indices = ProjectUtil.get_indices_to_generate(self.project)
        self.original_queued_indices = queued_indices & set(self.all_phrase_indices)
        self.staged_queued_indices = set(self.original_queued_indices)
        self.refresh_phrase_classifications()
        self.update_queued_status()
        return self.get_filtered_phrase_indices()

    def initial_selected_phrase_index(self) -> int | None:
        """Restore the phrase selected before a completed quick generation."""
        return self.item_index_for_phrase(self.quick_gen_restore_phrase_index)

    def on_content_loaded(self) -> None:
        """Play a restored quick-generation item after its row is mounted."""
        # The snapshot accelerates construction of every initial Option. Later
        # row refreshes query the live catalog, while the derived index sets are
        # retained for queue operations.
        self.phrase_segment_statuses.clear()
        if self.quick_gen_restore_phrase_index is not None and self.is_running:
            self.call_after_refresh(self.play_quick_generated_item)

    def play_quick_generated_item(self) -> None:
        """Play the quick-generated item after its restored row is mounted."""
        phrase_index = self.quick_gen_restore_phrase_index
        self.quick_gen_restore_phrase_index = None
        if (
            phrase_index is None
            or self.selected_index is None
            or self.highlighted_content_line_index() != phrase_index
        ):
            return
        self.action_play_sound()

    def update_queued_status(self) -> None:
        """Show how many ungenerated lines are currently queued."""
        all_text = (
            " (all)"
            if self.ungenerated_indices
            and self.queued_ungenerated_count == len(self.ungenerated_indices)
            else ""
        )
        status_text = (
            f"{self.queued_ungenerated_count} lines queued for generation{all_text}"
        )
        if self.is_running:
            self.set_pinned_text(status_text)
        else:
            self.pinned_text = status_text

    @property
    def find_label_text(self) -> str:
        return "Search text: "

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(0.1, self.update_playback_status)

    def on_unmount(self) -> None:
        """Stop playback started by this editor when the editor closes."""
        self.stop_tracked_playback()

    def clear_playback_status(self) -> None:
        """Clear the editor's tracked playback and its status text."""
        self.playing_sound_id = ""
        self.playing_sound_path = ""
        self.playing_phrase_index = None
        self.update_selection_status()

    def stop_tracked_playback(self) -> None:
        """Stop playback owned by this editor and clear its tracked state."""
        if (
            self.playing_sound_id
            and PlaySoundUtil.current_sound_id() == self.playing_sound_id
        ):
            PlaySoundUtil.stop_sound_async()
        self.clear_playback_status()

    def update_playback_status(self) -> None:
        """Clear playback status once the tracked sound has finished."""
        if not self.playing_sound_id:
            return
        if PlaySoundUtil.current_sound_id() != self.playing_sound_id:
            self.clear_playback_status()

    def show_playback_status(self) -> None:
        """Show the currently tracked phrase without querying playback state."""
        if self.playing_phrase_index is not None:
            self.set_selected_text(f"Playing line {self.playing_phrase_index + 1}")

    def update_selection_status(self) -> None:
        """Keep playback visible, otherwise show the current selection count."""
        if self.playing_phrase_index is not None:
            self.show_playback_status()
        else:
            super().update_selection_status()

    def get_phrase_segment_status(self, phrase_index: int) -> PhraseSegmentStatus:
        """Return the best segment and derived state for a project phrase."""
        cached_status = self.phrase_segment_statuses.get(phrase_index)
        if cached_status is not None:
            return cached_status
        return PhraseSegmentStatus(
            project=self.project,
            phrase_index=phrase_index,
            best_segment=self.project.sound_segments.get_best_item_for(phrase_index),
        )

    def refresh_phrase_classifications(self) -> None:
        """Snapshot segment state and rebuild index sets in one project pass."""
        statuses = {
            phrase_index: PhraseSegmentStatus(
                project=self.project,
                phrase_index=phrase_index,
                best_segment=self.project.sound_segments.get_best_item_for(
                    phrase_index
                ),
            )
            for phrase_index in self.all_phrase_indices
        }
        self.phrase_segment_statuses = statuses
        self.generated_indices = {
            phrase_index
            for phrase_index, status in statuses.items()
            if status.is_generated
        }
        self.generated_with_errors_indices = {
            phrase_index
            for phrase_index, status in statuses.items()
            if status.has_errors
        }
        self.failed_indices = {
            phrase_index
            for phrase_index, status in statuses.items()
            if status.is_failed
        }
        self.ungenerated_indices = set(self.all_phrase_indices) - self.generated_indices
        self.queued_ungenerated_count = len(
            self.staged_queued_indices & self.ungenerated_indices
        )

    def mark_phrases_ungenerated(self, phrase_indices: set[int]) -> None:
        """Update cached classifications after generated files are deleted."""
        newly_ungenerated = phrase_indices & self.generated_indices
        self.generated_indices.difference_update(phrase_indices)
        self.generated_with_errors_indices.difference_update(phrase_indices)
        self.failed_indices.difference_update(phrase_indices)
        self.ungenerated_indices.update(phrase_indices)
        self.queued_ungenerated_count += len(
            newly_ungenerated & self.staged_queued_indices
        )
        for phrase_index in phrase_indices:
            self.phrase_segment_statuses.pop(phrase_index, None)

    def get_highlighted_phrase_index(self) -> int | None:
        """Return the highlighted project phrase when content actions are available."""
        if (
            not self.content_initialized
            or self.find_active
            or self.selected_index is None
        ):
            return None
        return self.highlighted_content_line_index()

    def make_phrase_status_ansi(
        self,
        phrase_index: int,
        segment_status: PhraseSegmentStatus,
    ) -> str:
        """Build the generated or queued status displayed before a phrase."""
        best_segment = segment_status.best_segment
        if best_segment is None:
            should_queue = phrase_index in self.staged_queued_indices
            queued_text = "Queued   " if should_queue else "         "
            queued_color = COL_ACCENT if should_queue else ""
            return f"{COL_DIM}[{queued_color}{queued_text}{COL_DIM}]"

        status_ansi = f"{COL_DIM}[generated]"
        if best_segment.num_errors == -1:
            return status_ansi
        failed_marker = (
            f" {COL_ERROR}*{COL_DEFAULT}" if segment_status.is_failed else ""
        )
        return f"{status_ansi} {COL_DIM}[word errors: {best_segment.num_errors}{failed_marker}{COL_DIM}]"

    def format_line(self, index: int) -> HangingIndentText:
        list_item = self.list_items[self.phrase_indices[index]]
        if isinstance(list_item, GenerateSectionItem):
            return self.format_section_list_item(list_item.display_text, index)

        is_find_match = index == self.find_match_index
        style = f"{STYLE_DIM} reverse" if is_find_match else ""
        phrase_index = list_item.phrase_index
        segment_status = self.get_phrase_segment_status(phrase_index)
        status_ansi = self.make_phrase_status_ansi(phrase_index, segment_status)
        prefix_ansi = f"{COL_DIM}{self.format_line_number(phrase_index + 1)} {status_ansi} "
        presentable_text = self.presentable_texts.get(phrase_index)
        if presentable_text is None:
            presentable_text = self.project.phrase_groups[phrase_index].presentable_text
            self.presentable_texts[phrase_index] = presentable_text
        return HangingIndentText.from_ansi_prefix(
            f"{prefix_ansi}{Ansi.RESET}",
            presentable_text,
            max_lines=3,
            style=style,
        )

    def option_id(self, index: int) -> str:
        """Use project identities so options survive filter and status changes."""
        return self.stable_option_id(self.list_items[self.phrase_indices[index]])

    @staticmethod
    def stable_option_id(item: GenerateListItem) -> str:
        """Return a stable identity for one phrase or structural section row."""
        if isinstance(item, GenerateSectionItem):
            return f"generate-section-{item.ordinal}"
        return f"generate-phrase-{item.phrase_index}"

    def make_reconcile_items(
        self,
        old_items_by_id: dict[str, GenerateListItem],
        changed_phrase_indices: set[int] | None = None,
    ) -> list[OptionReconcileItem]:
        """Format only new rows and rows whose visible generation state changed."""
        changed_phrase_indices = changed_phrase_indices or set()
        items: list[OptionReconcileItem] = []
        for visible_index, item_index in enumerate(self.phrase_indices):
            item = self.list_items[item_index]
            option_id = self.stable_option_id(item)
            old_item = old_items_by_id.get(option_id)
            prompt_changed = old_item is None
            reflow = old_item is None
            if isinstance(item, GenerateSectionItem):
                if not isinstance(old_item, GenerateSectionItem):
                    prompt_changed = True
                    reflow = True
                elif item.display_text != old_item.display_text:
                    prompt_changed = True
                    reflow = True
            elif item.phrase_index in changed_phrase_indices:
                prompt_changed = True
                reflow = True

            items.append(
                (
                    option_id,
                    self.format_line(visible_index) if prompt_changed else None,
                    reflow,
                )
            )
        return items

    def visible_items_by_id(self) -> dict[str, GenerateListItem]:
        """Snapshot only rows which currently have mounted list options."""
        return {
            self.stable_option_id(self.list_items[item_index]): self.list_items[
                item_index
            ]
            for item_index in self.phrase_indices
        }

    def replace_filtered_phrase_indices(
        self,
        phrase_indices: list[int],
        selected_item_index: int | None,
        old_items_by_id: dict[str, GenerateListItem],
        changed_phrase_indices: set[int] | None = None,
    ) -> None:
        """Install rebuilt filter rows while retaining unchanged option content."""
        self.phrase_indices = phrase_indices
        self.replace_phrase_indices(
            phrase_indices,
            selected_item_index,
            self.make_reconcile_items(old_items_by_id, changed_phrase_indices),
        )

    def find_text_strings(self, item_index: int) -> Sequence[str]:
        """Search visible phrase text and complete section heading text."""
        item = self.list_items[item_index]
        if isinstance(item, GenerateSectionItem):
            return [item.display_text]
        return [
            self.format_line_number(item.phrase_index + 1),
            self.project.phrase_groups[item.phrase_index].presentable_text,
        ]

    def content_line_index(self, item_index: int) -> int | None:
        """Map phrase rows to Project lines while excluding section headings."""
        item = self.list_items[item_index]
        if isinstance(item, GenerateSectionItem):
            return None
        return item.phrase_index

    def item_index_for_phrase(self, phrase_index: int | None) -> int | None:
        """Return the current backing item for a visible project phrase."""
        if phrase_index is None:
            return None
        return next(
            (
                item_index
                for item_index, item in enumerate(self.list_items)
                if isinstance(item, GeneratePhraseGroupItem)
                and item.phrase_index == phrase_index
            ),
            None,
        )

    def apply_queue_state_to_selected_phrases(self, should_queue: bool) -> None:
        """Queue or unqueue selected phrases, leaving generated phrases unqueued."""

        def set_flag(_visible_index: int, phrase_index: int) -> bool:
            was_queued = phrase_index in self.staged_queued_indices
            new_flag = False if phrase_index in self.generated_indices else should_queue
            if was_queued == new_flag:
                return False
            if new_flag:
                self.staged_queued_indices.add(phrase_index)
            else:
                self.staged_queued_indices.discard(phrase_index)
            if phrase_index in self.ungenerated_indices:
                self.queued_ungenerated_count += 1 if new_flag else -1
            return True

        changed_indices = self.mutate_selected_items(set_flag, reflow=False)
        if changed_indices:
            self.update_queued_status()

    def action_toggle_queued(self) -> None:
        """Toggle selected ungenerated rows and clear generated rows' flags."""
        if (
            not self.content_initialized
            or self.find_active
            or not self.selected_indices
        ):
            return
        selected_phrase_indices = self.selected_content_line_indices()
        if not selected_phrase_indices:
            return
        all_selected_ungenerated_are_marked = all(
            phrase_index in self.staged_queued_indices
            for phrase_index in selected_phrase_indices
            if phrase_index in self.ungenerated_indices
        )
        self.apply_queue_state_to_selected_phrases(
            not all_selected_ungenerated_are_marked
        )

    def action_show_filter(self) -> None:
        """Show the line-filter selection dialog."""
        if not self.content_initialized or self.find_active:
            return
        self.refresh_phrase_classifications()
        line_counts = {
            FilterType.ALL: len(self.all_phrase_indices),
            FilterType.UNGENERATED: len(self.ungenerated_indices),
            FilterType.GENERATED: len(self.generated_indices),
            FilterType.GENERATED_WITH_ERRORS: len(self.generated_with_errors_indices),
            FilterType.FAILED: len(self.failed_indices),
        }
        self.push_screen(
            FilterDialog(self.filter_type, line_counts),
            self.handle_filter_selection,
        )

    def handle_filter_selection(self, filter_type: FilterType | None) -> None:
        """Apply a selected filter immediately; ignore dialog cancellation."""
        if filter_type is None:
            return
        selected_phrase_index = self.highlighted_content_line_index()
        old_items_by_id = self.visible_items_by_id()
        self.filter_type = filter_type
        self.update_header(self.make_editor_header_lines())
        phrase_indices = self.get_filtered_phrase_indices()
        selected_item_index = self.item_index_for_phrase(selected_phrase_index)
        self.replace_filtered_phrase_indices(
            phrase_indices,
            selected_item_index if selected_item_index in phrase_indices else None,
            old_items_by_id,
        )
        self.phrase_segment_statuses.clear()

    def get_filtered_phrase_indices(
        self, filter_type: FilterType | None = None
    ) -> list[int]:
        """Rebuild rows and return backing indices matching the active filter."""
        filter_type = filter_type or self.filter_type
        if filter_type == FilterType.ALL:
            matching_indices = set(self.all_phrase_indices)
        else:
            matching_indices = {
                FilterType.UNGENERATED: self.ungenerated_indices,
                FilterType.GENERATED: self.generated_indices,
                FilterType.GENERATED_WITH_ERRORS: self.generated_with_errors_indices,
                FilterType.FAILED: self.failed_indices,
            }[filter_type]
        matching_phrase_indices = [
            phrase_index
            for phrase_index in self.all_phrase_indices
            if phrase_index in matching_indices
        ]
        book = getattr(self.project, "book", None)
        if book is None:
            self.list_items = [
                GeneratePhraseGroupItem(phrase_index)
                for phrase_index in self.all_phrase_indices
            ]
            return matching_phrase_indices

        show_sections = len(book.sections) > 1
        section_count = len(book.sections)
        next_phrase_index = 0
        list_items: list[GenerateListItem] = []
        for ordinal, section in enumerate(book.sections, start=1):
            section_phrase_indices = range(
                next_phrase_index,
                next_phrase_index + len(section.phrase_groups),
            )
            visible_phrase_indices = [
                phrase_index
                for phrase_index in section_phrase_indices
                if phrase_index in matching_indices
            ]
            next_phrase_index += len(section.phrase_groups)
            if not visible_phrase_indices:
                continue
            if show_sections:
                list_items.append(
                    GenerateSectionItem(
                        ordinal=ordinal,
                        section_count=section_count,
                        title=section.title,
                        line_count=len(visible_phrase_indices),
                    )
                )
            list_items.extend(
                GeneratePhraseGroupItem(phrase_index)
                for phrase_index in visible_phrase_indices
            )
        self.list_items = list_items
        return list(range(len(list_items)))

    def action_play_sound(self) -> None:
        """Play the highlighted segment, or stop it when already playing."""
        phrase_index = self.get_highlighted_phrase_index()
        if phrase_index is None:
            return

        self.play_sound_for_phrase(phrase_index)

    def play_sound_for_phrase(self, phrase_index: int) -> None:
        """Play a phrase's segment, or stop it when already playing."""
        sound_segment = self.get_phrase_segment_status(phrase_index).best_segment
        if sound_segment is None:
            return
        sound_path = str(
            Path(self.project.sound_segments_path) / sound_segment.file_name
        )
        current_sound_id = PlaySoundUtil.current_sound_id()
        if (
            sound_path == self.playing_sound_path
            and current_sound_id
            and current_sound_id == self.playing_sound_id
        ):
            self.stop_tracked_playback()
            return
        sound_id, error = PlaySoundUtil.play_sound_file_async(sound_path)
        if error:
            self.notify(f"Couldn't play sound segment: {error}", severity="error")
            return
        self.playing_sound_id = sound_id
        self.playing_sound_path = sound_path
        self.playing_phrase_index = phrase_index
        self.show_playback_status()

    def action_quick_generate(self) -> None:
        """Regenerate the highlighted item outside the full-screen editor."""
        phrase_index = self.get_highlighted_phrase_index()
        if phrase_index is None:
            return

        self.quick_generate_phrase(phrase_index)

    def quick_generate_phrase(self, phrase_index: int) -> None:
        """Regenerate a specific phrase outside the full-screen editor."""

        blocker_text = readiness.get_generate_blocker_text(self.state, verbose=True)
        if blocker_text:
            self.push_screen(
                AlertDialog(
                    title="Cannot generate audio",
                    copy=blocker_text,
                )
            )
            return

        save_error = self.persist_staged_queue()
        if self.playing_phrase_index == phrase_index:
            self.stop_tracked_playback()
        self.project.sound_segments.delete_by_indices({phrase_index})
        self.project.sound_segments.force_invalidate()
        self.exit(QuickGenerationRequested(phrase_index, save_error))

    def action_show_info(self) -> None:
        """Show dialog with info for the highlighted segment."""
        phrase_index = self.get_highlighted_phrase_index()
        if phrase_index is None:
            return
        sound_segment = self.get_phrase_segment_status(phrase_index).best_segment
        if sound_segment is None:
            return
        lines = SegmentTranscriptUtil.make_info_text_lines(
            phrase_index, self.project, is_for_dialog=True
        )

        # Insert duration info after the first line
        sound_path = Path(self.project.sound_segments_path) / sound_segment.file_name
        try:
            duration = AudioMetaUtil.get_audio_duration(str(sound_path))
        except Exception:
            duration = None
        if duration is not None:
            lines.insert(1, f"Duration: {duration:.1f}s")
            lines.insert(2, "")

        self.push_screen(
            SegmentInfoDialog(
                text_util.combine_ansi_lines(lines),
                lambda: self.play_sound_for_phrase(phrase_index),
            ),
            lambda action: self.handle_segment_info_action(action, phrase_index),
        )

    def handle_segment_info_action(
        self,
        action: SegmentInfoAction | None,
        phrase_index: int,
    ) -> None:
        """Apply a closing info-dialog action to its snapshotted phrase."""
        if action == SegmentInfoAction.DELETE_GENERATED:
            self.confirm_delete_generated({phrase_index}, phrase_index)
        elif action == SegmentInfoAction.QUICK_GENERATE:
            self.quick_generate_phrase(phrase_index)

    def action_delete_generated(self) -> None:
        """Confirm deletion of generated sound for the selected visible lines."""
        if (
            not self.content_initialized
            or self.find_active
            or not self.selected_indices
        ):
            return
        phrase_indices = {
            phrase_index
            for phrase_index in self.selected_content_line_indices()
            if phrase_index in self.generated_indices
        }
        if not phrase_indices:
            return

        selected_phrase_index = self.highlighted_content_line_index()
        self.confirm_delete_generated(phrase_indices, selected_phrase_index)

    def confirm_delete_generated(
        self,
        phrase_indices: set[int],
        selected_phrase_index: int | None,
    ) -> None:
        """Ask for confirmation before deleting the specified generated phrases."""
        segment_word = "segment" if len(phrase_indices) == 1 else "segments"
        selected_visible_index = self.selected_index
        self.push_screen(
            SaveChangesDialog(
                [f"Delete {len(phrase_indices)} generated sound {segment_word}?"]
            ),
            lambda decision: self.handle_delete_generated_decision(
                decision,
                phrase_indices,
                selected_phrase_index,
                selected_visible_index,
            ),
        )

    def handle_delete_generated_decision(
        self,
        decision: ExitDecision | None,
        phrase_indices: set[int],
        selected_phrase_index: int | None,
        selected_visible_index: int | None,
    ) -> None:
        """Delete the snapshotted selected segments after explicit confirmation."""
        if decision != ExitDecision.CONFIRM:
            return
        if self.playing_phrase_index in phrase_indices:
            self.stop_tracked_playback()

        self.project.sound_segments.delete_by_indices(phrase_indices)
        self.project.sound_segments.force_invalidate()
        self.mark_phrases_ungenerated(phrase_indices)

        old_items_by_id = self.visible_items_by_id()
        visible_phrase_indices = self.get_filtered_phrase_indices()
        selected_item_index = self.item_index_for_phrase(selected_phrase_index)
        if selected_item_index not in visible_phrase_indices:
            selected_item_index = None
        if selected_item_index is None:
            phrase_rows = [
                (visible_index, item_index)
                for visible_index, item_index in enumerate(visible_phrase_indices)
                if isinstance(self.list_items[item_index], GeneratePhraseGroupItem)
            ]
            selected_item_index = (
                min(
                    phrase_rows,
                    key=lambda row: abs(
                        row[0] - (selected_visible_index or 0)
                    ),
                )[1]
                if phrase_rows
                else None
            )
        self.replace_filtered_phrase_indices(
            visible_phrase_indices,
            selected_item_index,
            old_items_by_id,
            phrase_indices,
        )
        self.update_queued_status()

    def apply_staged_queue_to_project(self) -> None:
        """Apply the staged queue to the project's in-memory generation range."""
        self.project.generate_range_string = RangeStringUtil.make_ranges_string(
            self.staged_queued_indices, len(self.project.phrase_groups)
        )

    def persist_staged_queue(self) -> str:
        """Apply and save the staged queue as the project's generation range."""
        self.apply_staged_queue_to_project()
        error = self.project.save()
        return f"Save failed: {error}" if error else ""

    def exit_without_confirmation(self) -> None:
        """Persist the queue and exit without showing a confirmation dialog."""
        save_error = self.persist_staged_queue()
        self.exit(EditorSaveFailed(save_error) if save_error else EditorClosed())
