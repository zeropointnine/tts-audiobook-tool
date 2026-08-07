from pathlib import Path
from typing import ClassVar

from textual.binding import Binding, BindingType
from textual.css.errors import StylesheetError
from textual.widgets import Static

from tts_audiobook_tool import text_util
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.segment_transcript_util import (
    SegmentTranscriptUtil,
)
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.textual.content_textual_app import ContentTextualApp
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.segment_info_dialog import SegmentInfoDialog
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    STYLE_DIM,
)
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.util import make_error_string, print_feedback


class SoundSegmentsEditorTextualApp(ContentTextualApp):

    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        Binding("x", "toggle_deletion", show=False),
        Binding("p", "play_sound", show=False),
        Binding("i", "show_info", show=False),
        Binding("e", "toggle_word_errors_filter", show=False),
    ]

    def __init__(self, project: Project) -> None:
        self.all_phrase_indices: list[int] = []
        self.deletion_flag_indices: dict[int, int] = {}
        self.original_deletion_flags: list[bool] = []
        self.staged_deletion_flags: list[bool] = []
        self.word_errors_filter_active = False
        self.did_save_changes = False
        self.deleted_sound_segment_count = 0
        self.save_error = ""
        self.playing_sound_id = ""
        self.playing_sound_path = ""
        self.playing_phrase_index: int | None = None
        header_lines = [
            f"{COL_ACCENT}Review/delete sound segments",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]",
            f"{COL_DIM}- Select multiple lines by holding [SHIFT] + navigation keys",
            f"{COL_DIM}- Press [{COL_ACCENT}X{COL_DIM}] to mark/unmark selected line/s for deletion",
            f"{COL_DIM}- Press [{COL_ACCENT}P{COL_DIM}] to play  [I] Info  [E] Show only word errors",
            f"{COL_DIM}- Press [ESC] to finish  - [CTRL-F] Find text",
        ]
        super().__init__(
            project,
            header_lines,
            empty_state_text="No sound segments",
            loading_state_text="...",
        )

    def initialize_content(self) -> list[int]:
        """Discover generated segments and stage deletion flags for their rows."""
        phrase_group_count = len(self.project.phrase_groups)
        self.all_phrase_indices = [
            index
            for index in sorted(self.project.sound_segments.get_existing_indices())
            if 0 <= index < phrase_group_count
        ]
        self.deletion_flag_indices = {
            phrase_index: index
            for index, phrase_index in enumerate(self.all_phrase_indices)
        }
        self.original_deletion_flags = [False] * len(self.all_phrase_indices)
        self.staged_deletion_flags = list(self.original_deletion_flags)
        return self.all_phrase_indices

    @property
    def find_label_text(self) -> str:
        return "Search text: "

    @property
    def has_changes(self) -> bool:
        return (
            self.content_initialized
            and self.staged_deletion_flags != self.original_deletion_flags
        )

    def compose_status_widgets(self):
        yield Static("", id="playing-status", markup=False)
        yield from super().compose_status_widgets()

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(0.1, self.update_playback_status)

    def on_unmount(self) -> None:
        """Stop playback started by this editor when the editor closes."""
        if (
            self.playing_sound_id
            and PlaySoundUtil.current_sound_id() == self.playing_sound_id
        ):
            PlaySoundUtil.stop_sound_async()
        self.clear_playback_status()

    def clear_playback_status(self) -> None:
        """Clear the editor's tracked playback and its status text."""
        self.playing_sound_id = ""
        self.playing_sound_path = ""
        self.playing_phrase_index = None
        status_widgets = self.query("#playing-status")
        if status_widgets:
            status_widgets.first(Static).update("")

    def update_playback_status(self) -> None:
        """Clear playback status once the tracked sound has finished."""
        if not self.playing_sound_id:
            return
        if PlaySoundUtil.current_sound_id() != self.playing_sound_id:
            self.clear_playback_status()

    def show_playback_status(self) -> None:
        """Show the currently tracked phrase without querying playback state."""
        status_widgets = self.query("#playing-status")
        if self.playing_phrase_index is not None and status_widgets:
            status_widgets.first(Static).update(
                f"Playing line {self.playing_phrase_index + 1}"
            )

    def format_line(self, index: int) -> HangingIndentText:
        phrase_index = self.phrase_indices[index]
        phrase_group = self.project.phrase_groups[phrase_index]
        best_sound_segment = self.project.sound_segments.get_best_item_for(phrase_index)
        deletion_flag_index = self.deletion_flag_indices[phrase_index]
        should_delete = self.staged_deletion_flags[deletion_flag_index]
        is_find_match = index == self.find_match_index
        style = f"{STYLE_DIM} reverse" if is_find_match else ""
        deletion_text = "DELETE" if should_delete else " " * 6
        deletion_color = COL_ERROR if should_delete else ""
        prefix_ansi = (
            f"{COL_DIM}[{phrase_index + 1:05d}] "
            f"{COL_DIM}[{deletion_color}{deletion_text}{COL_DIM}] "
        )
        content_ansi = Ansi.RESET
        if best_sound_segment is not None and best_sound_segment.num_errors > 0:
            failed_marker = ""
            if self.project.sound_segments.is_segment_failed(
                phrase_index, best_sound_segment
            ):
                failed_marker = f" {COL_ERROR}*{COL_ORANGE}"
            content_ansi += (
                f"{COL_ORANGE}[word errors: {best_sound_segment.num_errors}"
                f"{failed_marker}]{Ansi.RESET} "
            )
        content_ansi += phrase_group.presentable_text
        return HangingIndentText.from_ansi(
            prefix_ansi + content_ansi,
            content_start=len(prefix_ansi),
            max_lines=3,
            style=style,
        )

    def set_selected_deletion_flag(self, should_delete: bool) -> None:
        """Set the deletion flag for all selected generated phrase groups."""

        def set_flag(_visible_index: int, phrase_index: int) -> bool:
            flag_index = self.deletion_flag_indices[phrase_index]
            if self.staged_deletion_flags[flag_index] == should_delete:
                return False
            self.staged_deletion_flags[flag_index] = should_delete
            return True

        self.mutate_selected_items(set_flag)

    def action_toggle_deletion(self) -> None:
        """Toggle deletion off when all selected rows are marked, or on otherwise."""
        if (
            not self.content_initialized
            or self.find_active
            or not self.selected_indices
        ):
            return
        all_selected_are_marked = all(
            self.staged_deletion_flags[
                self.deletion_flag_indices[self.phrase_indices[index]]
            ]
            for index in self.selected_indices
        )
        self.set_selected_deletion_flag(not all_selected_are_marked)

    def action_toggle_word_errors_filter(self) -> None:
        """Toggle between all generated lines and only lines with word errors."""
        if not self.content_initialized or self.find_active:
            return
        selected_phrase_index = (
            self.phrase_indices[self.selected_index]
            if self.selected_index is not None
            else None
        )
        self.word_errors_filter_active = not self.word_errors_filter_active
        if self.word_errors_filter_active:
            phrase_indices = [
                phrase_index
                for phrase_index in self.all_phrase_indices
                if (
                    (sound_segment := self.project.sound_segments.get_best_item_for(
                        phrase_index
                    ))
                    is not None
                    and sound_segment.num_errors > 0
                )
            ]
        else:
            phrase_indices = self.all_phrase_indices
        self.replace_phrase_indices(phrase_indices, selected_phrase_index)

    def action_play_sound(self) -> None:
        """Play the highlighted segment, or stop it when already playing."""
        if (
            not self.content_initialized
            or self.find_active
            or self.selected_index is None
        ):
            return
        phrase_index = self.phrase_indices[self.selected_index]
        sound_segment = self.project.sound_segments.get_best_item_for(phrase_index)
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
            PlaySoundUtil.stop_sound_async()
            self.clear_playback_status()
            return
        sound_id, error = PlaySoundUtil.play_sound_file_async(sound_path)
        if error:
            self.notify(f"Couldn't play sound segment: {error}", severity="error")
            return
        self.playing_sound_id = sound_id
        self.playing_sound_path = sound_path
        self.playing_phrase_index = phrase_index
        self.show_playback_status()

    def action_show_info(self) -> None:
        """Show dialog with info for the highlighted segment."""
        if (
            not self.content_initialized
            or self.find_active
            or self.selected_index is None
        ):
            return
        phrase_index = self.phrase_indices[self.selected_index]
        lines = SegmentTranscriptUtil.make_info_text_lines(
            phrase_index, self.project, is_for_dialog=True
        )

        # Insert duration info after the first line
        sound_segment = self.project.sound_segments.get_best_item_for(phrase_index)
        if sound_segment is not None:
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
                text_util.combine_ansi_lines(lines)
            )
        )

    def make_confirmation_dialog(self) -> SaveChangesDialog:
        if not self.content_initialized:
            return SaveChangesDialog()
        sound_segment_file_count = sum(
            len(self.project.sound_segments.sound_segments_map.get(phrase_index, []))
            for phrase_index, should_delete in zip(
                self.all_phrase_indices, self.staged_deletion_flags, strict=True
            )
            if should_delete
        )
        file_word = "file" if sound_segment_file_count == 1 else "files"
        return SaveChangesDialog(
            f"Delete the {sound_segment_file_count} sound segment {file_word} marked for deletion?"
        )

    def commit_changes_and_exit(self) -> None:
        """Apply the staged deletion flags and exit."""
        if not self.content_initialized:
            self.exit()
            return
        deleted_sound_segment_count = 0
        try:
            indices_to_delete = [
                self.all_phrase_indices[index]
                for index, should_delete in enumerate(self.staged_deletion_flags)
                if should_delete
            ]
            current_segments = self.project.sound_segments.sound_segments_map
            existing_indices_to_delete = [
                index for index in indices_to_delete if index in current_segments
            ]
            deleted_sound_segment_count = sum(
                len(current_segments[index]) for index in existing_indices_to_delete
            )
            self.project.sound_segments.delete_by_indices(existing_indices_to_delete)
            error = ""
        except Exception as exception:
            error = make_error_string(exception)
        if error:
            self.save_error = f"Save failed: {error}"
        else:
            self.did_save_changes = True
            self.deleted_sound_segment_count = deleted_sound_segment_count
        self.exit()

    @classmethod
    def start(cls, project: Project) -> None:
        """Run a sound segment reviewer for a project and report its result."""
        if not cls.check_terminal_support():
            return
        app = cls(project)
        app.run(inline=False)
        if isinstance(app._exception, StylesheetError):
            print_feedback("Couldn't load textual css", is_error=True)
        elif app.save_error:
            print_feedback(app.save_error, is_error=True)
        elif app.did_save_changes:
            segment_word = (
                "segment" if app.deleted_sound_segment_count == 1 else "segments"
            )
            print_feedback(
                f"Deleted {app.deleted_sound_segment_count} sound {segment_word}",
                long_pause=True,
            )
