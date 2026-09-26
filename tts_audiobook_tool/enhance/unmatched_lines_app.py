"""Read-only, full-screen review of unmatched audiobook text ranges."""

from collections.abc import Sequence
import math
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.message import Message

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.textual.content_textual_app import ContentTextualApp, EditorClosed
from tts_audiobook_tool.textual.textual_shared import STYLE_DIM
from tts_audiobook_tool.util import time_stamp


class PreviewLoaded(Message):
    """Result of a bounded audiobook decode on a background thread."""

    def __init__(self, request_id: int, range_index: int, result: Sound | str) -> None:
        super().__init__()
        self.request_id = request_id
        self.range_index = range_index
        self.result = result


class UnmatchedLinesApp(ContentTextualApp[EditorClosed]):
    """Show one selectable row per consecutive run of unaligned phrases."""

    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        Binding("p", "play_sound", show=False),
    ]

    def __init__(
        self,
        project: Project,
        timed_phrases: list[TimedPhrase],
        audio_path: Path,
        audio_duration: float | None,
    ) -> None:
        self.audio_path = audio_path
        self.audio_duration = (
            audio_duration
            if audio_duration is not None and math.isfinite(audio_duration) and audio_duration > 0
            else None
        )
        self.timed_phrases = timed_phrases
        self.ranges = TimedPhrase.get_discontinuities(timed_phrases)
        self.playing_sound_id = ""
        self.playing_range_index: int | None = None
        self.pending_range_index: int | None = None
        self.preview_request_id = 0
        self.preview_closed = False
        super().__init__(
            project,
            [
                f"{COL_ACCENT}Unmatched text lines",
                f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]  - [CTRL-F] Find text",
                f"{COL_DIM}- [P] Play highlighted audio range",
                f"{COL_DIM}- Press [ESC] to close",
            ],
            phrase_indices=range(len(self.ranges)),
            empty_state_text="No unmatched text lines",
            multi_select_enabled=False,
        )

    def range_times(self, start: int, end: int) -> tuple[float, float]:
        """Use aligned neighbors, or the file duration for a trailing range."""
        start_time = self.timed_phrases[start - 1].time_end if start else 0.0
        end_time = (
            self.timed_phrases[end + 1].time_start
            if end + 1 < len(self.timed_phrases)
            else max(start_time, self.audio_duration or start_time)
        )
        return start_time, end_time

    def time_range(self, start: int, end: int) -> str:
        start_time, end_time = self.range_times(start, end)
        return f"{time_stamp(start_time, with_tenth=False)}-{time_stamp(end_time, with_tenth=False)}"

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(0.1, self.update_playback_status)

    def on_unmount(self) -> None:
        self.preview_closed = True
        self.preview_request_id += 1
        self.pending_range_index = None
        self.stop_tracked_playback()

    def update_selection_status(self) -> None:
        if self.playing_range_index is not None:
            self.set_selected_status(right=f"Playing range {self.playing_range_index + 1}")
        elif self.pending_range_index is not None:
            self.set_selected_status(right=f"Loading range {self.pending_range_index + 1} audio")
        else:
            super().update_selection_status()

    def stop_tracked_playback(self) -> None:
        """Stop only the sound started by this review screen."""
        if self.playing_sound_id and PlaySoundUtil.current_sound_id() == self.playing_sound_id:
            PlaySoundUtil.stop_sound_async()
        self.playing_sound_id = ""
        self.playing_range_index = None
        self.update_selection_status()

    def update_playback_status(self) -> None:
        if self.playing_sound_id and PlaySoundUtil.current_sound_id() != self.playing_sound_id:
            self.playing_sound_id = ""
            self.playing_range_index = None
            self.update_selection_status()

    def action_play_sound(self) -> None:
        """Toggle a bounded preview of the highlighted unmatched range."""
        if self.find_active or self.selected_index is None or self.preview_closed:
            return
        range_index = self.phrase_indices[self.selected_index]
        if self.pending_range_index == range_index:
            self.preview_request_id += 1
            self.pending_range_index = None
            self.update_selection_status()
            return
        if (
            self.playing_range_index == range_index
            and self.playing_sound_id
            and PlaySoundUtil.current_sound_id() == self.playing_sound_id
        ):
            self.stop_tracked_playback()
            return

        self.preview_request_id += 1
        request_id = self.preview_request_id
        self.pending_range_index = None
        self.stop_tracked_playback()
        start, end = self.range_times(*self.ranges[range_index])
        if end <= start:
            self.show_status_toast(right="No playable audio interval for this range")
            return
        self.pending_range_index = range_index
        self.update_selection_status()
        self.run_worker(
            lambda: self._load_preview(request_id, range_index, start, end),
            name="unmatched-audio-preview",
            thread=True,
        )

    def _load_preview(
        self, request_id: int, range_index: int, start: float, end: float
    ) -> None:
        result = SoundFileUtil.load_range(str(self.audio_path), start, end)
        self.post_message(PreviewLoaded(request_id, range_index, result))

    def on_preview_loaded(self, event: PreviewLoaded) -> None:
        if self.preview_closed or event.request_id != self.preview_request_id:
            return
        self.pending_range_index = None
        if isinstance(event.result, str):
            self.update_selection_status()
            self.notify(f"Couldn't play audio range: {event.result}", severity="error")
            return
        self.playing_sound_id = PlaySoundUtil.play_sound_async(event.result)
        self.playing_range_index = event.range_index
        self.update_selection_status()

    def row_fields(self, item_index: int) -> tuple[str, str, list[str]]:
        start, end = self.ranges[item_index]
        count = end - start + 1
        label = (
            f"Line {start + 1}"
            if count == 1
            else f"Lines {start + 1} to {end + 1} ({count} lines)"
        )
        first = self.timed_phrases[start].text.strip()
        previews = (
            [f"Line: {first}"]
            if count == 1
            else [
                f"First line: {first}",
                f"Last line: {self.timed_phrases[end].text.strip()}",
            ]
        )
        return self.time_range(start, end), label, previews

    def format_line(self, index: int) -> Text:
        timestamp, label, previews = self.row_fields(self.phrase_indices[index])
        row = Text.from_ansi(
            f"{COL_ACCENT}{timestamp}{COL_DEFAULT} ",
            no_wrap=True,
            overflow="ellipsis",
        )
        row.append(label, style="default")
        for preview in previews:
            row.append(f"\n  {preview}", style=STYLE_DIM)
        if index == self.find_match_index:
            row.stylize(f"{STYLE_DIM} reverse")
        return row

    def find_text_strings(self, item_index: int) -> Sequence[str]:
        timestamp, label, previews = self.row_fields(item_index)
        return [timestamp, label, *(preview.strip() for preview in previews)]

    def content_line_index(self, item_index: int) -> None:
        """Ranges are not actionable lines in the unrelated current project."""
        return None

