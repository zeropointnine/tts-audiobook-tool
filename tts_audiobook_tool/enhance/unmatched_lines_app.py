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

# Spans shorter than this are shown as a single approximate point
NO_SPAN_SECONDS = 0.5
# A range starting this close to the end of the file is labeled "end of audio"
AUDIO_END_SLACK_SECONDS = 3.0
# Seconds of surrounding audio played on each side of a range with no span
CONTEXT_SECONDS = 3.0


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
        self.playing_is_context = False
        self.pending_is_context = False
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

    def is_no_span(self, start_time: float, end_time: float) -> bool:
        return end_time - start_time < NO_SPAN_SECONDS

    def is_at_audio_end(self, start_time: float) -> bool:
        return (
            self.audio_duration is not None
            and start_time >= self.audio_duration - AUDIO_END_SLACK_SECONDS
        )

    def time_parts(self, start: int, end: int) -> tuple[str, str]:
        """
        Timestamp text plus an optional dim note.

        A degenerate span (the aligned neighbors touch) or one pinned to the end
        of the file is shown as a single approximate point, since a range like
        "00:10:52-00:10:52" reads as a display bug.
        """
        start_time, end_time = self.range_times(start, end)
        if self.is_at_audio_end(start_time):
            return f"≈{time_stamp(start_time, with_tenth=False)}", "(end of audio)"
        if self.is_no_span(start_time, end_time):
            return f"≈{time_stamp(start_time, with_tenth=False)}", "(no audio span)"
        return (
            f"{time_stamp(start_time, with_tenth=False)}-{time_stamp(end_time, with_tenth=False)}",
            "",
        )

    def play_times(self, start: int, end: int) -> tuple[float, float, bool]:
        """
        Interval to preview, and whether it is surrounding context rather than
        the range itself (used when the range has no usable span of its own).
        """
        start_time, end_time = self.range_times(start, end)
        if not self.is_no_span(start_time, end_time):
            return start_time, end_time, False
        context_start = max(0.0, start_time - CONTEXT_SECONDS)
        context_end = end_time + CONTEXT_SECONDS
        if self.audio_duration is not None:
            context_end = min(context_end, self.audio_duration)
        return context_start, context_end, True

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
            what = "audio around range" if self.playing_is_context else "range"
            self.set_selected_status(right=f"Playing {what} {self.playing_range_index + 1}")
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
        start, end, is_context = self.play_times(*self.ranges[range_index])
        if end <= start:
            self.show_status_toast(right="No playable audio interval for this range")
            return
        self.pending_range_index = range_index
        self.pending_is_context = is_context
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
        self.playing_is_context = self.pending_is_context
        self.update_selection_status()

    def row_fields(self, item_index: int) -> tuple[tuple[str, str], str, list[str]]:
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
        stamp, note = self.time_parts(start, end)
        return (stamp, note), label, previews

    def format_line(self, index: int) -> Text:
        (stamp, note), label, previews = self.row_fields(self.phrase_indices[index])
        row = Text.from_ansi(
            f"{COL_ACCENT}{stamp}{COL_DEFAULT} ",
            no_wrap=True,
            overflow="ellipsis",
        )
        if note:
            row.append(f"{note} ", style=STYLE_DIM)
        row.append(label, style="default")
        for preview in previews:
            row.append(f"\n  {preview}", style=STYLE_DIM)
        if index == self.find_match_index:
            row.stylize(f"{STYLE_DIM} reverse")
        return row

    def find_text_strings(self, item_index: int) -> Sequence[str]:
        (stamp, note), label, previews = self.row_fields(item_index)
        return [f"{stamp} {note}".strip(), label, *(preview.strip() for preview in previews)]

    def content_line_index(self, item_index: int) -> None:
        """Ranges are not actionable lines in the unrelated current project."""
        return None

