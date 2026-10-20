from pathlib import Path
from threading import Event
from typing import cast
from unittest.mock import patch

import numpy as np
from rich.console import Console
from textual.widgets import Input, OptionList, Static

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.enhance.unmatched_lines_app import UnmatchedLinesApp
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.textual.content_textual_app import EditorClosed
from textual_editor_stubs import StubProject, run


def make_app(phrases: list[TimedPhrase], audio_duration: float | None = None) -> UnmatchedLinesApp:
    return UnmatchedLinesApp(cast(Project, StubProject([])), phrases, Path("book.abr.m4b"), audio_duration)


def test_range_rows_format_text_and_colors() -> None:
    app = make_app([
        TimedPhrase("  First alone  ", 0, 0),
        TimedPhrase("Matched", 5, 10),
        TimedPhrase("  First in run  ", 0, 0),
        TimedPhrase("x" * 55, 0, 0),
        TimedPhrase("Matched", 25, 30),
        TimedPhrase("  Trailing  ", 0, 0),
    ])
    assert app.ranges == [(0, 0), (2, 3), (5, 5)]
    assert app.format_line(0).plain == "00:00:00-00:00:05 Line 1\n  Line: First alone"
    row = app.format_line(1)
    assert row.plain == (
        "00:00:10-00:00:25 Lines 3 to 4 (2 lines)\n"
        "  First line: First in run\n"
        "  Last line: " + "x" * 55
    )
    assert app.format_line(2).plain == "00:00:30-00:00:30 Line 6\n  Line: Trailing"
    console = Console()
    assert row.get_style_at_offset(console, 0).color.name == "#ffaa44"
    assert row.get_style_at_offset(console, 18).color.name == "default"
    assert row.get_style_at_offset(console, row.plain.index("First line:")).color.name == "#888888"
    assert app.find_match_indices("First in run") == [1]
    assert app.find_match_indices("x" * 55) == [1]
    assert app.find_match_indices("00:00:30") == [2]
    assert app.content_line_index(1) is None
    assert app.multi_select_enabled is False


def test_all_unmatched_and_no_unmatched_rows() -> None:
    all_unmatched = make_app([TimedPhrase("One", 0, 0), TimedPhrase("Two", 0, 0)])
    assert all_unmatched.format_line(0).plain.startswith("00:00:00-00:00:00 Lines 1 to 2 (2 lines)")
    empty = make_app([TimedPhrase("Aligned", 0, 5)])
    assert empty.phrase_indices == []
    assert empty.empty_state_text == "No unmatched text lines"


def test_trailing_unmatched_range_ends_at_audio_duration() -> None:
    phrases = [
        TimedPhrase("Unmatched", 0, 0),
        TimedPhrase("Matched", 5, 10),
        TimedPhrase("Trailing", 0, 0),
    ]
    app = make_app(phrases, audio_duration=95.5)
    assert app.range_times(*app.ranges[0]) == (0, 5)
    assert app.range_times(*app.ranges[1]) == (10, 95.5)
    assert app.format_line(1).plain.startswith("00:00:10-00:01:35 Line 3")
    assert make_app(phrases, audio_duration=None).range_times(2, 2) == (10, 10)
    assert make_app(phrases, audio_duration=5).range_times(2, 2) == (10, 10)

    all_unmatched = make_app([TimedPhrase("One", 0, 0), TimedPhrase("Two", 0, 0)], 95.5)
    assert all_unmatched.range_times(0, 1) == (0, 95.5)
    assert all_unmatched.format_line(0).plain.startswith("00:00:00-00:01:35")


def test_previews_ellipsize_at_list_boundary_and_reflow_on_resize() -> None:
    app = make_app([
        TimedPhrase("S" * 90, 0, 0),
        TimedPhrase("Aligned", 1, 2),
        TimedPhrase("A" * 90, 0, 0),
        TimedPhrase("Middle", 0, 0),
        TimedPhrase("B" * 90, 0, 0),
    ])

    async def exercise() -> None:
        async with app.run_test(size=(35, 20)) as pilot:
            await pilot.pause()
            options = app.query_one("#line-list", OptionList)
            for terminal_width in (35, 65, 110, 35):
                await pilot.resize_terminal(terminal_width, 20)
                await pilot.pause()
                width = options.scrollable_content_region.width
                assert options.render_line(0).text.startswith("00:00:00-00:00:01 Line 1")
                assert options.render_line(2).text.startswith("00:00:02-00:00:02 Lines 3 to 5")
                for line_number, prefix, letter in (
                    (1, "  Line: ", "S"),
                    (3, "  First line: ", "A"),
                    (4, "  Last line: ", "B"),
                ):
                    full_line = prefix + letter * 90
                    expected = (
                        full_line
                        if len(full_line) <= width
                        else full_line[: width - 1] + "…"
                    )
                    assert options.render_line(line_number).text.rstrip() == expected

    run(exercise())


def test_navigation_find_and_escape_close() -> None:
    app = make_app([
        TimedPhrase("First", 0, 0),
        TimedPhrase("Aligned", 1, 2),
        TimedPhrase("Second", 0, 0),
    ])

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#header-line-0", Static).render().plain == "Unmatched text lines"
            options = app.query_one("#line-list", OptionList)
            assert options.option_count == 2
            await pilot.press("down")
            assert options.highlighted == 1
            await pilot.press("ctrl+f")
            assert app.query_one("#find-input", Input).has_focus
            await pilot.press("F", "i", "r", "s", "t", "enter")
            assert options.highlighted == 0
            await pilot.press("escape")
            assert not app.find_active
            await pilot.press("escape")
        assert isinstance(app.return_value, EditorClosed)

    run(exercise())


def test_empty_state_is_shown() -> None:
    app = make_app([])

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#empty-state", Static).display
            assert not app.query_one("#line-list", OptionList).display
            await pilot.press("escape")
        assert isinstance(app.return_value, EditorClosed)

    run(exercise())


def test_p_plays_highlighted_precise_interval_and_toggles_off() -> None:
    app = make_app([
        TimedPhrase("Matched", 1, 2.25),
        TimedPhrase("Missing", 0, 0),
        TimedPhrase("Matched", 4.75, 6),
        TimedPhrase("Missing", 0, 0),
        TimedPhrase("Matched", 12.5, 14),
    ])
    sound = Sound(np.zeros(10, dtype=np.float32), 48000)
    with (
        patch.object(SoundFileUtil, "load_range", return_value=sound) as load,
        patch.object(PlaySoundUtil, "play_sound_async", return_value="preview-id") as play,
        patch.object(PlaySoundUtil, "current_sound_id", return_value="preview-id"),
        patch.object(PlaySoundUtil, "stop_sound_async") as stop,
    ):
        async def exercise() -> None:
            async with app.run_test() as pilot:
                await pilot.press("p")
                for _ in range(10):
                    await pilot.pause()
                    if play.called:
                        break
                load.assert_called_once_with("book.abr.m4b", 2.25, 4.75)
                play.assert_called_once_with(sound)
                assert app.query_one("#status-right", Static).render().plain == "Playing range 1"
                await pilot.press("p")
                stop.assert_called_once_with()
                assert app.playing_sound_id == ""
                await pilot.press("down", "p")
                for _ in range(10):
                    await pilot.pause()
                    if play.call_count == 2:
                        break
                assert load.call_args_list[-1].args == ("book.abr.m4b", 6, 12.5)
                assert play.call_count == 2
                await pilot.press("escape")
            assert stop.call_count == 2

        run(exercise())


def test_p_plays_trailing_range_through_audio_end() -> None:
    app = make_app([
        TimedPhrase("Matched", 5, 10),
        TimedPhrase("Trailing", 0, 0),
    ], audio_duration=80)
    sound = Sound(np.zeros(10, dtype=np.float32), 48000)
    with (
        patch.object(SoundFileUtil, "load_range", return_value=sound) as load,
        patch.object(PlaySoundUtil, "play_sound_async", return_value="preview-id") as play,
        patch.object(PlaySoundUtil, "current_sound_id", return_value="preview-id"),
    ):
        async def exercise() -> None:
            async with app.run_test() as pilot:
                await pilot.press("p")
                for _ in range(10):
                    await pilot.pause()
                    if play.called:
                        break
                load.assert_called_once_with("book.abr.m4b", 10, 80)

        run(exercise())


def test_p_ignores_find_empty_and_zero_length_ranges() -> None:
    app = make_app([TimedPhrase("Unmatched", 0, 0)])
    with patch.object(SoundFileUtil, "load_range") as load:
        async def exercise() -> None:
            async with app.run_test() as pilot:
                await pilot.press("p")
                assert "No playable audio interval" in app.toast_status.right
                await pilot.press("ctrl+f", "p")
                assert app.query_one("#find-input", Input).value == "p"
                load.assert_not_called()

        run(exercise())

    empty = make_app([])
    with patch.object(SoundFileUtil, "load_range") as load:
        async def exercise_empty() -> None:
            async with empty.run_test() as pilot:
                await pilot.press("p")
                load.assert_not_called()

        run(exercise_empty())


def test_preview_errors_and_finished_sound_clear_status() -> None:
    app = make_app([
        TimedPhrase("Missing", 0, 0),
        TimedPhrase("Matched", 2, 3),
    ])
    sound = Sound(np.zeros(10, dtype=np.float32), 48000)
    active_id = "preview-id"

    def current_id() -> str:
        return active_id

    with (
        patch.object(SoundFileUtil, "load_range", side_effect=["decode failed", sound]),
        patch.object(PlaySoundUtil, "play_sound_async", return_value="preview-id") as play,
        patch.object(PlaySoundUtil, "current_sound_id", side_effect=current_id),
        patch.object(PlaySoundUtil, "stop_sound_async") as stop,
        patch.object(app, "notify") as notify,
    ):
        async def exercise() -> None:
            nonlocal active_id
            async with app.run_test() as pilot:
                await pilot.press("p")
                for _ in range(10):
                    await pilot.pause()
                    if app.pending_range_index is None:
                        break
                assert app.pending_range_index is None
                assert app.playing_sound_id == ""
                notify.assert_called_once_with(
                    "Couldn't play audio range: decode failed", severity="error"
                )
                play.assert_not_called()
                await pilot.press("p")
                for _ in range(10):
                    await pilot.pause()
                    if play.called:
                        break
                assert app.playing_sound_id == "preview-id"
                active_id = ""
                app.update_playback_status()
                assert app.playing_sound_id == ""
                assert "Playing range" not in app.query_one("#status-right", Static).render().plain
            stop.assert_not_called()

        run(exercise())


def test_p_cancels_pending_decode_and_close_discards_its_result() -> None:
    app = make_app([
        TimedPhrase("Missing", 0, 0), TimedPhrase("Matched", 3, 4),
    ])
    started = Event()
    release = Event()
    sound = Sound(np.zeros(10, dtype=np.float32), 48000)

    def load(_path: str, _start: float, _end: float) -> Sound:
        started.set()
        assert release.wait(timeout=5)
        return sound

    with (
        patch.object(SoundFileUtil, "load_range", side_effect=load),
        patch.object(PlaySoundUtil, "play_sound_async") as play,
    ):
        async def exercise() -> None:
            async with app.run_test() as pilot:
                await pilot.press("p")
                assert started.wait(timeout=5)
                await pilot.press("p")
                assert app.pending_range_index is None
                await pilot.press("escape")
            assert app.preview_closed
            release.set()
            play.assert_not_called()

        try:
            run(exercise())
        finally:
            release.set()


def test_stale_decodes_do_not_play_after_new_request_or_close() -> None:
    app = make_app([
        TimedPhrase("First", 0, 0), TimedPhrase("Matched", 1, 2),
        TimedPhrase("Second", 0, 0), TimedPhrase("Matched", 3, 4),
    ])
    first_started = Event()
    release_first = Event()
    sound = Sound(np.zeros(10, dtype=np.float32), 48000)

    def load(_path: str, start: float, _end: float) -> Sound:
        if start == 0:
            first_started.set()
            assert release_first.wait(timeout=5)
        return sound

    with (
        patch.object(SoundFileUtil, "load_range", side_effect=load),
        patch.object(PlaySoundUtil, "play_sound_async", return_value="preview-id") as play,
        patch.object(PlaySoundUtil, "current_sound_id", return_value="preview-id"),
        patch.object(PlaySoundUtil, "stop_sound_async") as stop,
    ):
        async def exercise() -> None:
            async with app.run_test() as pilot:
                await pilot.press("p")
                assert first_started.wait(timeout=5)
                await pilot.press("down", "p")
                for _ in range(10):
                    await pilot.pause()
                    if play.called:
                        break
                assert play.call_count == 1
                release_first.set()
                await pilot.pause()
                assert play.call_count == 1
                await pilot.press("escape")
            stop.assert_called_once_with()
            assert app.preview_closed

        try:
            run(exercise())
        finally:
            release_first.set()
