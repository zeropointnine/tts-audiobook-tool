import asyncio
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from rich.style import Style
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Button, Static

from tts_audiobook_tool.project_support.segment_transcript_util import (
    SegmentTranscriptUtil,
)
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.textual.sound_segments_editor import (
    SoundSegmentsEditorTextualApp,
)
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.segment_info_dialog import SegmentInfoDialog
from tts_audiobook_tool.textual.textual_shared import NonWrappingOptionList
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.text_util import make_terminal_hyperlink


@dataclass
class StubPhraseGroup:
    presentable_text: str


@dataclass
class StubSoundSegment:
    file_name: str
    num_errors: int = -1


@dataclass
class StubSoundSegments:
    sound_segments_map: dict[int, list[StubSoundSegment]]
    deletion_calls: list[list[int]] = field(default_factory=list)
    failed_segment_files: set[str] = field(default_factory=set)

    def get_existing_indices(self) -> set[int]:
        return set(self.sound_segments_map)

    def get_best_item_for(self, index: int) -> StubSoundSegment | None:
        items = self.sound_segments_map.get(index, [])
        return min(
            items,
            key=lambda item: item.num_errors if item.num_errors != -1 else 10_000,
            default=None,
        )

    def is_segment_failed(self, index: int, item: StubSoundSegment) -> bool:
        return item.file_name in self.failed_segment_files

    def delete_by_indices(self, indices: list[int]) -> None:
        self.deletion_calls.append(indices)


@dataclass
class StubProject:
    phrase_groups: list[StubPhraseGroup]
    sound_segments: StubSoundSegments
    sound_segments_path: str = "/project/segments"


def make_app(
    num_lines: int = 12,
    generated_indices: set[int] | None = None,
) -> tuple[SoundSegmentsEditorTextualApp, StubProject]:
    if generated_indices is None:
        generated_indices = set(range(num_lines))
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(num_lines)],
        StubSoundSegments(
            {
                index: [StubSoundSegment(f"segment-{index}.flac")]
                for index in generated_indices
            }
        ),
    )
    app = SoundSegmentsEditorTextualApp(project)  # type: ignore[arg-type]
    app.load_content()
    return app, project


def run(coroutine) -> None:
    asyncio.run(coroutine)


def render_line(app: SoundSegmentsEditorTextualApp, index: int, width: int) -> str:
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
    )
    console.print(app.format_line(index), end="")
    return output.getvalue()


def test_segment_discovery_and_rows_are_deferred_until_after_first_draw() -> None:
    project = StubProject(
        [StubPhraseGroup("Line 1")],
        StubSoundSegments({0: [StubSoundSegment("segment-0.flac")]}),
    )
    app = SoundSegmentsEditorTextualApp(project)  # type: ignore[arg-type]

    assert app.content_initialized is False
    assert app.phrase_indices == []
    assert app.all_phrase_indices == []
    assert app.staged_deletion_flags == []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.content_initialized is True
            assert app.phrase_indices == [0]
            assert app.all_phrase_indices == [0]
            assert app.staged_deletion_flags == [False]

    run(exercise())


def test_segment_actions_are_ignored_before_deferred_content_loads() -> None:
    project = StubProject(
        [StubPhraseGroup("Line 1")],
        StubSoundSegments({0: [StubSoundSegment("segment-0.flac")]}),
    )
    app = SoundSegmentsEditorTextualApp(project)  # type: ignore[arg-type]

    app.action_toggle_deletion()
    app.action_toggle_word_errors_filter()
    app.action_play_sound()
    app.action_show_info()
    app.commit_changes_and_exit()

    assert app.all_phrase_indices == []
    assert app.staged_deletion_flags == []
    assert app.word_errors_filter_active is False
    assert project.sound_segments.deletion_calls == []
    assert app.has_changes is False


def test_only_generated_phrases_are_displayed_once_in_project_order() -> None:
    app, project = make_app(6, {5, 1})
    project.sound_segments.sound_segments_map[5].append(
        StubSoundSegment("second-generation-for-line-6.flac")
    )
    project.sound_segments.sound_segments_map[-1] = [StubSoundSegment("invalid.flac")]
    project.sound_segments.sound_segments_map[12] = [StubSoundSegment("stale.flac")]

    app = SoundSegmentsEditorTextualApp(project)  # type: ignore[arg-type]
    app.load_content()

    assert app.phrase_indices == [1, 5]
    assert app.staged_deletion_flags == [False, False]
    assert str(app.format_line(0)) == "[00002] [      ] Line 2"
    assert str(app.format_line(1)) == "[00006] [      ] Line 6"


def test_only_positive_best_segment_word_error_count_is_displayed() -> None:
    app, project = make_app(3)
    project.sound_segments.sound_segments_map[0] = [
        StubSoundSegment("first.flac", num_errors=3),
        StubSoundSegment("best.flac", num_errors=1),
        StubSoundSegment("unknown.flac"),
    ]
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("zero-errors.flac", num_errors=0),
    ]

    assert str(app.format_line(0)) == (
        "[00001] [      ] [word errors: 1] Line 1"
    )
    formatted_line = app.format_line(0)
    word_errors_span = next(
        span
        for span in formatted_line.spans
        if str(formatted_line)[span.start : span.end] == "[word errors: 1]"
    )
    assert isinstance(word_errors_span.style, Style)
    assert word_errors_span.style.color
    assert word_errors_span.style.color.get_truecolor() == (255, 175, 0)
    assert str(app.format_line(1)) == "[00002] [      ] Line 2"
    assert str(app.format_line(2)) == "[00003] [      ] Line 3"


def test_failed_word_error_count_has_error_colored_asterisk() -> None:
    app, project = make_app(1)
    failed_segment = StubSoundSegment("failed.flac", num_errors=2)
    project.sound_segments.sound_segments_map[0] = [failed_segment]
    project.sound_segments.failed_segment_files.add(failed_segment.file_name)

    formatted_line = app.format_line(0)

    assert str(formatted_line) == "[00001] [      ] [word errors: 2 *] Line 1"
    asterisk_span = next(
        span
        for span in formatted_line.spans
        if str(formatted_line)[span.start : span.end] == "*"
    )
    assert isinstance(asterisk_span.style, Style)
    assert asterisk_span.style.color
    assert asterisk_span.style.color.get_truecolor() == (255, 0, 0)


def test_long_text_wraps_with_hanging_indent_and_is_limited_to_three_lines() -> None:
    app, project = make_app(1)
    project.phrase_groups[0].presentable_text = "one two three four five six seven"

    rendered = render_line(app, 0, width=25)

    assert rendered.splitlines() == [
        "[00001] [      ] one two",
        "                 three",
        "                 four…",
    ]


def test_inactive_selected_line_dim_background_extends_to_full_row_width() -> None:
    app, _ = make_app(2)

    async def exercise() -> None:
        async with app.run_test(size=(30, 20)) as pilot:
            await pilot.press("shift+down")
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            line = option_list.render_line(0)
            assert line.cell_length == option_list.scrollable_content_region.width
            assert all(
                segment.style is not None
                and segment.style.reverse
                and segment.style.color is not None
                and tuple(segment.style.color.get_truecolor()) == (136, 136, 136)
                for segment in line
            )

    run(exercise())


def test_deletion_toggle_applies_to_every_selected_generated_phrase() -> None:
    app, _ = make_app(6)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "shift+down", "x")
            assert app.staged_deletion_flags == [True, True, True, False, False, False]
            first_line = app.format_line(0)
            assert str(first_line).startswith("[00001] [DELETE]")
            delete_span = next(
                span
                for span in first_line.spans
                if str(first_line)[span.start : span.end] == "DELETE"
            )
            assert delete_span.style
            assert app.selected_indices == {2}
            assert app.has_changes is True

            await pilot.press("shift+up", "x")
            assert app.staged_deletion_flags == [True, False, False, False, False, False]
            assert str(app.format_line(1)).startswith("[00002] [      ]")
            assert app.selected_indices == {1}

            await pilot.press("home", "x")
            assert app.staged_deletion_flags == [False] * 6
            assert app.has_changes is False

    run(exercise())


def test_e_toggles_word_error_filter_and_preserves_original_line_numbers() -> None:
    app, project = make_app(6)
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("line-2.flac", num_errors=2)
    ]
    project.sound_segments.sound_segments_map[4] = [
        StubSoundSegment("line-5.flac", num_errors=1)
    ]

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            assert app.word_errors_filter_active is True
            assert app.phrase_indices == [1, 4]
            assert str(app.format_line(0)).startswith("[00002] [      ] [word errors: 2]")
            assert str(app.format_line(1)).startswith("[00005] [      ] [word errors: 1]")

            await pilot.press("e")
            assert app.word_errors_filter_active is False
            assert app.phrase_indices == [0, 1, 2, 3, 4, 5]

    run(exercise())


def test_word_error_filter_can_be_empty_and_deletion_flags_survive_toggle() -> None:
    app, project = make_app(3)
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("line-2.flac", num_errors=1)
    ]

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "x", "e")
            assert app.phrase_indices == [1]
            assert str(app.format_line(0)).startswith("[00002] [DELETE]")

            project.sound_segments.sound_segments_map[1][0].num_errors = 0
            await pilot.press("e", "e")
            assert app.phrase_indices == []
            assert app.selected_index is None
            assert app.selected_indices == set()

            await pilot.press("e")
            assert app.phrase_indices == [0, 1, 2]
            assert str(app.format_line(1)).startswith("[00002] [DELETE]")

    run(exercise())


def test_p_plays_highlighted_best_sound_segment() -> None:
    app, project = make_app(3)
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("worse.flac", num_errors=2),
        StubSoundSegment("best.flac", num_errors=0),
    ]

    async def exercise() -> None:
        with (
            patch.object(
                PlaySoundUtil, "current_sound_id", return_value="second-sound-id"
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("second-sound-id", ""),
            ) as play_sound,
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "p")
                play_sound.assert_called_once_with("/project/segments/best.flac")
                assert app.playing_sound_id == "second-sound-id"
                assert app.playing_sound_path == "/project/segments/best.flac"
                assert app.playing_phrase_index == 1
                playing_status = app.query_one("#playing-status", Static)
                assert str(playing_status.render()) == "Playing line 2"

    run(exercise())


def test_p_on_current_sound_stops_it_instead_of_restarting() -> None:
    app, _ = make_app(1)

    async def exercise() -> None:
        with (
            patch.object(
                PlaySoundUtil,
                "current_sound_id",
                return_value="sound-id",
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ) as play_sound,
            patch.object(PlaySoundUtil, "stop_sound_async") as stop_sound,
        ):
            async with app.run_test() as pilot:
                await pilot.press("p")
                await pilot.press("p")
                play_sound.assert_called_once_with("/project/segments/segment-0.flac")
                stop_sound.assert_called_once_with()
                assert app.playing_sound_id == ""
                assert app.playing_sound_path == ""
                assert app.playing_phrase_index is None
                playing_status = app.query_one("#playing-status", Static)
                assert str(playing_status.render()) == ""

    run(exercise())


def test_p_on_different_sound_starts_new_playback() -> None:
    app, _ = make_app(2)
    current_sound_id = "first-sound-id"

    async def exercise() -> None:
        def get_current_sound_id() -> str:
            return current_sound_id

        def play_sound(path: str) -> tuple[str, str]:
            nonlocal current_sound_id
            current_sound_id = (
                "second-sound-id" if path.endswith("segment-1.flac") else "first-sound-id"
            )
            return current_sound_id, ""

        with (
            patch.object(
                PlaySoundUtil,
                "current_sound_id",
                side_effect=get_current_sound_id,
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                side_effect=play_sound,
            ) as play_sound,
            patch.object(PlaySoundUtil, "stop_sound_async"),
        ):
            async with app.run_test() as pilot:
                await pilot.press("p")
                await pilot.press("down", "p")
                assert play_sound.call_args_list == [
                    (("/project/segments/segment-0.flac",), {}),
                    (("/project/segments/segment-1.flac",), {}),
                ]
                assert app.playing_sound_id == "second-sound-id"
                assert app.playing_sound_path == "/project/segments/segment-1.flac"
                playing_status = app.query_one("#playing-status", Static)
                assert str(playing_status.render()) == "Playing line 2"

    run(exercise())


def test_i_shows_ansi_formatted_segment_info_for_selected_phrase(
    tmp_path: Path,
) -> None:
    app, project = make_app(2)
    project.sound_segments_path = str(tmp_path)
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("worse.flac", num_errors=2),
        StubSoundSegment("best.flac", num_errors=0),
    ]
    filename = make_terminal_hyperlink(
        str(tmp_path / "best.flac"), "best.flac", is_file=True
    )
    info_lines = [
        "\033[31mLine: 2, word errors detected: 0\033[0m",
        f"Filename: {filename}",
        "\033[2;3mSource text: The selected line\033[0m",
    ]

    async def exercise() -> None:
        with (
            patch.object(
                SegmentTranscriptUtil,
                "make_info_text_lines",
                return_value=info_lines,
            ) as make_info_text_lines,
            patch.object(
                AudioMetaUtil, "get_audio_duration", return_value=10.84
            ) as get_audio_duration,
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "i")
                assert isinstance(app.screen, SegmentInfoDialog)
                make_info_text_lines.assert_called_once_with(
                    1, project, is_for_dialog=True
                )
                get_audio_duration.assert_called_once_with(str(tmp_path / "best.flac"))

                rendered_info = app.screen.query_one(
                    "#segment-info-content", Static
                ).render()
                assert isinstance(rendered_info, Content)
                assert str(rendered_info) == (
                    "Line: 2, word errors detected: 0\n"
                    "Duration: 10.8s\n"
                    "\n"
                    "Filename: best.flac\n"
                    "Source text: The selected line"
                )
                styles = [span.style for span in app.screen.info_text.spans]
                assert any(getattr(style, "color", None) is not None for style in styles)
                assert any(getattr(style, "italic", None) for style in styles)
                assert any(
                    getattr(style, "link", None) == f"file://{tmp_path / 'best.flac'}"
                    for style in styles
                )

    run(exercise())


def test_info_dialog_omits_duration_when_audio_load_fails() -> None:
    app, project = make_app(1)
    info_lines = ["\033[31mCould not load segment STT info: invalid JSON\033[0m"]

    async def exercise() -> None:
        with (
            patch.object(
                SegmentTranscriptUtil,
                "make_info_text_lines",
                return_value=info_lines,
            ),
            patch.object(
                AudioMetaUtil,
                "get_audio_duration",
                side_effect=OSError("couldn't load audio"),
            ),
        ):
            async with app.run_test() as pilot:
                await pilot.press("i")
                assert isinstance(app.screen, SegmentInfoDialog)
                assert str(
                    app.screen.query_one("#segment-info-content", Static).render()
                ) == "Could not load segment STT info: invalid JSON"

    run(exercise())


def test_info_dialog_is_height_limited_scrollable_and_escape_dismisses(
) -> None:
    app, _ = make_app(1)
    info_lines = [f"metadata {index}" for index in range(100)]

    async def exercise() -> None:
        with patch.object(
            SegmentTranscriptUtil, "make_info_text_lines", return_value=info_lines
        ):
            async with app.run_test(size=(80, 18)) as pilot:
                await pilot.press("i")
                assert isinstance(app.screen, SegmentInfoDialog)
                dialog = app.screen.query_one("#segment-info-dialog", Vertical)
                scroll = app.screen.query_one("#segment-info-scroll", VerticalScroll)
                assert dialog.region.x == 0
                assert dialog.region.width == 80
                assert dialog.region.y >= 2
                assert dialog.region.bottom <= 16
                assert scroll.virtual_size.height > scroll.region.height

                await pilot.press("escape")
                assert not isinstance(app.screen, SegmentInfoDialog)

    run(exercise())


def test_info_dialog_inside_click_stays_open_and_outside_click_dismisses(
) -> None:
    app, _ = make_app(1)

    async def exercise() -> None:
        with patch.object(
            SegmentTranscriptUtil,
            "make_info_text_lines",
            return_value=["Segment info"],
        ):
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("i")
                assert isinstance(app.screen, SegmentInfoDialog)
                dialog = app.screen.query_one("#segment-info-dialog", Vertical)

                await pilot.click(dialog)
                assert isinstance(app.screen, SegmentInfoDialog)

                await pilot.click(offset=(0, 0))
                assert not isinstance(app.screen, SegmentInfoDialog)

    run(exercise())


def test_playback_and_selection_status_coexist_but_find_bar_replaces_them() -> None:
    app, _ = make_app(4)

    async def exercise() -> None:
        with (
            patch.object(PlaySoundUtil, "current_sound_id", return_value="sound-id"),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ),
            patch.object(PlaySoundUtil, "stop_sound_async"),
        ):
            async with app.run_test() as pilot:
                await pilot.press("p", "shift+down", "shift+down")
                status_bar = app.query_one("#status-bar", Horizontal)
                playing_status = app.query_one("#playing-status", Static)
                selection_status = app.query_one("#selection-status", Static)
                find_bar = app.query_one("#find-bar", Horizontal)
                assert str(playing_status.render()) == "Playing line 1"
                assert str(selection_status.render()) == "3 lines selected"
                assert status_bar.display is True
                assert find_bar.display is False

                await pilot.press("ctrl+f")
                assert status_bar.display is False
                assert find_bar.display is True

                await pilot.press("escape")
                assert status_bar.display is True
                assert find_bar.display is False

    run(exercise())


def test_playback_status_clears_dynamically_when_sound_finishes() -> None:
    app, _ = make_app(1)
    current_sound_id = "sound-id"

    async def exercise() -> None:
        def get_current_sound_id() -> str:
            return current_sound_id

        with (
            patch.object(
                PlaySoundUtil, "current_sound_id", side_effect=get_current_sound_id
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ),
        ):
            async with app.run_test() as pilot:
                nonlocal current_sound_id
                await pilot.press("p")
                assert str(app.query_one("#playing-status", Static).render()) == (
                    "Playing line 1"
                )

                current_sound_id = ""
                await pilot.pause(0.2)
                assert str(app.query_one("#playing-status", Static).render()) == ""
                assert app.playing_phrase_index is None

    run(exercise())


def test_playback_timer_does_not_rewrite_unchanged_status() -> None:
    app, _ = make_app(1)
    app.playing_sound_id = "sound-id"
    app.playing_phrase_index = 0

    async def exercise() -> None:
        with (
            patch.object(PlaySoundUtil, "current_sound_id", return_value="sound-id"),
            patch.object(app, "show_playback_status") as show_playback_status,
        ):
            async with app.run_test():
                app.update_playback_status()
                show_playback_status.assert_not_called()

    run(exercise())


def test_empty_generated_set_is_safe_and_exits_cleanly() -> None:
    app, _ = make_app(3, set())

    assert app.phrase_indices == []
    assert app.selected_index is None
    assert app.selected_indices == set()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+a", "x", "escape")
            await pilot.pause()
            assert app.is_running is False

    run(exercise())


def test_dirty_escape_cancel_returns_to_reviewer() -> None:
    app, project = make_app(2)
    project.sound_segments.sound_segments_map[0] = [
        StubSoundSegment(f"segment-{index}.flac") for index in range(50)
    ]

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x", "escape")
            assert isinstance(app.screen, SaveChangesDialog)
            question = app.screen.query_one("#save-changes-copy-line-1", Static)
            assert str(question.render()) == (
                "Delete the 50 sound segment files marked for deletion?"
            )

            await pilot.press("escape")
            assert not isinstance(app.screen, SaveChangesDialog)
            assert app.is_running is True
            assert app.has_changes is True

    run(exercise())


def test_save_deletes_original_phrase_indices() -> None:
    app, project = make_app(7, {1, 4, 6})
    original_map = {
        index: list(items)
        for index, items in project.sound_segments.sound_segments_map.items()
    }

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "x", "escape")
            assert isinstance(app.screen, SaveChangesDialog)

            yes_button = app.screen.query_one("#yes", Button)
            await pilot.click(yes_button)
            await pilot.pause()
            assert app.is_running is False

    run(exercise())
    assert project.sound_segments.deletion_calls == [[1, 4]]
    assert project.sound_segments.sound_segments_map == original_map
    assert app.did_save_changes is True
    assert app.save_error == ""


def test_save_ignores_marked_phrase_that_disappeared_while_editor_was_open() -> None:
    app, project = make_app(2)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x")
            del project.sound_segments.sound_segments_map[0]
            app.commit_changes_and_exit()
            await pilot.pause()
            assert app.is_running is False

    run(exercise())
    assert project.sound_segments.deletion_calls == [[]]
    assert app.did_save_changes is True
    assert app.save_error == ""
