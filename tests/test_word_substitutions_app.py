from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
from rich.text import Text
from textual.timer import Timer
from textual.widgets import Input, OptionList, Rule, Static

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    GenerationTerminalStatus,
    TtsPreviewFinished,
)
from tts_audiobook_tool.textual import worker_app as worker_app_module
from tts_audiobook_tool.textual import word_substitutions_app as word_subs_module
from tts_audiobook_tool.textual.quick_gen_modal import QuickGenJob, QuickGenModal
from tts_audiobook_tool.textual.content_textual_app import (
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.textual_shared import (
    STYLE_HIGHLIGHT,
    NonWrappingOptionList,
)
from tts_audiobook_tool.textual.word_substitutions_app import (
    ADD_ITEM_LABEL,
    NO_ITEMS_LABEL,
    WordSubstitutionItem,
    WordSubstitutionSentinel,
    WordSubstitutionsApp,
    fit_cell,
    make_word_substitution_preview_prompt,
)
from tts_audiobook_tool.textual.word_substitutions_dialog import (
    WordSubstitutionEdit,
    WordSubstitutionsDialog,
)

from textual_editor_stubs import run


@dataclass
class StubWordSubstitutionsProject:
    word_substitutions: dict[str, str] = field(default_factory=dict)
    save_error: str = ""
    save_calls: int = 0
    # Playback-shaping inputs read by the preview autoplay path.
    high_shelf: HighShelfEq = HighShelfEq.DISABLED
    limit_silence_gaps: bool = False
    limit_silence_gaps_duration: float = 0.5

    def save(self) -> str:
        self.save_calls += 1
        return self.save_error

    # Read by the quick-generation modal's worker-log chrome.
    dir_path: str = ""

    def get_tts_model_type(self) -> TtsModelType:
        return TtsModelType.require_by_id("none")

    def get_high_shelf(self) -> HighShelfEq:
        return self.high_shelf


def make_app(
    values: dict[str, str] | None = None,
    save_error: str = "",
) -> tuple[WordSubstitutionsApp, StubWordSubstitutionsProject]:
    project = StubWordSubstitutionsProject(
        dict(values or {}), save_error=save_error
    )
    state = cast(State, SimpleNamespace(project=cast(Project, project)))
    app = WordSubstitutionsApp(state)
    return app, project


def originals(app: WordSubstitutionsApp) -> list[str]:
    return [
        item.original
        for item in app.list_items
        if isinstance(item, WordSubstitutionItem)
    ]


def test_empty_project_shows_no_items_sentinel_then_add_sentinel() -> None:
    app, _ = make_app()

    assert app.list_items == [
        WordSubstitutionSentinel(NO_ITEMS_LABEL, is_add=False),
        WordSubstitutionSentinel(ADD_ITEM_LABEL, is_add=True),
    ]
    assert app.content_line_index(0) is None
    assert app.content_line_index(1) is None


def test_rows_sort_case_insensitively_with_one_indexed_ordinals() -> None:
    app, _ = make_app({"Zebra": "z", "apple": "a", "Mango": "m"})

    assert originals(app) == ["apple", "Mango", "Zebra"]
    assert app.list_items[0] == WordSubstitutionItem("apple", "a", 1)
    assert app.list_items[1] == WordSubstitutionItem("Mango", "m", 2)
    assert app.list_items[2] == WordSubstitutionItem("Zebra", "z", 3)
    assert app.list_items[-1] == WordSubstitutionSentinel(ADD_ITEM_LABEL, is_add=True)
    assert app.content_line_index(0) == 0
    assert app.content_line_index(3) is None


def test_table_header_and_row_share_column_layout_at_width_60() -> None:
    app, _ = make_app({"Ariekei": "AriaKay"})

    header = app.table_header_row().render_line(60).plain
    row = app.format_line(0).render_line(60).plain

    # ordinal (3) + ordinal gap (2) + original (27) + gap (1) + substitution (27)
    assert len(header) == 60
    assert len(row) == 60
    assert header[:3] == "   "  # no "#" label above the ordinal column
    assert header[5:32] == "Original word:".ljust(27)
    assert header[33:] == "Substitution word:".ljust(27)
    assert row[:3] == "  1"
    assert row[5:32] == "Ariekei".ljust(27)
    assert row[33:] == "AriaKay".ljust(27)


def test_long_values_are_ellipsized_with_a_margin_to_the_exact_row_width() -> None:
    app, _ = make_app({"a" * 40: "b" * 40})

    row = app.format_line(0).render_line(45).plain

    assert len(row) == 45
    assert "a" * 20 not in row  # column width is (45 - 6) // 2 == 19
    # Each cell ends with "…" plus a blank margin cell
    assert row[5:24] == ("a" * 17) + "\u2026 "
    assert row[25:45] == ("b" * 18) + "\u2026 "


def test_fit_cell_ellipsis_reserves_a_trailing_margin_cell() -> None:
    cell = fit_cell("a" * 30, 10, ellipsis=True)

    assert Text(cell).cell_len == 10
    assert cell == ("a" * 8) + "\u2026 "

    # A one-cell column has no room for both text and a margin.
    assert fit_cell("a" * 30, 1, ellipsis=True) == "\u2026"


def test_fit_cell_ellipsis_leaves_values_that_fit_alone() -> None:
    assert fit_cell("a" * 10, 10, ellipsis=True) == "a" * 10
    assert fit_cell("a" * 9, 10, ellipsis=True) == ("a" * 9) + " "
    assert "\u2026" not in fit_cell("a" * 10, 10, ellipsis=True)


def test_values_that_fill_their_column_exactly_are_not_ellipsized() -> None:
    app, _ = make_app({"a" * 19: "b" * 20})

    row = app.format_line(0).render_line(45).plain

    assert row[5:24] == "a" * 19
    assert row[25:45] == "b" * 20
    assert "\u2026" not in row


def test_add_item_sentinel_uses_highlight_color() -> None:
    app, _ = make_app()
    add_index = len(app.list_items) - 1

    row = app.format_line(add_index).render_line(40)

    assert row.plain.strip() == ADD_ITEM_LABEL
    assert row.spans
    assert str(row.spans[0].style) == STYLE_HIGHLIGHT


def test_handle_dialog_result_adds_item_and_selects_it() -> None:
    app, _ = make_app()

    app.handle_dialog_result(WordSubstitutionEdit("Ariekei", "AriaKay"))

    assert app.staged == {"Ariekei": "AriaKay"}
    assert app.list_items[0] == WordSubstitutionItem("Ariekei", "AriaKay", 1)
    assert app.selected_index == 0


def test_handle_dialog_result_rename_resorts_and_follows_key() -> None:
    app, _ = make_app({"apple": "a", "zebra": "z"})

    app.handle_dialog_result(
        WordSubstitutionEdit("aardvark", "zz", previous_original="zebra")
    )

    assert app.staged == {"apple": "a", "aardvark": "zz"}
    assert originals(app) == ["aardvark", "apple"]
    assert app.selected_index == 0


def test_handle_dialog_result_ignores_cancellation() -> None:
    app, _ = make_app({"apple": "a"})

    app.handle_dialog_result(None)

    assert app.staged == {"apple": "a"}
    assert app.has_changes is False


def test_has_changes_tracks_staged_dictionary() -> None:
    app, _ = make_app({"apple": "a"})
    assert app.has_changes is False

    app.handle_dialog_result(WordSubstitutionEdit("apple", "b"))

    assert app.has_changes is True


def test_commit_saves_and_exits_saved() -> None:
    app, project = make_app()
    app.handle_dialog_result(WordSubstitutionEdit("Ariekei", "AriaKay"))
    exits: list[object] = []
    app.exit = exits.append  # type: ignore[method-assign]

    app.commit_changes_and_exit()

    assert project.word_substitutions == {"Ariekei": "AriaKay"}
    assert project.save_calls == 1
    assert exits == [EditorSaved()]


def test_commit_rolls_back_project_on_save_failure() -> None:
    app, project = make_app({"apple": "a"}, save_error="disk full")
    app.handle_dialog_result(WordSubstitutionEdit("apple", "b"))
    exits: list[object] = []
    app.exit = exits.append  # type: ignore[method-assign]

    app.commit_changes_and_exit()

    assert project.word_substitutions == {"apple": "a"}
    assert exits == [EditorSaveFailed("Save failed: disk full")]


def test_mounted_app_shows_header_and_no_items_sentinel_does_nothing() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.query_one("#table-header", Static)
            assert app.query_one("#table-header-divider", Rule)
            assert app.query_one("#status-left", Static).render() == ""
            assert app.query_one("#status-right", Static).render() == ""
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            assert option_list.option_count == 2

            # Row 0 is the "No items currently" sentinel.
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, WordSubstitutionsDialog)

            # Row 1 is "Add item".
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert "Add item" in str(
                app.screen.query_one("#word-substitutions-title", Static).render()
            )
            assert app.screen.query_one("#original-input", Input).value == ""

    run(exercise())


def test_mounted_app_opens_edit_dialog_and_stages_committed_change() -> None:
    app, _ = make_app({"Ariekei": "AriaKay"})

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.query_one("#line-list", OptionList).option_count == 2

            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert "Edit item" in str(
                app.screen.query_one("#word-substitutions-title", Static).render()
            )
            original_input = app.screen.query_one("#original-input", Input)
            assert original_input.value == "Ariekei"

            original_input.value = "Ariekei"
            app.screen.query_one("#replacement-input", Input).value = "Aria-Kay"
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, WordSubstitutionsDialog)
            assert app.staged == {"Ariekei": "Aria-Kay"}
            assert app.has_changes is True

    run(exercise())


def test_pressing_x_deletes_current_row_keeps_position_and_shows_toast() -> None:
    app, _ = make_app({"apple": "a", "Banana": "b", "cherry": "c"})

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            assert originals(app) == ["apple", "Banana", "cherry"]
            await pilot.press("down")  # highlight "Banana"

            await pilot.press("x")
            await pilot.pause()

            assert app.staged == {"apple": "a", "cherry": "c"}
            assert app.has_changes is True
            assert originals(app) == ["apple", "cherry"]
            assert app.list_items[app.selected_index] == WordSubstitutionItem(
                "cherry", "c", 2
            )
            assert str(app.query_one("#status-left", Static).render()) == "Deleted item"

    run(exercise())


def test_pressing_x_on_sentinel_rows_deletes_nothing() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            # Row 0 is "No items currently", row 1 is "Add item".
            await pilot.press("x")
            await pilot.pause()
            assert app.staged == {}
            assert app.has_changes is False
            assert str(app.query_one("#status-left", Static).render()) == ""

            await pilot.press("down")
            await pilot.press("x")
            await pilot.pause()
            assert app.staged == {}
            assert str(app.query_one("#status-left", Static).render()) == ""

    run(exercise())


def test_preview_prompt_uses_both_literal_row_values() -> None:
    assert make_word_substitution_preview_prompt("Ariekei", "AriaKay") == (
        "Original word: Ariekei. Substitute word: AriaKay"
    )


def patch_preview_job(
    sound: Sound | None,
    *,
    problems: tuple[str, ...] = (),
    status: GenerationTerminalStatus = GenerationTerminalStatus.COMPLETED,
):
    """Patch the preview job factory with a fake worker job.

    Returns the prompts submitted and the patchers to apply; the fake worker
    answers each poll with one canned finish event.
    """
    prompts: list[str] = []

    def make_job(_state, title: str, prompt: str) -> QuickGenJob:
        def submit() -> str:
            prompts.append(prompt)
            return "job"

        return QuickGenJob(title=title, submit=submit, noun="Preview generation", problems=problems)

    events = [[TtsPreviewFinished("job", status, sound=sound)]]
    patchers = [
        patch.object(word_subs_module, "make_preview_job", make_job),
        patch.object(
            ModelWorker,
            "drain_events",
            staticmethod(lambda max_events=1000: events.pop(0) if events else []),
        ),
        patch.object(worker_app_module, "EVENT_POLL_SECONDS", 0.02),
    ]
    return prompts, patchers


def run_patched(patchers, coroutine) -> None:
    from contextlib import ExitStack

    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        run(coroutine)


def test_pressing_q_previews_in_place_with_unsaved_staged_state() -> None:
    # The preview modal opens over the live editor, using the highlighted
    # row's literal values (including unsaved staged edits); nothing is saved
    # and the editor keeps running with its staged state intact.
    app, project = make_app({"apple": "a", "zebra": "z"})
    app.handle_dialog_result(WordSubstitutionEdit("apple", "ay"))
    prompts, patchers = patch_preview_job(
        None, status=GenerationTerminalStatus.FAILED
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause(0.3)
            assert isinstance(app.screen, QuickGenModal)
            assert app.is_running is True
            assert app.staged == {"apple": "ay", "zebra": "z"}

    run_patched(patchers, exercise())
    assert project.save_calls == 0
    assert prompts == ["Original word: apple. Substitute word: ay"]


def test_pressing_q_on_sentinel_rows_does_nothing() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            assert app.preview_status_timer is None
            await pilot.press("q")
            await pilot.pause()
            assert app.is_running is True
            assert len(app.screen_stack) == 1

            await pilot.press("down", "q")
            await pilot.pause()
            assert app.is_running is True
            assert len(app.screen_stack) == 1

    run(exercise())


def test_preview_preflight_problems_show_in_modal_and_do_not_submit() -> None:
    # Pre-flight failures appear inside the modal and nothing is submitted;
    # "q" cannot stack a second session while it is open.
    app, _ = make_app({"apple": "ay"})
    prompts, patchers = patch_preview_job(None, problems=("Choose a voice",))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause(0.2)
            assert isinstance(app.screen, QuickGenModal)

            await pilot.press("q")
            await pilot.pause()
            assert len(app.screen_stack) == 2

            await pilot.press("escape")
            await pilot.pause()
            assert len(app.screen_stack) == 1
            assert app.modal_session_active is False

    run_patched(patchers, exercise())
    assert prompts == []


def test_completed_preview_autoplays_in_memory_sound() -> None:
    # On success the modal disappears and the returned sound plays, shaped for
    # playback (resampled to the app rate) rather than the raw model output.
    project = StubWordSubstitutionsProject({"apple": "a", "Banana": "b"})
    state = cast(State, SimpleNamespace(project=cast(Project, project)))
    sound = Sound(np.zeros(2_400, dtype=np.float32), 24_000)
    app = WordSubstitutionsApp(state)
    _, patchers = patch_preview_job(sound)

    async def exercise() -> None:
        with (
            patch.object(
                PlaySoundUtil, "play_sound_async", return_value="preview-id"
            ) as play,
            patch.object(
                PlaySoundUtil, "current_sound_id", return_value="preview-id"
            ),
            patch.object(PlaySoundUtil, "stop_sound_async", return_value=True),
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "q")
                await pilot.pause(0.6)
                assert not isinstance(app.screen, QuickGenModal)
                assert app.selected_index == 1
                played = play.call_args.args[0]
                assert played is not sound
                assert played.sr == APP_SAMPLE_RATE
                assert len(played.data) == 2 * len(sound.data)
                assert app.preview_sound_id == "preview-id"
                assert app.preview_status_timer is not None

    run_patched(patchers, exercise())


def test_overlapping_previews_stop_the_previous_status_timer() -> None:
    # Replacing active playback must retire its timer, so completion only
    # cleans up the current timer rather than leaving an orphan interval.
    app, _ = make_app({"apple": "a", "Banana": "b"})
    sound = Sound(np.zeros(240, dtype=np.float32), APP_SAMPLE_RATE)
    first_timer = Mock(spec=Timer)
    second_timer = Mock(spec=Timer)
    with (
        patch.object(app, "set_interval", side_effect=[first_timer, second_timer]),
        patch.object(
            SoundPipeline, "prepare_generated_sound_for_playback", return_value=sound
        ),
        patch.object(PlaySoundUtil, "play_sound_async", side_effect=["first", "second"]),
        patch.object(PlaySoundUtil, "current_sound_id", return_value="second") as current,
        patch.object(PlaySoundUtil, "stop_sound_async") as stop_sound,
    ):
        app.play_preview_sound(sound, "apple")
        app.play_preview_sound(sound, "Banana")
        first_timer.stop.assert_called_once_with()
        second_timer.stop.assert_not_called()
        assert app.preview_status_timer is second_timer
        assert app.preview_sound_id == "second"
        assert app.selected_status.right == "Playing pronunciation preview: Banana"

        app.update_preview_playback_status()
        assert app.preview_status_timer is second_timer
        current.return_value = ""
        app.update_preview_playback_status()
        second_timer.stop.assert_called_once_with()
        assert app.preview_status_timer is None
        assert app.preview_sound_id == ""
        assert app.selected_status.right == ""
        app.on_unmount()
        first_timer.stop.assert_called_once_with()
        second_timer.stop.assert_called_once_with()
        stop_sound.assert_not_called()


def test_preview_shapes_playback_with_project_settings() -> None:
    """The preview applies interactive playback shaping like the other paths."""
    project = StubWordSubstitutionsProject(
        {"apple": "a"},
        high_shelf=HighShelfEq.MODERATE,
        limit_silence_gaps=True,
        limit_silence_gaps_duration=1.25,
    )
    state = cast(State, SimpleNamespace(project=cast(Project, project)))
    sound = Sound(np.zeros(240, dtype=np.float32), 24_000)
    app = WordSubstitutionsApp(state)
    shaped = Sound(np.zeros(480, dtype=np.float32), APP_SAMPLE_RATE)
    _, patchers = patch_preview_job(sound)

    async def exercise() -> None:
        with (
            patch.object(
                SoundPipeline,
                "prepare_generated_sound_for_playback",
                return_value=shaped,
            ) as prepare,
            patch.object(
                PlaySoundUtil, "play_sound_async", return_value="preview-id"
            ) as play,
            patch.object(
                PlaySoundUtil, "current_sound_id", return_value="preview-id"
            ),
            patch.object(PlaySoundUtil, "stop_sound_async", return_value=True),
        ):
            async with app.run_test() as pilot:
                await pilot.press("q")
                await pilot.pause(0.6)
                prepare.assert_called_once_with(
                    sound,
                    high_shelf=HighShelfEq.MODERATE,
                    limit_silence_gaps=True,
                    limit_silence_gaps_duration=1.25,
                )
                play.assert_called_once_with(shaped)

    run_patched(patchers, exercise())


def test_pressing_x_on_only_item_restores_empty_state() -> None:
    app, _ = make_app({"apple": "a"})

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("x")
            await pilot.pause()

            assert app.staged == {}
            assert app.has_changes is True
            assert app.list_items == [
                WordSubstitutionSentinel(NO_ITEMS_LABEL, is_add=False),
                WordSubstitutionSentinel(ADD_ITEM_LABEL, is_add=True),
            ]
            assert str(app.query_one("#status-left", Static).render()) == "Deleted item"

    run(exercise())
