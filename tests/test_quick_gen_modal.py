"""QuickGenModal: in-place quick job hosted on a ModalScreen over a live app."""

import asyncio
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from textual.app import App
from textual.widgets import Static

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.generation_events import GenerationTimedOut, ModelUnhealthy
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleOutput,
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    TtsPreviewFinished,
)
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.worker_reset import HardResetCause, HardResetRequest
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual import quick_gen_modal as quick_gen_module
from tts_audiobook_tool.textual import worker_app as worker_app_module
from tts_audiobook_tool.textual.quick_gen_modal import (
    QuickGenJob,
    QuickGenModal,
    QuickGenResult,
    QuickGenReview,
)
from tts_audiobook_tool.textual.worker_content import WorkerLogContentArea
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class HostApp(App[None]):
    """Stands in for an editor: a live app that the modal is pushed over."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[QuickGenResult | None] = []
        self.editor_keys: list[str] = []

    def on_key(self, event) -> None:
        self.editor_keys.append(event.key)


def make_state() -> State:
    project = SimpleNamespace(
        get_tts_model_type=lambda: TtsModelType.require_by_id("none"),
        dir_path="",
    )
    prefs = SimpleNamespace(save_gen_log=False)
    return cast(State, SimpleNamespace(project=project, prefs=prefs))


def make_job(problems: tuple[str, ...] = ()) -> QuickGenJob:
    return QuickGenJob(
        title="Quick generate - line 3",
        submit=lambda: "job",
        problems=problems,
    )


def feed_events(monkeypatch, batches: list[list]) -> None:
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(lambda max_events=1000: batches.pop(0) if batches else []),
    )
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)


def run(coroutine) -> None:
    asyncio.run(coroutine)


def log_text(modal: QuickGenModal) -> str:
    return "\n".join(modal.query_one(WorkerLogContentArea).worker_log.line_texts())


def header_text(modal: QuickGenModal) -> str:
    return " ".join(
        str(modal.query_one(selector, Static).render())
        for selector in ("#quick-gen-title", "#quick-gen-status", "#quick-gen-prompt-row")
    )


def test_clean_success_dismisses_immediately(monkeypatch) -> None:
    # A cleanly completed job closes the modal with no review step.
    feed_events(monkeypatch, [[GenerationFinished("job", GenerationTerminalStatus.COMPLETED, "")]])
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            assert "Quick generate - line 3" in header_text(modal)
            assert "to cancel" in header_text(modal)
            modal._drain_worker_events()
            await pilot.pause(0.4)
            assert len(app.results) == 1
            assert app.results[0] is not None and app.results[0].completed_cleanly
            assert len(app.screen_stack) == 1

    run(exercise())


def test_preview_success_returns_sound(monkeypatch) -> None:
    # A preview job hands its in-memory sound back through the result.
    sound = Sound(np.zeros(24, dtype=np.float32), 24_000)
    feed_events(monkeypatch, [[TtsPreviewFinished("job", GenerationTerminalStatus.COMPLETED, sound=sound)]])
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)

    run(exercise())
    assert app.results[0] is not None and app.results[0].sound is sound


def test_failure_stays_open_until_escape(monkeypatch) -> None:
    # Anything but a clean success waits for the user: output stays reviewable.
    feed_events(monkeypatch, [[
        ConsoleOutput("job", "stdout", "some worker output\n"),
        GenerationFinished("job", GenerationTerminalStatus.FAILED, "", message="boom"),
    ]])
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)
            assert app.results == []
            assert len(app.screen_stack) == 2
            assert "some worker output" in log_text(modal)
            assert "boom" in log_text(modal)
            assert "to close" in header_text(modal)
            await pilot.press("escape")
            await pilot.pause()
            assert len(app.results) == 1
            assert app.results[0] is not None
            assert app.results[0].status is GenerationTerminalStatus.FAILED

    run(exercise())


def test_completed_with_word_errors_stays_open(monkeypatch) -> None:
    # A completed job whose item was tagged failed is not "clean": the modal
    # stays up for review, and ENTER closes it.
    from tts_audiobook_tool.generation_events import GenerationProgress
    from tts_audiobook_tool.model_worker_protocol import GenerationUpdate

    feed_events(monkeypatch, [[
        GenerationUpdate("job", GenerationProgress(1, 0, 1, failed=1)),
        GenerationFinished("job", GenerationTerminalStatus.COMPLETED, ""),
    ]])
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)
            assert app.results == []
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.results) == 1
            assert app.results[0] is not None and app.results[0].completed
            assert not app.results[0].completed_cleanly

    run(exercise())


def test_preflight_problems_are_listed_and_nothing_is_submitted(monkeypatch) -> None:
    # Pre-flight failures never reach the worker; every problem is shown.
    submitted: list[bool] = []
    feed_events(monkeypatch, [])
    job = QuickGenJob(
        title="Quick generate - line 3",
        submit=lambda: submitted.append(True) or "job",
        problems=("Voice file a.wav not found", "Voice file b.wav has no transcript"),
    )
    app = HostApp()
    modal = QuickGenModal(make_state(), job)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause(0.3)
            text = log_text(modal)
            assert "Voice file a.wav not found" in text
            assert "Voice file b.wav has no transcript" in text
            assert "to close" in header_text(modal)
            await pilot.press("escape")
            await pilot.pause()
            assert len(app.results) == 1
            assert app.results[0] is not None and not app.results[0].submitted

    run(exercise())
    assert submitted == []


def test_keys_do_not_reach_the_editor_beneath(monkeypatch) -> None:
    # While the modal is up, ordinary keys must not leak to the host app's
    # key handling (the editors bind q/x/space etc. at app level).
    feed_events(monkeypatch, [])
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            for key in ("q", "x", "space"):
                await pilot.press(key)
            await pilot.pause()
            assert len(app.screen_stack) == 2
            assert app.results == []

    run(exercise())


def test_watchdog_timeout_hard_resets_worker_and_waits_for_review(monkeypatch) -> None:
    # The worker-health path of the shared controller works hosted on a screen:
    # a watchdog event hard-resets (thread worker reporting back to the app),
    # and the modal then stays open showing the cause.
    resets: list[int] = []
    delivered = {"on": False}
    events = [GenerationUpdate("job", GenerationTimedOut(180.0))]
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(lambda max_events=1000: [events.pop(0)] if delivered["on"] and events else []),
    )
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(lambda: resets.append(1) or ""))
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            delivered["on"] = True
            modal._drain_worker_events()
            await pilot.pause(0.5)
            assert resets == [1]
            assert modal.terminal_result is not None
            assert modal.terminal_result.status is GenerationTerminalStatus.WORKER_RESET
            assert modal.terminal_result.hard_reset_cause is HardResetCause.GENERATION_TIMEOUT
            assert "GEN_TIMEOUT" in modal.terminal_result.message
            assert app.results == []
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.results) == 1

    run(exercise())


def test_late_updates_do_not_preempt_a_reset_in_progress(monkeypatch) -> None:
    # A second health event arriving mid-reset must not start another reset.
    resets: list[int] = []
    pending = {"count": 0}
    events = [
        GenerationUpdate("job", GenerationTimedOut(180.0)),
        GenerationUpdate("job", ModelUnhealthy(reason="boom")),
    ]

    def drain(max_events=1000):
        if pending["count"] > 0:
            pending["count"] -= 1
            return [events.pop(0)]
        return []

    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(drain))
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(lambda: resets.append(1) or ""))
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            pending["count"] = 2
            modal._drain_worker_events()
            assert modal.reset_in_progress is True
            modal._drain_worker_events()
            await pilot.pause(0.5)
            assert resets == [1]
            assert modal.terminal_result.hard_reset_cause is HardResetCause.GENERATION_TIMEOUT

    run(exercise())


def test_escape_cancels_then_second_escape_hard_resets(monkeypatch) -> None:
    # The cancel -> hard-reset ladder: first ESC requests cooperative cancel
    # (and the prompt switches to the kill hint), second ESC resets. Running
    # jobs cannot be dismissed any other way.
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    cancellations: list[str] = []
    resets: list[int] = []
    feed_events(monkeypatch, [])
    monkeypatch.setattr(
        ModelWorker, "request_cancel",
        staticmethod(lambda operation_id: cancellations.append(operation_id) or True),
    )
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(lambda: resets.append(1) or ""))
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            await pilot.press("enter")
            assert app.results == []
            await pilot.press("escape")
            assert cancellations == ["job"]
            assert "kill process" in header_text(modal)
            assert resets == []
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert resets == [1]
            assert modal.terminal_result.hard_reset_cause is HardResetCause.USER_ESCALATION
            assert app.results == []

    run(exercise())


def test_job_height_sets_dialog_height(monkeypatch) -> None:
    # Previews print little, so their jobs ask for a shorter dialog.
    feed_events(monkeypatch, [])
    job = QuickGenJob(title="Preview", submit=lambda: "job", height="16")
    app = HostApp()
    modal = QuickGenModal(make_state(), job)

    async def exercise() -> None:
        async with app.run_test(size=(100, 40)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause(0.3)
            assert modal.query_one("#quick-gen-dialog").region.height == 16

    run(exercise())


def test_status_is_right_justified_on_the_title_row(monkeypatch) -> None:
    # Row 1 holds the title at the left and the status flush against the right
    # edge of the header.
    feed_events(monkeypatch, [])
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause(0.3)
            title = modal.query_one("#quick-gen-title", Static)
            status = modal.query_one("#quick-gen-status", Static)
            header = modal.query_one("#quick-gen-header")
            assert "Status:" in str(status.render())
            assert title.region.y == status.region.y == header.region.y
            assert title.region.x < status.region.x
            # Header has 1 cell of right padding.
            assert status.region.right == header.region.right - 1

    run(exercise())


@pytest.mark.parametrize("exit_kind", ["palette_quit", "interface_error"])
def test_abnormal_host_exit_resets_owned_job_and_reconciles(monkeypatch, exit_kind) -> None:
    # Palette Quit and UI exceptions bypass dismissal; teardown must terminate
    # owned inference before re-reading the audio that survived on disk.
    feed_events(monkeypatch, [])
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ModelWorker, "request_cancel",
        staticmethod(lambda operation_id: calls.append(("cancel", operation_id)) or True),
    )
    monkeypatch.setattr(
        ModelWorker, "reset", staticmethod(lambda: calls.append(("reset", "")) or "")
    )
    job = replace(
        make_job(),
        reconcile=lambda remaining: calls.append(("reconcile", remaining)) or "",
    )
    app = HostApp()
    modal = QuickGenModal(make_state(), job)
    lock = Mock()
    monkeypatch.setattr(quick_gen_module, "SystemSleepLock", lambda: lock)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            if exit_kind == "palette_quit":
                await pilot.press("ctrl+p", "q", "u", "i", "t", "enter")
            else:
                def fail_interface() -> None:
                    raise RuntimeError("broken interface")

                app.call_later(fail_interface)
            await pilot.pause()

    if exit_kind == "interface_error":
        with pytest.raises(RuntimeError, match="broken interface"):
            run(exercise())
    else:
        run(exercise())
    assert calls == [("cancel", "job"), ("reset", ""), ("reconcile", "")]
    assert app.results == []
    assert modal.terminal_result is not None
    assert modal.terminal_result.hard_reset_cause is HardResetCause.INTERFACE_FAILURE
    lock.release.assert_called()


def test_abnormal_exit_after_terminal_event_reconciles_without_reset(monkeypatch) -> None:
    # Closing a review screen via Quit still needs queue reconciliation, but
    # must not reset a job that has already reached its terminal result.
    feed_events(monkeypatch, [[
        GenerationFinished("job", GenerationTerminalStatus.FAILED, "2-3"),
    ]])
    reconcile = Mock(return_value="")
    cancel = Mock(return_value=True)
    reset = Mock(return_value="")
    monkeypatch.setattr(ModelWorker, "request_cancel", cancel)
    monkeypatch.setattr(ModelWorker, "reset", reset)
    app = HostApp()
    modal = QuickGenModal(make_state(), replace(make_job(), reconcile=reconcile))

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)
            await pilot.press("ctrl+p", "q", "u", "i", "t", "enter")
            await pilot.pause()

    run(exercise())
    reconcile.assert_called_once_with("2-3")
    cancel.assert_not_called()
    reset.assert_not_called()


def test_normal_dismissal_leaves_reconciliation_to_editor(monkeypatch) -> None:
    # A normally delivered result is reconciled by the editor, not a second
    # teardown path which could overwrite newer queue edits.
    feed_events(monkeypatch, [[
        GenerationFinished("job", GenerationTerminalStatus.COMPLETED, "2"),
    ]])
    reconcile = Mock(return_value="")
    cancel = Mock(return_value=True)
    monkeypatch.setattr(ModelWorker, "request_cancel", cancel)
    app = HostApp()
    modal = QuickGenModal(make_state(), replace(make_job(), reconcile=reconcile))

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)

    run(exercise())
    assert len(app.results) == 1
    reconcile.assert_not_called()
    cancel.assert_not_called()


def test_abnormal_exit_does_not_reset_an_unowned_operation(monkeypatch) -> None:
    # A stale screen whose operation no longer owns the worker must never
    # reset a successor; the operation-id-guarded cancellation rejects it.
    feed_events(monkeypatch, [])
    cancel = Mock(return_value=False)
    reset = Mock(return_value="")
    monkeypatch.setattr(ModelWorker, "request_cancel", cancel)
    monkeypatch.setattr(ModelWorker, "reset", reset)
    app = HostApp()
    modal = QuickGenModal(make_state(), make_job())

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            app.exit()

    run(exercise())
    cancel.assert_called_once_with("job")
    reset.assert_not_called()


def test_exit_during_reset_finishes_existing_reset_once(monkeypatch) -> None:
    # Textual cancels thread workers on exit, but their process reset keeps
    # running. Teardown must join it, not concurrently reset its replacement.
    feed_events(monkeypatch, [])
    resetting = Event()
    unmounting = Event()
    calls: list[str] = []

    def reset() -> str:
        calls.append("reset")
        resetting.set()
        assert unmounting.wait(5.0)
        calls.append("reset finished")
        return ""

    monkeypatch.setattr(ModelWorker, "reset", staticmethod(reset))
    app = HostApp()
    class ResetExitModal(QuickGenModal):
        async def on_unmount(self) -> None:
            # Textual dispatches class-level handlers through the MRO, then
            # invokes QuickGenModal's cleanup; instance monkeypatches are ignored.
            unmounting.set()

    modal = ResetExitModal(
        make_state(),
        replace(make_job(), reconcile=lambda _: calls.append("reconcile") or ""),
    )

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._begin_hard_reset(HardResetRequest(HardResetCause.USER_ESCALATION))
            assert await asyncio.to_thread(resetting.wait, 5.0)
            app.exit()

    run(exercise())
    assert calls == ["reset", "reset finished", "reconcile"]
    assert modal.terminal_result is not None
    assert modal.terminal_result.hard_reset_cause is HardResetCause.USER_ESCALATION


def test_generation_job_is_staged_and_needs_no_reconciliation(monkeypatch) -> None:
    # A line regeneration writes only to its own staging directory, so there is
    # no project state to reconcile; the submit forwards that directory.
    monkeypatch.setattr(quick_gen_module, "collect_preflight_problems", lambda *_, **__: ())
    submit = Mock(return_value="job")
    monkeypatch.setattr(ModelWorker, "submit_generation", submit)
    state = make_state()
    state.project.dir_path = "/project"  # type: ignore[misc]
    state.project.sound_segments_path = "/project/segments"  # type: ignore[attr-defined]
    state.project.sound_segments = SimpleNamespace(  # type: ignore[attr-defined]
        get_best_item_for=lambda _index: SimpleNamespace(file_name="old.flac")
    )
    monkeypatch.setattr(
        quick_gen_module.ProjectVoiceUtil, "get_batch_size", staticmethod(lambda _p: 1)
    )
    job = quick_gen_module.make_generation_job(state, 0)
    assert job.reconcile is None
    assert job.review is not None
    assert job.review.staging_dir.startswith("/project/segments_temp/")
    assert job.review.original_path == "/project/segments/old.flac"
    assert job.submit() == "job"
    assert submit.call_args.kwargs["staging_dir"] == job.review.staging_dir
    assert quick_gen_module.make_preview_job(state, "Preview", "test").review is None


def make_staged_job(tmp_path, *, original: bool = True) -> tuple[QuickGenJob, str, str]:
    staging_dir = tmp_path / "segments_temp" / "job"
    staged = staging_dir / "[00003] [0123456789abcdef] [none] [voice] [2] New.flac"
    original_path = tmp_path / "segments" / "[00003] [0123456789abcdef] [none] [voice] [5] Old.flac"

    def submit() -> str:
        staging_dir.mkdir(parents=True)
        staged.write_bytes(b"new")
        return "job"

    job = QuickGenJob(
        title="Quick generate - line 3",
        submit=submit,
        review=QuickGenReview(
            str(staging_dir), 2, str(original_path) if original else ""
        ),
    )
    return job, str(staged), str(original_path)


def review_text(modal: QuickGenModal) -> str:
    return str(modal.query_one("#quick-gen-review-text", Static).render())


@pytest.mark.parametrize(("key", "accepted"), [("enter", True), ("escape", False)])
def test_staged_job_shows_review_and_returns_decision(monkeypatch, tmp_path, key, accepted) -> None:
    # A clean staged job does not auto-dismiss: the review panel appears, the
    # new sound autoplays, and ENTER keeps / ESC discards the staged take.
    feed_events(monkeypatch, [[GenerationFinished("job", GenerationTerminalStatus.COMPLETED, "")]])
    play = Mock(return_value=("sound", ""))
    monkeypatch.setattr(quick_gen_module.PlaySoundUtil, "play_sound_file_async", play)
    monkeypatch.setattr(quick_gen_module.PlaySoundUtil, "current_sound_id", lambda: "sound")
    monkeypatch.setattr(quick_gen_module.PlaySoundUtil, "stop_sound_async", Mock())
    job, staged, _original = make_staged_job(tmp_path)
    app = HostApp()
    modal = QuickGenModal(make_state(), job)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)
            assert app.results == []
            assert modal.review_pending
            assert modal.query_one("#quick-gen-review").display
            text = review_text(modal)
            assert "[1] Play original sound  (5 word errors)" in text
            assert "[2] Play new sound  (2 word errors)  playing" in text
            assert "to keep new sound" in text
            # The review panel carries the prompt; the header row is blank.
            assert "to close" not in header_text(modal)
            play.assert_called_once_with(staged)
            await pilot.press(key)
            await pilot.pause()

    run(exercise())
    assert len(app.results) == 1
    result = app.results[0]
    assert result is not None
    assert result.staged_path == staged
    assert result.accepted is accepted


def test_review_keys_switch_and_toggle_playback(monkeypatch, tmp_path) -> None:
    # [1] plays the original over the new sound; pressing it again stops it.
    feed_events(monkeypatch, [[GenerationFinished("job", GenerationTerminalStatus.COMPLETED, "")]])
    play = Mock(return_value=("sound", ""))
    stop = Mock()
    monkeypatch.setattr(quick_gen_module.PlaySoundUtil, "play_sound_file_async", play)
    monkeypatch.setattr(quick_gen_module.PlaySoundUtil, "current_sound_id", lambda: "sound")
    monkeypatch.setattr(quick_gen_module.PlaySoundUtil, "stop_sound_async", stop)
    job, staged, original = make_staged_job(tmp_path)
    app = HostApp()
    modal = QuickGenModal(make_state(), job)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)
            await pilot.press("1")
            assert modal.review_playing == "original"
            await pilot.press("1")
            assert modal.review_playing is None

    run(exercise())
    assert [call.args[0] for call in play.call_args_list] == [staged, original]
    assert stop.call_count == 2


def test_review_without_original_shows_it_unavailable(monkeypatch, tmp_path) -> None:
    # A never-generated line has no original to compare against.
    feed_events(monkeypatch, [[GenerationFinished("job", GenerationTerminalStatus.COMPLETED, "")]])
    monkeypatch.setattr(
        quick_gen_module.PlaySoundUtil, "play_sound_file_async", Mock(return_value=("s", ""))
    )
    monkeypatch.setattr(quick_gen_module.PlaySoundUtil, "current_sound_id", lambda: "s")
    job, _staged, _original = make_staged_job(tmp_path, original=False)
    app = HostApp()
    modal = QuickGenModal(make_state(), job)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)
            assert "[1] Play original sound (none)" in review_text(modal)

    run(exercise())


def test_cancelled_staged_job_offers_no_review(monkeypatch, tmp_path) -> None:
    # Only a completed job is reviewable, even if a take was already staged.
    feed_events(monkeypatch, [[GenerationFinished("job", GenerationTerminalStatus.CANCELLED, "")]])
    job, _staged, _original = make_staged_job(tmp_path)
    app = HostApp()
    modal = QuickGenModal(make_state(), job)

    async def exercise() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(modal, app.results.append)
            await pilot.pause()
            modal._drain_worker_events()
            await pilot.pause(0.4)
            assert not modal.review_pending
            assert "to close" in header_text(modal)
            await pilot.press("enter")
            await pilot.pause()

    run(exercise())
    assert app.results[0] is not None
    assert app.results[0].staged_path == ""
    assert not app.results[0].accepted
