"""Keyboard contract shared by the generation and realtime playback sessions."""

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from textual.widgets import Input, Static

from tts_audiobook_tool import text_util
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import ConsoleOutput
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual import worker_app as worker_app_module
from tts_audiobook_tool.textual import worker_session
from tts_audiobook_tool.textual.generation_app import GenerationApp, GenerationTranscript
from tts_audiobook_tool.textual.real_time_playback_app import RealTimePlaybackApp
from tts_audiobook_tool.textual.worker_content import WorkerLogContentArea
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.worker_reset import HardResetCause


@pytest.fixture(params=["generation", "realtime"])
def session(request, monkeypatch):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    for method in ("submit_generation", "submit_realtime_playback"):
        monkeypatch.setattr(ModelWorker, method, staticmethod(lambda **_: "job"))
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(lambda: []))
    cancellations: list[str] = []
    resets: list[bool] = []
    monkeypatch.setattr(
        ModelWorker, "request_cancel",
        staticmethod(lambda operation_id: cancellations.append(operation_id) or True),
    )
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(lambda: resets.append(True) or ""))
    state = cast(State, SimpleNamespace(project=SimpleNamespace(
        get_tts_model_type=lambda: TtsModelType.require_by_id("none"),
        generate_range_string="all", gen_auto_concat=False,
    )))
    if request.param == "generation":
        app = GenerationApp(state, {0}, 1, GenerationTranscript("", enabled=False))
    else:
        app = RealTimePlaybackApp(state, [], None)
    return app, cancellations, resets


def hotkey_text(app) -> str:
    hotkey_id = "#realtime-hotkey" if isinstance(app, RealTimePlaybackApp) else "#generation-hotkey"
    return str(app.query_one(hotkey_id, Static).render())


def log_document(app) -> str:
    return "\n".join(app.query_one(WorkerLogContentArea).worker_log.line_texts())


def test_escape_cancels_then_resets_but_ctrl_c_does_neither(session, monkeypatch, tmp_path) -> None:
    # The full cancel -> hard-reset ladder, for each host app: the first ESC
    # requests a cooperative cancel, the second resets the worker at once
    # (no confirmation dialog), and the reset summary links the worker log.
    app, cancellations, resets = session
    log_path = str(tmp_path / "worker log.log")  # a path with a space must still link
    monkeypatch.setattr(worker_session, "make_worker_log_file_path", lambda: log_path)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            assert "[ESC] to interrupt" in hotkey_text(app)
            await pilot.press("ctrl+c")
            assert cancellations == []
            assert app.is_running
            assert app.return_value is None

            await pilot.press("escape")
            assert cancellations == ["job"]
            assert app.cancel_requested
            assert resets == []
            assert "[ESC] to kill process" in hotkey_text(app)
            assert "[ESC] again to hard-reset" in log_document(app)
            assert "CTRL-C" not in log_document(app)

            await pilot.press("ctrl+c")
            assert resets == []
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert cancellations == ["job"]
            assert resets == [True]
            assert len(app.screen_stack) == 1
            result = app.terminal_result
            assert result.status.name == "WORKER_RESET"
            assert result.hard_reset_cause is HardResetCause.USER_ESCALATION
            assert f"Worker log: {text_util.make_terminal_hyperlink(log_path, is_file=True)}" in result.message
            # Once stopped, the prompt moves on to ENTER (continue/finish).
            assert "[ENTER] to" in hotkey_text(app)

    asyncio.run(exercise())


def test_remote_mode_escape_cancels_without_offering_hard_reset(session, monkeypatch) -> None:
    # A remote (SGL-Omni) worker holds no local model memory, so the hard reset
    # is not offered: ESC only requests a cancel and further presses do nothing.
    app, cancellations, resets = session
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            assert cancellations == ["job"]
            assert app.cancel_requested
            assert "Cancellation requested" in log_document(app)
            assert "hard-reset" not in log_document(app)
            assert "kill process" not in hotkey_text(app)
            assert "Waiting for the current generation to stop" in hotkey_text(app)

            await pilot.press("escape", "escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert cancellations == ["job"]
            assert resets == []
            assert not app.reset_in_progress
            assert app.terminal_result is None

    asyncio.run(exercise())


@pytest.mark.parametrize("backend_mode", [TtsRuntimeMode.LOCAL, TtsRuntimeMode.REMOTE_CLIENT])
def test_cancellation_notice_leaves_one_blank_before_worker_summary(
    session, monkeypatch, backend_mode
) -> None:
    # App notices must not add padding on top of the worker's normal blank line.
    app, cancellations, resets = session
    monkeypatch.setattr(Tts, "_backend_mode", backend_mode)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app._handle_console_output(ConsoleOutput("job", "stdout", "request details\n"))
            await pilot.press("escape")
            assert cancellations == ["job"]
            assert resets == []
            log = app.query_one(WorkerLogContentArea).worker_log
            notice_lines = ["", "Cancellation requested, please wait..."]
            if backend_mode is TtsRuntimeMode.LOCAL:
                notice_lines.append("Or press [ESC] again to hard-reset")
            assert log.line_texts() == ["request details", *notice_lines, ""]

            # Inference finishes after cancellation, then prints its usual separator.
            app._handle_console_output(
                ConsoleOutput("job", "stdout", "\nGenerated audio in 4.2s\n")
            )
            assert log.line_texts() == [
                "request details", *notice_lines, "", "Generated audio in 4.2s", ""
            ]

    asyncio.run(exercise())


def test_escape_closes_find_without_cancelling_or_escalating(session) -> None:
    app, cancellations, resets = session

    async def exercise() -> None:
        async with app.run_test() as pilot:
            # Test both a running job and one awaiting cooperative cancellation.
            for pending in (False, True):
                app.cancel_requested = pending
                await pilot.press("ctrl+f")
                find_input = app.query_one("#find-input", Input)
                find_input.value = "needle"
                find_input.select_all()
                await pilot.press("ctrl+c")
                assert app.clipboard == "needle"
                await pilot.press("escape")
                assert not app.find_active
                assert app.is_running
                assert app.cancel_requested is pending
                assert cancellations == []
                assert resets == []

    asyncio.run(exercise())


@pytest.mark.parametrize("blocked_attribute", ["finishing", "reset_in_progress"])
def test_escape_is_ignored_during_settle_or_reset(session, blocked_attribute) -> None:
    app, cancellations, resets = session

    async def exercise() -> None:
        async with app.run_test() as pilot:
            setattr(app, blocked_attribute, True)
            for pending in (False, True):
                app.cancel_requested = pending
                await pilot.press("escape")
                assert cancellations == []
                assert resets == []
                assert app.return_value is None
            # Even a stale summary must not dismiss a flow owned by the guard.
            app.terminal_result = app.make_worker_reset_result("stale", HardResetCause.USER_ESCALATION)
            await pilot.press("escape")
            assert app.return_value is None
            setattr(app, blocked_attribute, False)

    asyncio.run(exercise())


def test_escape_dismisses_summary_without_cancelling(session) -> None:
    app, cancellations, resets = session
    result = app.make_worker_reset_result("stopped", HardResetCause.USER_ESCALATION)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app._show_terminal_summary(result)
            await pilot.press("ctrl+c")
            assert app.return_value is None
            await pilot.press("escape")
            assert cancellations == []
            assert resets == []

    asyncio.run(exercise())
    assert app.return_value is result


def test_escape_without_an_operation_does_not_exit_or_cancel(session) -> None:
    app, cancellations, resets = session

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.operation_id = None
            await pilot.press("escape")
            assert cancellations == []
            assert resets == []
            assert app.return_value is None
            assert app.is_running

    asyncio.run(exercise())
