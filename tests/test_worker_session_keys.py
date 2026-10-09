"""Keyboard contract shared by the generation and realtime playback sessions."""

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from textual.widgets import Input, Static

from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import ConsoleOutput
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual import worker_app as worker_app_module
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


def test_escape_cancels_then_resets_but_ctrl_c_does_neither(session) -> None:
    app, cancellations, resets = session

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+c")
            assert cancellations == []
            assert app.is_running
            assert app.return_value is None

            await pilot.press("escape")
            assert cancellations == ["job"]
            assert app.cancel_requested
            assert resets == []
            hotkey = app.query_one("#realtime-hotkey" if isinstance(app, RealTimePlaybackApp)
                                   else "#generation-hotkey", Static)
            assert "[ESC] to kill process" in str(hotkey.render())
            document = "\n".join(app.query_one(WorkerLogContentArea).worker_log.line_texts())
            assert "[ESC] again to hard-reset" in document
            assert "CTRL-C" not in document

            await pilot.press("ctrl+c")
            assert resets == []
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert cancellations == ["job"]
            assert resets == [True]
            assert app.terminal_result.hard_reset_cause is HardResetCause.USER_ESCALATION

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
            notice_lines = ["", "Cancellation requested, please wait"]
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
