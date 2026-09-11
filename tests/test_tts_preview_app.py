import asyncio
from types import SimpleNamespace
from typing import cast

import numpy as np
from textual.widgets import Static

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.generation_events import GenerationTimedOut, ModelUnhealthy
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    GenerationTerminalStatus,
    GenerationUpdate,
    TtsPreviewFinished,
)
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual import worker_app as worker_app_module
from tts_audiobook_tool.textual.tts_preview_app import (
    TtsPreviewApp,
    run_tts_preview_app,
)
from tts_audiobook_tool.worker_reset import HardResetCause


def run(coroutine) -> None:
    asyncio.run(coroutine)


def make_state() -> State:
    return cast(State, SimpleNamespace(project=SimpleNamespace()))


def test_tts_preview_app_auto_returns_in_memory_sound(monkeypatch) -> None:
    sound = Sound(np.zeros(24, dtype=np.float32), 24_000)
    queued_events = [
        [
            TtsPreviewFinished(
                "job",
                GenerationTerminalStatus.COMPLETED,
                sound=sound,
            )
        ]
    ]
    monkeypatch.setattr(
        ModelWorker,
        "submit_tts_preview",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(
            lambda max_events=1000: queued_events.pop(0)
            if queued_events
            else []
        ),
    )
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    app = TtsPreviewApp(make_state(), "Original word: one. Substitute word: two")

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            assert "Generating preview" in str(
                app.query_one("#generation-title", Static).render()
            )
            app._drain_worker_events()
            await pilot.pause(0.3)
            assert app.terminal_result is not None
            assert app.auto_exit is True

    run(exercise())
    assert app.return_value is not None
    assert app.return_value.completed
    assert app.return_value.sound is sound


def test_tts_preview_app_waits_for_enter_after_failure(monkeypatch) -> None:
    queued_events = [
        [
            TtsPreviewFinished(
                "job",
                GenerationTerminalStatus.FAILED,
                message="preview failed",
            )
        ]
    ]
    monkeypatch.setattr(
        ModelWorker,
        "submit_tts_preview",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(
            lambda max_events=1000: queued_events.pop(0)
            if queued_events
            else []
        ),
    )
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    app = TtsPreviewApp(make_state(), "preview prompt")

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            app._drain_worker_events()
            await pilot.pause(0.3)
            assert app.terminal_result is not None
            assert app.terminal_result.message == "preview failed"
            assert app.return_value is None
            assert app.is_running is True
            await pilot.press("enter")
            await pilot.pause()

    run(exercise())
    assert app.return_value is not None
    assert app.return_value.status is GenerationTerminalStatus.FAILED


def test_tts_preview_app_hard_resets_worker_on_gen_timeout(monkeypatch) -> None:
    """The preview's watchdog event takes the shared hard-reset path."""
    reset_calls: list[int] = []
    latch = {"deliver": False}
    events = [GenerationUpdate("job", GenerationTimedOut(180.0))]

    def fake_drain_events() -> list:
        if latch["deliver"] and events:
            return [events.pop(0)]
        return []

    def fake_reset() -> str:
        reset_calls.append(1)
        return ""

    monkeypatch.setattr(
        ModelWorker,
        "submit_tts_preview",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(fake_drain_events))
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(fake_reset))
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    app = TtsPreviewApp(make_state(), "preview prompt")

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            latch["deliver"] = True
            app._drain_worker_events()
            await pilot.pause(0.4)

            assert reset_calls == [1]
            assert app.terminal_result is not None
            assert (
                app.terminal_result.status
                is GenerationTerminalStatus.WORKER_RESET
            )
            assert (
                app.terminal_result.hard_reset_cause
                is HardResetCause.GENERATION_TIMEOUT
            )
            # The summary cites the GEN_TIMEOUT cap that tripped.
            assert "GEN_TIMEOUT" in app.terminal_result.message
            assert "180" in app.terminal_result.message
            await pilot.press("enter")
            await pilot.pause()

    run(exercise())
    assert app.return_value is not None
    assert app.return_value.status is GenerationTerminalStatus.WORKER_RESET
    assert not app.return_value.completed
    assert app.return_value.sound is None


def test_tts_preview_app_ignores_updates_once_reset_owns_the_session(
    monkeypatch,
) -> None:
    """A late queued update must not restart or preempt an in-flight reset."""
    reset_calls: list[int] = []
    latch = {"deliver": 0}
    events = [
        GenerationUpdate("job", GenerationTimedOut(180.0)),
        GenerationUpdate("job", ModelUnhealthy(reason="boom")),
    ]

    def fake_drain_events() -> list:
        if latch["deliver"] > 0:
            latch["deliver"] -= 1
            return [events.pop(0)]
        return []

    monkeypatch.setattr(
        ModelWorker,
        "submit_tts_preview",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(fake_drain_events))
    monkeypatch.setattr(
        ModelWorker,
        "reset",
        staticmethod(lambda: reset_calls.append(1) or ""),
    )
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    app = TtsPreviewApp(make_state(), "preview prompt")

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            # Both updates arrive while the reset started by the first one is
            # still in progress.
            latch["deliver"] = 2
            app._drain_worker_events()
            assert app.reset_in_progress is True
            app._drain_worker_events()
            await pilot.pause(0.4)

            assert reset_calls == [1]
            assert (
                app.terminal_result.hard_reset_cause
                is HardResetCause.GENERATION_TIMEOUT
            )
            await pilot.press("enter")
            await pilot.pause()

    run(exercise())
    assert app.return_value is not None
    assert app.return_value.status is GenerationTerminalStatus.WORKER_RESET


def test_interface_failure_cleanup_reports_worker_restart_failure(monkeypatch) -> None:
    """A preview screen that raised takes the shared failure classification.

    The interface failed while its worker was mid-job, so the worker is
    hard-reset and the result reports an interface failure, exactly as the
    generation and realtime runners do.
    """
    reset_calls: list[None] = []

    def failing_run(app, **_kwargs):
        app.operation_id = "job"
        raise RuntimeError("preview interface exploded")

    monkeypatch.setattr(ModelWorker, "start", staticmethod(lambda: ""))
    monkeypatch.setattr(
        ModelWorker,
        "reset",
        staticmethod(
            lambda: reset_calls.append(None) or "replacement worker failed to start"
        ),
    )
    monkeypatch.setattr(TtsPreviewApp, "run", failing_run)

    result = run_tts_preview_app(make_state(), "preview prompt")

    assert reset_calls == [None]
    assert result.status is GenerationTerminalStatus.FAILED
    assert result.hard_reset_cause is HardResetCause.INTERFACE_FAILURE
    assert "RuntimeError: preview interface exploded" in result.message
    assert "replacement worker failed to start" in result.message
    assert result.sound is None
