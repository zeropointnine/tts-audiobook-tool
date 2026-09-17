from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tts_audiobook_tool import concat_util
from tts_audiobook_tool.app_support import system_sleep
from tts_audiobook_tool.app_support.system_sleep import prevent_system_sleep
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    GenerationTerminalStatus,
    RealTimePlaybackFinished,
    RealTimePlaybackTerminalStatus,
    RealTimePlaybackUpdate,
)
from tts_audiobook_tool.real_time_playback_events import RealTimePlaybackAwaitingContinue
from tts_audiobook_tool.textual import generation_app, real_time_playback_app


class FakeMode:
    """Stand-in for a wakepy Mode that records enter/exit calls."""

    def __init__(
        self,
        *,
        active: bool = True,
        enter_error: Exception | None = None,
        exit_error: Exception | None = None,
    ) -> None:
        self.active = active
        self.active_method = SimpleNamespace(name="fake-method") if active else None
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self) -> FakeMode:
        self.enter_count += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, *_exc_info: Any) -> bool:
        self.exit_count += 1
        if self.exit_error is not None:
            raise self.exit_error
        return False


class FakeKeep:
    """Stand-in for the wakepy ``keep`` module."""

    def __init__(
        self,
        mode: FakeMode | None = None,
        *,
        mode_factory: Callable[[], FakeMode] | None = None,
        running_error: Exception | None = None,
    ) -> None:
        self._mode = mode
        self._mode_factory = mode_factory
        self._running_error = running_error
        self.running_kwargs: list[dict[str, Any]] = []
        self.created: list[FakeMode] = []

    def running(self, **kwargs: Any) -> FakeMode:
        self.running_kwargs.append(kwargs)
        if self._running_error is not None:
            raise self._running_error
        mode = self._mode_factory() if self._mode_factory else self._mode
        assert mode is not None
        self.created.append(mode)
        return mode


# --- prevent_system_sleep: context manager semantics ---


def test_holds_and_releases_around_block(monkeypatch) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    events: list[str] = []
    with prevent_system_sleep():
        events.append("body")

    assert events == ["body"]
    assert keep.running_kwargs == [{"on_fail": "pass"}]
    assert [mode.enter_count for mode in keep.created] == [1]
    assert [mode.exit_count for mode in keep.created] == [1]


def test_releases_and_propagates_when_body_raises(monkeypatch) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    with pytest.raises(ValueError, match="boom"):
        with prevent_system_sleep():
            raise ValueError("boom")

    assert [mode.exit_count for mode in keep.created] == [1]


def test_missing_wakepy_runs_block_uninhibited(monkeypatch) -> None:
    monkeypatch.setattr(system_sleep, "_wakepy_keep", None)

    events: list[str] = []
    with prevent_system_sleep():
        events.append("body")

    assert events == ["body"]


def test_running_failure_runs_block_uninhibited(monkeypatch) -> None:
    keep = FakeKeep(running_error=RuntimeError("no session bus"))
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    events: list[str] = []
    with prevent_system_sleep():
        events.append("body")

    assert events == ["body"]
    assert keep.created == []


def test_enter_failure_runs_block_uninhibited(monkeypatch) -> None:
    mode = FakeMode(enter_error=RuntimeError("activation failed"))
    keep = FakeKeep(mode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    events: list[str] = []
    with prevent_system_sleep():
        events.append("body")

    assert events == ["body"]
    assert mode.enter_count == 1
    # A mode whose __enter__ raised was never entered, so it must not be exited.
    assert mode.exit_count == 0


def test_inactive_mode_still_runs_block_and_releases(monkeypatch) -> None:
    mode = FakeMode(active=False)
    keep = FakeKeep(mode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    events: list[str] = []
    with prevent_system_sleep():
        events.append("body")

    assert events == ["body"]
    assert mode.exit_count == 1


def test_release_failure_does_not_break_the_block(monkeypatch) -> None:
    mode = FakeMode(exit_error=RuntimeError("release failed"))
    keep = FakeKeep(mode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    events: list[str] = []
    with prevent_system_sleep():
        events.append("body")

    assert events == ["body"]
    assert mode.exit_count == 1


def test_release_failure_does_not_mask_body_exception(monkeypatch) -> None:
    mode = FakeMode(exit_error=RuntimeError("release failed"))
    keep = FakeKeep(mode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    with pytest.raises(ValueError, match="boom"):
        with prevent_system_sleep():
            raise ValueError("boom")

    assert mode.exit_count == 1


def test_decorator_scope_wraps_every_call(monkeypatch) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)
    calls: list[int] = []

    @prevent_system_sleep()
    def work(value: int) -> int:
        calls.append(value)
        return value * 2

    assert work(1) == 2
    assert work(2) == 4
    assert calls == [1, 2]
    assert [mode.enter_count for mode in keep.created] == [1, 1]
    assert [mode.exit_count for mode in keep.created] == [1, 1]


# --- wiring: the long-running entry points acquire and release the inhibitor ---


def test_run_generation_app_holds_sleep_inhibitor(monkeypatch, tmp_path) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)
    monkeypatch.setattr(ModelWorker, "start", staticmethod(lambda: "worker unavailable"))
    monkeypatch.setattr(generation_app, "_present_console_result", lambda *_args: None)

    state = SimpleNamespace(
        project=SimpleNamespace(dir_path=str(tmp_path), generate_range_string="all"),
        prefs=SimpleNamespace(save_gen_log=False),
    )
    result = generation_app.run_generation_app(state, {0}, 1, False)

    assert result.status is GenerationTerminalStatus.FAILED
    assert [mode.enter_count for mode in keep.created] == [1]
    assert [mode.exit_count for mode in keep.created] == [1]


def test_run_real_time_playback_modal_holds_sleep_inhibitor(monkeypatch) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)
    monkeypatch.setattr(ModelWorker, "start", staticmethod(lambda: "worker unavailable"))
    monkeypatch.setattr(
        real_time_playback_app, "_present_console_result", lambda *_args: None
    )
    monkeypatch.setattr(real_time_playback_app.ask, "can_hotkey", False)

    result = real_time_playback_app.run_real_time_playback_modal(
        state=SimpleNamespace(),
        phrase_groups=[],
        line_range=None,
    )

    assert result.status is RealTimePlaybackTerminalStatus.FAILED
    assert [mode.enter_count for mode in keep.created] == [1]
    assert [mode.exit_count for mode in keep.created] == [1]


def test_make_file_holds_sleep_inhibitor(monkeypatch) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    def boom(_project):
        raise RuntimeError("stop before encode")

    monkeypatch.setattr(
        concat_util.ProjectTextIOUtil, "load_raw_text", staticmethod(boom)
    )

    with pytest.raises(RuntimeError, match="stop before encode"):
        ConcatUtil.make_file(SimpleNamespace(project=SimpleNamespace()), 0, 0, [], "/tmp/stem")

    assert [mode.enter_count for mode in keep.created] == [1]
    assert [mode.exit_count for mode in keep.created] == [1]


# --- early release: the lock must not cover the blocking ENTER prompts ---


def test_system_sleep_lock_release_is_idempotent(monkeypatch) -> None:
    mode = FakeMode()
    keep = FakeKeep(mode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)

    lock = system_sleep.SystemSleepLock()
    assert mode.enter_count == 1

    lock.release()
    lock.release()

    assert mode.exit_count == 1


def test_run_generation_app_releases_before_console_prompt(monkeypatch, tmp_path) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)
    monkeypatch.setattr(ModelWorker, "start", staticmethod(lambda: "worker unavailable"))

    exits_at_prompt: list[int] = []

    def present(*_args: Any) -> None:
        exits_at_prompt.append(keep.created[0].exit_count)

    monkeypatch.setattr(generation_app, "_present_console_result", present)

    state = SimpleNamespace(
        project=SimpleNamespace(dir_path=str(tmp_path), generate_range_string="all"),
        prefs=SimpleNamespace(save_gen_log=False),
    )
    generation_app.run_generation_app(state, {0}, 1, False)

    # The lock is already released by the time the prompt is presented.
    assert exits_at_prompt == [1]


def test_generation_app_releases_lock_at_terminal_summary() -> None:
    released: list[str] = []
    transcript = generation_app.GenerationTranscript("", enabled=False)
    app = generation_app.GenerationApp(
        SimpleNamespace(project=SimpleNamespace(gen_auto_concat=False)),
        {0},
        1,
        False,
        transcript,
        on_job_end=lambda: released.append("released"),
    )

    app._pre_terminal_summary(
        generation_app.GenerationModalResult(GenerationTerminalStatus.COMPLETED, "", "")
    )

    assert released == ["released"]


def test_real_time_playback_app_releases_lock_at_terminal_summary() -> None:
    released: list[str] = []
    app = real_time_playback_app.RealTimePlaybackApp(
        SimpleNamespace(project=SimpleNamespace(gen_auto_concat=False)),
        [],
        None,
        on_job_end=lambda: released.append("released"),
    )

    app._pre_terminal_summary(
        real_time_playback_app.RealTimePlaybackModalResult(
            RealTimePlaybackTerminalStatus.COMPLETED, ""
        )
    )

    assert released == ["released"]


def test_realtime_console_releases_before_enter_prompt(monkeypatch) -> None:
    keep = FakeKeep(mode_factory=FakeMode)
    monkeypatch.setattr(system_sleep, "_wakepy_keep", keep)
    monkeypatch.setattr(ModelWorker, "start", staticmethod(lambda: ""))
    monkeypatch.setattr(real_time_playback_app, "can_textual", lambda: False)
    monkeypatch.setattr(
        real_time_playback_app, "_present_console_result", lambda *_args: None
    )
    monkeypatch.setattr(
        ModelWorker, "submit_realtime_playback", staticmethod(lambda **_: "job")
    )
    monkeypatch.setattr(
        ModelWorker,
        "continue_realtime_playback",
        staticmethod(lambda _operation_id: True),
    )
    events: list[Any] = [
        RealTimePlaybackUpdate("job", RealTimePlaybackAwaitingContinue(3.0, False)),
        RealTimePlaybackFinished("job", RealTimePlaybackTerminalStatus.COMPLETED, ""),
    ]
    monkeypatch.setattr(
        ModelWorker,
        "get_event",
        staticmethod(lambda timeout=0.1: events.pop(0) if events else None),
    )

    exits_at_prompt: list[int] = []

    def enter() -> None:
        exits_at_prompt.append(keep.created[0].exit_count)

    monkeypatch.setattr(real_time_playback_app.ask, "ask_enter_to_continue", enter)

    real_time_playback_app.run_real_time_playback_modal(
        state=SimpleNamespace(),
        phrase_groups=[],
        line_range=None,
    )

    # The lock is already released when the worker's continue prompt appears.
    assert exits_at_prompt == [1]
