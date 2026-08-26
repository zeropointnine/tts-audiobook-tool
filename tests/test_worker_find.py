"""Tests for the Ctrl+F find bar shared by the worker session apps."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from rich.segment import Segment
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.strip import Strip
from textual.widgets import Input, Static

from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual import worker_app as worker_app_module
from tts_audiobook_tool.textual.worker_app import WorkerTextualApp, worker_app_css
from tts_audiobook_tool.textual.worker_content import WorkerLog
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@dataclass(frozen=True)
class _StubResult:
    status: object = None
    message: str = ""


def make_state() -> State:
    return cast(State, SimpleNamespace())


class StubWorkerApp(WorkerTextualApp[_StubResult]):
    """Minimal concrete worker app for exercising the shared find bar."""

    DIVIDER_ID = "stub-divider"
    OUTPUT_SHELL_ID = "stub-output-shell"
    HEADER_UPDATE_SECONDS = 3600.0
    CSS = worker_app_css("stub-divider")

    def __init__(self) -> None:
        super().__init__(make_state())
        self.continued = 0
        self.cancelled = 0

    def compose_header(self) -> ComposeResult:
        yield Static("HEADER")

    def submit_worker_job(self) -> str:
        return "op"

    def _on_submit_failure(self, message: str) -> None:
        pass

    def _handle_session_event(self, event: object) -> None:
        pass

    def _handle_update(self, update: object) -> None:
        pass

    def _on_worker_command_failed(self, message: str) -> None:
        pass

    def _on_worker_exit(self, event: object) -> None:
        pass

    def terminal_label(self, result: _StubResult) -> str:
        return "done"

    def make_worker_reset_result(self, message: str) -> _StubResult:
        return _StubResult()

    def _update_header(self) -> None:
        pass

    def action_continue(self) -> None:
        self.continued += 1

    def action_cancel_or_reset(self) -> None:
        self.cancelled += 1


def run(coroutine) -> None:
    asyncio.run(coroutine)


def _seed(log: WorkerLog, lines: list[str]) -> None:
    log.feed_console(list(lines), "")


def test_worker_app_uses_active_model_output_filters(monkeypatch) -> None:
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)

    async def exercise() -> None:
        Tts._type = TtsModelType.MIRA
        mira_app = StubWorkerApp()
        async with mira_app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            mira_log = mira_app._worker_log()
            assert mira_log.output_filters == ("smem_size",)
            mira_log.feed_console(["kernel smem_size=123"], "")
            assert mira_log.line_texts() == [""]

        Tts._type = TtsModelType.CHATTERBOX
        normal_app = StubWorkerApp()
        async with normal_app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            normal_log = normal_app._worker_log()
            assert normal_log.output_filters == ()
            normal_log.feed_console(["kernel smem_size=123"], "")
            assert normal_log.line_texts() == ["kernel smem_size=123", ""]

    run(exercise())


def test_find_bar_searches_and_gates_session_keys(monkeypatch) -> None:
    """Ctrl+F opens the overlay bar; Enter advances matches without continuing,
    CTRL-C is ignored, Escape closes and clears the highlight."""

    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)

    async def exercise() -> None:
        app = StubWorkerApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            log = app._worker_log()
            _seed(log, ["one two", "three two", "four"])
            await pilot.pause()

            find_bar = app.query_one("#find-bar", Horizontal)
            assert find_bar.display is False

            await pilot.press("ctrl+f")
            await pilot.pause()
            assert app.find_active
            assert find_bar.display is True
            assert app.focused is app.query_one("#find-input", Input)

            await pilot.press("t", "w", "o")
            await pilot.press("enter")
            await pilot.pause()
            # Enter was consumed by the find input, not the session's Enter.
            assert app.continued == 0
            assert log.highlight_line_index == 1
            assert app.query_one("#find-result", Static).content == "2 of 2"

            # A second Enter wraps forward to the first match.
            await pilot.press("enter")
            await pilot.pause()
            assert log.highlight_line_index == 0
            assert app.query_one("#find-result", Static).content == "1 of 2"

            # Backward navigation wraps the other way.
            app.action_find_previous()
            await pilot.pause()
            assert log.highlight_line_index == 1

            # CTRL-C is ignored while the find input owns focus.
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.cancelled == 0

            await pilot.press("escape")
            await pilot.pause()
            assert not app.find_active
            assert find_bar.display is False
            assert log.highlight_line_index is None
            assert app.focused is log

    run(exercise())


def test_find_match_indices_and_wrap(monkeypatch) -> None:
    """Matching is case-insensitive and relative matches wrap past the edges."""

    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)

    async def exercise() -> None:
        app = StubWorkerApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            log = app._worker_log()
            _seed(log, ["Alpha", "beta", "ALPHA", "gamma"])
            await pilot.pause()

            assert app.find_match_indices("alpha") == [0, 2]
            assert app.find_match_indices("") == []
            assert app.find_match_indices("zzz") == []

            app.find_search_start_index = 2
            assert app.find_relative_match([0, 2], 1) == 0
            assert app.find_relative_match([0, 2], -1) == 0

    run(exercise())


def test_worker_log_line_accessors_and_highlight() -> None:
    """WorkerLog exposes line text/geometry helpers and reverse-videos the
    highlighted find line."""

    async def exercise() -> None:
        app = StubWorkerApp()
        async with app.run_test(size=(40, 12)) as pilot:
            await pilot.pause()
            log = app._worker_log()
            _seed(log, ["short", "x" * 100, "tail"])
            await pilot.pause()

            assert log.line_count == 4
            assert log.line_texts() == ["short", "x" * 100, "tail", ""]

            # The long middle line wraps, so its top row is > 0 and the tail's
            # top row comes after it.
            width = log.scrollable_content_region.width
            middle_rows = -(-100 // width)
            assert log.line_scroll_y(2) == 1 + middle_rows
            assert log.line_index_at_scroll_y(0) == 0
            assert log.line_index_at_scroll_y(1) == 1
            assert log.line_index_at_scroll_y(1 + middle_rows) == 2

            log.highlight_line_index = 1
            plain = Strip([Segment("match")])
            highlighted = log._style_row(plain, 1)
            assert any(segment.style.reverse for segment in highlighted)
            assert log._style_row(plain, 0) is plain

    run(exercise())


def test_escape_does_not_interrupt_generation(monkeypatch) -> None:
    """Escape no longer cancels or hard-resets a running generation session;
    CTRL-C is the sole interrupt key."""

    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)

    async def exercise() -> None:
        app = StubWorkerApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            # Not in find mode and no terminal result: Escape must be a no-op
            # rather than forwarding to cancel_or_reset.
            await pilot.press("escape")
            await pilot.pause()
            assert app.cancelled == 0
            assert app.find_active is False

    run(exercise())
