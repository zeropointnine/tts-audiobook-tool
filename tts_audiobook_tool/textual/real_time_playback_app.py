from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult

from tts_audiobook_tool import ask, util
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.generation_events import GenerationPhase, GenerationTimedOut
from tts_audiobook_tool.gen_timeout_util import make_gen_timeout_message
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleFlush,
    ConsoleOutput,
    ModelWorkerEvent,
    RealTimePlaybackFinished,
    RealTimePlaybackTerminalStatus,
    RealTimePlaybackUpdate,
    WorkerCommandFailed,
    WorkerExited,
)
from tts_audiobook_tool.real_time_playback_events import (
    RealTimePlaybackAwaitingContinue,
    RealTimePlaybackBuffer,
    RealTimePlaybackProgress,
    RealTimePlaybackSegmentText,
    RealTimePlaybackStarted,
)
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.textual.real_time_playback_header import (
    PromptMode,
    RealTimePlaybackHeader,
    RealTimePlaybackSourceText,
)
from tts_audiobook_tool.textual.textual_shared import can_textual
from tts_audiobook_tool.textual.worker_app import (
    FINAL_OUTPUT_SETTLE_SECONDS,
    WorkerTextualApp,
    worker_app_css,
)

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types.phrase import PhraseGroup
    from tts_audiobook_tool.state import State


@dataclass(frozen=True)
class RealTimePlaybackModalResult:
    status: RealTimePlaybackTerminalStatus
    message: str = ""

    @property
    def completed(self) -> bool:
        return self.status == RealTimePlaybackTerminalStatus.COMPLETED


_TERMINAL_LABELS: dict[RealTimePlaybackTerminalStatus, str] = {
    RealTimePlaybackTerminalStatus.COMPLETED: "Realtime playback completed",
    RealTimePlaybackTerminalStatus.CANCELLED: "Realtime playback cancelled",
    RealTimePlaybackTerminalStatus.ABORTED: "Realtime playback stopped",
    RealTimePlaybackTerminalStatus.FAILED: "Realtime playback failed",
    RealTimePlaybackTerminalStatus.WORKER_RESET: "Realtime playback stopped; model hard-reset"
}


class RealTimePlaybackApp(WorkerTextualApp[RealTimePlaybackModalResult]):
    """Full-screen realtime playback session: header, divider, dimmed
    source-text band, live worker log."""

    CSS = worker_app_css("realtime-divider")

    DIVIDER_ID: ClassVar[str] = "realtime-divider"
    OUTPUT_SHELL_ID: ClassVar[str] = "realtime-output-shell"
    HEADER_UPDATE_SECONDS: ClassVar[float] = 0.25

    def __init__(
        self,
        state: State,
        phrase_groups: list[PhraseGroup],
        line_range: tuple[int, int] | None,
    ) -> None:
        super().__init__(state)
        self.phrase_groups = phrase_groups
        self.line_range = line_range
        self.processed = 0
        self.total = 0
        self.buffer_seconds = 0.0
        self.buffer_updated_at = self.started_at
        # (text, start_sample, end_sample) for each segment whose audio was
        # added to the stream; used to show which source text is being played.
        self.spoken_segments: list[tuple[str, int, int]] = []
        # (monotonic timestamp, played_samples, total_samples_added) of the
        # most recent segment-anchor event; the playhead extrapolates from it.
        self.play_anchor: tuple[float, int, int] | None = None
        self.waiting_for_continue = False
        self.exit_after_terminal = False
        self.teardown_in_progress = False

    def compose_header(self) -> ComposeResult:
        yield RealTimePlaybackHeader(id="realtime-header")

    def compose_below_divider(self) -> ComposeResult:
        # The band sits below the shared divider and closes itself with its
        # own rule, so the two strokes frame it between the header chrome
        # and the worker log.
        yield RealTimePlaybackSourceText(id="realtime-source")

    def submit_worker_job(self) -> str:
        return ModelWorker.submit_realtime_playback(
            state=self.state,
            phrase_groups=self.phrase_groups,
            line_range=self.line_range,
        )

    def _on_submit_failure(self, message: str) -> None:
        # A submission failure happens before the session has rendered any
        # worker output, so finalize without the usual settle step.
        self._show_terminal_summary(
            RealTimePlaybackModalResult(
                RealTimePlaybackTerminalStatus.FAILED,
                message,
            )
        )

    def _handle_session_event(self, event: ModelWorkerEvent) -> None:
        if isinstance(event, RealTimePlaybackUpdate):
            self._handle_update(event.update)
        elif isinstance(event, RealTimePlaybackFinished):
            self._begin_finish(event)

    def _handle_update(self, update: object) -> None:
        if self.teardown_in_progress:
            return
        if isinstance(update, GenerationPhase):
            self.phase = update.label
        elif isinstance(update, RealTimePlaybackStarted):
            self.total = update.total
            self.processed = 0
            self.spoken_segments = []
            self.play_anchor = None
        elif isinstance(update, RealTimePlaybackProgress):
            self.processed = update.processed
            self.total = update.total
        elif isinstance(update, RealTimePlaybackBuffer):
            self._record_buffer_duration(update.duration_seconds)
        elif isinstance(update, RealTimePlaybackSegmentText):
            self._record_segment(update)
        elif isinstance(update, RealTimePlaybackAwaitingContinue):
            self._record_buffer_duration(update.duration_seconds)
            if not self.waiting_for_continue:
                self._append_lines(
                    [
                        "",
                        f"Press {util.make_hotkey_string('ENTER')} to finish",
                    ]
                )
            self.waiting_for_continue = True
            self.phase = (
                "Playback interrupted"
                if update.interrupted
                else "Playback generation complete"
            )
        elif isinstance(update, GenerationTimedOut):
            # A gen timeout hard-resets the worker. Deliberately not gated on
            # cancel_requested/cancel_pending: a pending single-CTRL-C cancel
            # must not suppress the timeout; only an already-finalized session
            # (a reset in progress or a terminal result) skips it.
            if self.terminal_result is None and not self.reset_in_progress:
                self._begin_hard_reset(
                    make_gen_timeout_message(update.timeout_seconds)
                )
        self._update_header()

    def _begin_finish(self, event: RealTimePlaybackFinished) -> None:
        if self.finishing or self.terminal_result is not None:
            return
        self.finishing = True
        if not self.teardown_in_progress:
            self.phase = event.status.value.replace("_", " ").title()
            self._update_header()
        self.set_timer(
            FINAL_OUTPUT_SETTLE_SECONDS,
            lambda: self._finish_from_worker_event(event),
        )

    def _finish_from_worker_event(self, event: RealTimePlaybackFinished) -> None:
        if not self.teardown_in_progress:
            self._finalize_console()
        self._show_terminal_summary(
            RealTimePlaybackModalResult(event.status, event.message)
        )

    def _finalize_failure(self, message: str) -> None:
        if self.terminal_result is not None:
            return
        self._finalize_console()
        self._show_terminal_summary(
            RealTimePlaybackModalResult(
                RealTimePlaybackTerminalStatus.FAILED,
                message,
            )
        )

    def _on_worker_command_failed(self, message: str) -> None:
        self._finalize_failure(message)

    def _on_worker_exit(self, event: WorkerExited) -> None:
        # Synthesized by the client when the worker process died; it is the
        # single death signal (no liveness polling here). A hard reset in
        # progress owns the terminal summary instead.
        self._finalize_worker_exit(
            event.message or "Model worker exited unexpectedly"
        )

    def _finalize_worker_exit(self, message: str) -> None:
        if self.reset_in_progress or self.terminal_result is not None:
            return
        self._finalize_console()
        self._show_terminal_summary(
            RealTimePlaybackModalResult(
                RealTimePlaybackTerminalStatus.WORKER_RESET,
                message,
            )
        )

    def _skip_log_updates(self) -> bool:
        return self.terminal_result is not None or self.teardown_in_progress

    def terminal_label(self, result: RealTimePlaybackModalResult) -> str:
        return _TERMINAL_LABELS[result.status]

    def terminal_summary_extra_lines(
        self, result: RealTimePlaybackModalResult
    ) -> list[str]:
        if self.exit_after_terminal:
            return []
        return [
            "",
            f"Press {util.make_hotkey_string('ENTER')} to finish",
        ]

    def _pre_terminal_summary(self, result: RealTimePlaybackModalResult) -> None:
        self.waiting_for_continue = False
        self._record_buffer_duration(0.0)

    def _post_terminal_summary(self, result: RealTimePlaybackModalResult) -> None:
        if self.exit_after_terminal:
            self.set_timer(0.05, lambda: self.exit(result))

    def _suppress_terminal_summary_ui(self) -> bool:
        return self.teardown_in_progress

    def make_worker_reset_result(self, message: str) -> RealTimePlaybackModalResult:
        return RealTimePlaybackModalResult(
            RealTimePlaybackTerminalStatus.WORKER_RESET,
            message,
        )

    @property
    def cancel_pending(self) -> bool:
        return super().cancel_pending and not self.waiting_for_continue

    @property
    def cancel_or_reset_blocked(self) -> bool:
        return super().cancel_or_reset_blocked or self.waiting_for_continue

    @property
    def prompt_mode(self) -> PromptMode:
        if self.terminal_result is not None:
            return "finished"
        if self.waiting_for_continue:
            return "awaiting_continue"
        if self.cancel_pending:
            return "cancel_pending"
        return "default"

    def _record_buffer_duration(self, duration_seconds: float) -> None:
        self.buffer_seconds = max(0.0, duration_seconds)
        self.buffer_updated_at = time.monotonic()

    @property
    def interpolated_buffer_seconds(self) -> float:
        """Estimate worker-owned buffer drain between authoritative events."""
        elapsed_since_update = max(0.0, time.monotonic() - self.buffer_updated_at)
        return max(0.0, self.buffer_seconds - elapsed_since_update)

    def _record_segment(self, update: RealTimePlaybackSegmentText) -> None:
        anchor = self.play_anchor
        if anchor is not None and update.end_sample < anchor[2]:
            # The stream buffer was reset mid-run (total_samples_added went
            # backwards), so previous sample ranges are stale; start fresh.
            self.spoken_segments = []
        self.spoken_segments.append(
            (update.text, update.start_sample, update.end_sample)
        )
        self.play_anchor = (time.monotonic(), update.played_samples, update.end_sample)

    @property
    def interpolated_played_samples(self) -> int | None:
        """
        Estimate the playback cursor between segment-anchor events.

        The worker owns the audio stream.  Between anchors, samples drain at
        exactly 1x device time while buffered data remains (the stream
        callback never skips or repeats samples), so extrapolate linearly
        from the most recent anchor and clamp at its added total: when the
        buffer drains (underflow) the cursor freezes until the next segment
        arrives.
        """
        anchor = self.play_anchor
        if anchor is None:
            return None
        anchor_at, anchor_played, anchor_added = anchor
        elapsed = max(0.0, time.monotonic() - anchor_at)
        return min(anchor_added, anchor_played + int(elapsed * APP_SAMPLE_RATE))

    def _current_playing_text(self) -> tuple[str, bool]:
        """
        (text, finished) of the segment the estimated playhead is inside.
        When the cursor is not inside any segment (typically frozen at the
        end of the last one while underflowed), the last segment is returned
        as finished.
        """
        pos = self.interpolated_played_samples
        if pos is None or not self.spoken_segments:
            return "", False
        active_idx = next(
            (
                i
                for i, (_, start, end) in enumerate(self.spoken_segments)
                if start <= pos < end
            ),
            None,
        )
        if active_idx is not None:
            return self.spoken_segments[active_idx][0], False
        return self.spoken_segments[-1][0], True

    def _update_header(self) -> None:
        if not self.is_mounted or self.teardown_in_progress:
            return
        header = self.query_one(RealTimePlaybackHeader)
        header.update_memory_text()
        header.update_status(self.phase)
        header.update_stats(
            self.processed,
            self.total,
            self.interpolated_buffer_seconds,
            zero_buffer_is_error=(
                not self.waiting_for_continue
                and not self.exit_after_terminal
                and self.terminal_result is None
            ),
        )
        # The source-text band renders uniformly dim, so the finished flag
        # (playhead past the last segment) no longer affects rendering.
        playing_text, _ = self._current_playing_text()
        self.query_one(RealTimePlaybackSourceText).update_playing_text(playing_text)
        header.update_hotkey(self.prompt_mode)

    def action_continue(self) -> None:
        if self.waiting_for_continue and self.operation_id is not None:
            if ModelWorker.continue_realtime_playback(self.operation_id):
                self.waiting_for_continue = False
                self.exit_after_terminal = True
                # Freeze the last rendered header while the worker closes its
                # audio stream and sends the terminal event. Event polling must
                # continue, but teardown should not produce transient status or
                # prompt changes immediately before the full-screen app exits.
                self.teardown_in_progress = True
            return
        if self.terminal_result is not None:
            self.exit(self.terminal_result)

    def action_cancel_or_reset(self) -> None:
        super().action_cancel_or_reset()
        # CTRL-C also snaps the log back to its end: a user who scrolled
        # up to read earlier output still sees the cancellation notice and
        # the latest worker lines.
        self._snap_log_to_tail()

    def action_cancel_or_continue(self) -> None:
        # Escape dismisses the find bar first, and otherwise only finishes a
        # completed (or waiting-to-finish) session. It no longer interrupts
        # realtime playback: CTRL-C is the sole interrupt/hard-reset key.
        if self.find_active:
            self.close_find()
            return
        if self.terminal_result is not None or self.waiting_for_continue:
            self.action_continue()


def _run_realtime_playback_console(
    state: State,
    phrase_groups: list[PhraseGroup],
    line_range: tuple[int, int] | None,
) -> RealTimePlaybackModalResult:
    try:
        operation_id = ModelWorker.submit_realtime_playback(
            state=state,
            phrase_groups=phrase_groups,
            line_range=line_range,
        )
    except Exception as exception:
        return RealTimePlaybackModalResult(
            RealTimePlaybackTerminalStatus.FAILED,
            f"{type(exception).__name__}: {exception}",
        )

    interrupts = Interrupts()
    interrupts.set("model worker realtime playback")
    cancellation_sent = False
    try:
        while True:
            if interrupts.did_interrupt and not cancellation_sent:
                cancellation_sent = ModelWorker.request_cancel(operation_id)
            event = ModelWorker.get_event(timeout=0.1)
            if event is None or getattr(event, "operation_id", None) != operation_id:
                continue
            if isinstance(event, ConsoleOutput):
                target = sys.stderr if event.stream == "stderr" else sys.stdout
                target.write(event.text)
                target.flush()
            elif isinstance(event, ConsoleFlush):
                target = sys.stderr if event.stream == "stderr" else sys.stdout
                target.flush()
            elif isinstance(event, RealTimePlaybackUpdate):
                if isinstance(event.update, RealTimePlaybackAwaitingContinue):
                    interrupts.clear()
                    ask.ask_enter_to_continue()
                    ModelWorker.continue_realtime_playback(operation_id)
                elif isinstance(event.update, GenerationTimedOut):
                    # A gen timeout hard-resets the worker even when a cancel
                    # was already requested (cancellation_sent stays True).
                    message = make_gen_timeout_message(event.update.timeout_seconds)
                    print(message)
                    ModelWorker.reset()
                    return RealTimePlaybackModalResult(
                        RealTimePlaybackTerminalStatus.WORKER_RESET,
                        message,
                    )
            elif isinstance(event, RealTimePlaybackFinished):
                return RealTimePlaybackModalResult(event.status, event.message)
            elif isinstance(event, WorkerCommandFailed):
                return RealTimePlaybackModalResult(
                    RealTimePlaybackTerminalStatus.FAILED,
                    event.message,
                )
            elif isinstance(event, WorkerExited):
                return RealTimePlaybackModalResult(
                    RealTimePlaybackTerminalStatus.WORKER_RESET,
                    event.message or "Model worker exited unexpectedly",
                )
    finally:
        interrupts.clear()


def _present_console_result(result: RealTimePlaybackModalResult) -> None:
    labels = {
        RealTimePlaybackTerminalStatus.COMPLETED: "Realtime playback completed.",
        RealTimePlaybackTerminalStatus.CANCELLED: "Realtime playback cancelled.",
        RealTimePlaybackTerminalStatus.ABORTED: "Realtime playback stopped.",
        RealTimePlaybackTerminalStatus.FAILED: "Realtime playback failed.",
        RealTimePlaybackTerminalStatus.WORKER_RESET: "Model worker was reset.",
    }
    print(labels[result.status])
    if result.message:
        print(result.message)


def run_real_time_playback_modal(
    state: State,
    phrase_groups: list[PhraseGroup],
    line_range: tuple[int, int] | None,
) -> RealTimePlaybackModalResult:
    start_error = ModelWorker.start()
    if start_error:
        result = RealTimePlaybackModalResult(
            RealTimePlaybackTerminalStatus.FAILED,
            start_error,
        )
        _present_console_result(result)
        if ask.can_hotkey:
            ask.ask_enter_to_continue()
        return result

    if not can_textual():
        result = _run_realtime_playback_console(state, phrase_groups, line_range)
        _present_console_result(result)
        return result

    app = RealTimePlaybackApp(state, phrase_groups, line_range)
    try:
        result = app.run(inline=False)
    except Exception as exception:
        if app.terminal_result is not None:
            return app.terminal_result
        if app.operation_id is not None:
            ModelWorker.reset()
        result = RealTimePlaybackModalResult(
            RealTimePlaybackTerminalStatus.FAILED,
            f"{type(exception).__name__}: {exception}",
        )
        _present_console_result(result)
        return result
    if result is not None:
        return result
    if app.terminal_result is not None:
        return app.terminal_result
    if app.operation_id is not None:
        ModelWorker.reset()
    result = RealTimePlaybackModalResult(
        RealTimePlaybackTerminalStatus.FAILED,
        "Realtime playback interface closed without a result",
    )
    _present_console_result(result)
    return result
