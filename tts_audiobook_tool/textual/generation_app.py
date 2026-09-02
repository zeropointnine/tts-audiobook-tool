from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult

from tts_audiobook_tool import ask, text_util, util
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support import make_worker_log_file_path
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.constants import (
    COL_DEFAULT,
    COL_DIM_ITALICS,
    PROJECT_GEN_LOG_SUBDIR,
    PROJECT_JSON_FILE_NAME,
)
from tts_audiobook_tool.generation_events import (
    GenerationPhase,
    GenerationProgress,
    GenerationStarted,
    GenerationStats,
    GenerationTimedOut,
)
from tts_audiobook_tool.gen_timeout_util import make_gen_timeout_message
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleOutput,
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    ModelWorkerEvent,
    WorkerCommandFailed,
    WorkerExited,
)
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.textual.generation_header import GenerationHeader, PromptMode
from tts_audiobook_tool.textual.textual_shared import can_textual
from tts_audiobook_tool.textual.worker_app import (
    FINAL_OUTPUT_SETTLE_SECONDS,
    ConsoleLineAssembler,
    WorkerTextualApp,
    _split_pending_control,
    worker_app_css,
)
if TYPE_CHECKING:
    from tts_audiobook_tool.state import State


# ConsoleLineAssembler and make_worker_log_file_path live in the worker-app
# base and in app support; the test suite imports (and patches) them from
# this module, so both are re-exported here.
__all__ = [
    "ConsoleLineAssembler",
    "GenerationApp",
    "GenerationModalResult",
    "GenerationTranscript",
    "make_generation_transcript_path",
    "make_worker_log_file_path",
    "run_generation_app",
]


@dataclass(frozen=True)
class GenerationModalResult:
    status: GenerationTerminalStatus
    remaining_range_string: str
    transcript_path: str
    message: str = ""
    failed_items: int = 0
    errored_items: int = 0

    @property
    def completed(self) -> bool:
        return self.status == GenerationTerminalStatus.COMPLETED

    @property
    def completed_cleanly(self) -> bool:
        """Whether the job completed without any item needing attention."""
        return self.completed and not (self.failed_items or self.errored_items)


class GenerationTranscript:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.path = path if enabled else ""
        self._enabled = enabled
        self._file = None
        if not enabled:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8", newline="\n")
        self._pending_control = ""

    def write_chunk(self, text: str) -> None:
        if self._file is None or not text:
            return
        text = self._pending_control + text
        self._pending_control, text = _split_pending_control(text)
        plain_text = text_util.strip_ansi_codes(text)
        # Retain each dynamic progress update as a readable transcript line,
        # even though carriage returns replace one live row in the UI.
        plain_text = plain_text.replace("\r\n", "\n").replace("\r", "\n")
        self._file.write(plain_text)
        self._file.flush()

    def write_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        self.write_chunk("".join(f"{line}\n" for line in lines))

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()


_TERMINAL_LABELS: dict[GenerationTerminalStatus, str] = {
    GenerationTerminalStatus.COMPLETED: "Generation completed.",
    GenerationTerminalStatus.CANCELLED: "Generation cancelled.",
    GenerationTerminalStatus.ABORTED: "Generation stopped.",
    GenerationTerminalStatus.FAILED: "Generation failed.",
    GenerationTerminalStatus.WORKER_RESET: "Generation stopped; model worker was reset.",
}


class GenerationApp(WorkerTextualApp[GenerationModalResult]):
    """Full-screen generation session: header, divider, live worker log."""

    CSS = worker_app_css("generation-divider")

    DIVIDER_ID: ClassVar[str] = "generation-divider"
    OUTPUT_SHELL_ID: ClassVar[str] = "generation-output-shell"
    HEADER_UPDATE_SECONDS: ClassVar[float] = 1.0

    def __init__(
        self,
        state: State,
        indices: set[int],
        batch_size: int,
        is_regen: bool,
        transcript: GenerationTranscript,
    ) -> None:
        super().__init__(state)
        self.indices = set(indices)
        self.batch_size = batch_size
        self.is_regen = is_regen
        self.transcript = transcript
        self.progress = GenerationProgress(0, len(indices), len(indices))
        self.stats: GenerationStats | None = None
        # True once a generation that should return on its own reaches its
        # terminal summary: the app then exits immediately, without waiting
        # for ENTER. That covers any quick generation that completed (it
        # returns to the editor) and a completed regular generation with
        # gen_auto_concat enabled (it proceeds to concatenation).
        self.auto_continue = False

    def compose_header(self) -> ComposeResult:
        # (bottom prompt row removed; its trigger points are retained in
        # terminal_summary_extra_lines and action_cancel_or_reset)
        # yield Static("[CTRL-C] Request cancellation", id="generation-prompt", markup=False)
        yield GenerationHeader(
            title="Quick generate" if self.is_regen else "Generating audio...",
            id="generation-header",
        )

    def submit_worker_job(self) -> str:
        return ModelWorker.submit_generation(
            state=self.state,
            indices=self.indices,
            batch_size=self.batch_size,
            is_regen=self.is_regen,
        )

    def _on_submit_failure(self, message: str) -> None:
        self._finalize_local_failure(message)

    def _handle_session_event(self, event: ModelWorkerEvent) -> None:
        if isinstance(event, GenerationUpdate):
            self._handle_update(event.update)
        elif isinstance(event, GenerationFinished):
            self._begin_finish(event)

    def _handle_update(self, update: object) -> None:
        # `update` is a GenerationEvent (typed in the IPC protocol); dispatch
        # on the concrete type.
        if isinstance(update, GenerationPhase):
            self.phase = update.label
        elif isinstance(update, GenerationStarted):
            self.progress = GenerationProgress(0, update.total, update.total)
        elif isinstance(update, GenerationProgress):
            self.progress = update
        elif isinstance(update, GenerationStats):
            self.stats = update
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

    def _begin_finish(self, event: GenerationFinished) -> None:
        if self.finishing or self.terminal_result is not None:
            return
        self.finishing = True
        self.phase = event.status.value.replace("_", " ").title()
        self._update_header()
        self.set_timer(
            FINAL_OUTPUT_SETTLE_SECONDS,
            lambda: self._finish_from_worker_event(event),
        )

    def _finish_from_worker_event(self, event: GenerationFinished) -> None:
        self._finalize_console()
        self._show_terminal_summary(
            GenerationModalResult(
                status=event.status,
                remaining_range_string=event.remaining_range_string,
                transcript_path=self.transcript.path,
                message=event.message,
                failed_items=self.progress.failed,
                errored_items=self.progress.errored,
            )
        )

    def _finalize_local_failure(self, message: str) -> None:
        if self.terminal_result is not None:
            return
        self.finishing = True
        self._finalize_console()
        self._show_terminal_summary(
            GenerationModalResult(
                status=GenerationTerminalStatus.FAILED,
                remaining_range_string=self.state.project.generate_range_string,
                transcript_path=self.transcript.path,
                message=message,
            )
        )

    def _on_worker_command_failed(self, message: str) -> None:
        self._finalize_local_failure(message)

    def _on_worker_exit(self, event: WorkerExited) -> None:
        # Synthesized by the client when the worker process died; it
        # is the single death signal (no liveness polling here). A
        # hard reset in progress owns the terminal summary instead.
        if not self.reset_in_progress:
            self._finalize_worker_exit(
                event.message or "Model worker exited unexpectedly"
            )

    def _finalize_worker_exit(self, message: str) -> None:
        if self.terminal_result is not None:
            return
        self.finishing = True
        self._finalize_console()
        self._show_terminal_summary(
            GenerationModalResult(
                status=GenerationTerminalStatus.WORKER_RESET,
                remaining_range_string=_read_persisted_range_string(self.state),
                transcript_path=self.transcript.path,
                message=message,
            )
        )

    def _record_console_output(self, text: str) -> None:
        self.transcript.write_chunk(text)

    def _append_application_lines(self, lines: list[str]) -> None:
        self._append_lines(lines)
        self.transcript.write_lines(lines)

    def terminal_label(self, result: GenerationModalResult) -> str:
        if self._suppresses_completion_banner(result):
            return ""
        return _TERMINAL_LABELS[result.status]

    def terminal_display_label(self, result: GenerationModalResult) -> str:
        if self._suppresses_completion_banner(result):
            return ""
        return _styled_terminal_label(result.status, _TERMINAL_LABELS)

    def _suppresses_completion_banner(self, result: GenerationModalResult) -> bool:
        """Whether the terminal summary must omit its status banner.

        A quick generation that completed returns straight to the editor,
        so the banner (and the phase it would set) would only flash for the
        brief auto-return delay. Interrupted or failed quick generations
        still show their labels.
        """
        return self.is_regen and result.completed_cleanly

    def terminal_summary_extra_lines(self, result: GenerationModalResult) -> list[str]:
        lines = []
        if result.transcript_path:
            lines.append(
                f"Transcript: {text_util.make_terminal_hyperlink(result.transcript_path, is_file=True)}"
            )
        if self.auto_continue:
            # A regular generation proceeds to concatenation. Quick generation
            # instead returns directly to the editor without displaying a
            # misleading concatenation handoff.
            if not self.is_regen:
                lines.extend(["", "Proceeding to concatenation..."])
        else:
            lines.extend(["", f"Press {util.make_hotkey_string('ENTER')} to continue"])
        return lines

    def _pre_terminal_summary(self, result: GenerationModalResult) -> None:
        # Quick generation returns immediately only when its one item needs no
        # attention. A generated segment tagged as failed (excess word errors),
        # or an item that exhausted generation retries, keeps the result open
        # for review just like an interrupted job.
        quick_return = self.is_regen and result.completed_cleanly
        regular_auto_concat = (
            not self.is_regen
            and result.completed
            and self.state.project.gen_auto_concat
        )
        self.auto_continue = quick_return or regular_auto_concat

    def _post_terminal_summary(self, result: GenerationModalResult) -> None:
        if result.status in (GenerationTerminalStatus.COMPLETED, GenerationTerminalStatus.ABORTED) \
                and not self.is_regen:
            app_support.play_done_sound()

        if self.auto_continue:
            # Return immediately, with no ceremony and no ENTER wait, so the
            # caller's flow (editor reopen or concatenation) continues.
            self.action_continue()

    @property
    def prompt_mode(self) -> PromptMode:
        """State driving the header's bottom prompt line: the default
        interrupt hint, the kill-process hint while the worker waits for a
        safe boundary, and the continue hint once the job has stopped."""
        if self.terminal_result is not None:
            if self.auto_continue:
                return "auto_return" if self.is_regen else "auto_continue"
            return "finished"
        if self.cancel_pending:
            return "cancel_pending"
        return "default"

    def _update_header(self) -> None:
        if not self.is_mounted:
            return
        now = self.finished_at if self.finished_at is not None else time.monotonic()
        elapsed = max(0.0, now - self.started_at)
        header = self.query_one(GenerationHeader)
        header.update_memory_text()
        header.update_status(self.phase)
        header.update_stats(
            self.progress.processed,
            self.progress.total,
            elapsed,
            eta_seconds=self.progress.eta_seconds,
        )
        header.update_hotkey(self.prompt_mode)

    def action_cancel_or_reset(self) -> None:
        super().action_cancel_or_reset()
        # CTRL-C also snaps the log back to its end: a user who scrolled
        # up to read earlier output still sees the cancellation notice and
        # the latest worker lines.
        self._snap_log_to_tail()

    def make_worker_reset_result(self, message: str) -> GenerationModalResult:
        return GenerationModalResult(
            status=GenerationTerminalStatus.WORKER_RESET,
            remaining_range_string=_read_persisted_range_string(self.state),
            transcript_path=self.transcript.path,
            message=message,
        )


def make_generation_transcript_path(project_dir_path: str) -> str:
    directory = os.path.join(project_dir_path, PROJECT_GEN_LOG_SUBDIR)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return os.path.join(directory, f"generation-{timestamp}.log")


def _read_persisted_range_string(state: State) -> str:
    try:
        path = os.path.join(state.project.dir_path, PROJECT_JSON_FILE_NAME)
        with open(path, "r", encoding="utf-8") as file:
            value = json.load(file).get("generate_range_string")
        return value if isinstance(value, str) else state.project.generate_range_string
    except (AttributeError, OSError, ValueError, TypeError):
        return state.project.generate_range_string


def _reconcile_generation_result(state: State, result: GenerationModalResult) -> None:
    if result.remaining_range_string:
        state.project.generate_range_string = result.remaining_range_string
    state.project.sound_segments.force_invalidate()
    # The file-based segment catalog is the source of truth for what was
    # actually written: the worker's in-memory range update only reaches disk
    # if the worker ran to its save point, so after a hard reset or worker
    # crash it is stale. Re-derive the range string from the (just
    # invalidated) catalog so persisted state matches the audio on disk.
    save_error = ProjectUtil.persist_range_without_generated_items(state.project)
    if save_error:
        ask.ask_error(save_error)


def _run_generation_console(
    state: State,
    indices: set[int],
    batch_size: int,
    is_regen: bool,
    transcript: GenerationTranscript,
) -> GenerationModalResult:
    assembler = ConsoleLineAssembler()
    progress = GenerationProgress(0, len(indices), len(indices))
    try:
        operation_id = ModelWorker.submit_generation(
            state=state,
            indices=indices,
            batch_size=batch_size,
            is_regen=is_regen,
        )
    except Exception as exception:
        return GenerationModalResult(
            GenerationTerminalStatus.FAILED,
            state.project.generate_range_string,
            transcript.path,
            f"{type(exception).__name__}: {exception}",
        )

    interrupts = Interrupts()
    interrupts.set("model worker generation")
    cancellation_sent = False
    try:
        # The loop is guaranteed to terminate: a healthy worker answers with
        # GenerationFinished or WorkerCommandFailed, and if the process dies
        # the drainer synthesizes WorkerExited for this operation.
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
                transcript.write_chunk(event.text)
                assembler.feed(event.text)
            elif isinstance(event, GenerationUpdate):
                if isinstance(event.update, GenerationProgress):
                    progress = event.update
                elif isinstance(event.update, GenerationTimedOut):
                    # A gen timeout hard-resets the worker even when a cancel
                    # was already requested (cancellation_sent stays True).
                    message = make_gen_timeout_message(event.update.timeout_seconds)
                    print(message)
                    ModelWorker.reset()
                    return GenerationModalResult(
                        GenerationTerminalStatus.WORKER_RESET,
                        _read_persisted_range_string(state),
                        transcript.path,
                        message,
                    )
            elif isinstance(event, GenerationFinished):
                assembler.finish()
                return GenerationModalResult(
                    event.status,
                    event.remaining_range_string,
                    transcript.path,
                    event.message,
                    failed_items=progress.failed,
                    errored_items=progress.errored,
                )
            elif isinstance(event, WorkerCommandFailed):
                assembler.finish()
                return GenerationModalResult(
                    GenerationTerminalStatus.FAILED,
                    state.project.generate_range_string,
                    transcript.path,
                    event.message,
                )
            elif isinstance(event, WorkerExited):
                assembler.finish()
                return GenerationModalResult(
                    GenerationTerminalStatus.WORKER_RESET,
                    _read_persisted_range_string(state),
                    transcript.path,
                    event.message or "Model worker exited unexpectedly",
                )
    finally:
        interrupts.clear()


def _styled_terminal_label(
    status: GenerationTerminalStatus,
    labels: dict[GenerationTerminalStatus, str],
) -> str:
    label = labels[status]
    if status is GenerationTerminalStatus.CANCELLED:
        return f"{COL_DIM_ITALICS}{label}{COL_DEFAULT}"
    return label


def _present_console_result(
    state: State,
    result: GenerationModalResult,
    transcript: GenerationTranscript,
    is_regen: bool,
) -> None:
    # Match the Textual quick-generation path: a clean completion returns to
    # the editor without adding a terminal summary or waiting for ENTER.
    if is_regen and result.completed_cleanly:
        return

    labels = {
        GenerationTerminalStatus.COMPLETED: "Generation completed.",
        GenerationTerminalStatus.CANCELLED: "Generation cancelled.",
        GenerationTerminalStatus.ABORTED: "Generation stopped.",
        GenerationTerminalStatus.FAILED: "Generation failed.",
        GenerationTerminalStatus.WORKER_RESET: "Model worker was reset.",
    }
    lines = ["", _styled_terminal_label(result.status, labels)]
    if result.message:
        lines.append(result.message)
    if result.transcript_path:
        lines.append(f"Transcript: {text_util.make_terminal_hyperlink(result.transcript_path, is_file=True)}")
    transcript.write_lines(lines)
    print("\n".join(lines))
    if result.completed and state.project.gen_auto_concat and not is_regen:
        # Auto-concat is enabled: no ENTER wait; program flow resumes at the
        # concatenation step.
        print("Proceeding to concatenation...")
    elif (not result.completed or is_regen) and ask.can_hotkey:
        # Reaching this branch for quick generation means either the job
        # stopped or its item was tagged as failed/errored.
        ask.ask_enter_to_continue()


def run_generation_app(
    state: State,
    indices: set[int],
    batch_size: int,
    is_regen: bool,
) -> GenerationModalResult:
    transcript = GenerationTranscript(
        make_generation_transcript_path(state.project.dir_path),
        enabled=state.prefs.save_gen_log,
    )
    app: GenerationApp | None = None
    try:
        start_error = ModelWorker.start()
        if start_error:
            result = GenerationModalResult(
                GenerationTerminalStatus.FAILED,
                state.project.generate_range_string,
                transcript.path,
                start_error,
            )
            _present_console_result(state, result, transcript, is_regen)
            return result
        if not can_textual():
            result = _run_generation_console(
                state,
                indices,
                batch_size,
                is_regen,
                transcript,
            )
            _reconcile_generation_result(state, result)
            _present_console_result(state, result, transcript, is_regen)
            return result

        app = GenerationApp(state, indices, batch_size, is_regen, transcript)
        try:
            result = app.run(inline=False)
        except Exception as exception:
            if app.terminal_result is not None:
                result = app.terminal_result
            else:
                if app.operation_id is not None:
                    ModelWorker.reset()
                result = GenerationModalResult(
                    GenerationTerminalStatus.FAILED,
                    _read_persisted_range_string(state),
                    transcript.path,
                    f"{type(exception).__name__}: {exception}",
                )
            _reconcile_generation_result(state, result)
            _present_console_result(state, result, transcript, is_regen)
            return result
        if result is None:
            if app.terminal_result is not None:
                result = app.terminal_result
            else:
                if app.operation_id is not None:
                    ModelWorker.reset()
                result = GenerationModalResult(
                    GenerationTerminalStatus.FAILED,
                    _read_persisted_range_string(state),
                    transcript.path,
                    "Generation interface closed without a result",
                )
        _reconcile_generation_result(state, result)
        if result.status == GenerationTerminalStatus.FAILED and app.terminal_result is None:
            _present_console_result(state, result, transcript, is_regen)
        return result
    finally:
        transcript.close()
