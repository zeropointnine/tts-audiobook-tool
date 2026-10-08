from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType

from tts_audiobook_tool import ask, text_util, util
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support import make_worker_log_file_path
from tts_audiobook_tool.app_support.system_sleep import SystemSleepLock
from tts_audiobook_tool.constants import (
    COL_DEFAULT,
    COL_DIM_ITALICS,
    PROJECT_GEN_LOG_SUBDIR,
    PROJECT_JSON_FILE_NAME,
)
from tts_audiobook_tool.generation_events import (
    GenerationPhase,
    GenerationProgress,
    GenerationRunEnded,
    GenerationStarted,
    GenerationStats,
)
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    ModelWorkerEvent,
    WorkerExited,
)
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.textual.generation_header import GenerationHeader, PromptMode
from tts_audiobook_tool.textual.worker_app import (
    FINAL_OUTPUT_SETTLE_SECONDS,
    ConsoleLineAssembler,
    WorkerTextualApp,
    _split_pending_control,
    session_failure_result,
    worker_app_css,
)
from tts_audiobook_tool.worker_reset import HardResetCause
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
    hard_reset_cause: HardResetCause | None = None

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
    BINDINGS: ClassVar[list[BindingType]] = [
        *WorkerTextualApp.BINDINGS,
        Binding("c", "toggle_auto_concat", show=False, priority=True),
    ]

    DIVIDER_ID: ClassVar[str] = "generation-divider"
    OUTPUT_SHELL_ID: ClassVar[str] = "generation-output-shell"
    HEADER_UPDATE_SECONDS: ClassVar[float] = 1.0

    def __init__(
        self,
        state: State,
        indices: set[int],
        batch_size: int,
        transcript: GenerationTranscript,
        on_job_end: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(state)
        self.indices = set(indices)
        self.batch_size = batch_size
        self.transcript = transcript
        self.progress = GenerationProgress(0, len(indices), len(indices))
        self.stats: GenerationStats | None = None
        self._auto_concat_changed = False
        # Called once the worker job reaches a terminal result, before the
        # summary's ENTER wait. Releases the caller's system-sleep lock so an
        # idle machine is not held awake while the user reads the summary.
        self.on_job_end = on_job_end

    def compose_header(self) -> ComposeResult:
        # (bottom prompt row removed; its trigger points are retained in
        # terminal_summary_extra_lines and action_cancel_or_reset)
        # yield Static("[ESC] Request cancellation", id="generation-prompt", markup=False)
        yield GenerationHeader(
            title="Generating audio",
            id="generation-header",
        )

    def submit_worker_job(self) -> str:
        return ModelWorker.submit_generation(
            state=self.state,
            indices=self.indices,
            batch_size=self.batch_size,
            is_regen=False,
        )

    def _on_submit_failure(self, message: str) -> None:
        self._finalize_local_failure(message)

    def _handle_session_event(self, event: ModelWorkerEvent) -> None:
        if isinstance(event, GenerationUpdate):
            self._handle_update(event.update)
        elif isinstance(event, GenerationFinished):
            self._begin_finish(event)

    def _handle_update(self, update: object) -> None:
        # Program-requested resets deliberately override a pending cooperative
        # cancellation and are handled by the shared worker-session lifecycle.
        if self._begin_hard_reset_for_update(update):
            self._update_header()
            return

        # `update` is a GenerationEvent (typed in the IPC protocol); dispatch
        # on the concrete type.
        if isinstance(update, GenerationPhase):
            self.phase = update.label
        elif isinstance(update, GenerationStarted):
            self.progress = GenerationProgress(0, update.total, update.total)
        elif isinstance(update, GenerationProgress):
            if update.current_indices:
                self._expect_separator()
            self.progress = update
        elif isinstance(update, GenerationRunEnded):
            # No further batches: the run's closing rule precedes the trailing
            # summary block, whose header line is already a dash rule.
            self._arm_trailing_divider()
        elif isinstance(update, GenerationStats):
            self.stats = update
        self._update_header()

    def _begin_finish(self, event: GenerationFinished) -> None:
        # A hard reset in progress owns the terminal summary: the run's own
        # finished event may still arrive (an unhealthy-model abort emits its
        # event just before the loop breaks) and must not preempt the reset.
        if self.reset_in_progress or self.finishing or self.terminal_result is not None:
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
                status=GenerationTerminalStatus.FAILED,
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
        return _TERMINAL_LABELS[result.status]

    def terminal_display_label(self, result: GenerationModalResult) -> str:
        return _styled_terminal_label(result.status, _TERMINAL_LABELS)

    def terminal_summary_extra_lines(self, result: GenerationModalResult) -> list[str]:
        lines = []
        if result.transcript_path:
            lines.append(
                f"Transcript: {text_util.make_terminal_hyperlink(result.transcript_path, is_file=True)}"
            )
        if self.auto_exit:
            lines.extend(["", "Proceeding to concatenation..."])
        else:
            lines.extend(["", f"Press {util.make_hotkey_string('ENTER')} to continue"])
        return lines

    def should_auto_exit(self, result: GenerationModalResult) -> bool:
        """A completed generation proceeds to concatenation when that is enabled."""
        return result.completed and self.state.project.gen_auto_concat

    def _pre_terminal_summary(self, result: GenerationModalResult) -> None:
        # The worker job is over. Release the system-sleep lock now, so the
        # summary's ENTER wait (and the concatenation handoff that follows it)
        # does not hold an idle machine awake.
        if self.on_job_end is not None:
            self.on_job_end()
        super()._pre_terminal_summary(result)

    def _post_terminal_summary(self, result: GenerationModalResult) -> None:
        fatal_statuses = {
            GenerationTerminalStatus.ABORTED,
            GenerationTerminalStatus.FAILED,
        }
        reset_should_alert = (
            result.status is GenerationTerminalStatus.WORKER_RESET
            and result.hard_reset_cause is not None
            and result.hard_reset_cause.should_alert
        )
        if result.status in fatal_statuses or reset_should_alert:
            app_support.play_fatal_gen_sound()
        elif result.status is GenerationTerminalStatus.COMPLETED:
            app_support.play_done_sound()

        # Return immediately, with no ceremony and no ENTER wait, so the
        # caller's flow (concatenation) continues.
        super()._post_terminal_summary(result)

    @property
    def prompt_mode(self) -> PromptMode:
        """State driving the header's bottom prompt line: the default
        interrupt hint, the kill-process hint while the worker waits for a
        safe boundary, and the continue hint once the job has stopped."""
        if self.terminal_result is not None:
            return "auto_continue" if self.auto_exit else "finished"
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
        header.update_auto_concat(self.state.project.gen_auto_concat)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "toggle_auto_concat" and self.find_active:
            return False
        return super().check_action(action, parameters)

    def action_toggle_auto_concat(self) -> None:
        project = self.state.project
        previous = project.gen_auto_concat
        project.gen_auto_concat = not previous
        error = project.save()
        if error:
            project.gen_auto_concat = previous
            self.notify(f"Couldn't save concatenate setting: {error}", severity="error")
        else:
            self._auto_concat_changed = True
        self._update_header()

    def action_cancel_or_reset(self) -> None:
        super().action_cancel_or_reset()
        # ESC also snaps the log back to its end: a user who scrolled
        # up to read earlier output still sees the cancellation notice and
        # the latest worker lines.
        self._snap_log_to_tail()

    def make_worker_reset_result(
        self,
        message: str,
        cause: HardResetCause,
    ) -> GenerationModalResult:
        return GenerationModalResult(
            status=GenerationTerminalStatus.WORKER_RESET,
            remaining_range_string=_read_persisted_range_string(self.state),
            transcript_path=self.transcript.path,
            message=message,
            hard_reset_cause=cause,
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


def reconcile_generation_state(state: State, remaining_range_string: str) -> str:
    """Re-sync project state with what the worker actually wrote.

    Returns an error message when the corrected range could not be saved
    (empty on success); the caller decides how to present it.
    """
    if remaining_range_string:
        state.project.generate_range_string = remaining_range_string
    state.project.sound_segments.force_invalidate()
    # The file-based segment catalog is the source of truth for what was
    # actually written: the worker's in-memory range update only reaches disk
    # if the worker ran to its save point, so after a hard reset or worker
    # crash it is stale. Re-derive the range string from the (just
    # invalidated) catalog so persisted state matches the audio on disk.
    return ProjectUtil.persist_range_without_generated_items(state.project)


def _reconcile_generation_result(state: State, result: GenerationModalResult) -> None:
    save_error = reconcile_generation_state(state, result.remaining_range_string)
    if save_error:
        ask.ask_error(save_error)


def _persist_auto_concat_after_worker(app: GenerationApp) -> None:
    # The worker loaded its own project before the toggle and may have saved
    # its stale setting when updating the generation range. Re-save the main
    # project's reconciled state after the worker has stopped.
    if app._auto_concat_changed:
        error = app.state.project.save()
        if error:
            ask.ask_error(f"Couldn't save concatenate setting: {error}")


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
) -> None:
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
    if result.completed and state.project.gen_auto_concat:
        # Auto-concat is enabled: no ENTER wait; program flow resumes at the
        # concatenation step.
        print("Proceeding to concatenation...")
    elif not result.completed and ask.can_hotkey:
        ask.ask_enter_to_continue()


def run_generation_app(
    state: State,
    indices: set[int],
    batch_size: int,
) -> GenerationModalResult:
    transcript = GenerationTranscript(
        make_generation_transcript_path(state.project.dir_path),
        enabled=state.prefs.save_gen_log,
    )
    # The lock covers the worker job only. The textual session releases it as
    # soon as the job reaches a terminal result (before the summary's ENTER
    # wait); every console handoff below releases before it prompts. The
    # finally is the safety net for the remaining paths.
    sleep_lock = SystemSleepLock()
    try:
        start_error = ModelWorker.start()
        if start_error:
            result = GenerationModalResult(
                GenerationTerminalStatus.FAILED,
                state.project.generate_range_string,
                transcript.path,
                start_error,
            )
            sleep_lock.release()
            _present_console_result(state, result, transcript)
            return result

        app = GenerationApp(
            state,
            indices,
            batch_size,
            transcript,
            on_job_end=sleep_lock.release,
        )

        def make_failure_result(
            message: str, reset_cause: HardResetCause | None
        ) -> GenerationModalResult:
            return GenerationModalResult(
                GenerationTerminalStatus.FAILED,
                _read_persisted_range_string(state),
                transcript.path,
                message,
                hard_reset_cause=reset_cause,
            )

        try:
            result = app.run(inline=False)
        except Exception as exception:
            result = session_failure_result(
                app,
                make_failure_result,
                f"{type(exception).__name__}: {exception}",
            )
            sleep_lock.release()
            _reconcile_generation_result(state, result)
            _persist_auto_concat_after_worker(app)
            _present_console_result(state, result, transcript)
            return result
        if result is None:
            result = session_failure_result(
                app,
                make_failure_result,
                "Generation interface closed without a result",
            )
        _reconcile_generation_result(state, result)
        _persist_auto_concat_after_worker(app)
        if result.status == GenerationTerminalStatus.FAILED and app.terminal_result is None:
            sleep_lock.release()
            _present_console_result(state, result, transcript)
        return result
    finally:
        sleep_lock.release()
        transcript.close()
