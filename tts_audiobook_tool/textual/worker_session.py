"""Model-worker session lifecycle shared by every worker-session host.

``WorkerSessionMixin`` owns the lifecycle of one worker job: submission, the
``operation_id``-filtered event drain, console plumbing into a
``WorkerLogContentArea``, the cancel -> hard-reset ladder, and the terminal
summary flow. It is host-agnostic: it only needs the scheduling and query
methods that both a Textual ``App`` and a ``Screen`` provide, so the
full-screen session apps (``WorkerTextualApp``) and in-app modal sessions
(``QuickGenModal``) share one implementation.

The host supplies the session-specific pieces through small hooks (job
submission, event dispatch, terminal result and summary formatting, header
rendering) and decides how a finished session leaves (``action_continue``).
"""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from tts_audiobook_tool.app_support import make_worker_log_file_path
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleFlush,
    ConsoleOutput,
    ModelWorkerEvent,
    WorkerCommandFailed,
    WorkerExited,
)
from tts_audiobook_tool.textual.worker_content import WorkerLogContentArea
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.worker_reset import (
    HardResetCause,
    HardResetOutcome,
    HardResetRequest,
    hard_reset_request_from_generation_update,
    perform_hard_reset,
)

if TYPE_CHECKING:
    from textual.app import App
    from textual.dom import DOMNode

    from tts_audiobook_tool.state import State

    # The type checker sees the mixin as a DOM node, so the scheduling and
    # query methods the lifecycle relies on resolve with Textual's own
    # signatures (``app``, ``is_mounted``, ``query_one``, ``set_timer``,
    # ``set_interval``, ``run_worker``).
    _SessionHostBase = DOMNode
else:
    class _SessionHostBase:
        """Runtime placeholder: the concrete App / Screen host supplies the
        Textual methods the lifecycle uses."""


# Once the terminal worker event arrives, wait this long before presenting
# the summary so a final console flush can still reach the log.
FINAL_OUTPUT_SETTLE_SECONDS = 0.1


_CURSOR_HOME_RE = re.compile(r"\x1b\[(?:1)?G")
_ERASE_LINE_RE = re.compile(r"\x1b\[[0-9;?]*K")
_INCOMPLETE_CSI_RE = re.compile(r"\x1b(?:\[[0-9;?]*)?$")
_OSC_SEQUENCE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _preserve_hyperlink_osc(match: re.Match[str]) -> str:
    """Keep OSC 8 hyperlinks for Rich; discard non-display OSC controls."""
    sequence = match.group(0)
    return sequence if sequence.startswith("\x1b]8;") else ""


def _incomplete_osc_start(text: str) -> int:
    """Start index of a trailing OSC sequence lacking its BEL/ST terminator."""
    start = text.rfind("\x1b]")
    if start == -1:
        return -1
    tail = text[start:]
    if "\x07" in tail or "\x1b\\" in tail:
        return -1
    return start


def _split_pending_control(text: str) -> tuple[str, str]:
    """
    Returns (pending_control, remaining_text). A trailing control sequence
    which may continue in a later chunk is held back and returned first.
    """
    osc_start = _incomplete_osc_start(text)
    if osc_start != -1:
        return text[osc_start:], text[:osc_start]
    match = _INCOMPLETE_CSI_RE.search(text)
    if match is not None:
        return match.group(0), text[: match.start()]
    return "", text


class ConsoleLineAssembler:
    """Convert stream chunks into append-only lines plus one replaceable live line."""

    def __init__(self) -> None:
        self.current_line = ""
        self._pending_control = ""

    @staticmethod
    def _normalize_cursor_controls(text: str) -> str:
        text = text.replace("\r\n", "\n")
        # Rich converts OSC 8 hyperlinks to clickable Text spans. Preserve those
        # while discarding non-display OSC controls such as terminal titles.
        text = _OSC_SEQUENCE_RE.sub(_preserve_hyperlink_osc, text)
        text = _CURSOR_HOME_RE.sub("\r", text)
        return _ERASE_LINE_RE.sub("", text)

    def feed(self, text: str) -> tuple[list[str], str]:
        text = self._pending_control + text
        self._pending_control, text = _split_pending_control(text)

        completed: list[str] = []
        for character in self._normalize_cursor_controls(text):
            if character == "\n":
                completed.append(self.current_line)
                self.current_line = ""
            elif character == "\r":
                self.current_line = ""
            else:
                self.current_line += character
        return completed, self.current_line

    def finish(self) -> list[str]:
        if not self.current_line:
            return []
        line = self.current_line
        self.current_line = ""
        return [line]


class WorkerModalResult(Protocol):
    """Shape of a worker session's terminal result.

    Each session type uses its own frozen dataclass (the generation result
    additionally carries the remaining range and the transcript path); the
    base only reads the status and the message.
    """

    @property
    def status(self) -> object: ...

    @property
    def message(self) -> str: ...


ResultT = TypeVar("ResultT", bound=WorkerModalResult)


class WorkerSessionMixin(Generic[ResultT], _SessionHostBase):
    """Worker-job lifecycle for a Textual host (``App`` or ``Screen``).

    Hosts call ``_init_session`` from their ``__init__`` and
    ``_start_session`` once mounted, and implement the hooks below.
    """

    if TYPE_CHECKING:
        # Provided by the concrete Textual host (App / Screen).
        @property
        def is_mounted(self) -> bool: ...

    def _init_session(self, state: State) -> None:
        self.state = state
        self.assembler = ConsoleLineAssembler()
        self.operation_id: str | None = None
        self.started_at = time.monotonic()
        self.finished_at: float | None = None
        self.phase = "Starting worker job"
        self.cancel_requested = False
        self.reset_in_progress = False
        self.reset_request: HardResetRequest | None = None
        self.reset_outcome: HardResetOutcome | None = None
        # Teardown may need to join a reset whose Textual worker was cancelled.
        # Serialize the actual process reset and retain its outcome before the
        # UI callback, so closing a host cannot reset its replacement twice.
        self._reset_lock = threading.Lock()
        self._session_closed = False
        self.finishing = False
        self.terminal_result: ResultT | None = None
        # Set by ``_pre_terminal_summary`` from ``should_auto_exit``: whether
        # this result ends the session on its own instead of waiting for ENTER.
        self.auto_exit = False
        # Whether this session has already placed its closing rule before the
        # terminal summary block; the end-of-run signals set it as soon as the
        # last batch is known to be done, and ``_show_terminal_summary`` is the
        # safety net for the paths that never emit one.
        self._trailing_divider_added = False

    def _start_session(self, poll_seconds: float, header_update_seconds: float) -> None:
        """Submit the worker job and start the event-poll and header timers."""
        self._update_header()
        try:
            self.operation_id = self.submit_worker_job()
        except Exception as exception:
            self._on_submit_failure(f"{type(exception).__name__}: {exception}")
            return
        self.set_interval(poll_seconds, self._drain_worker_events)
        self.set_interval(header_update_seconds, self._update_header)

    def submit_worker_job(self) -> str:
        """Submit the worker job and return its operation id."""
        raise NotImplementedError

    def _on_submit_failure(self, message: str) -> None:
        """Handle a worker submission that failed before any job existed."""
        raise NotImplementedError

    # ----------------------------------------------------------------
    # worker event drain
    # ----------------------------------------------------------------

    def _drain_worker_events(self) -> None:
        """Drain the worker event queue once and dispatch each event to its
        handler.

        The queue is polled on an interval (see ``on_mount``), not a
        ``run_worker``, so the Textual event loop is never blocked. Events
        belonging to a different (older) operation are ignored: a stale
        terminal event must never finalize a newer session.
        """
        operation_id = self.operation_id
        if self._session_closed or operation_id is None:
            return
        for event in ModelWorker.drain_events():
            if getattr(event, "operation_id", None) != operation_id:
                continue
            if isinstance(event, ConsoleOutput):
                self._handle_console_output(event)
            elif isinstance(event, ConsoleFlush):
                self._handle_console_flush(event)
            elif self.reset_in_progress or self.terminal_result is not None:
                # A reset/terminal result owns the session. Keep recording old
                # worker console output, but ignore queued status and terminal
                # events that could otherwise preempt or overwrite it.
                continue
            elif isinstance(event, WorkerCommandFailed):
                self._on_worker_command_failed(event.message)
            elif isinstance(event, WorkerExited):
                self._on_worker_exit(event)
            else:
                self._handle_session_event(event)

    def _handle_session_event(self, event: ModelWorkerEvent) -> None:
        """Dispatch this session's update and finished protocol events."""
        raise NotImplementedError

    def _handle_console_output(self, event: ConsoleOutput) -> None:
        """Relay one console chunk: record it, then feed it to the log's
        current line."""
        self._record_console_output(event.text)
        if self._skip_log_updates():
            return
        completed, live = self.assembler.feed(event.text)
        self._feed_console(completed, live)

    def _handle_console_flush(self, event: ConsoleFlush) -> None:
        """A flush carries no new text, so there is nothing to do: the
        log's current line already mirrors the assembler's current line
        (or an app line committed it, in which case the terminal
        overwrote the bar and it is gone for good)."""

    def _record_console_output(self, text: str) -> None:
        """Record console text outside the on-screen log (no-op by default)."""

    def _skip_log_updates(self) -> bool:
        """Whether console text is still recorded but no longer shown in the
        log."""
        return self.terminal_result is not None

    def _handle_update(self, update: object) -> None:
        """React to one structured worker update."""
        raise NotImplementedError

    def _on_worker_command_failed(self, message: str) -> None:
        """Handle a worker command that failed (the worker is still alive)."""
        raise NotImplementedError

    def _on_worker_exit(self, event: WorkerExited) -> None:
        """React to the synthesized worker-death event."""
        raise NotImplementedError

    # ----------------------------------------------------------------
    # log presentation
    # ----------------------------------------------------------------

    def _expect_separator(self) -> None:
        """Arm full-width rendering of the next printed console divider."""
        self.query_one(WorkerLogContentArea).expect_separator()

    def _append_trailing_divider(self) -> None:
        """Insert the run's closing rule now.

        Used by sessions whose end-of-run signal has no console dash line of
        its own (realtime playback's await-continue boundary), and as the
        safety net for terminal paths that never emit such a signal.
        """
        self._trailing_divider_added = True
        if not self.is_mounted:
            return
        self.query_one(WorkerLogContentArea).append_separator()

    def _arm_trailing_divider(self) -> None:
        """Promote the next printed dash line to the run's closing rule.

        The generation run-end boundary already prints a dash rule in the
        console stream (the summary block's header line), so it is promoted
        to a full-width rule rather than duplicated.
        """
        self._trailing_divider_added = True
        if not self.is_mounted:
            return
        self._expect_separator()

    def _feed_console(self, completed: list[str], live: str) -> None:
        """Apply one console chunk to the log's current line."""
        self.query_one(WorkerLogContentArea).feed(completed, live)

    def _append_lines(self, lines: list[str]) -> None:
        """Append app-generated lines to the log; the current line is
        committed first, so the lines enter the document flow."""
        if lines:
            self.query_one(WorkerLogContentArea).append_lines(lines)

    def _finalize_console(self) -> None:
        """Commit the log's current line and drain the assembler's
        remaining state."""
        self.assembler.finish()
        self.query_one(WorkerLogContentArea).finalize()

    def _append_application_lines(self, lines: list[str]) -> None:
        """Present lines produced by the app rather than the worker."""
        self._append_lines(lines)

    # ----------------------------------------------------------------
    # cancellation and hard reset
    # ----------------------------------------------------------------

    @property
    def cancel_pending(self) -> bool:
        """The worker was asked to stop but has not reached a safe boundary
        yet (and has not died or been hard-reset in the meantime)."""
        return (
            self.cancel_requested
            and not self.finishing
            and not self.reset_in_progress
            and self.terminal_result is None
        )

    @property
    def cancel_or_reset_blocked(self) -> bool:
        """Whether cancellation/escalation is blocked because a summary,
        settle, or hard reset currently owns the session flow."""
        return (
            self.terminal_result is not None
            or self.finishing
            or self.reset_in_progress
        )

    @property
    def hard_reset_available(self) -> bool:
        """Whether the second-ESC hard reset is offered.

        The hard reset dumps the worker to clear its resident *local* model
        memory; in SGL-Omni backend mode inference is remote and the worker
        holds no local TTS model memory, so the escape hatch is not offered
        there. The GEN_TIMEOUT watchdog remains the automatic hang backstop
        in both modes.
        """
        return not Tts.is_remote_mode()

    def _snap_log_to_tail(self) -> None:
        """Scroll the worker log to the bottom and resume tail following.

        A session is usually cancelled from the live view, but the user
        may have scrolled up to read earlier output (manual scrolling
        detaches from the tail). ESC snaps the log back to its end so
        the cancellation notice and the latest worker output are visible
        immediately.
        """
        if not self.is_mounted:
            return
        log = self.query_one(WorkerLogContentArea).worker_log
        log.follow_tail = True
        log.scroll_end(animate=False, immediate=True, x_axis=False)

    def action_cancel_or_reset(self) -> None:
        if self.cancel_or_reset_blocked:
            return
        operation_id = self.operation_id
        if operation_id is None:
            return
        if not self.cancel_requested:
            self.cancel_requested = ModelWorker.request_cancel(operation_id)
            if self.cancel_requested:
                self.phase = "Cancellation requested"
                # These are logical lines, not print() arguments. Leave trailing
                # spacing to the worker so its next blank line is not duplicated.
                lines = ["", f"{COL_ERROR}Cancellation requested, please wait"]
                if self.hard_reset_available:
                    lines.append(
                        f"{COL_ERROR}Or press [{COL_DEFAULT}ESC{COL_ERROR}] again to hard-reset"
                    )
                self._append_application_lines(lines)
                self._update_header()
            return
        if not self.hard_reset_available:
            # SGL-Omni mode: no local model memory to clear, so there is
            # nothing to dump; the pending cooperative cancel takes effect
            # at the worker's next safe boundary.
            return
        self._begin_hard_reset(
            HardResetRequest(HardResetCause.USER_ESCALATION)
        )

    def _begin_hard_reset_for_update(self, update: object) -> bool:
        """Start a program-requested reset for a recognized generation update.

        Returns True whenever ``update`` is a reset request, including when an
        existing reset or terminal result already owns the session.
        """
        request = hard_reset_request_from_generation_update(update)
        if request is None:
            return False
        if (
            self.terminal_result is None
            and not self.finishing
            and not self.reset_in_progress
        ):
            self._begin_hard_reset(request)
        return True

    def _begin_hard_reset(self, request: HardResetRequest) -> None:
        """Start a hard reset and retain its explicit trigger metadata."""
        self.reset_in_progress = True
        self.reset_request = request
        self.reset_outcome = None
        self.phase = "Hard-resetting model worker"
        # The reason is shown once, in the terminal summary beside the outcome.
        lines = ["", f"{COL_ERROR}Terminating and hard-resetting models...", ""]
        self._append_application_lines(lines)
        self._update_header()
        # Captured here on the UI thread: the reset runs on a worker thread and
        # reports back through the owning app's thread-safe call.
        ui_app = self.app
        self.run_worker(
            lambda: self._hard_reset_worker(request, ui_app),
            name="hard-reset-model-worker",
            thread=True,
            exclusive=True,
        )

    def _perform_hard_reset_once(self, request: HardResetRequest) -> HardResetOutcome:
        """Finish one process reset even when the host closes during it."""
        with self._reset_lock:
            if self.reset_outcome is None:
                self.reset_outcome = perform_hard_reset(request)
            return self.reset_outcome

    def _hard_reset_worker(self, request: HardResetRequest, ui_app: App) -> None:
        outcome = self._perform_hard_reset_once(request)
        if self._session_closed:
            return
        try:
            ui_app.call_from_thread(self._hard_reset_finished, outcome)
        except RuntimeError:
            if not self._session_closed:
                raise

    def _hard_reset_finished(self, outcome: HardResetOutcome) -> None:
        if self._session_closed:
            return
        self.reset_in_progress = False
        self.reset_outcome = outcome
        # Colored per line: the log stores one color-scoped line per row.
        message = "\n".join(
            f"{COL_ERROR}{line}{COL_DEFAULT}" for line in outcome.message.splitlines()
        )
        log_path = make_worker_log_file_path()
        if log_path:
            message = (
                f"{message}\nWorker log: {log_path}" if message else f"Worker log: {log_path}"
            )
        self._show_terminal_summary(
            self.make_worker_reset_result(message, outcome.request.cause)
        )

    # ----------------------------------------------------------------
    # terminal summary
    # ----------------------------------------------------------------

    def _show_terminal_summary(self, result: ResultT) -> None:
        """Record the terminal result and render its summary block."""
        self._pre_terminal_summary(result)
        self.finishing = False
        self.finished_at = time.monotonic()
        self.terminal_result = result
        # An empty display label suppresses the banner line (and its
        # separator); the summary then carries only the message and the
        # session's extra lines, if any.
        label = self.terminal_display_label(result)
        lines = ["", label] if label else []
        if result.message:
            lines.append(result.message)
        lines.extend(self.terminal_summary_extra_lines(result))
        if not self._suppress_terminal_summary_ui():
            # Every session ends with the same boundary: the run's closing
            # rule separates the last batch's output from the summary block
            # below. Sessions that already placed it at their end-of-run
            # signal (generation, realtime) skip this.
            if not self._trailing_divider_added:
                self._append_trailing_divider()
            self._append_application_lines(lines)
            # An empty plain label leaves the current phase unchanged.
            phase_label = self.terminal_label(result)
            if phase_label:
                self.phase = phase_label.rstrip(".")
            self._update_header()
        self._post_terminal_summary(result)

    def _pre_terminal_summary(self, result: ResultT) -> None:
        """Reset session-specific state before the result is recorded, and
        record whether the result ends the session on its own."""
        self.auto_exit = self.should_auto_exit(result)

    def should_auto_exit(self, result: ResultT) -> bool:
        """Whether this terminal result leaves the screen on its own instead
        of waiting for the user to press ENTER.

        The generation session (auto-concatenation) and the quick-generation
        modal (clean completion) override this rather than waiting for a
        keypress. A session that returns True must render a prompt that
        reflects the automatic exit and must not print an ENTER hint.

        Realtime playback does not participate: its exit follows a worker
        continue handshake, not a completion predicate.
        """
        return False

    def _post_terminal_summary(self, result: ResultT) -> None:
        """Run session-specific effects after the summary is rendered, and
        leave immediately when the result needs no review."""
        if self.auto_exit:
            self.action_continue()

    def _suppress_terminal_summary_ui(self) -> bool:
        """Whether the summary must not touch the on-screen UI or header."""
        return False

    def terminal_label(self, result: ResultT) -> str:
        """Plain terminal status label (also used for the phase text). An
        empty string leaves the current phase unchanged."""
        raise NotImplementedError

    def terminal_display_label(self, result: ResultT) -> str:
        """Label as rendered in the summary block. An empty string
        suppresses the banner line (and its separator)."""
        return self.terminal_label(result)

    def terminal_summary_extra_lines(self, result: ResultT) -> list[str]:
        """Session-specific lines appended after the message (transcript
        link, continue hint, ...)."""
        return []

    def make_worker_reset_result(
        self,
        message: str,
        cause: HardResetCause,
    ) -> ResultT:
        """Build the terminal result after reset and replacement were attempted."""
        raise NotImplementedError

    # ----------------------------------------------------------------
    # header and exit
    # ----------------------------------------------------------------

    def _update_header(self) -> None:
        """Render the app's header."""
        raise NotImplementedError

    def action_continue(self) -> None:
        if self.terminal_result is not None:
            self._leave_session(self.terminal_result)

    def _leave_session(self, result: ResultT) -> None:
        """Leave the finished session with its result (app exit or screen
        dismissal, depending on the host)."""
        raise NotImplementedError

    def _close_find_if_open(self) -> bool:
        """Close an open find bar; return whether one was open. Hosts without
        a find bar keep this default."""
        return False

    def action_cancel_or_continue(self) -> None:
        # Escape closes search first, dismisses a summary, or requests
        # cancellation (escalating a pending local cancellation to hard reset).
        if self._close_find_if_open():
            return
        if self.finishing or self.reset_in_progress:
            return
        if self.terminal_result is not None:
            self.action_continue()
            return
        self.action_cancel_or_reset()
