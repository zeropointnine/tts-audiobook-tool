"""Base for the full-screen model-worker session apps.

The phrase-generation and realtime-playback Textual apps share the same
session shell and lifecycle: the key bindings, the screen chrome (header,
divider rule, worker log), worker submission and the event-drain loop,
console-output plumbing, cancellation and hard-reset handling, and the
terminal summary flow. This module is their common base.

A concrete app subclasses ``WorkerTextualApp`` and supplies the
session-specific pieces: the class variables name the chrome elements and
the header refresh rate, and small hooks provide the worker job to submit,
the session protocol events to react to, the terminal result type, and the
summary formatting.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.widgets import Input, Rule, Static

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
from tts_audiobook_tool.textual.worker_content import WorkerLog, WorkerLogContentArea
from tts_audiobook_tool.tts import Tts

if TYPE_CHECKING:
    from tts_audiobook_tool.state import State


# The worker event queue is polled on this interval.
EVENT_POLL_SECONDS = 0.03
# Once the terminal worker event arrives, wait this long before presenting
# the summary so a final console flush can still reach the log.
FINAL_OUTPUT_SETTLE_SECONDS = 0.1


WORKER_APP_SCREEN_CSS = """
    Screen {
        background: ansi_default;
        color: ansi_default;
    }
"""


# The find bar overlays the bottom row of the worker log (``overlay: screen``
# removes it from height resolution, so the ``1fr`` log never shrinks, and
# ``offset-y: -1`` pulls it up over the log's last row). Its opaque background
# covers the log text underneath. The color variables match the shared palette
# in ``textual_shared.py``, which the worker apps do not load.
WORKER_APP_FIND_CSS = """
    $col-accent: #ffaa44;
    $col-dim: #888888;
    $col-default: ansi_default;

    #find-bar {
        display: none;
        overlay: screen;
        offset-y: -1;
        height: 1;
        width: 100%;
        layout: horizontal;
        background: ansi_default;
    }

    #find-label {
        width: auto;
        height: 1;
        color: $col-accent;
        text-style: italic;
    }

    #find-input {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0;
        background: ansi_default;
        background-tint: transparent;
    }

    #find-result {
        width: 12;
        height: 1;
        color: $col-dim;
        text-style: italic;
        content-align: right middle;
    }

    #find-input:focus {
        border: none;
        color: $col-default;
        background: ansi_default;
        background-tint: transparent;
    }
"""


def worker_app_css(divider_id: str) -> str:
    """Screen CSS for a worker session app (see ``WORKER_APP_SCREEN_CSS``)."""
    return "\n".join(
        (
            WORKER_APP_SCREEN_CSS,
            WORKER_APP_FIND_CSS,
            "    # User CSS overrides the `Rule.-horizontal` DEFAULT_CSS margins (1 row",
            "    # above and below the divider), so the divider block is exactly 1 row.",
            f"    #{divider_id} {{ color: #888888; margin: 0; }}",
        )
    )


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


class WorkerTextualApp(App[ResultT], Generic[ResultT]):
    """Base app for a full-screen model-worker session.

    The base owns the shared session shell and lifecycle: the key bindings,
    the screen chrome, worker submission and the event-drain loop, console
    output plumbing, cancellation and hard-reset handling, and the terminal
    summary flow. Concrete apps supply the session-specific pieces through
    class variables (the divider and output ids, the header refresh rate)
    and small hooks (job submission, session-event dispatch, update
    handling, terminal result and summary formatting, header rendering).
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "cancel_or_reset", show=False, priority=True),
        Binding("escape", "cancel_or_continue", show=False, priority=True),
        Binding("enter", "continue", show=False, priority=True),
        # Textual binds Ctrl+Q to app quit by default; the session owns its
        # own termination path, so the binding is shadowed here.
        Binding("ctrl+q", "ignore_ctrl_q", show=False, priority=True),
        Binding("ctrl+f", "open_find", show=False, priority=True),
        # Many terminals report Shift+Enter as plain Enter, so this binding only
        # works where the terminal emits a distinct key sequence.
        Binding("shift+enter", "find_previous", show=False, priority=True),
        Binding("ctrl+a", "select_all", show=False, priority=True),
    ]

    # Identity of the session chrome; concrete apps define these class
    # variables.
    DIVIDER_ID: ClassVar[str]
    OUTPUT_SHELL_ID: ClassVar[str]
    HEADER_UPDATE_SECONDS: ClassVar[float]

    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state
        self.assembler = ConsoleLineAssembler()
        self.operation_id: str | None = None
        self.started_at = time.monotonic()
        self.finished_at: float | None = None
        self.phase = "Starting worker job"
        self.cancel_requested = False
        self.reset_in_progress = False
        self.finishing = False
        self.terminal_result: ResultT | None = None
        # Ctrl+F opens the bottom find bar over the log; typing edits the
        # query, Enter submits it (next match), Shift+Enter goes back. While
        # find owns focus, the session's Enter/CTRL-C bindings are disabled in
        # ``check_action`` so they never fire from the find input.
        self.find_active = False
        self.find_search_start_index: int | None = None
        self.find_query_submitted = False
        self.find_match_index: int | None = None

    # ----------------------------------------------------------------
    # composition and startup
    # ----------------------------------------------------------------

    def compose_header(self) -> ComposeResult:
        """Yield the app's header widget."""
        raise NotImplementedError

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        yield Rule(id=self.DIVIDER_ID)
        yield from self.compose_below_divider()
        yield WorkerLogContentArea(
            output_filters=Tts.get_type().value.output_filters,
            id=self.OUTPUT_SHELL_ID,
        )
        yield Horizontal(
            Static(self.find_label_text, id="find-label", markup=False),
            Input(id="find-input", compact=True, select_on_focus=False),
            Static("", id="find-result", markup=False),
            id="find-bar",
        )

    def compose_below_divider(self) -> ComposeResult:
        """Extra screen content between the shared divider and the worker log.

        Defaults to nothing. An app that wants a dedicated region there (the
        realtime app's source-text band, framed between this divider and the
        band's own closing rule) yields it from here.
        """
        yield from ()

    def on_mount(self) -> None:
        self.theme = "ansi-dark"
        self.query_one(WorkerLogContentArea).worker_log.focus()
        self._update_header()
        try:
            self.operation_id = self.submit_worker_job()
        except Exception as exception:
            self._on_submit_failure(f"{type(exception).__name__}: {exception}")
            return
        self.set_interval(EVENT_POLL_SECONDS, self._drain_worker_events)
        self.set_interval(self.HEADER_UPDATE_SECONDS, self._update_header)

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
        if operation_id is None:
            return
        for event in ModelWorker.drain_events():
            if getattr(event, "operation_id", None) != operation_id:
                continue
            if isinstance(event, ConsoleOutput):
                self._handle_console_output(event)
            elif isinstance(event, ConsoleFlush):
                self._handle_console_flush(event)
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
        """Whether CTRL-C must be ignored because a summary, settle, or hard
        reset currently owns the session flow."""
        return (
            self.terminal_result is not None
            or self.finishing
            or self.reset_in_progress
        )

    def _snap_log_to_tail(self) -> None:
        """Scroll the worker log to the bottom and resume tail following.

        A session is usually cancelled from the live view, but the user
        may have scrolled up to read earlier output (manual scrolling
        detaches from the tail). CTRL-C snaps the log back to its end so
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
                self._append_application_lines(
                    [
                        "",
                        f"{COL_ERROR}Cancellation requested, please wait\n",
                        f"{COL_ERROR}Or press [{COL_DEFAULT}CTRL-C{COL_ERROR}] again to hard-reset\n",
                        " \n"
                    ]
                )
                # self.query_one("#generation-prompt", Static).update(
                #     "[CTRL-C] Hard-reset option"
                # )
                pass
                self._update_header()
            return
        self._begin_hard_reset()

    def _begin_hard_reset(self) -> None:
        self.reset_in_progress = True
        self.phase = "Hard-resetting model worker"
        self._append_application_lines(["", f"{COL_ERROR}Terminating and hard-resetting models\n", "\n "])
        self._update_header()
        self.run_worker(
            self._hard_reset_worker,
            name="hard-reset-model-worker",
            thread=True,
            exclusive=True,
        )

    def _hard_reset_worker(self) -> None:
        error = ModelWorker.reset()
        self.call_from_thread(self._hard_reset_finished, error)

    def _hard_reset_finished(self, error: str) -> None:
        self.reset_in_progress = False
        message = error
        log_path = make_worker_log_file_path()
        if log_path:
            message = (
                f"{message}\nWorker log: {log_path}" if message else f"Worker log: {log_path}"
            )
        self._show_terminal_summary(self.make_worker_reset_result(message))

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
            self._append_application_lines(lines)
            # An empty plain label leaves the current phase unchanged.
            phase_label = self.terminal_label(result)
            if phase_label:
                self.phase = phase_label.rstrip(".")
            self._update_header()
        self._post_terminal_summary(result)

    def _pre_terminal_summary(self, result: ResultT) -> None:
        """Reset session-specific state before the result is recorded."""

    def _post_terminal_summary(self, result: ResultT) -> None:
        """Run session-specific effects after the summary is rendered."""

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

    def make_worker_reset_result(self, message: str) -> ResultT:
        """Build the terminal result for a hard reset that completed
        successfully."""
        raise NotImplementedError

    # ----------------------------------------------------------------
    # header and exit
    # ----------------------------------------------------------------

    def _update_header(self) -> None:
        """Render the app's header."""
        raise NotImplementedError

    def action_continue(self) -> None:
        if self.terminal_result is not None:
            self.exit(self.terminal_result)

    def action_cancel_or_continue(self) -> None:
        # Escape dismisses the find bar while it is open, and otherwise only
        # finishes a completed session. It no longer interrupts the generation
        # loop: CTRL-C is the sole interrupt/hard-reset key.
        if self.find_active:
            self.close_find()
            return
        if self.terminal_result is not None:
            self.exit(self.terminal_result)

    def action_ignore_ctrl_q(self) -> None:
        """Override Textual's built-in Ctrl+Q quit binding."""

    # ----------------------------------------------------------------
    # find bar
    # ----------------------------------------------------------------

    def _worker_log(self) -> WorkerLog:
        """The worker log searched by the find bar."""
        return self.query_one(WorkerLogContentArea).worker_log

    @property
    def find_label_text(self) -> str:
        return "Search text: "

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Disable session bindings that would steal keys from the find bar.

        While find owns focus, Enter and CTRL-C must fall through to the find
        input (or be ignored) instead of continuing or cancelling the worker
        session. Escape stays enabled and is handled by
        ``action_cancel_or_continue``.
        """
        if self.find_active and action in ("continue", "cancel_or_reset"):
            return False
        return True

    def action_open_find(self) -> None:
        """Open find at the current log position, retaining and selecting its query."""
        find_input = self.query_one("#find-input", Input)
        if not self.find_active:
            self.find_active = True
            self.find_search_start_index = self._current_log_line_index()
            self.find_query_submitted = False
            self.query_one("#find-bar", Horizontal).display = True
            self.query_one("#find-result", Static).update("")
        find_input.focus()
        find_input.select_all()

    def close_find(self) -> None:
        """Hide the find bar and return keyboard control to the worker log."""
        if not self.find_active:
            return
        self.find_match_index = None
        self.find_active = False
        self.query_one("#find-bar", Horizontal).display = False
        log = self._worker_log()
        log.highlight_line_index = None
        log.focus()
        log.refresh()

    def _current_log_line_index(self) -> int | None:
        """The logical line the next search starts after: the current match if
        one is shown, otherwise the line at the top of the viewport."""
        if self.find_match_index is not None:
            return self.find_match_index
        log = self._worker_log()
        return log.line_index_at_scroll_y(int(log.scroll_offset.y))

    def find_match_indices(self, query: str) -> list[int]:
        """Return logical line indices containing a case-insensitive query."""
        if not query:
            return []
        folded_query = query.casefold()
        return [
            index
            for index, text in enumerate(self._worker_log().line_texts())
            if folded_query in text.casefold()
        ]

    def find_relative_match(
        self, match_indices: list[int], direction: int
    ) -> int | None:
        """Find a match in one direction, wrapping past the search start."""
        if not match_indices:
            return None
        line_count = self._worker_log().line_count
        search_start = self.find_search_start_index
        if search_start is None or not 0 <= search_start < line_count:
            return match_indices[0]
        match_index_set = set(match_indices)
        indices = (
            (search_start + (direction * offset)) % line_count
            for offset in range(1, line_count + 1)
        )
        return next((index for index in indices if index in match_index_set), None)

    def advance_find(self, query: str, direction: int) -> None:
        """Advance through matches and update the right-aligned feedback."""
        match_indices = self.find_match_indices(query)
        if not match_indices:
            self.query_one("#find-result", Static).update("No matches")
            return
        self.find_search_start_index = self._current_log_line_index()
        match_index = self.find_relative_match(match_indices, direction)
        if match_index is not None:
            match_number = match_indices.index(match_index) + 1
            self.show_find_match(match_index, match_number, len(match_indices))

    def show_find_match(
        self, match_index: int, match_number: int, match_count: int
    ) -> None:
        """Highlight and scroll to one find result while retaining input focus."""
        self.find_match_index = match_index
        log = self._worker_log()
        log.highlight_line_index = match_index
        log.scroll_to(y=log.line_scroll_y(match_index), animate=False)
        log.follow_tail = False
        log.refresh()
        self.query_one("#find-result", Static).update(
            f"{match_number} of {match_count}"
        )

    def action_find_previous(self) -> None:
        """Move backward after the current query has been submitted."""
        if not self.find_active or not self.find_query_submitted:
            return
        self.advance_find(self.query_one("#find-input", Input).value, -1)

    def action_select_all(self) -> None:
        """While find owns focus, select the input's query text."""
        if self.find_active:
            self.query_one("#find-input", Input).select_all()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Clear stale match feedback without moving the log."""
        if event.input.id == "find-input" and self.find_active:
            self.find_query_submitted = False
            self.find_match_index = None
            log = self._worker_log()
            log.highlight_line_index = None
            log.refresh()
            self.query_one("#find-result", Static).update("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Advance to the next match while retaining find focus."""
        if event.input.id != "find-input" or not self.find_active:
            return
        self.find_query_submitted = True
        self.advance_find(event.value, 1)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        if event.input.id == "find-input":
            self.close_find()

    def on_click(self, event: events.Click) -> None:
        """Dismiss find mode for clicks anywhere outside its text input."""
        if self.find_active and event.widget is not self.query_one("#find-input", Input):
            self.close_find()
