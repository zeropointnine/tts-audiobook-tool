"""Base for the full-screen model-worker session apps.

The phrase-generation and realtime-playback Textual apps share the same
session shell: the key bindings, the screen chrome (header, divider rule,
worker log, find bar) and the app exit. The worker-job lifecycle itself
(submission, event drain, console plumbing, cancellation and hard reset,
terminal summary) lives in ``worker_session.WorkerSessionMixin``, which this
base hosts; in-app modal sessions host the same mixin on a ``Screen``.

A concrete app subclasses ``WorkerTextualApp`` and supplies the
session-specific pieces: the class variables name the chrome elements and
the header refresh rate, and small hooks provide the worker job to submit,
the session protocol events to react to, the terminal result type, and the
summary formatting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.widgets import Input, Rule, Static

from tts_audiobook_tool.textual.textual_shared import TEXTUAL_SCROLLBAR_CSS
from tts_audiobook_tool.textual.worker_content import WorkerLog, WorkerLogContentArea
from tts_audiobook_tool.textual.worker_session import (
    FINAL_OUTPUT_SETTLE_SECONDS,
    ConsoleLineAssembler,
    ResultT,
    WorkerModalResult,
    WorkerSessionMixin,
    _split_pending_control,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.worker_reset import (
    HardResetCause,
    HardResetRequest,
    perform_hard_reset,
)

if TYPE_CHECKING:
    from tts_audiobook_tool.state import State

# Names re-exported for callers and tests that import them from this module.
__all__ = [
    "EVENT_POLL_SECONDS",
    "FINAL_OUTPUT_SETTLE_SECONDS",
    "ConsoleLineAssembler",
    "WorkerModalResult",
    "WorkerTextualApp",
    "_split_pending_control",
    "session_failure_result",
    "worker_app_css",
]

# The worker event queue is polled on this interval.
EVENT_POLL_SECONDS = 0.03


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
            TEXTUAL_SCROLLBAR_CSS,
            "    # User CSS overrides the `Rule.-horizontal` DEFAULT_CSS margins (1 row",
            "    # above and below the divider), so the divider block is exactly 1 row.",
            f"    #{divider_id} {{ color: #888888; margin: 0; }}",
        )
    )



class WorkerTextualApp(WorkerSessionMixin[ResultT], App[ResultT]):
    """Base app for a full-screen model-worker session.

    The worker-job lifecycle comes from ``WorkerSessionMixin``; this class
    adds the shared session shell: key bindings, screen chrome, the find bar,
    and the app exit.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        # Shadow Textual's default Ctrl+C action; Escape owns interruption.
        Binding("ctrl+c", "ignore_ctrl_c", show=False, priority=True),
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
        self._init_session(state)
        # Ctrl+F opens the bottom find bar over the log; typing edits the
        # query, Enter submits it (next match), Shift+Enter goes back. While
        # find owns focus, ``check_action`` lets Enter submit the query and
        # Ctrl+C copy input text; Escape closes search without cancelling.
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
            output_filters=Tts.get_info(self.state.project).output_filters,
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
        self._start_session(EVENT_POLL_SECONDS, self.HEADER_UPDATE_SECONDS)

    def _leave_session(self, result: ResultT) -> None:
        self.exit(result)

    def _close_find_if_open(self) -> bool:
        if not self.find_active:
            return False
        self.close_find()
        return True


    def action_ignore_ctrl_c(self) -> None:
        """Override Textual's built-in Ctrl+C action outside the find input."""

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

        While find owns focus, Enter submits the query and Ctrl+C copies the
        input selection rather than invoking session actions. Escape stays
        enabled to close search through ``action_cancel_or_continue``.
        """
        if self.find_active and action in ("continue", "cancel_or_reset", "ignore_ctrl_c"):
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


def session_failure_result(
    app: WorkerTextualApp[ResultT],
    make_result: Callable[[str, HardResetCause | None], ResultT],
    message: str,
) -> ResultT:
    """Classify a worker session that raised or returned no result.

    Returns the session's own terminal result when it recorded one before
    failing, so a caller never re-derives an outcome the session already
    presented. Otherwise the failure is attributed to the interface and the
    model worker is hard-reset first: the interface died while its worker was
    mid-job, and leaving it running would strand the inference and its
    resident model memory.

    ``make_result`` builds the caller's own failure result, because each
    session type has its own result dataclass (generation additionally
    carries the remaining range and the transcript path). The caller decides
    how the result is presented; this helper only classifies it.
    """
    if app.terminal_result is not None:
        return app.terminal_result
    reset_cause: HardResetCause | None = None
    if app.operation_id is not None:
        reset_cause = HardResetCause.INTERFACE_FAILURE
        message = perform_hard_reset(HardResetRequest(reset_cause, message)).message
    return make_result(message, reset_cause)
