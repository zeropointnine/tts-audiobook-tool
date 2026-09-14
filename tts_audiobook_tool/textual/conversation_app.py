from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Callable, cast

import numpy as np
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.css.errors import StylesheetError
from textual.message import Message
from textual.widgets import Rule, Static

from tts_audiobook_tool import ask, readiness
from tts_audiobook_tool.app_support import app_hint_util
from tts_audiobook_tool.l import L
from tts_audiobook_tool.app_types import Segment
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_DIM_ITALICS
from tts_audiobook_tool.conversation.conversation import (
    ConversationInitialization,
    ConversationRuntime,
)
from tts_audiobook_tool.conversation.conversation_types import (
    ChatInputMode,
    ResponseResult,
    ResponseSnapshot,
)
from tts_audiobook_tool.conversation.prompt_draft import PromptDraft, PromptSubmission
from tts_audiobook_tool.conversation.response_session import ResponseSession
from tts_audiobook_tool.model_worker_protocol import ConsoleFlush, ConsoleOutput
from tts_audiobook_tool.state import State
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual.conversation_widgets import (
    CONVERSATION_CSS,
    ConversationEntry,
    ConversationInputArea,
    ConversationRole,
    ConversationTextEditor,
    ConversationTranscript,
)
from tts_audiobook_tool.textual.textual_shared import (
    STYLE_DEFAULT,
    STYLE_DIM,
    TEXTUAL_SHARED_CSS,
    can_textual,
)
from tts_audiobook_tool.textual.worker_app import ConsoleLineAssembler


class ConversationPhase(str, Enum):
    INITIALIZING = "initializing"
    CANCELLING_INITIALIZATION = "cancelling_initialization"
    INITIALIZATION_CANCELLED = "initialization_cancelled"
    AWAITING_INPUT = "awaiting_input"
    RESPONDING = "responding"
    FAILED = "failed"
    EXITING = "exiting"


class ConversationAppStatus(str, Enum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    DECLINED = "declined"
    FAILED = "failed"


@dataclass(frozen=True)
class ConversationAppResult:
    status: ConversationAppStatus
    message: str = ""


class InitializationFinished(Message):
    def __init__(
        self, session_id: int, result: ConversationInitialization | None, error: str
    ) -> None:
        self.session_id = session_id
        self.result = result
        self.error = error
        super().__init__()


class WorkerConsoleMessage(Message):
    def __init__(self, session_id: int, event: ConsoleOutput | ConsoleFlush) -> None:
        self.session_id = session_id
        self.event = event
        super().__init__()


class InitializationCancelled(Message):
    def __init__(self, reset_error: str) -> None:
        self.reset_error = reset_error
        super().__init__()


class TranscriptionMessage(Message):
    def __init__(
        self, session_id: int, segments: list[Segment], audio: object | None
    ) -> None:
        self.session_id = session_id
        self.segments = segments
        self.audio = audio
        super().__init__()


class RuntimeFailureMessage(Message):
    def __init__(self, session_id: int, turn_id: int | None, error: str) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.error = error
        super().__init__()


class ResponseFinishedMessage(Message):
    def __init__(self, session_id: int, turn_id: int, result: ResponseResult) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.result = result
        super().__init__()


class RuntimeClosedMessage(Message):
    pass


class ConversationInputDivider(Rule):
    """A fixed-height rule that stays blank until initialization finishes."""

    def __init__(self) -> None:
        super().__init__(id="input-divider")
        self.stroke_visible = False

    def show_stroke(self) -> None:
        self.stroke_visible = True
        self.refresh()

    def render(self) -> RenderResult:
        if not self.stroke_visible:
            return Text(" " * self.content_size.width)
        return super().render()


RuntimeFactory = Callable[..., ConversationRuntime]


class ConversationTextualApp(App[ConversationAppResult]):
    """Dedicated multi-turn Textual LLM voice-chat interface."""

    CSS = (
        TEXTUAL_SHARED_CSS
        + CONVERSATION_CSS
        + """
    Screen { background: ansi_default; color: ansi_default; }
    """
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "cancel_initialization", show=False, priority=True),
        Binding("escape", "escape", show=False, priority=True),
        Binding("ctrl+q", "ignore", show=False, priority=True),
        Binding("left", "draft_left", show=False),
        Binding("right", "draft_right", show=False),
        Binding("delete", "draft_delete", show=False),
        Binding("backspace", "draft_delete", show=False),
        Binding("enter", "enter", show=False),
    ]

    def __init__(
        self,
        state: State,
        *,
        blocker_messages: tuple[str, ...] = (),
        runtime_factory: RuntimeFactory = ConversationRuntime,
        runtime: ConversationRuntime | None = None,
    ) -> None:
        super().__init__()
        self.theme = "ansi-dark"
        self.state = state
        self.blocker_messages = blocker_messages
        self.draft = PromptDraft()
        self.phase = ConversationPhase.INITIALIZING
        self.turn_id = 0
        self.active_turn_id: int | None = None
        self.active_response: ResponseSession | None = None
        self.active_llm_entry: ConversationEntry | None = None
        self._cleanup_started = False
        self._console_assembler = ConsoleLineAssembler()
        self._initialization_lines: list[str] = []
        self._initialization_entry: ConversationEntry | None = None
        self._turn_started_at: float | None = None
        self._last_snapshot_key: tuple | None = None
        self._last_snapshot_change_at: float = time.monotonic()
        self._stall_notice_shown = False
        self._interrupt_requested_turn_id: int | None = None
        self.runtime = runtime or runtime_factory(
            state,
            on_transcription=self._post_transcription,
            on_transcription_error=self._post_transcription_error,
        )
        self.header_lines = self.make_header_lines()

    def _set_phase(self, phase: ConversationPhase, note: str = "") -> None:
        if self.phase is phase:
            return
        L.i(f"[chat] phase {self.phase.value} -> {phase.value}" + (f" ({note})" if note else ""))
        self.phase = phase

    @property
    def transcript(self) -> ConversationTranscript:
        return self.query_one("#conversation-transcript", ConversationTranscript)

    @property
    def input_area(self) -> ConversationInputArea:
        return self.query_one("#conversation-input-area", ConversationInputArea)

    def make_header_lines(self) -> list[str]:
        """Build the app's fixed instruction header for the active input mode."""
        title = "LLM voice chat (echo only)" if getattr(
            self.runtime, "echo_override", False
        ) else "LLM voice chat"
        match self.runtime.input_mode:
            case ChatInputMode.TEXT:
                return [
                    f"{COL_ACCENT}{title}",
                    f"{COL_DIM}- Press [ESC] to interrupt/close",
                ]
            case ChatInputMode.MIC_IMMEDIATE:
                return [
                    f"{COL_ACCENT}{title}",
                    f"{COL_DIM}- Press [ESC] to interrupt/close",
                ]
            case ChatInputMode.MIC_ENTER:
                return [
                    f"{COL_ACCENT}{title}",
                    f"{COL_DIM}Speak into the microphone to build your prompt",
                    f"{COL_DIM}- Press [{COL_ACCENT}ENTER{COL_DIM}] to submit  - [LEFT/RIGHT/DEL] Edit transcribed chunks",
                    f"{COL_DIM}- Press [ESC] to interrupt/close",
                ]

    def compose(self) -> ComposeResult:
        header = Vertical(
            *(
                Static(
                    Text.from_ansi(f"{Ansi.RESET}{line}"),
                    id=f"header-line-{index}",
                    classes="header-line",
                    markup=False,
                )
                for index, line in enumerate(self.header_lines)
            ),
            id="header",
        )
        header.styles.height = len(self.header_lines)
        yield header
        yield Rule(id="header-divider")
        yield ConversationTranscript()
        yield ConversationInputDivider()
        # "Microphone (submit by pressing ENTER)" encloses the selected
        # draft phrase in dim brackets at the default color; the immediate
        # mic mode keeps the accent-italic highlight instead.
        yield ConversationInputArea(
            self.draft,
            selection_brackets=self.runtime.is_microphone_input
            and not self.runtime.is_immediate_input,
            microphone_input=self.runtime.is_microphone_input,
            immediate_input=self.runtime.is_immediate_input,
        )

    def on_mount(self) -> None:
        self.input_area.show_disabled()
        if getattr(self.runtime, "echo_override", False):
            self.transcript.append_entry(
                ConversationRole.SYSTEM,
                f'{COL_DIM_ITALICS}Currently in "echo mode": Input is simply echoed, skipping LLM',
                parse_ansi=True,
                gap=True,
            )
        self.set_interval(0.05, self._refresh_dynamic_ui)
        self.call_after_refresh(self._start_initialization)

    def _start_initialization(self) -> None:
        if self.phase != ConversationPhase.INITIALIZING:
            return
        self.run_worker(
            self._initialize_runtime,
            thread=True,
            name="conversation-initialization",
            exclusive=True,
        )

    def _initialize_runtime(self) -> None:
        session_id = self.runtime.session_id
        if self.blocker_messages:
            self.post_message(
                InitializationFinished(
                    session_id, None, "\n".join(self.blocker_messages)
                )
            )
            return
        try:

            def handle_console(event: ConsoleOutput | ConsoleFlush) -> None:
                self.post_message(WorkerConsoleMessage(session_id, event))

            result = self.runtime.initialize(console_handler=handle_console)
        except Exception as exception:
            self.post_message(
                InitializationFinished(
                    session_id,
                    None,
                    f"{type(exception).__name__}: {exception}",
                )
            )
        else:
            self.post_message(InitializationFinished(session_id, result, ""))

    def _update_initialization_content(self, content: str) -> bool:
        """Show the worker's initialization console output in the transcript.

        Mirrors the generation session: the app itself never adds an
        "Initializing..." placeholder. The worker only emits lines for work it
        actually performs (e.g. print_init's dim-italic "Warming up models..."
        and "Initializing <model> model (...)" lines), and this entry relays
        them with their original ANSI styling, exactly as the generation
        worker log does. When the models are already warm the worker prints
        nothing and no entry appears.
        """
        if not content:
            return False
        entry = self._initialization_entry
        if entry is None:
            entry = self.transcript.append_entry(
                ConversationRole.SYSTEM,
                content,
                parse_ansi=True,
                entry_id="initialization-output",
            )
            self._initialization_entry = entry
            return True
        if entry.raw_content == content:
            return False
        entry.set_content(content, parse_ansi=True)
        return True

    def on_worker_console_message(self, message: WorkerConsoleMessage) -> None:
        if (
            message.session_id != self.runtime.session_id
            or self.phase != ConversationPhase.INITIALIZING
        ):
            return
        event = message.event
        if isinstance(event, ConsoleFlush):
            return
        complete, live = self._console_assembler.feed(event.text)
        self._initialization_lines.extend(complete)
        if self._update_initialization_content(
            "\n".join((*self._initialization_lines, live))
        ):
            self.transcript._follow_after_update()

    def on_initialization_finished(self, message: InitializationFinished) -> None:
        if (
            message.session_id != self.runtime.session_id
            or self.phase != ConversationPhase.INITIALIZING
        ):
            return
        self.query_one("#input-divider", ConversationInputDivider).show_stroke()
        self._initialization_lines.extend(self._console_assembler.finish())
        if self._update_initialization_content("\n".join(self._initialization_lines)):
            self.transcript._follow_after_update()
        if message.error:
            self._fail(message.error)
            return
        assert message.result is not None
        for warning in message.result.warnings:
            self.transcript.append_entry(ConversationRole.SYSTEM, warning)
        if message.result.input_device:
            self.transcript.append_entry(
                ConversationRole.SYSTEM,
                Text.from_ansi(
                    f"{COL_DIM_ITALICS}Using sound input device:\n  {message.result.input_device}\n"
                ),
            )
        self.transcript.append_entry(
            ConversationRole.SYSTEM,
            Text("Ready", style=f"italic {STYLE_DIM}"),
            gap=True,
        )
        self._set_phase(ConversationPhase.AWAITING_INPUT, "initialization complete")
        self._show_input()

    def _cancel_initialization(self) -> None:
        try:
            reset_error = self.runtime.cancel_initialization()
        except Exception as exception:
            reset_error = f"{type(exception).__name__}: {exception}"
        self.post_message(InitializationCancelled(reset_error))

    def on_initialization_cancelled(self, message: InitializationCancelled) -> None:
        if self.phase != ConversationPhase.CANCELLING_INITIALIZATION:
            return
        content = f"\n{COL_DEFAULT}{Ansi.ITALICS}Reset model/s. Press a key."
        self.transcript.append_entry(
            ConversationRole.SYSTEM,
            content,
            parse_ansi=True,
        )
        if message.reset_error:
            self.transcript.append_entry(
                ConversationRole.SYSTEM,
                message.reset_error,
                error=True,
            )
        self._set_phase(
            ConversationPhase.INITIALIZATION_CANCELLED,
            "initialization reset complete",
        )
        self.input_area.show_disabled()
        self.screen.set_focus(None)

    def _fail(self, error: str) -> None:
        self.transcript.append_entry(ConversationRole.SYSTEM, error, error=True)
        self._set_phase(ConversationPhase.FAILED, error[:120])
        self.input_area.show_disabled()

    def _show_input(self) -> None:
        self.input_area.clear()
        if self.runtime.is_text_input:
            self.input_area.show_text()
        else:
            self.input_area.show_microphone()

    def on_conversation_text_editor_submitted(
        self, message: ConversationTextEditor.Submitted
    ) -> None:
        self._submit(PromptSubmission(message.value))

    def _submit(self, submission: PromptSubmission) -> None:
        if (
            self.phase != ConversationPhase.AWAITING_INPUT
            or not submission.text.strip()
        ):
            return
        self.turn_id += 1
        turn_id = self.turn_id
        self.active_turn_id = turn_id
        self._interrupt_requested_turn_id = None
        L.i(
            f"[chat] turn {turn_id} submitted:"
            f" mode={self.runtime.input_mode.id} chars={len(submission.text)}"
            f" audio={'yes' if submission.audio is not None else 'no'}"
        )
        self.transcript.attach_tail()
        echo_override = getattr(self.runtime, "echo_override", False)
        if not echo_override:
            self.transcript.append_entry(
                ConversationRole.YOU, submission.text, gap=True
            )
        response_role = (
            ConversationRole.ECHO if echo_override else ConversationRole.LLM
        )
        self.active_llm_entry = self.transcript.append_entry(
            response_role, "...", entry_id=f"llm-turn-{turn_id}", gap=True
        )
        self.input_area.clear()
        self.input_area.show_disabled()
        self._set_phase(ConversationPhase.RESPONDING, f"turn {turn_id}")
        self._turn_started_at = time.monotonic()
        self._last_snapshot_change_at = time.monotonic()
        self._last_snapshot_key = None
        self._stall_notice_shown = False
        try:

            def handle_response_error(error: str) -> None:
                self.post_message(
                    RuntimeFailureMessage(self.runtime.session_id, turn_id, error)
                )

            response = self.runtime.create_response(on_error=handle_response_error)
        except Exception as exception:
            self._fail(f"{type(exception).__name__}: {exception}")
            return
        self.active_response = response
        self.run_worker(
            lambda: self._run_response(turn_id, response, submission),
            thread=True,
            name=f"conversation-response-{turn_id}",
            exclusive=True,
        )

    def _run_response(
        self,
        turn_id: int,
        response: ResponseSession,
        submission: PromptSubmission,
    ) -> None:
        try:
            result = self.runtime.run_response(response, submission)
        except Exception as exception:
            result = ResponseResult(
                text=response.snapshot().text,
                interrupted=response.snapshot().interrupted,
                error=f"{type(exception).__name__}: {exception}",
            )
        self.post_message(
            ResponseFinishedMessage(self.runtime.session_id, turn_id, result)
        )

    def _refresh_dynamic_ui(self) -> None:
        self._refresh_active_response()
        self._refresh_input_level()

    def _refresh_input_level(self) -> None:
        if (
            self.phase != ConversationPhase.AWAITING_INPUT
            or not self.runtime.is_microphone_input
        ):
            return
        self.input_area.set_input_level(self.runtime.input_level_db)

    def _refresh_active_response(self) -> None:
        response = self.active_response
        entry = self.active_llm_entry
        if (
            self.phase != ConversationPhase.RESPONDING
            or response is None
            or entry is None
        ):
            return
        snapshot = response.snapshot()
        snapshot_key = (
            snapshot.spoken_segments,
            snapshot.pending_sentences,
            snapshot.render_buffer,
            snapshot.playback_done,
            snapshot.llm_content_received,
        )
        now = time.monotonic()
        if snapshot_key != self._last_snapshot_key:
            self._last_snapshot_key = snapshot_key
            self._last_snapshot_change_at = now
        elif (
            not self._stall_notice_shown
            and now - self._last_snapshot_change_at > 60.0
            and not snapshot.llm_content_received
        ):
            self._stall_notice_shown = True
            L.w(
                f"[chat] turn {self.active_turn_id}: no LLM content for "
                f"{now - self._last_snapshot_change_at:.0f}s - still waiting"
            )
            self.transcript.append_entry(
                ConversationRole.SYSTEM,
                "Still waiting for the LLM response... Esc to interrupt.",
            )
        self.transcript.update_entry(entry, self._render_snapshot(snapshot))

    @staticmethod
    def _render_snapshot(snapshot: ResponseSnapshot) -> Text:
        result = Text()
        position = snapshot.play_position_samples
        active_index = (
            None
            if snapshot.playback_done
            else next(
                (
                    index
                    for index, (_text, start, end) in enumerate(
                        snapshot.spoken_segments
                    )
                    if start <= position < end
                ),
                None,
            )
        )
        for index, (text, _start, end) in enumerate(snapshot.spoken_segments):
            style = STYLE_DEFAULT if index == active_index else STYLE_DIM
            if end <= position or snapshot.playback_done:
                style = STYLE_DIM
            result.append(text, style=style)
        for index, text in enumerate(snapshot.pending_sentences):
            style = f"bold italic {STYLE_DIM}" if index == 0 else STYLE_DIM
            result.append(text, style=style)
        if snapshot.render_buffer:
            result.append(snapshot.render_buffer, style=STYLE_DIM)
        return result

    def on_response_finished_message(self, message: ResponseFinishedMessage) -> None:
        if (
            message.session_id != self.runtime.session_id
            or message.turn_id != self.active_turn_id
            or self.phase
            not in (ConversationPhase.RESPONDING, ConversationPhase.EXITING)
        ):
            return
        entry = self.active_llm_entry
        response = self.active_response
        if entry is not None and response is not None:
            snapshot = response.snapshot()
            if snapshot.llm_content_received or message.result.text:
                # result.text is a TTS-oriented flattening (parts joined with
                # a single space), which loses the line breaks of the streamed
                # text; snapshot.text reconstructs exactly what the realtime
                # renderer displayed.
                final_text = snapshot.text
                final_content = (
                    Text(final_text, style=STYLE_DIM)
                    if final_text
                    else self._render_snapshot(snapshot)
                )
                entry.set_content(final_content)
            else:
                self.transcript.remove_entry(entry)
        if message.result.error:
            self.transcript.append_entry(
                ConversationRole.SYSTEM, message.result.error, error=True
            )
        self.active_response = None
        self.active_llm_entry = None
        self.active_turn_id = None
        self._interrupt_requested_turn_id = None
        if self.phase == ConversationPhase.EXITING:
            return
        L.i(
            f"[chat] turn {message.turn_id} finished:"
            f" interrupted={message.result.interrupted}"
            f" error={message.result.error[:120]!r}"
            f" chars={len(message.result.text)}"
        )
        self._set_phase(ConversationPhase.AWAITING_INPUT, f"turn {message.turn_id} done")
        self._show_input()

    def _post_transcription(
        self, session_id: int, segments: list[Segment], audio: object | None
    ) -> None:
        self.post_message(TranscriptionMessage(session_id, segments, audio))

    def _post_transcription_error(self, session_id: int, error: str) -> None:
        self.post_message(RuntimeFailureMessage(session_id, None, error))

    def on_transcription_message(self, message: TranscriptionMessage) -> None:
        if (
            message.session_id != self.runtime.session_id
            or self.phase != ConversationPhase.AWAITING_INPUT
            or not self.runtime.is_microphone_input
        ):
            return
        audio = cast(np.ndarray | None, message.audio)
        if not self.draft.add_transcription(message.segments, audio):
            return
        self.input_area.mic_view.refresh_draft()
        if self.runtime.is_immediate_input:
            submission = self.draft.finalize()
            if submission is not None:
                self._submit(submission)

    def on_runtime_failure_message(self, message: RuntimeFailureMessage) -> None:
        if message.session_id != self.runtime.session_id:
            return
        if message.turn_id is not None and message.turn_id != self.active_turn_id:
            return
        self.transcript.append_entry(ConversationRole.SYSTEM, message.error, error=True)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        # Once cancellation has finished, bound keys must fall through to
        # on_key so every key—not only unbound character keys—exits the app.
        if self.phase == ConversationPhase.INITIALIZATION_CANCELLED:
            return False
        return True

    def on_key(self, event: events.Key) -> None:
        if self.phase != ConversationPhase.INITIALIZATION_CANCELLED:
            return
        event.stop()
        event.prevent_default()
        self.exit(ConversationAppResult(ConversationAppStatus.COMPLETED))

    def action_cancel_initialization(self) -> None:
        if self.phase != ConversationPhase.INITIALIZING:
            return
        L.i("[chat] ctrl-c: cancelling initialization and resetting worker")
        self._set_phase(
            ConversationPhase.CANCELLING_INITIALIZATION,
            "user cancelled initialization",
        )
        self.transcript.append_entry(
            ConversationRole.SYSTEM,
            f"{COL_DIM_ITALICS}\nCancelling...",
            parse_ansi=True,
        )
        self.input_area.show_disabled()
        self.run_worker(
            self._cancel_initialization,
            thread=True,
            name="conversation-initialization-cancel",
            exclusive=False,
        )

    def action_draft_left(self) -> None:
        if (
            self.phase == ConversationPhase.AWAITING_INPUT
            and self.runtime.is_microphone_input
        ):
            self.draft.select_previous()
            self.input_area.mic_view.refresh_draft()

    def action_draft_right(self) -> None:
        if (
            self.phase == ConversationPhase.AWAITING_INPUT
            and self.runtime.is_microphone_input
        ):
            self.draft.select_next()
            self.input_area.mic_view.refresh_draft()

    def action_draft_delete(self) -> None:
        if (
            self.phase == ConversationPhase.AWAITING_INPUT
            and self.runtime.is_microphone_input
        ):
            self.draft.delete_selected()
            self.input_area.mic_view.refresh_draft()

    def action_enter(self) -> None:
        if self.phase == ConversationPhase.FAILED:
            self._begin_exit()
        elif (
            self.phase == ConversationPhase.AWAITING_INPUT
            and self.runtime.is_microphone_input
        ):
            submission = self.draft.finalize()
            if submission is not None:
                self._submit(submission)

    def action_escape(self) -> None:
        if self.phase == ConversationPhase.INITIALIZING:
            return
        if self.phase == ConversationPhase.AWAITING_INPUT:
            L.i("[chat] esc: exiting session")
            self._begin_exit()
        elif self.phase == ConversationPhase.RESPONDING:
            turn_id = self.active_turn_id
            if turn_id is None or self._interrupt_requested_turn_id == turn_id:
                return
            L.i(f"[chat] esc: interrupting turn {turn_id}")
            self._interrupt_requested_turn_id = turn_id
            self.input_area.show_waiting()
            # Closing an HTTP response can block until the provider releases
            # its stream. Keep that work off Textual's event loop so the
            # waiting message is painted immediately.
            self.run_worker(
                lambda: self.runtime.request_interrupt(),
                thread=True,
                name=f"conversation-interrupt-{turn_id}",
                exclusive=False,
            )

    def action_ignore(self) -> None:
        pass

    def _begin_exit(self) -> None:
        if self._cleanup_started:
            return
        self._cleanup_started = True
        self._set_phase(ConversationPhase.EXITING, "user exit")
        self.input_area.show_disabled()
        self.runtime.request_interrupt()
        self.run_worker(self._close_runtime, thread=True, name="conversation-cleanup")

    def _close_runtime(self) -> None:
        self.runtime.close()
        self.post_message(RuntimeClosedMessage())

    def on_runtime_closed_message(self, _message: RuntimeClosedMessage) -> None:
        self.exit(ConversationAppResult(ConversationAppStatus.COMPLETED))

    def on_unmount(self) -> None:
        # A stylesheet/runtime exception may bypass the normal binding path.
        self.runtime.close()


def run_conversation_app(state: State) -> ConversationAppResult:
    """Run chat in Textual, with no raw-terminal fallback."""
    if not can_textual():
        message = "Full-screen Textual chat is unavailable in this terminal."
        ask.ask_error(message)
        return ConversationAppResult(ConversationAppStatus.UNAVAILABLE, message)

    blockers = tuple(item.verbose for item in readiness.get_chat_blockers(state))
    if not blockers and not app_hint_util.show_pre_inference_hints(
        state.prefs, state.project
    ):
        return ConversationAppResult(ConversationAppStatus.DECLINED)

    app = ConversationTextualApp(state, blocker_messages=blockers)
    try:
        result = app.run(inline=False)
    except Exception as exception:
        message = f"{type(exception).__name__}: {exception}"
        ask.ask_error(message)
        app.runtime.close()
        return ConversationAppResult(ConversationAppStatus.FAILED, message)

    exception = app._exception
    if isinstance(exception, StylesheetError):
        message = "Couldn't load textual css"
        ask.ask_error(message)
        app.runtime.close()
        return ConversationAppResult(ConversationAppStatus.FAILED, message)
    if exception is not None:
        message = f"{type(exception).__name__}: {exception}"
        ask.ask_error(message)
        app.runtime.close()
        return ConversationAppResult(ConversationAppStatus.FAILED, message)
    if result is None:
        message = "Conversation closed without returning a result"
        ask.ask_error(message)
        app.runtime.close()
        return ConversationAppResult(ConversationAppStatus.FAILED, message)
    return result


__all__ = [
    "ConversationAppResult",
    "ConversationAppStatus",
    "ConversationPhase",
    "ConversationTextualApp",
    "run_conversation_app",
]
