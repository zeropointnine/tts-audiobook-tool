import asyncio
import threading
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual.widgets import Static

from tts_audiobook_tool.conversation.conversation import ConversationInitialization
from tts_audiobook_tool.conversation.conversation_types import (
    ChatInputMode,
    ResponseResult,
    ResponseSnapshot,
)
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM_ITALICS
from tts_audiobook_tool.conversation.prompt_draft import PromptSubmission
from tts_audiobook_tool.model_worker_protocol import ConsoleOutput
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.textual.conversation_app import (
    ConversationAppResult,
    ConversationAppStatus,
    ConversationInputDivider,
    ConversationPhase,
    ConversationTextualApp,
)
from tts_audiobook_tool.textual.conversation_widgets import ConversationRole
from tts_audiobook_tool.textual.textual_shared import STYLE_DIM


def run(coroutine):
    return asyncio.run(coroutine)


class FakeResponse:
    def __init__(self) -> None:
        self.value = ResponseSnapshot()

    def snapshot(self) -> ResponseSnapshot:
        return self.value

    def request_interrupt(self) -> None:
        self.value = ResponseSnapshot(
            render_buffer="partial reply",
            llm_content_received=True,
            interrupted=True,
        )


class FakeRuntime:
    session_id = 17
    input_mode = ChatInputMode.TEXT
    is_text_input = True
    is_microphone_input = False
    is_immediate_input = False
    input_level_db: float | None = None

    def __init__(self) -> None:
        self.response = FakeResponse()
        self.release = threading.Event()
        self.closed = 0
        self.interrupts = 0
        self.submissions: list[PromptSubmission] = []

    def initialize(self, console_handler=None) -> ConversationInitialization:
        return ConversationInitialization(("startup warning",), "")

    def create_response(self, on_error=None) -> FakeResponse:
        return self.response

    def run_response(
        self, response: FakeResponse, submission: PromptSubmission
    ) -> ResponseResult:
        self.submissions.append(submission)
        response.value = ResponseSnapshot(
            render_buffer="partial reply", llm_content_received=True
        )
        self.release.wait(2)
        return ResponseResult("partial reply", interrupted=response.value.interrupted)

    def request_interrupt(self) -> None:
        self.interrupts += 1
        self.response.request_interrupt()
        self.release.set()

    def close(self) -> None:
        self.closed += 1
        self.release.set()


def make_state():
    return SimpleNamespace()


def test_app_initializes_after_mount_and_has_fixed_shell() -> None:
    async def exercise() -> None:
        runtime = FakeRuntime()
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.phase == ConversationPhase.AWAITING_INPUT
            assert app.query_one("#header").size.height == 2
            assert app.query_one("#conversation-input-prefix").display
            assert not app.input_area.level_meter.display
            title = app.query_one("#header-line-0", Static).content
            assert isinstance(title, Text)
            assert title.plain == "LLM voice chat"
            assert str(title.spans[0].style) == "#ffaa44"
            exit_line = app.query_one("#header-line-1", Static).content
            assert isinstance(exit_line, Text)
            assert exit_line.plain == "- Press [ESC] to interrupt/close"
            assert str(exit_line.spans[0].style) == "#888888"
            assert app.query_one("#conversation-input-area").size.height == 3
            # The worker emitted no initialization console output, so there
            # is no initialization entry at all - only the warning and
            # "Ready" entries appear (no "Initializing..." placeholder).
            contents = [entry.raw_content for entry in app.transcript.entries]
            assert contents == ["startup warning", "Ready"]
            ready = next(
                entry for entry in app.transcript.entries if entry.raw_content == "Ready"
            )
            assert ready.has_class("entry-gap")
            assert str(ready.content.spans[0].style) == f"italic {STYLE_DIM}"

    run(exercise())


def test_echo_mode_header_initial_notice_and_response_label() -> None:
    async def exercise() -> None:
        runtime = FakeRuntime()
        runtime.echo_override = True
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            title = app.query_one("#header-line-0", Static).content
            assert isinstance(title, Text)
            assert title.plain == "LLM voice chat (echo only)"
            notice_entry = app.transcript.entries[0]
            notice = notice_entry.content
            assert isinstance(notice, Text)
            assert notice.plain == (
                'Currently in "echo mode": Input is simply echoed, skipping LLM'
            )
            assert notice_entry.has_class("entry-gap")

            app._submit(PromptSubmission("repeat me"))
            await pilot.pause()
            entry = app.active_llm_entry
            assert entry is not None
            assert entry.role is ConversationRole.ECHO
            assert all(
                transcript_entry.role is not ConversationRole.YOU
                for transcript_entry in app.transcript.entries
            )
            rendered = entry.content
            assert isinstance(rendered, Text)
            assert rendered.plain.startswith("Echo: ")
            runtime.release.set()

    run(exercise())


def make_mode_runtime(text_input: bool, immediate_input: bool) -> FakeRuntime:
    runtime = FakeRuntime()
    runtime.input_mode = (
        ChatInputMode.TEXT
        if text_input
        else ChatInputMode.MIC_IMMEDIATE
        if immediate_input
        else ChatInputMode.MIC_ENTER
    )
    runtime.is_text_input = text_input
    runtime.is_immediate_input = immediate_input
    runtime.is_microphone_input = not text_input
    runtime.input_level_db = None if text_input else -5.0
    return runtime


def test_microphone_immediate_header_matches_content_app_conventions() -> None:
    async def exercise() -> None:
        app = ConversationTextualApp(
            make_state(), runtime=make_mode_runtime(False, True)
        )  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            await pilot.pause(0.06)
            assert app.query_one("#header").size.height == 2
            assert app.input_area.size.height == 1
            assert not app.query_one("#conversation-input-prefix").display
            assert app.input_area.level_meter.display
            assert app.input_area.level_meter.content == "████████  -5 dB"
            title = app.query_one("#header-line-0", Static).content
            assert isinstance(title, Text)
            assert title.plain == "LLM voice chat"
            assert str(title.spans[0].style) == "#ffaa44"
            exit_line = app.query_one("#header-line-1", Static).content
            assert isinstance(exit_line, Text)
            assert exit_line.plain == "- Press [ESC] to interrupt/close"
            assert str(exit_line.spans[0].style) == "#888888"

    run(exercise())


def test_mic_phrase_header_matches_content_app_conventions() -> None:
    async def exercise() -> None:
        app = ConversationTextualApp(
            make_state(), runtime=make_mode_runtime(False, False)
        )  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            await pilot.pause(0.06)
            assert app.query_one("#header").size.height == 4
            assert app.input_area.size.height == 3
            assert not app.query_one("#conversation-input-prefix").display
            assert app.input_area.level_meter.display
            assert app.input_area.level_meter.content == "████████  -5 dB"
            lines = [
                app.query_one(f"#header-line-{index}", Static).content
                for index in range(4)
            ]
            assert all(isinstance(line, Text) for line in lines)
            assert [line.plain for line in lines] == [
                "LLM voice chat",
                "Speak into the microphone to build your prompt",
                "- Press [ENTER] to submit  - [LEFT/RIGHT/DEL] Edit transcribed chunks",
                "- Press [ESC] to interrupt/close",
            ]
            assert str(lines[0].spans[0].style) == "#ffaa44"
            assert all(str(line.spans[0].style) == "#888888" for line in lines[1:])
            assert any(
                span.start == 9
                and span.end == 14
                and str(span.style) == "#ffaa44"
                for span in lines[2].spans
            )

    run(exercise())


def _draft_content(app) -> Text:
    app.draft.add_phrase("first")
    app.draft.add_phrase("second")
    app.input_area.mic_view.refresh_draft()
    content = app.input_area.mic_view.content
    assert isinstance(content, Text)
    return content


def _style_at(text: Text, position: int):
    from rich.style import Style

    for start, end, style in text.spans:
        if start <= position < end:
            return Style.parse(style) if isinstance(style, str) else style
    return None


def test_mic_enter_selection_is_default_enclosed_by_dim_brackets() -> None:
    from rich.color import Color

    async def exercise() -> None:
        # "Microphone (submit by pressing ENTER)": the selected phrase keeps
        # the default color, enclosed by dim square brackets.
        app = ConversationTextualApp(
            make_state(), runtime=make_mode_runtime(False, False)
        )  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            content = _draft_content(app)
            await pilot.pause()
            # The highlight was on the last chunk, so it followed "second".
            assert content.plain == "first [second]"
            selected = _style_at(content, content.plain.find("second"))
            assert selected is not None
            assert selected.color == Color.default()
            assert not selected.italic
            for probe in ("first", "[", "]"):
                style = _style_at(content, content.plain.find(probe))
                assert style is not None, probe
                assert style.color == Color.parse("#888888"), probe
                assert not style.italic, probe

    run(exercise())


def test_mic_immediate_selection_keeps_accent_italic() -> None:
    from rich.color import Color

    async def exercise() -> None:
        app = ConversationTextualApp(
            make_state(), runtime=make_mode_runtime(False, True)
        )  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            content = _draft_content(app)
            await pilot.pause()
            assert content.plain == "first second"
            # The highlight was on the last chunk, so it followed "second".
            selected = _style_at(content, content.plain.find("second"))
            assert selected is not None
            assert selected.italic
            assert selected.color == Color.parse("#ffaa44")
            other = _style_at(content, content.plain.find("first"))
            assert other is not None
            assert not other.italic
            assert other.color == Color.parse("#888888")

    run(exercise())


def test_microphone_meter_is_hidden_during_llm_turn() -> None:
    async def exercise() -> None:
        runtime = make_mode_runtime(False, False)
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            await pilot.pause(0.06)
            assert app.input_area.level_meter.display
            assert app.input_area.level_meter.content == "████████  -5 dB"

            app._submit(PromptSubmission("hello"))
            await pilot.pause()
            assert app.phase == ConversationPhase.RESPONDING
            assert not app.input_area.level_meter.display
            assert app.input_area.level_meter.content == ""

            await pilot.press("escape")
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            await pilot.pause(0.06)
            assert app.phase == ConversationPhase.AWAITING_INPUT
            assert app.input_area.level_meter.display
            assert app.input_area.level_meter.content == "████████  -5 dB"

    run(exercise())


class FakeInitConsoleRuntime:
    session_id = 19
    input_mode = ChatInputMode.TEXT
    is_text_input = True
    is_microphone_input = False
    is_immediate_input = False

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def initialize(self, console_handler=None) -> ConversationInitialization:
        if console_handler is not None:
            # What the worker's print_init emits on a truecolor terminal:
            # dim color + italic, reset, the line, then a trailing blank
            # line. A cold mic-mode chat warm-up emits the banner and, after
            # the (silent) TTS load, the STT model's own init line.
            console_handler(
                ConsoleOutput(
                    "op-init",
                    "stdout",
                    "\033[38;2;136;136;136m\033[3mWarming up models...\033[0m\n\n",
                )
            )
            console_handler(
                ConsoleOutput(
                    "op-init",
                    "stdout",
                    "\033[38;2;136;136;136m\033[3m"
                    "Initializing faster-whisper model "
                    "(large-v3, cpu, int8_float32, 8 threads)...\033[0m\n\n",
                )
            )
        return ConversationInitialization((), "")


def test_initialization_entry_mirrors_worker_console_and_styling() -> None:
    from rich.color import Color
    from rich.style import Style

    async def exercise() -> None:
        runtime = FakeInitConsoleRuntime()
        app = ConversationTextualApp(
            make_state(), runtime=runtime  # type: ignore[arg-type]
        )
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            init_entry = app.query_one("#initialization-output")
            # The worker's dim-italic print_init line is relayed verbatim,
            # original ANSI styling included (not stripped to plain dim).
            assert (
                "\033[38;2;136;136;136m\033[3mWarming up models...\033[0m"
                in init_entry.raw_content
            )
            assert "Initializing faster-whisper model" in init_entry.raw_content
            content = init_entry.content

            def style_at(pos: int):
                for start, end, style in content.spans:
                    if start <= pos < end:
                        return Style.parse(style) if isinstance(style, str) else style
                return None

            for probe in (
                "Warming up models...",
                "Initializing faster-whisper model",
            ):
                style = style_at(content.plain.find(probe))
                assert style is not None, probe
                assert style.italic, probe
                assert style.color == Color.parse("#888888"), probe
            ready = next(
                entry for entry in app.transcript.entries if entry.raw_content == "Ready"
            )
            assert ready.has_class("entry-gap")
            assert not any(
                "Initializing..." in entry.raw_content
                for entry in app.transcript.entries
            )

    run(exercise())


class FakeBlockingInitializationRuntime(FakeRuntime):
    def __init__(self, reset_error: str = "") -> None:
        super().__init__()
        self.reset_error = reset_error
        self.initialization_started = threading.Event()
        self.initialization_can_return = threading.Event()
        self.cancel_started = threading.Event()
        self.cancel_can_return = threading.Event()
        self.cancel_calls = 0

    def initialize(self, console_handler=None) -> ConversationInitialization:
        self.initialization_started.set()
        self.initialization_can_return.wait(2.0)
        return ConversationInitialization((), "")

    def cancel_initialization(self) -> str:
        self.cancel_calls += 1
        self.cancel_started.set()
        self.initialization_can_return.set()
        self.cancel_can_return.wait(2.0)
        return self.reset_error


def test_input_divider_is_blank_during_initialization_then_visible() -> None:
    async def exercise() -> None:
        runtime = FakeBlockingInitializationRuntime()
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if runtime.initialization_started.is_set():
                    break
            assert runtime.initialization_started.is_set()
            divider = app.query_one("#input-divider", ConversationInputDivider)
            assert divider.size.height == 1
            assert divider.visible
            assert not divider.stroke_visible
            blank_row = divider.render()
            assert isinstance(blank_row, Text)
            assert not blank_row.plain.strip()

            runtime.initialization_can_return.set()
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            assert divider.visible
            assert divider.stroke_visible
            assert divider.styles.color.hex == STYLE_DIM

    run(exercise())


@pytest.mark.parametrize(
    ("exit_key", "reset_error"),
    [("x", ""), ("enter", "replacement failed")],
)
def test_ctrl_c_cancels_initialization_then_next_key_exits(
    exit_key: str,
    reset_error: str,
) -> None:
    async def exercise() -> None:
        runtime = FakeBlockingInitializationRuntime(reset_error)
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if runtime.initialization_started.is_set():
                    break
            assert runtime.initialization_started.is_set()
            assert app.phase == ConversationPhase.INITIALIZING

            await pilot.press("ctrl+c")
            await pilot.pause()
            assert runtime.cancel_started.wait(1.0)
            assert app.phase == ConversationPhase.CANCELLING_INITIALIZATION
            assert [entry.raw_content for entry in app.transcript.entries] == [
                f"{COL_DIM_ITALICS}\nCancelling..."
            ]

            # Further CTRL-C presses do not start a second reset while the
            # first reset is still in progress.
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert runtime.cancel_calls == 1

            # The old initialization may finish after cancellation begins;
            # its stale completion must not make the app ready.
            runtime.cancel_can_return.set()
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.INITIALIZATION_CANCELLED:
                    break
            assert app.phase == ConversationPhase.INITIALIZATION_CANCELLED
            assert runtime.cancel_calls == 1
            expected_contents = [
                f"{COL_DIM_ITALICS}\nCancelling...",
                f"\n{COL_DEFAULT}{Ansi.ITALICS}Reset model/s. Press a key.",
            ]
            if reset_error:
                expected_contents.append(reset_error)
            assert [
                entry.raw_content for entry in app.transcript.entries
            ] == expected_contents
            assert app.return_value is None

            # Both unbound characters and normally-bound keys are accepted as
            # the one keypress that exits the cancelled app.
            await pilot.press(exit_key)
            await pilot.pause()
            assert app.return_value == ConversationAppResult(
                ConversationAppStatus.COMPLETED
            )

    run(exercise())


class FakeDelayedInterruptRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.interrupt_started = threading.Event()
        self.interrupt_can_return = threading.Event()

    def request_interrupt(self) -> None:
        self.interrupts += 1
        self.response.request_interrupt()
        self.interrupt_started.set()
        self.interrupt_can_return.wait(2.0)


def test_escape_shows_italic_waiting_message_until_response_finishes() -> None:
    async def exercise() -> None:
        runtime = FakeDelayedInterruptRuntime()
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app._submit(PromptSubmission("hello"))
            await pilot.pause()
            assert app.phase == ConversationPhase.RESPONDING

            await pilot.press("escape")
            await pilot.pause()
            assert runtime.interrupt_started.wait(1.0)
            assert app.phase == ConversationPhase.RESPONDING
            assert not app.input_area.prefix.display
            assert (
                app.input_area.disabled_view.region.x == app.input_area.region.x
            )
            assert app.input_area.has_class("is-waiting")
            waiting = app.input_area.disabled_view.content
            assert isinstance(waiting, Text)
            assert waiting.plain == "Please wait"
            assert str(waiting.style) == "italic"

            runtime.interrupt_can_return.set()
            runtime.release.set()
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            assert not app.input_area.disabled_view.display

    run(exercise())


def test_submission_updates_one_llm_entry_and_escape_interrupts() -> None:
    async def exercise() -> None:
        runtime = FakeRuntime()
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app._submit(PromptSubmission("hello"))
            await pilot.pause()
            assert app.phase == ConversationPhase.RESPONDING
            assert [entry.role for entry in app.transcript.entries][-2:] == [
                ConversationRole.YOU,
                ConversationRole.LLM,
            ]
            you_entry = app.transcript.entries[-2]
            llm_entry = app.active_llm_entry
            assert llm_entry is not None
            assert you_entry.has_class("entry-gap")
            assert llm_entry.has_class("entry-gap")
            await pilot.pause(0.08)
            assert "partial reply" in llm_entry.raw_content

            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert runtime.interrupts == 1
            assert app.phase == ConversationPhase.AWAITING_INPUT
            assert llm_entry in app.transcript.entries
            assert llm_entry.raw_content == "partial reply"

    run(exercise())


def test_escape_exits_immediately_while_awaiting_input() -> None:
    async def exercise() -> None:
        runtime = FakeRuntime()
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.phase == ConversationPhase.AWAITING_INPUT
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.phase == ConversationPhase.AWAITING_INPUT
            assert runtime.closed == 0
            await pilot.press("escape")
            await pilot.pause()
            assert app.phase == ConversationPhase.EXITING
            for _ in range(100):
                await pilot.pause()
                if app.return_value is not None:
                    break
            assert app.return_value == ConversationAppResult(
                ConversationAppStatus.COMPLETED
            )
            assert runtime.closed >= 1

    run(exercise())


def test_ctrl_c_is_ignored_while_responding_and_escape_interrupts() -> None:
    async def exercise() -> None:
        runtime = FakeRuntime()
        app = ConversationTextualApp(make_state(), runtime=runtime)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app._submit(PromptSubmission("hello"))
            await pilot.pause()
            assert app.phase == ConversationPhase.RESPONDING
            await pilot.press("ctrl+c")
            await pilot.pause()
            await pilot.pause()
            assert app.phase == ConversationPhase.RESPONDING
            assert runtime.interrupts == 0
            assert runtime.closed == 0
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert runtime.interrupts == 1
            assert app.phase == ConversationPhase.AWAITING_INPUT

    run(exercise())


class FakeMultilineResponse:
    def __init__(self) -> None:
        self.value = ResponseSnapshot()

    def snapshot(self) -> ResponseSnapshot:
        return self.value

    def request_interrupt(self) -> None:
        pass


class FakeMultilineRuntime:
    session_id = 18
    input_mode = ChatInputMode.TEXT
    is_text_input = True
    is_microphone_input = False
    is_immediate_input = False

    def __init__(self) -> None:
        self.response = FakeMultilineResponse()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def initialize(self, console_handler=None) -> ConversationInitialization:
        return ConversationInitialization((), "")

    def create_response(self, on_error=None) -> FakeMultilineResponse:
        return self.response

    def run_response(
        self, response: FakeMultilineResponse, submission: PromptSubmission
    ) -> ResponseResult:
        response.value = ResponseSnapshot(
            spoken_segments=(
                ("first line\n", 0, 100),
                ("second line\n", 100, 200),
                ("third line", 200, 300),
            ),
            llm_content_received=True,
        )
        return ResponseResult("first line second line third line")


def test_final_entry_preserves_realtime_linebreaks() -> None:
    async def exercise() -> None:
        runtime = FakeMultilineRuntime()
        app = ConversationTextualApp(
            make_state(), runtime=runtime  # type: ignore[arg-type]
        )
        async with app.run_test(size=(60, 18)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.phase == ConversationPhase.AWAITING_INPUT
            app._submit(PromptSubmission("hello"))
            for _ in range(100):
                await pilot.pause()
                if app.phase == ConversationPhase.AWAITING_INPUT:
                    break
            assert app.phase == ConversationPhase.AWAITING_INPUT
            entry = app.query_one("#llm-turn-1")
            assert entry.raw_content == "first line\nsecond line\nthird line"

    run(exercise())


def test_render_snapshot_styles_pending_sentence_dim_italic() -> None:
    from rich.color import Color
    from rich.style import Style

    snapshot = ResponseSnapshot(
        spoken_segments=(("spoken part", 0, 100),),
        pending_sentences=("pending one", "pending two"),
        render_buffer="preview",
        play_position_samples=50,
        llm_content_received=True,
    )
    text = ConversationTextualApp._render_snapshot(snapshot)

    def style_at(position: int):
        for start, end, style in text.spans:
            if start <= position < end:
                return Style.parse(style) if isinstance(style, str) else style
        return None

    active_style = style_at(text.plain.find("spoken part"))
    assert active_style is not None
    assert active_style.color == Color.default()
    first_pending_style = style_at(text.plain.find("pending one"))
    assert first_pending_style is not None
    assert first_pending_style.italic
    assert first_pending_style.color == Color.parse("#888888")
    for marker in ("pending two", "preview"):
        other_style = style_at(text.plain.find(marker))
        assert other_style is not None
        assert not other_style.italic
        assert other_style.color == Color.parse("#888888")


def test_blockers_fail_inside_textual_and_enter_returns() -> None:
    async def exercise() -> None:
        runtime = FakeRuntime()
        app = ConversationTextualApp(
            make_state(),
            runtime=runtime,  # type: ignore[arg-type]
            blocker_messages=("missing model",),
        )
        async with app.run_test(size=(60, 18)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.phase == ConversationPhase.FAILED
            assert any(
                entry.error and "missing model" in entry.raw_content
                for entry in app.transcript.entries
            )
            await pilot.press("enter")
            await pilot.pause()

        assert runtime.closed >= 1

    run(exercise())
