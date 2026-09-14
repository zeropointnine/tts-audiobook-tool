import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import has_selection
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import DummyHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style


_SELECTION_STYLE = Style.from_dict({"selected": "reverse"})
_ESCAPE_SEQUENCE_TIMEOUT_SECONDS = 0.05
_HISTORY_KEYS = (
    "up",
    "down",
    "pageup",
    "pagedown",
    "c-p",
    "c-n",
    "c-r",
    "c-s",
)


class AskAdvanced:
    @staticmethod
    def ask(message: str = "", prefill: str = "") -> str:
        """Behave like input(), with editable, initially selected prefill text."""
        if not isinstance(message, str):
            raise TypeError("prompt must be a string")

        if not isinstance(prefill, str):
            raise TypeError("prefilled_input must be a string")

        try:
            if any(
                character < " " or character == "\x7f"
                for character in prefill
            ):
                # Cannot safely display embedded terminal controls in prefill text.
                return input(message)

            if not (_is_tty(sys.stdin) and _is_tty(sys.stdout)):
                return input(message)

            return _ask_with_prompt_toolkit(message, prefill)
        except KeyboardInterrupt:
            # Cancel only the active text input; Ctrl-C outside AskAdvanced
            # retains its normal interrupt behavior. Escape uses this path too.
            sys.stdout.write("\n")
            sys.stdout.flush()
            return ""


def _is_tty(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return False


def _create_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    def ignore_history_key(event: KeyPressEvent) -> None:
        """Keep history navigation/search keys inert for input()-style prompts."""

    for key in _HISTORY_KEYS:
        bindings.add(key)(ignore_history_key)

    @bindings.add(Keys.BracketedPaste, filter=has_selection)
    def replace_selection_with_paste(event: KeyPressEvent) -> None:
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        event.current_buffer.cut_selection()
        event.current_buffer.insert_text(data)

    @bindings.add("escape")
    @bindings.add("c-c")
    @bindings.add(Keys.SIGINT)
    def cancel(event: KeyPressEvent) -> None:
        event.current_buffer.text = ""
        # Return normally instead of raising through prompt-toolkit's event
        # loop. This avoids invoking its asynchronous exception renderer while
        # the loop is shutting down (which can emit an un-awaited coroutine
        # warning on Python 3.11).
        event.app.exit(result="", style="class:aborting")

    return bindings


def _ask_with_prompt_toolkit(
    prompt: str,
    prefilled_input: str,
    *,
    prompt_input: Input | None = None,
    prompt_output: Output | None = None,
) -> str:
    """Read one prompt-toolkit line with selected prefilled text."""
    session = PromptSession[str](
        ANSI(prompt),
        editing_mode=EditingMode.EMACS,
        multiline=False,
        complete_while_typing=False,
        validate_while_typing=False,
        enable_history_search=False,
        enable_system_prompt=False,
        enable_suspend=False,
        enable_open_in_editor=False,
        completer=None,
        auto_suggest=None,
        reserve_space_for_menu=0,
        mouse_support=False,
        history=DummyHistory(),
        key_bindings=_create_key_bindings(),
        style=_SELECTION_STYLE,
        input=prompt_input,
        output=prompt_output,
    )
    session.app.ttimeoutlen = _ESCAPE_SEQUENCE_TIMEOUT_SECONDS

    def select_prefill() -> None:
        if not prefilled_input:
            return

        buffer = get_app().current_buffer
        buffer.cursor_position = 0
        buffer.start_selection()
        selection = buffer.selection_state
        if selection is None:
            return
        selection.enter_shift_mode()
        buffer.cursor_position = len(buffer.text)

    return session.prompt(
        default=prefilled_input,
        pre_run=select_prefill,
        # A one-line library prompt should not replace the host process's
        # asyncio exception handler. prompt-toolkit's handler can itself try
        # to schedule a coroutine after loop shutdown on Python 3.11.
        set_exception_handler=False,
    )
