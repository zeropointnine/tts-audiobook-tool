"""
User-input helpers for the interactive application.

This module centralizes the app's standard console/file-dialog prompting
behavior, including line input, hotkey input, path selection, and simple
save-back helpers for project/prefs values.
"""

import os
import re
import sys
from typing import Callable
from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_types import Saveable
from tts_audiobook_tool.ask_advanced import AskAdvanced
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.util import *


# Whether single-key hotkey input is currently available.
# This starts from the TTY capability check, and can later be downgraded
# to False if a low-level hotkey read fails unexpectedly.
can_hotkey = sys.stdin.isatty()

# Common ANSI/VT terminal escape sequences produced by cursor/function keys.
terminal_escape_sequence_pattern = re.compile(
    r"\x1b(?:"
    r"[@-Z\\-_]"
    r"|\[[0-?]*[ -/]*[@-~]"
    r"|\][^\x1b\x07]*(?:\x07|\x1b\\)"
    r"|[PX^_].*?(?:\x1b\\)"
    r")"
)


def ask_input(message: str="", lower: bool=True, extra_line: bool=True, prefill: str="") -> str:
    """
    App-standard way of getting user text input.
    Should behave like a drop-in replacement for `input()`.
    Prints extra line after the input by default.
    """

    if not DEV:
        _clear_input_buffer()

    message = f"{message}{Ansi.RESET}{COL_INPUT}"

    prefill = prefill.strip()
    if lower:
        prefill = prefill.lower()

    try:
        inp = AskAdvanced.ask(message, prefill)

    except Exception as e:
        # If running env cannot handle app's standard input mechanism, this is in the almost-fatal category
        # Might as well print out error in-place
        printt(COL_ERROR + make_error_string(e))
        return ""

    inp = _strip_terminal_escape_sequences(inp)
    inp = inp.strip()
    if lower:
        inp = inp.lower()

    print(Ansi.RESET, end="")
    if extra_line:
        printt()

    return inp

def ask_multiline() -> str:
    """ App standard way of getting multi-line string input from user. """
    return sys.stdin.read().strip()

# ---

def ask_hotkey(message: str="", lower: bool=True) -> str:
    """
    Blocks until a single hotkey press is read.
    Falls back to vanilla input()-based input if necessary.
    """
    global can_hotkey

    if not can_hotkey:
        return _ask_hotkey_vanilla(message, lower)

    if message:
        printt(message.strip() + " ", end="")

    if not DEV:
        _clear_input_buffer()

    try:
        if os.name == 'nt':
            s = _read_hotkey_windows()
        else:
            s = _read_hotkey_posix()
    except KeyboardInterrupt:
        return "\x03"
    except Exception:
        can_hotkey = False
        return _ask_hotkey_vanilla("", lower)

    if message:
        printt()

    if lower:
        s = s.lower()

    return s

def ask_enter_to_continue(value: str="", is_replacement: bool=False) -> None:
    if is_replacement:
        message = value
    else:
        message = f"{value}\nPress enter: "

    if can_hotkey:
        while True:
            key = ask_hotkey(message)
            if key in ["\r", "\n"]:
                printt()
                return
            message = ""
    else:
        ask_input(message)


def ask_confirm(message: str="") -> bool:
    if not message:
        message = f"Press {make_hotkey_string('Y')} to confirm: "
    inp = ask_hotkey(message)
    if can_hotkey:
        printt()
    return inp == "y"


def ask_error(error_message: str) -> None:
    """
    App-standard way of displaying a user-facing error message.
    Prints user-facing error message in red plus blank line, and asks for enter key
    """
    printt(f"{COL_ERROR}{error_message}")
    printt()
    ask_enter_to_continue()

# ---

def ask_file_path(
        console_message: str,
        dialog_title: str,
        filetypes: list[tuple[str, str]] = [],
        initialdir: str="",
        prefill: str=""
) -> str:
    """
    Gets a file path string from user using either gui file requestor or input().
    """
    try:
        from tkinter import filedialog
        printt(console_message)
        path = filedialog.askopenfilename(title=dialog_title, filetypes=filetypes, initialdir=initialdir)
        did_tk = True
        if isinstance(path, tuple):
            path = ""
    except Exception:
        path = ask_path_input(console_message, prefill=prefill)
        did_tk = False

    if not path:
        return ""
    path = os.path.normpath(os.path.abspath(path))

    if did_tk:
        printt(path)
    printt()
    return path

def ask_dir_path(
        console_message: str,
        dialog_title: str,
        initialdir: str = "",
        mustexist: bool = True,
) -> str:
    """
    Gets a dir path string from user using either gui file requestor or input().
    """
    try:
        from tkinter import filedialog
        printt(console_message)
        path = filedialog.askdirectory(title=dialog_title, initialdir=initialdir, mustexist=mustexist)
        did_tk = True
        if isinstance(path, tuple):
            path = ""
    except Exception:
        path = ask_path_input(console_message)
        did_tk = False

    if not path:
        return ""
    path = os.path.normpath(os.path.abspath(path))

    if did_tk:
        printt(path)
    printt()
    return path

def ask_path_input(message: str="", prefill: str="") -> str:
    """
    Get file/directory path. Strip outer quotes if necessary.
    """
    printt(message)
    inp = ask_input(lower=False, prefill=prefill)
    inp = text_util.strip_quotes_around_path_string(inp)
    return inp

def ask_number_and_save(
    saveable: Saveable,
    attr: str,
    prompt: str,
    min_value: float,
    max_value: float,
    default_value: float,
    success_prefix: str,
    is_int: bool=False,
    print_range_info: bool=True,
    is_minus_one_default: bool=False,
) -> None:
    """
    """
    from tts_audiobook_tool.prefs import Prefs
    from tts_audiobook_tool.project import Project
    if not isinstance(saveable, Project) and not isinstance(saveable, Prefs):
        raise ValueError(f"Not Project or Prefs: {saveable}")

    if not hasattr(saveable, attr):
        raise ValueError(f"No such attribute {attr}")

    if is_int:
        min_value = int(min_value)
        max_value = int(max_value)
        default_value = int(default_value)

    stored_value = getattr(saveable, attr)
    is_effective_default_prefill = is_minus_one_default and stored_value == -1
    prefill = default_value if is_effective_default_prefill else stored_value
    if is_int:
        prefill = int(prefill) # for good measure
    prefill = str(prefill)

    prompt = prompt.strip()
    if not prompt.endswith(":"):
        prompt += ":"
    if print_range_info:
        prompt += " " + f"{COL_DIM}(valid range: {min_value}-{max_value}; default: {default_value})"
    printt(prompt)

    value = ask_input(prefill=prefill)
    if not value:
        return
    if value == prefill:
        # Do not save an unchanged displayed value. This also preserves a
        # stored "use default" sentinel when its effective value was shown.
        return
    try:
        value = float(value)
    except Exception:
        ask_error("Bad value")
        return
    if is_int:
        value = int(value)
    if value == stored_value:
        return
    is_default_sentinel = is_minus_one_default and value == -1
    if not is_default_sentinel and not (min_value <= value <= max_value):
        ask_error("Out of range")
        return

    setattr(saveable, attr, value)
    saveable.save()

    print_feedback(success_prefix, str(value))

def ask_string_and_save(
    saveable: Saveable,
    prompt_line: str,
    attr: str,
    success_prefix: str,
    loop_on_error: bool=False,
    validator: Callable[[str], str] | None = None,
    normalizer: Callable[[str], str] | None = None,
) -> bool:
    """
    Helper to ask for a string value and save it to the project (Saveable).
    Prefills input with current value.

    :param validator: Takes in the user input string and returns error string if invalid (optional)
    :param normalizer: Normalizes the user input before validation and saving (optional)
    :return: Whether a value was successfully saved
    """
    if not hasattr(saveable, attr):
        raise ValueError(f"No such attribute {attr}")

    current_value = getattr(saveable, attr)
    prefill = current_value

    while True:
        printt(prompt_line)
        value = ask_input(prefill=prefill, lower=False)
        prefill = "" # ie, only prefill on first try
        if not value:
            return False
        if value == current_value:
            return False
        if normalizer:
            value = normalizer(value)
        if value == current_value:
            return False
        if validator:
            err = validator(value)
            if err:
                ask_error(err)
                if loop_on_error:
                    continue
                return False
        break

    setattr(saveable, attr, value)
    saveable.save()
    print_feedback(success_prefix, value)
    return True

# ---

def _clear_input_buffer() -> None:
    """Use before input reads to prevent buffered keystrokes from being consumed."""
    try:
        import msvcrt
        while msvcrt.kbhit(): # type: ignore
            msvcrt.getch() # type: ignore
    except ImportError:
        if not sys.stdin.isatty():
            return
        import termios
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH) # type: ignore
        except termios.error:
            return


def _strip_terminal_escape_sequences(value: str) -> str:
    """Remove ANSI/VT terminal escape sequences from line input."""
    return terminal_escape_sequence_pattern.sub("", value)


def _read_hotkey_windows(block: bool=True) -> str | None:
    """
    Read one hotkey on Windows and normalize common special keys.

    When `block` is False, returns None if no key is waiting.
    """
    import msvcrt

    if not block and not msvcrt.kbhit(): # type: ignore
        return None

    ch = msvcrt.getwch() # type: ignore
    if ch in ("\x00", "\xe0"):
        special = msvcrt.getwch() # type: ignore
        mapped = {
            "H": "\x1b[A",
            "P": "\x1b[B",
            "K": "\x1b[D",
            "M": "\x1b[C",
            "S": "\x1b[3~",
            "G": "\x1b[H",
            "O": "\x1b[F",
            "I": "\x1b[5~",
            "Q": "\x1b[6~",
        }.get(special)
        if mapped is not None:
            return mapped
        return ch + special

    if ch == "\r":
        return "\r"
    if ch == "\x08":
        return "\x08"
    if ch == "\x03":
        return "\x03"
    if ch == "\x1b":
        return "\x1b"
    return ch


def _read_hotkey_posix() -> str:
    """
    Read one hotkey on POSIX terminals.

    This treats a lone Escape keypress as Escape after a short timeout
    instead of always blocking for the next byte in a possible escape
    sequence.
    """
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        first = os.read(fd, 1)
        if first != b"\x1b":
            return first.decode("utf-8", errors="replace")

        sequence = first
        if not select.select([sys.stdin], [], [], 0.05)[0]:
            return "\x1b"

        for _ in range(8):
            if not select.select([sys.stdin], [], [], 0)[0]:
                break
            sequence += os.read(fd, 1)

        return sequence.decode("utf-8", errors="replace")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _ask_hotkey_vanilla(message: str="", lower: bool=True) -> str:
    inp = ask_input(message, lower, extra_line=True)
    if inp:
        inp = inp[0]
    return inp
