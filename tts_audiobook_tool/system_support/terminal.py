import os
import shutil
import sys
from typing import Any


def can_use_full_screen_terminal(
    *,
    stdin: Any | None = None,
    stdout: Any | None = None,
    os_name: str | None = None,
    term: str | None = None,
) -> bool:
    """Conservatively detect whether the interactive full-screen UI can run.

    A missing or unfamiliar ``TERM`` value is not treated as a failure: terminal
    capability detection is imperfect, and blocking a capable terminal is worse
    than allowing an uncertain one. We reject only non-interactive standard
    streams and the explicit Unix ``TERM=dumb`` capability declaration.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    os_name = os.name if os_name is None else os_name
    term = os.environ.get("TERM", "") if term is None else term

    if stdin is None or stdout is None:
        return False

    try:
        if not (stdin.isatty() and stdout.isatty()):
            return False
    except Exception:
        # Stream wrappers do not always implement isatty correctly. An unknown
        # result is not enough evidence to prevent the app from starting.
        return True

    if os_name != "nt" and term.strip().lower() == "dumb":
        return False

    return True


def get_terminal_width(fallback: int = 80) -> int:
    """Returns terminal width with a safe cross-platform fallback."""
    try:
        width = shutil.get_terminal_size(fallback=(fallback, 20)).columns
    except Exception:
        width = fallback
    return max(20, width)


def get_terminal_height(fallback: int = 24) -> int:
    """Returns terminal height with a safe cross-platform fallback."""
    try:
        height = shutil.get_terminal_size(fallback=(80, fallback)).lines
    except Exception:
        height = fallback
    return max(1, height)


def clear_input_buffer() -> None:
    """Use before input() to prevent buffered keystrokes from being consumed."""
    import sys

    try:
        import msvcrt

        while msvcrt.kbhit():  # type: ignore
            msvcrt.getch()  # type: ignore
    except ImportError:
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)  # type: ignore[arg-type]
