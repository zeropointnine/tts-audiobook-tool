import os
import shutil
import sys
from typing import Any

# Clears the screen and the scrollback, cursor home (as `cls` and `clear` do)
CLEAR_SCREEN_SEQUENCE = "\033[2J\033[3J\033[H"


def clear_screen() -> None:
    """
    Clears the terminal with an escape sequence.

    Not `os.system("cls")`: that launches cmd.exe on every menu draw, and after
    a control-c it sometimes never returned (blank screen, keys ignored, until
    another control-c killed it). Needs VT processing on Windows, see
    `enable_windows_vt_processing()`.
    """
    print(CLEAR_SCREEN_SEQUENCE, end="", flush=True)


def enable_windows_vt_processing() -> None:
    """
    Turns on escape-sequence handling for the Windows console's output.

    Windows Terminal has it on already. The classic console (conhost) has it
    off by default; the app's colors used to work there only because running
    `cls` through cmd.exe turns it on as a side effect. No-op elsewhere, and
    when output isn't a console.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        enable_virtual_terminal_processing = 0x0004
        for std_handle in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(std_handle)
            mode = wintypes.DWORD()
            if handle and kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing)
    except Exception:
        pass


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
