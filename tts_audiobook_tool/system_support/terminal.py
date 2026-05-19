import shutil


def get_terminal_width(fallback: int = 80) -> int:
    """Returns terminal width with a safe cross-platform fallback."""
    try:
        width = shutil.get_terminal_size(fallback=(fallback, 20)).columns
    except Exception:
        width = fallback
    return max(20, width)


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
