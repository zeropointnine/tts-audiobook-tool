import shutil


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
