import signal
import sys


class AskAdvanced:
    @staticmethod
    def ask(message: str = "", prefill: str = "") -> str:
        """
        Behave like input(), but with editable pre-filled text.

        - Linux and macOS: Uses readline
        - Windows: Uses the native console line editor through ReadConsoleW
        - Else: Falls back to input()
        """
        if not isinstance(message, str):
            raise TypeError("prompt must be a string")

        if not isinstance(prefill, str):
            raise TypeError("prefilled_input must be a string")

        try:
            if any(
                character < " " or character == "\x7f"
                for character in prefill
            ):
                # Cannot display embedded terminal controls in prefill text
                return input(message)

            if sys.platform in {"linux", "darwin"}:
                return _ask_with_posix_interruptible_input(message, prefill)

            if sys.platform == "win32":
                return _ask_with_windows_console(message, prefill)

            return input(message)
        except KeyboardInterrupt:
            # Cancel only the active text input; Ctrl-C outside AskAdvanced
            # retains its normal interrupt behavior.
            sys.stdout.write("\n")
            sys.stdout.flush()
            return ""


def _is_tty(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return False


def _ask_with_posix_interruptible_input(
    prompt: str, prefilled_input: str
) -> str:
    """Temporarily make SIGINT cancel input even if the app normally eats it."""
    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (OSError, ValueError):
        # Signal handlers can only be changed from Python's main thread.
        return _ask_with_readline(prompt, prefilled_input)

    try:
        return _ask_with_readline(prompt, prefilled_input)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _ask_with_readline(prompt: str, prefilled_input: str) -> str:
    if not (_is_tty(sys.stdin) and _is_tty(sys.stdout)):
        return input(prompt)

    try:
        import readline
    except ImportError:
        return input(prompt)

    if not all(hasattr(readline, name) for name in ("insert_text", "set_startup_hook")):
        return input(prompt)

    def insert_prefill() -> None:
        readline.insert_text(prefilled_input)

    readline.set_startup_hook(insert_prefill)
    try:
        return input(prompt)
    finally:
        # readline has no public getter with which to preserve an existing hook.
        readline.set_startup_hook(None)


def _write_windows_initial_line(prompt: str, prefilled_input: str) -> None:
    # nInitialChars preserves text in ReadConsoleW's edit buffer, but the
    # console does not initially echo those preserved characters. Write them
    # first so its physical cursor matches the logical edit cursor.
    sys.stdout.write(prompt + prefilled_input)
    sys.stdout.flush()


def _ask_with_windows_console(prompt: str, prefilled_input: str) -> str:
    """Read an editable prefilled line with Windows' native console editor."""
    if not (_is_tty(sys.stdin) and _is_tty(sys.stdout)):
        return input(prompt)

    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
    except (ImportError, AttributeError):
        return input(prompt)

    enable_processed_input = 0x0001
    enable_line_input = 0x0002
    enable_echo_input = 0x0004
    enable_virtual_terminal_input = 0x0200
    error_operation_aborted = 995
    ctrl_z = "\x1a"

    class _ConsoleReadControl(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.ULONG),
            ("nInitialChars", wintypes.ULONG),
            ("dwCtrlWakeupMask", wintypes.ULONG),
            ("dwControlKeyState", wintypes.ULONG),
        ]

    try:
        win_dll = getattr(ctypes, "WinDLL")
        get_last_error = getattr(ctypes, "get_last_error")
        get_osfhandle = getattr(msvcrt, "get_osfhandle")
        kernel32 = win_dll("kernel32", use_last_error=True)
        input_handle = wintypes.HANDLE(get_osfhandle(sys.stdin.fileno()))
    except (AttributeError, OSError, ValueError):
        return input(prompt)

    get_console_mode = kernel32.GetConsoleMode
    get_console_mode.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    get_console_mode.restype = wintypes.BOOL

    set_console_mode = kernel32.SetConsoleMode
    set_console_mode.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    set_console_mode.restype = wintypes.BOOL

    read_console = kernel32.ReadConsoleW
    read_console.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(_ConsoleReadControl),
    )
    read_console.restype = wintypes.BOOL

    input_mode = wintypes.DWORD()
    if not get_console_mode(input_handle, ctypes.byref(input_mode)):
        return input(prompt)

    original_input_mode = input_mode.value
    reading_input_mode = (
        original_input_mode
        | enable_processed_input
        | enable_line_input
        | enable_echo_input
    ) & ~enable_virtual_terminal_input

    # ReadConsoleW counts UTF-16 code units rather than Python code points.
    initial_chars = len(prefilled_input.encode("utf-16-le")) // 2
    # A generously sized buffer preserves normal interactive use while ensuring
    # nInitialChars is strictly less than nNumberOfCharsToRead as required.
    buffer_capacity = max(65536, initial_chars + 4096)
    buffer = ctypes.create_unicode_buffer(buffer_capacity)
    if initial_chars:
        ctypes.memmove(
            buffer,
            ctypes.c_wchar_p(prefilled_input),
            initial_chars * ctypes.sizeof(ctypes.c_wchar),
        )

    control = _ConsoleReadControl()
    control.nLength = ctypes.sizeof(control)
    control.nInitialChars = initial_chars
    # Make Ctrl-Z complete the read so it can retain input()'s EOF behavior.
    control.dwCtrlWakeupMask = 1 << ord(ctrl_z)
    control.dwControlKeyState = 0
    chars_read = wintypes.DWORD()

    mode_changed = False
    try:
        if reading_input_mode != original_input_mode:
            if not set_console_mode(input_handle, reading_input_mode):
                return input(prompt)
            mode_changed = True

        _write_windows_initial_line(prompt, prefilled_input)

        succeeded = read_console(
            input_handle,
            buffer,
            buffer_capacity,
            ctypes.byref(chars_read),
            ctypes.byref(control),
        )
        last_error = get_last_error()
        if last_error == error_operation_aborted:
            raise KeyboardInterrupt
        if not succeeded:
            raise OSError(last_error, "ReadConsoleW failed")

        value = ctypes.wstring_at(buffer, chars_read.value)
        if value.endswith(ctrl_z):
            value = value[:-1]
            if not value:
                raise EOFError

        if value.endswith("\r\n"):
            return value[:-2]
        if value.endswith(("\r", "\n")):
            return value[:-1]
        if not value:
            raise EOFError
        return value
    finally:
        if mode_changed:
            set_console_mode(input_handle, original_input_mode)
