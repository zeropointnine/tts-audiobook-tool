from __future__ import annotations

import os
import select
import sys
from abc import ABC, abstractmethod


KEY_LEFT = "\x1b[D"
KEY_RIGHT = "\x1b[C"
KEY_DEL = "\x7f"
KEY_DEL2 = "\x08"
KEY_FWDDEL = "\x1b[3~"
KEY_ENTER = "\n"
KEY_CTRL_C = "\x03"


class ConsoleSession(ABC):
    """
    Abstract console I/O session for platform-specific raw key handling.

    Concrete implementations configure the terminal or console for interactive
    single-key input, restore the original state when finished, and normalize
    platform-specific key sequences into the shared constants defined here.
    """

    @staticmethod
    def create(real_stdout: object, real_stderr: object) -> "ConsoleSession":
        if os.name == "nt":
            return WindowsConsoleSession(real_stdout=real_stdout, real_stderr=real_stderr)
        return PosixConsoleSession()

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def restore(self) -> None:
        pass

    @abstractmethod
    def read_key(self) -> str | None:
        pass


class PosixConsoleSession(ConsoleSession):
    def __init__(self) -> None:
        import termios

        self.fd = sys.stdin.fileno()
        self.old_settings = None
        self.termios = termios

    def start(self) -> None:
        import tty

        self.old_settings = self.termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def restore(self) -> None:
        if self.old_settings is not None:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)
            self.old_settings = None

    def read_key(self) -> str | None:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = os.read(self.fd, 1)
        if ch == b"\x1b" and select.select([sys.stdin], [], [], 0.05)[0]:
            ch += os.read(self.fd, 8)
        return ch.decode("utf-8", errors="replace")


class WindowsConsoleSession(ConsoleSession):
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

    def __init__(self, real_stdout: object, real_stderr: object) -> None:
        import ctypes
        import msvcrt

        self.ctypes = ctypes
        self.msvcrt = msvcrt
        self.real_stdout = real_stdout
        self.real_stderr = real_stderr
        self.kernel32 = ctypes.windll.kernel32 # type: ignore
        self.saved_console_modes: list[tuple[int, int]] = []
        self.vt_enabled = False

    def start(self) -> None:
        self._enable_vt_mode(self.real_stdout)
        self._enable_vt_mode(self.real_stderr)

    def restore(self) -> None:
        while self.saved_console_modes:
            handle, mode = self.saved_console_modes.pop()
            self.kernel32.SetConsoleMode(handle, mode)

    def read_key(self) -> str | None:
        if not self.msvcrt.kbhit(): # type: ignore
            return None

        ch = self.msvcrt.getwch() # type: ignore
        if ch in ("\x00", "\xe0"):
            special = self.msvcrt.getwch() # type: ignore
            return {
                "K": KEY_LEFT,
                "M": KEY_RIGHT,
                "S": KEY_FWDDEL,
            }.get(special)
        if ch == "\r":
            return KEY_ENTER
        if ch == "\x08":
            return KEY_DEL2
        if ch == "\x03":
            return KEY_CTRL_C
        return ch

    def _enable_vt_mode(self, stream: object) -> None:
        fileno = getattr(stream, "fileno", None)
        if not callable(fileno):
            return
        try:
            fd = fileno()
        except Exception:
            return

        handle = self.msvcrt.get_osfhandle(fd) # type: ignore
        mode = self.ctypes.c_uint()
        if not self.kernel32.GetConsoleMode(handle, self.ctypes.byref(mode)):
            return

        current_mode = int(mode.value)
        new_mode = current_mode | self.ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if new_mode == current_mode:
            self.vt_enabled = True
            return
        if self.kernel32.SetConsoleMode(handle, new_mode):
            self.saved_console_modes.append((handle, current_mode))
            self.vt_enabled = True
