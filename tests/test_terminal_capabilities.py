from __future__ import annotations

import pytest

from tts_audiobook_tool.start import Start
from tts_audiobook_tool.system_support import terminal


class StubStream:
    def __init__(self, is_tty: bool, raises: bool = False) -> None:
        self.is_tty = is_tty
        self.raises = raises

    def isatty(self) -> bool:
        if self.raises:
            raise OSError("capability unavailable")
        return self.is_tty


def test_full_screen_detector_accepts_ttys_without_term() -> None:
    assert terminal.can_use_full_screen_terminal(
        stdin=StubStream(True),
        stdout=StubStream(True),
        os_name="posix",
        term="",
    )


def test_full_screen_detector_accepts_unknown_term() -> None:
    assert terminal.can_use_full_screen_terminal(
        stdin=StubStream(True),
        stdout=StubStream(True),
        os_name="posix",
        term="unfamiliar-capable-terminal",
    )


def test_full_screen_detector_rejects_explicit_dumb_terminal() -> None:
    assert not terminal.can_use_full_screen_terminal(
        stdin=StubStream(True),
        stdout=StubStream(True),
        os_name="posix",
        term="dumb",
    )


def test_full_screen_detector_rejects_noninteractive_stream() -> None:
    assert not terminal.can_use_full_screen_terminal(
        stdin=StubStream(True),
        stdout=StubStream(False),
    )


def test_full_screen_detector_allows_unknown_isatty_result() -> None:
    assert terminal.can_use_full_screen_terminal(
        stdin=StubStream(True, raises=True),
        stdout=StubStream(True),
    )


def test_startup_blocker_exits_with_plain_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(terminal, "can_use_full_screen_terminal", lambda: False)
    start = object.__new__(Start)

    with pytest.raises(SystemExit) as exception_info:
        start.exit_on_unsupported_terminal()

    assert exception_info.value.code == 1
    output = capsys.readouterr().out
    assert "requires a full-featured interactive terminal" in output
    assert "current terminal is unsupported" in output
    assert "Exiting." in output
    assert "\x1b" not in output


def test_startup_blocker_allows_supported_terminal(monkeypatch, capsys) -> None:
    monkeypatch.setattr(terminal, "can_use_full_screen_terminal", lambda: True)
    start = object.__new__(Start)

    start.exit_on_unsupported_terminal()

    assert capsys.readouterr().out == ""
