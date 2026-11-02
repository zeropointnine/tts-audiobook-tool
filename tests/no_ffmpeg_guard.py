"""
Guard that makes any launch of an FFmpeg executable fail loudly.

Patches the in-process launch points — `subprocess.Popen` (which also covers
`subprocess.run`/`call`/`check_output`, `os.popen`, asyncio subprocesses,
ffmpeg-python, and pydub), `os.system`, `os.posix_spawn[p]`, and the
`os.spawn*` / `os.exec*` families — so that launching `ffmpeg`, `ffprobe`,
`ffplay`, `avconv`, or `avprobe` raises `FfmpegExecutableLaunched` instead.

Every blocked launch is also recorded, so a caller that swallows the exception
(production code often catches broad errors and returns a message string)
still fails at the end of the guarded block.

Limitation: only this interpreter is patched. Child Python processes (e.g.
model workers) need their own guard.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import os
import re
import subprocess
import traceback
from collections.abc import Iterator
from typing import Any

FFMPEG_EXECUTABLES = ("ffmpeg", "ffprobe", "ffplay", "avconv", "avprobe")

# An argv[0]/executable: bare or path-qualified, optionally `.exe`, and also
# versioned/bundled variants such as `ffmpeg-linux64-v4.2.2` (imageio-ffmpeg).
_PROGRAM = re.compile(rf"^({'|'.join(FFMPEG_EXECUTABLES)})(?:[-_.][\w.-]*)?$", re.IGNORECASE)
# A word inside a shell command line: exact names only, to avoid matching
# e.g. `ffmpeg_util.py`.
_SHELL_WORD = re.compile(rf"^({'|'.join(FFMPEG_EXECUTABLES)})(?:\.exe)?$", re.IGNORECASE)
_SHELLS = {"sh", "bash", "dash", "zsh", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}


class FfmpegExecutableLaunched(AssertionError):
    """Raised when guarded code tries to launch an FFmpeg executable."""


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return os.fsdecode(value)
    return os.fspath(value) if isinstance(value, os.PathLike) else str(value)


def _basename(value: Any) -> str:
    return re.split(r"[\\/]", _text(value))[-1]


def _shell_offender(command: Any) -> str | None:
    for word in re.split(r"[\s\"'&|;()<>]+", _text(command)):
        if word and _SHELL_WORD.match(_basename(word)):
            return word
    return None


def ffmpeg_offender(args: Any, executable: Any = None, shell: bool = False) -> str | None:
    """The FFmpeg program a launch would run, or None if it would run something else."""
    if executable is not None and _PROGRAM.match(_basename(executable)):
        return _text(executable)
    if isinstance(args, (str, bytes, os.PathLike)):
        if shell or (isinstance(args, (str, bytes)) and re.search(r"\s", _text(args))):
            return _shell_offender(args)
        return _text(args) if _PROGRAM.match(_basename(args)) else None
    argv = list(args or ())
    if not argv:
        return None
    if shell:
        return _shell_offender(" ".join(_text(arg) for arg in argv))
    if _PROGRAM.match(_basename(argv[0])):
        return _text(argv[0])
    if _basename(argv[0]).lower() in _SHELLS:
        return _shell_offender(" ".join(_text(arg) for arg in argv[1:]))
    return None


class NoFfmpegGuard:
    """Context manager blocking FFmpeg executable launches; see module docstring."""

    def __init__(self) -> None:
        self.violations: list[str] = []
        self._stack: contextlib.ExitStack | None = None

    def _block(self, launcher: str, offender: str) -> None:
        message = f"{launcher} tried to launch FFmpeg executable {offender!r}, from:\n{_caller_frames()}"
        self.violations.append(message)
        raise FfmpegExecutableLaunched(message)

    def __enter__(self) -> NoFfmpegGuard:
        stack = contextlib.ExitStack()
        self._stack = stack
        guard = self

        original_init = subprocess.Popen.__init__
        # Follows `__wrapped__`, so nested guards bind against Popen's real signature.
        signature = inspect.signature(original_init)

        @functools.wraps(original_init)
        def guarded_init(popen_self, *args, **kwargs):
            bound = signature.bind(popen_self, *args, **kwargs)
            offender = ffmpeg_offender(
                bound.arguments.get("args"),
                bound.arguments.get("executable"),
                bool(bound.arguments.get("shell", False)),
            )
            if offender is not None:
                guard._block("subprocess.Popen", offender)
            original_init(popen_self, *args, **kwargs)

        stack.enter_context(_patched(subprocess.Popen, "__init__", guarded_init))

        original_system = os.system

        def guarded_system(command):
            offender = _shell_offender(command)
            if offender is not None:
                guard._block("os.system", offender)
            return original_system(command)

        stack.enter_context(_patched(os, "system", guarded_system))

        # (name, index of program path, index of argv)
        launchers = [
            ("posix_spawn", 0, 1), ("posix_spawnp", 0, 1),
            ("execv", 0, 1), ("execve", 0, 1), ("execvp", 0, 1), ("execvpe", 0, 1),
            ("spawnv", 1, 2), ("spawnve", 1, 2), ("spawnvp", 1, 2), ("spawnvpe", 1, 2),
        ]
        for name, path_index, argv_index in launchers:
            if hasattr(os, name):
                stack.enter_context(_patched(os, name, self._wrap_launcher(name, path_index, argv_index)))
        return self

    def _wrap_launcher(self, name: str, path_index: int, argv_index: int):
        original = getattr(os, name)
        guard = self

        @functools.wraps(original)
        def guarded(*args, **kwargs):
            path = args[path_index] if len(args) > path_index else None
            argv = args[argv_index] if len(args) > argv_index else None
            offender = ffmpeg_offender(argv or [path], executable=path)
            if offender is not None:
                guard._block(f"os.{name}", offender)
            return original(*args, **kwargs)

        return guarded

    def __exit__(self, *_exc_info) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None

    def assert_clean(self) -> None:
        if self.violations:
            raise FfmpegExecutableLaunched(
                "FFmpeg executable launch(es) blocked by the no-FFmpeg guard:\n  "
                + "\n  ".join(self.violations)
            )


def _caller_frames(count: int = 6) -> str:
    """The innermost frames outside this module and the stdlib launchers."""
    skipped = (__file__, os.path.dirname(subprocess.__file__))
    frames = [
        frame for frame in traceback.extract_stack()
        if not frame.filename.startswith(skipped) or "site-packages" in frame.filename
    ]
    return "".join(
        f"    {frame.filename}:{frame.lineno} in {frame.name}\n" for frame in frames[-count:]
    ).rstrip("\n")


@contextlib.contextmanager
def no_ffmpeg_exe() -> Iterator[NoFfmpegGuard]:
    """Block FFmpeg launches in the block, then fail if any were attempted."""
    with NoFfmpegGuard() as guard:
        yield guard
    guard.assert_clean()


@contextlib.contextmanager
def _patched(target: Any, name: str, replacement: Any) -> Iterator[None]:
    original = target.__dict__[name] if isinstance(target, type) else getattr(target, name)
    setattr(target, name, replacement)
    try:
        yield
    finally:
        setattr(target, name, original)
