from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from tts_audiobook_tool.l import L


try:
    from wakepy import keep as _wakepy_keep
except ImportError:  # pragma: no cover - app startup already blocks on missing wakepy
    _wakepy_keep = None


def _log(message: str, *, is_warning: bool) -> None:
    """Writes to the app log; never raises (``L.logger`` exists only after init)."""
    if not hasattr(L, "logger"):
        return
    (L.w if is_warning else L.i)(message)


def _activate() -> Any | None:
    """
    Enters wakepy's ``keep.running`` mode.

    Returns the entered mode, or None when sleep inhibition is unavailable.
    Never raises: any failure is logged and reported as "unavailable" so that
    the caller's work still runs.
    """
    if _wakepy_keep is None:
        _log("wakepy is not installed; system sleep will not be prevented", is_warning=True)
        return None
    try:
        mode = _wakepy_keep.running(on_fail="pass")
        # Activation failure never raises with on_fail="pass"; the mode's
        # `active` flag tells us whether an inhibitor method succeeded.
        mode.__enter__()
    except Exception as exception:
        _log(
            f"Could not prevent system sleep: {type(exception).__name__}: {exception}",
            is_warning=True,
        )
        return None
    if mode.active:
        method = mode.active_method.name if mode.active_method else "unknown"
        _log(f"Preventing system sleep while running (wakepy method: {method})", is_warning=False)
    else:
        _log(
            "Could not prevent system sleep: no supported wakepy method on this system",
            is_warning=True,
        )
    return mode


def _deactivate(mode: Any) -> None:
    try:
        mode.__exit__(None, None, None)
    except Exception as exception:
        _log(
            f"Error releasing the system sleep inhibitor: "
            f"{type(exception).__name__}: {exception}",
            is_warning=True,
        )


class SystemSleepLock:
    """
    A system-sleep inhibitor that can be released before the enclosing call ends.

    ``prevent_system_sleep()`` is the usual entry point. This class exists for
    jobs whose work finishes before their function returns, such as a
    full-screen session that keeps its summary on screen (waiting for a keypress)
    after the worker job is already done.

    Acquires on construction; a no-op when sleep inhibition is unavailable.
    ``release()`` is safe to call repeatedly.
    """

    def __init__(self) -> None:
        self._mode = _activate()
        self._released = False

    def release(self) -> None:
        """Releases the inhibitor. Safe to call repeatedly."""
        if self._released:
            return
        self._released = True
        mode, self._mode = self._mode, None
        if mode is not None:
            _deactivate(mode)

    def __enter__(self) -> SystemSleepLock:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        self.release()
        return False


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    """
    Keeps the system from going to sleep while the wrapped work runs.

    Usable as a ``with`` block or as a decorator (``@prevent_system_sleep()``).
    Releases the inhibitor on every exit path, including exceptions and
    interrupts.

    Never raises and never prevents the wrapped work from running: when sleep
    inhibition is unavailable (wakepy missing, no supported inhibit method, any
    activation error) the block simply runs uninhibited, and the reason is
    written to the app log.
    """
    with SystemSleepLock():
        yield


# View sleep related system updates on Linux using:
# dbus-monitor --session "interface='org.gnome.SessionManager'"
