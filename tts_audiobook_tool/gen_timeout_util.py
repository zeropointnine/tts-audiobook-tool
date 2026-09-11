"""
GEN_TIMEOUT enforcement for the TTS generation loops.

A "generation step" (one generate-and-validate batch call, ie one TTS
inference) must finish within ``GEN_TIMEOUT`` seconds. A step that exceeds
the cap is treated as a hung or pathologically slow inference: the
generation loop aborts and the model worker is reset.

The watchdog exists because inference runs in third-party model/runtime code
outside the app's control, which may deadlock, stall, or otherwise stop
returning for reasons the app cannot prevent. This has occurred in practice;
for example, dots.tts inference has locked up for me more than once
on at least one system.

Because the loops run inside the model worker process, a call that never
returns would block detection on the calling thread. The watchdog therefore
runs on a helper thread: on expiry it reports the timeout to the worker
console (relayed to the main process log/transcript) and emits a structured
``GenerationTimedOut`` event through the active ``GenerationEvents`` sink,
which the main process answers by hard-resetting the worker.

The watchdog deliberately takes no notice of interrupt/cancel state: a
pending cancellation is only observable at loop boundaries, between calls,
so an in-flight inference stays armed until it finishes or times out.

The very first inference of a run is exempt (``GenTimeoutTracker``): it may
legitimately spend far longer than the cap on first-run model warm-up, lazy
loading, or a model download. Single-shot callers (the diagnostic TTS preview)
use the always-armed ``backend_gen_timeout_scope()`` instead and decide that
exemption for themselves.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationTimedOut
from tts_audiobook_tool.util import *


@dataclass
class GenTimeoutGuard:
    """
    Result handle for one ``gen_timeout_scope()`` usage.

    After the with-block, ``did_time_out`` is True when the guarded call
    exceeded the cap (the timeout has already been reported and the worker
    reset requested); the calling loop must abort.
    """

    did_time_out: bool = False


def make_gen_timeout_message(
    timeout_seconds: float,
    *,
    is_remote: bool = False,
) -> str:
    """Return backend-appropriate generation-timeout feedback."""
    if is_remote:
        return "Remote generation timed out; recycling client worker"
    return (
        f"TTS inference exceeded GEN_TIMEOUT ({timeout_seconds:g}s); "
        "generation loop aborted; model-worker hard reset required"
    )


@contextmanager
def gen_timeout_scope(
    timeout_seconds: float | None = None,
    *,
    is_remote: bool = False,
) -> Iterator[GenTimeoutGuard]:
    """
    Watch one generation step (one TTS inference call) for GEN_TIMEOUT.

    Params:
        timeout_seconds:
            Overrides the cap for this scope (used by tests). When None,
            ``GEN_TIMEOUT`` is read at call time so it can be patched.
    """

    timeout = GEN_TIMEOUT if timeout_seconds is None else timeout_seconds
    guard = GenTimeoutGuard()
    finished = threading.Event()
    # The GenerationEvents sink is contextvar-scoped; a fresh thread starts
    # with an empty context, so run the emit through a copy of this context
    # (the documented pattern in generation_events.py).
    emit_context = contextvars.copy_context()

    def watchdog() -> None:
        if finished.wait(timeout):
            return
        guard.did_time_out = True
        printt()
        printt(
            f"{COL_ERROR}{make_gen_timeout_message(timeout, is_remote=is_remote)}"
        )
        emit_context.run(
            GenerationEvents.emit,
            GenerationTimedOut(timeout_seconds=timeout, is_remote=is_remote),
        )

    thread = threading.Thread(target=watchdog, name="gen-timeout-watchdog", daemon=True)
    thread.start()
    try:
        yield guard
    finally:
        finished.set()
        thread.join(timeout=1.0)


class GenTimeoutTracker:
    """Applies the GEN_TIMEOUT watchdog to every generation step except the
    first one.

    The first inference of a run may legitimately take far longer than
    ``GEN_TIMEOUT``: first-run model warm-up, lazy loading, or even a model
    download. A fresh tracker exempts its first ``scope()`` from timing out;
    every later scope is armed.

    One tracker belongs to one generation run (its lifetime defines which
    step is "first"), so a run with retries shares it across all of them.
    """

    def __init__(
        self,
        timeout_seconds: float | None = None,
        *,
        is_remote: bool = False,
    ) -> None:
        self._did_first_gen = False
        self._timeout_seconds = timeout_seconds
        self._is_remote = is_remote

    @contextmanager
    def scope(self, timeout_seconds: float | None = None) -> Iterator[GenTimeoutGuard]:
        if not self._did_first_gen:
            # First gen of the run: untimed (warm-up/download may dominate).
            self._did_first_gen = True
            yield GenTimeoutGuard()
            return
        effective_timeout = (
            self._timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        with gen_timeout_scope(
            effective_timeout,
            is_remote=self._is_remote,
        ) as guard:
            yield guard


def get_backend_gen_timeout() -> tuple[float, bool]:
    """
    Return ``(timeout_seconds, is_remote)`` for the active backend.

    Local backends use ``GEN_TIMEOUT``; SGL-Omni inference is remote, so it
    gets a more generous last-resort client deadline and remote wording. Both
    values are read at call time so tests can patch them.
    """
    # Import lazily to avoid pulling the TTS model registry into this low-level
    # watchdog module during import initialization.
    from tts_audiobook_tool.tts import Tts

    if Tts.is_sgl_mode():
        return SGL_OMNI_GEN_TIMEOUT, True
    return GEN_TIMEOUT, False


def make_backend_gen_timeout_tracker() -> GenTimeoutTracker:
    """Create a tracker with the current backend's timeout policy."""
    timeout_seconds, is_remote = get_backend_gen_timeout()
    return GenTimeoutTracker(
        timeout_seconds=timeout_seconds,
        is_remote=is_remote,
    )


@contextmanager
def backend_gen_timeout_scope() -> Iterator[GenTimeoutGuard]:
    """
    Watch one *single-shot* generation step for GEN_TIMEOUT.

    A ``GenTimeoutTracker`` exempts the first step of a run, because that step
    may legitimately carry first-run model warm-up, lazy loading, or a
    download. Callers that make exactly one inference per call (the diagnostic
    TTS preview) have no "later step" to fall back on, so this scope is always
    armed; those callers decide the exemption themselves, from what they know
    about whether the upcoming call can still include model setup.
    """
    timeout_seconds, is_remote = get_backend_gen_timeout()
    with gen_timeout_scope(
        timeout_seconds=timeout_seconds,
        is_remote=is_remote,
    ) as guard:
        yield guard
