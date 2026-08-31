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
loading, or a model download.
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


def make_gen_timeout_message(timeout_seconds: float) -> str:
    """Single source of the gen-timeout feedback text, citing the cap value."""
    return (
        f"TTS inference exceeded GEN_TIMEOUT ({timeout_seconds:g}s); "
        f"generation loop aborted and model worker was reset"
    )


@contextmanager
def gen_timeout_scope(timeout_seconds: float | None = None) -> Iterator[GenTimeoutGuard]:
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
        printt(f"{COL_ERROR}{make_gen_timeout_message(timeout)}")
        emit_context.run(
            GenerationEvents.emit, GenerationTimedOut(timeout_seconds=timeout)
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

    def __init__(self) -> None:
        self._did_first_gen = False

    @contextmanager
    def scope(self, timeout_seconds: float | None = None) -> Iterator[GenTimeoutGuard]:
        if not self._did_first_gen:
            # First gen of the run: untimed (warm-up/download may dominate).
            self._did_first_gen = True
            yield GenTimeoutGuard()
            return
        with gen_timeout_scope(timeout_seconds) as guard:
            yield guard
