from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass(frozen=True)
class GenerationPhase:
    label: str


@dataclass(frozen=True)
class GenerationStarted:
    total: int


@dataclass(frozen=True)
class GenerationProgress:
    processed: int
    remaining: int
    total: int
    current_indices: tuple[int, ...] = ()
    passed: int = 0
    failed: int = 0
    errored: int = 0
    retries: int = 0


@dataclass(frozen=True)
class GenerationStats:
    generation_seconds: float
    audio_seconds: float
    realtime_factor: float
    speed_factor: float


@dataclass(frozen=True)
class GenerationTimedOut:
    """One generation step exceeded the GEN_TIMEOUT cap.

    Emitted from a watchdog thread while the inference call is still in
    flight; the recipient should abort the run and reset the model worker.
    """

    timeout_seconds: float


GenerationEvent = (
    GenerationPhase
    | GenerationStarted
    | GenerationProgress
    | GenerationStats
    | GenerationTimedOut
)
GenerationEventSink = Callable[[GenerationEvent], None]


class GenerationEvents:
    """Process-local, optional structured telemetry for synchronous generation.

    The active sink lives in a ``contextvar`` rather than a class attribute.
    For the synchronous, single-threaded ``generate_files()`` path the
    behavior of sequential and nested ``using_sink`` calls is identical to a
    plain attribute swap. The contextvar additionally scopes the sink to the
    running context: an unrelated thread never inherits it (a fresh thread
    starts with an empty context), so a stray emit from another thread cannot
    leak into the active sink; a model-library thread that propagates the
    parent context (``copy_context().run(...)``) still reaches it.
    """

    _sink: contextvars.ContextVar[GenerationEventSink | None] = contextvars.ContextVar(
        "generation-events-sink",
        default=None,
    )

    @classmethod
    def emit(cls, event: GenerationEvent) -> None:
        sink = cls._sink.get()
        if sink is not None:
            sink(event)

    @classmethod
    @contextmanager
    def using_sink(cls, sink: GenerationEventSink) -> Iterator[None]:
        token = cls._sink.set(sink)
        try:
            yield
        finally:
            cls._sink.reset(token)
