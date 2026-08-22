from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from tts_audiobook_tool.generation_events import GenerationPhase


@dataclass(frozen=True)
class RealTimePlaybackStarted:
    total: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class RealTimePlaybackProgress:
    processed: int
    total: int
    current_index: int | None = None


@dataclass(frozen=True)
class RealTimePlaybackBuffer:
    duration_seconds: float


@dataclass(frozen=True)
class RealTimePlaybackSegmentText:
    """
    The source text of one generated segment, emitted when that segment's
    audio is appended to the playback stream.

    start_sample/end_sample bound the segment (including any appended break
    sound) in the stream's cumulative sample timeline, and played_samples is
    the stream's played-sample count at emit time.  The receiving app treats
    each event as an anchor and extrapolates the playhead forward at 1x
    device time between anchors (see
    RealTimePlaybackApp.interpolated_played_samples).
    """

    index: int
    text: str
    start_sample: int
    end_sample: int
    played_samples: int


@dataclass(frozen=True)
class RealTimePlaybackAwaitingContinue:
    duration_seconds: float
    interrupted: bool


RealTimePlaybackEvent = (
    GenerationPhase
    | RealTimePlaybackStarted
    | RealTimePlaybackProgress
    | RealTimePlaybackBuffer
    | RealTimePlaybackSegmentText
    | RealTimePlaybackAwaitingContinue
)
RealTimePlaybackEventSink = Callable[[RealTimePlaybackEvent], None]


class RealTimePlaybackEvents:
    """Process-local structured telemetry for synchronous realtime playback."""

    _sink: contextvars.ContextVar[RealTimePlaybackEventSink | None] = (
        contextvars.ContextVar("realtime-playback-events-sink", default=None)
    )

    @classmethod
    def emit(cls, event: RealTimePlaybackEvent) -> None:
        sink = cls._sink.get()
        if sink is not None:
            sink(event)

    @classmethod
    @contextmanager
    def using_sink(cls, sink: RealTimePlaybackEventSink) -> Iterator[None]:
        token = cls._sink.set(sink)
        try:
            yield
        finally:
            cls._sink.reset(token)
