import contextvars
import threading

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.generate_util import print_speed_info
from tts_audiobook_tool.generation_events import (
    GenerationEvents,
    GenerationPhase,
    GenerationStats,
)


def test_generation_event_sink_is_scoped_and_restored() -> None:
    outer: list[object] = []
    inner: list[object] = []

    with GenerationEvents.using_sink(outer.append):
        GenerationEvents.emit(GenerationPhase("outer one"))
        with GenerationEvents.using_sink(inner.append):
            GenerationEvents.emit(GenerationPhase("inner"))
        GenerationEvents.emit(GenerationPhase("outer two"))

    GenerationEvents.emit(GenerationPhase("discarded"))

    assert outer == [GenerationPhase("outer one"), GenerationPhase("outer two")]
    assert inner == [GenerationPhase("inner")]


def test_print_speed_info_emits_structured_realtime_stats(capsys) -> None:
    events: list[object] = []
    sound = Sound(np.zeros(16_000, dtype=np.float32), 16_000)

    with GenerationEvents.using_sink(events.append):
        print_speed_info(2.0, [sound])

    assert events == [
        GenerationStats(
            generation_seconds=2.0,
            audio_seconds=1.0,
            realtime_factor=2.0,
            speed_factor=0.5,
        )
    ]
    assert "speed: 0.5x" in capsys.readouterr().out


def test_sink_does_not_leak_into_unrelated_threads() -> None:
    active: list[object] = []
    done = threading.Event()

    def unrelated_thread() -> None:
        # A fresh thread starts with an empty context, so it must not see
        # the active sink (the old class-attribute sink would have captured
        # this stray emit).
        GenerationEvents.emit(GenerationPhase("stray emit"))
        done.set()

    with GenerationEvents.using_sink(active.append):
        thread = threading.Thread(target=unrelated_thread)
        thread.start()
        assert done.wait(1.0)
        thread.join(1.0)

    assert active == []


def test_sink_reaches_thread_that_propagates_parent_context() -> None:
    active: list[object] = []
    done = threading.Event()

    with GenerationEvents.using_sink(active.append):
        # A well-behaved model-library thread copies the parent context and
        # runs inside it, so its emit still reaches the active sink.
        parent_context = contextvars.copy_context()

        def model_thread() -> None:
            parent_context.run(
                GenerationEvents.emit, GenerationPhase("from model thread")
            )
            done.set()

        thread = threading.Thread(target=model_thread)
        thread.start()
        assert done.wait(1.0)
        thread.join(1.0)

    assert active == [GenerationPhase("from model thread")]
