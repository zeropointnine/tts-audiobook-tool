import asyncio
import threading
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from textual.geometry import Size
from textual.widgets import Rule, Static

from tts_audiobook_tool import real_time_playback
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleOutput,
    RealTimePlaybackCommand,
    RealTimePlaybackFinished,
    RealTimePlaybackTerminalStatus,
    RealTimePlaybackUpdate,
)
from tts_audiobook_tool.real_time_playback_events import (
    RealTimePlaybackAwaitingContinue,
    RealTimePlaybackBuffer,
    RealTimePlaybackEvents,
    RealTimePlaybackProgress,
    RealTimePlaybackSegmentText,
    RealTimePlaybackStarted,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual.real_time_playback_app import (
    RealTimePlaybackApp,
    RealTimePlaybackModalResult,
    _run_realtime_playback_console,
)
from tts_audiobook_tool.textual.real_time_playback_header import (
    RealTimePlaybackSourceText,
)


def run(coroutine) -> None:
    asyncio.run(coroutine)


def make_state() -> State:
    return cast(State, SimpleNamespace(project=SimpleNamespace()))


def test_app_source_band_is_framed_below_the_shared_divider(monkeypatch) -> None:
    monkeypatch.setattr(
        ModelWorker,
        "submit_realtime_playback",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(lambda: []))

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # The base's shared divider separates the header from the
            # source-text band, and the band's own rule closes it below:
            # header, divider, band text, band rule, worker log.
            band = app.query_one("#realtime-source", RealTimePlaybackSourceText)
            header_y = app.query_one("#realtime-header").region.y
            divider_y = app.query_one("#realtime-divider", Rule).region.y
            band_text_y = app.query_one("#realtime-source-text", Static).region.y
            band_rule_y = app.query_one("#realtime-source-divider", Rule).region.y
            log_y = app.query_one("#realtime-output-shell").region.y
            assert header_y < divider_y < band_text_y < band_rule_y < log_y
            # Exactly two rule widgets: the shared divider and the band's own.
            assert len(app.query(Rule)) == 2
            # 2-line text area plus the 1-line closing rule.
            assert band.size == Size(80, 3)

    run(exercise())


def test_escape_does_not_interrupt_realtime_playback(monkeypatch) -> None:
    """Escape no longer cancels or hard-resets running realtime playback;
    CTRL-C is the sole interrupt key."""

    monkeypatch.setattr(
        ModelWorker,
        "submit_realtime_playback",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(lambda: []))
    cancel_calls: list[str] = []
    monkeypatch.setattr(
        ModelWorker,
        "request_cancel",
        staticmethod(lambda operation_id: cancel_calls.append(operation_id) or True),
    )

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Not in find mode, no terminal result, and not waiting to finish:
            # Escape must be a no-op rather than forwarding to cancel_or_reset.
            await pilot.press("escape")
            await pilot.pause()
            assert cancel_calls == []
            assert not app.find_active

    run(exercise())


def test_console_fallback_waits_in_main_process_and_signals_worker(
    monkeypatch, capsys
) -> None:
    events = iter(
        [
            ConsoleOutput("job", "stdout", "playing\n"),
            RealTimePlaybackUpdate(
                "job",
                RealTimePlaybackAwaitingContinue(2.5, False),
            ),
            RealTimePlaybackFinished(
                "job",
                RealTimePlaybackTerminalStatus.COMPLETED,
            ),
        ]
    )
    continued: list[str] = []
    prompted: list[bool] = []
    monkeypatch.setattr(
        ModelWorker,
        "submit_realtime_playback",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "get_event",
        staticmethod(lambda timeout=0.1: next(events)),
    )
    monkeypatch.setattr(
        ModelWorker,
        "continue_realtime_playback",
        staticmethod(lambda operation_id: continued.append(operation_id) or True),
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.textual.real_time_playback_app.ask.ask_enter_to_continue",
        lambda: prompted.append(True),
    )

    result = _run_realtime_playback_console(make_state(), [], None)

    assert result.status is RealTimePlaybackTerminalStatus.COMPLETED
    assert prompted == [True]
    assert continued == ["job"]
    assert "playing" in capsys.readouterr().out


def test_app_receives_structured_progress_buffer_and_waiting_events(monkeypatch) -> None:
    monkeypatch.setattr(
        ModelWorker,
        "submit_realtime_playback",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(lambda: []))
    continued: list[str] = []
    monkeypatch.setattr(
        ModelWorker,
        "continue_realtime_playback",
        staticmethod(lambda operation_id: continued.append(operation_id) or True),
    )

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_update(RealTimePlaybackStarted(4, 0, 3))
            app._handle_update(RealTimePlaybackProgress(2, 4, 2))
            app._handle_update(RealTimePlaybackBuffer(7.5))
            assert app.processed == 2
            assert app.total == 4
            assert app.buffer_seconds == 7.5
            assert app.spoken_segments == []
            assert app.play_anchor is None

            appended_lines: list[str] = []
            monkeypatch.setattr(app, "_append_lines", appended_lines.extend)
            app._handle_update(RealTimePlaybackAwaitingContinue(6.0, False))
            assert any("Press" in line and "to finish" in line for line in appended_lines)
            assert app.waiting_for_continue
            assert app.prompt_mode == "awaiting_continue"
            rendered_status = str(
                app.query_one("#realtime-status", Static).render()
            )

            app.action_continue()
            app.phase = "Transient teardown status"
            app._update_header()
            assert str(app.query_one("#realtime-status", Static).render()) == rendered_status
            assert continued == ["job"]
            assert app.exit_after_terminal
            assert app.teardown_in_progress
            assert not app.waiting_for_continue

            lines_before_terminal = list(appended_lines)
            app._show_terminal_summary(
                RealTimePlaybackModalResult(
                    RealTimePlaybackTerminalStatus.COMPLETED
                )
            )
            assert appended_lines == lines_before_terminal

    run(exercise())


def test_buffer_duration_interpolates_between_worker_events(monkeypatch) -> None:
    app = RealTimePlaybackApp(make_state(), [], None)
    app.buffer_seconds = 8.0
    app.buffer_updated_at = 100.0
    monkeypatch.setattr(
        "tts_audiobook_tool.textual.real_time_playback_app.time.monotonic",
        lambda: 102.5,
    )

    assert app.interpolated_buffer_seconds == 5.5

    app.buffer_updated_at = 90.0
    assert app.interpolated_buffer_seconds == 0.0


def test_submit_realtime_playback_serializes_text_and_control_events(monkeypatch) -> None:
    commands = []
    cancellation_event = threading.Event()
    continue_event = threading.Event()
    monkeypatch.setattr(ModelWorker, "start", classmethod(lambda cls: ""))
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(ModelWorker, "_command_queue", SimpleNamespace(put=commands.append))
    monkeypatch.setattr(ModelWorker, "_cancellation_event", cancellation_event)
    monkeypatch.setattr(ModelWorker, "_continue_event", continue_event)
    phrase_group = PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=2)
    state = SimpleNamespace(
        project=SimpleNamespace(dir_path="/project"),
        prefs=Prefs(project_dir="/project"),
    )

    operation_id = ModelWorker.submit_realtime_playback(
        state=state,
        phrase_groups=[phrase_group],
        line_range=(2, 4),
    )

    assert len(commands) == 1
    command = commands[0]
    assert isinstance(command, RealTimePlaybackCommand)
    assert command.operation_id == operation_id
    assert command.line_range == (2, 4)
    assert command.phrase_groups_json[0]["voice_index"] == 2
    assert command.phrase_groups_json[0]["phrases"] == [
        {"text": "Hello.", "reason": Reason.SENTENCE.json_value}
    ]
    assert ModelWorker.continue_realtime_playback(operation_id)
    assert continue_event.is_set()


def test_realtime_stream_owner_closes_stream_when_impl_raises(monkeypatch) -> None:
    from tts_audiobook_tool import real_time_playback

    closed: list[bool] = []
    continuation_cleared: list[bool] = []
    fake_stream = SimpleNamespace(shut_down=lambda: closed.append(True))
    monkeypatch.setattr(
        real_time_playback.Tts,
        "clear_continuation",
        lambda: continuation_cleared.append(True),
    )

    def fail_impl(state, phrase_groups, line_range, continue_event, stream_holder):
        stream_holder.append(fake_stream)
        raise RuntimeError("boom")

    monkeypatch.setattr(real_time_playback, "_start_impl", fail_impl)

    with pytest.raises(RuntimeError, match="boom"):
        real_time_playback.start(make_state(), [], None)

    assert closed == [True]
    assert continuation_cleared == [True]


def test_sound_stream_shutdown_closes_after_stop_failure() -> None:
    from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream

    calls: list[str] = []

    class FailingStream:
        def stop(self) -> None:
            calls.append("stop")
            raise RuntimeError("stop failed")

        def close(self) -> None:
            calls.append("close")

    streamer = SoundDeviceStream()
    streamer.stream = cast(object, FailingStream())  # type: ignore[assignment]

    streamer.shut_down()

    assert calls == ["stop", "close"]
    assert streamer.stream is None


def test_playing_text_tracks_extrapolated_playhead(monkeypatch) -> None:
    app = RealTimePlaybackApp(make_state(), [], None)
    now = 100.0
    monkeypatch.setattr(
        "tts_audiobook_tool.textual.real_time_playback_app.time.monotonic",
        lambda: now,
    )

    assert app._current_playing_text() == ("", False)

    app._record_segment(
        RealTimePlaybackSegmentText(0, "First segment", 0, 144000, 0)
    )
    # At the anchor moment the cursor sits at the segment start: active.
    assert app._current_playing_text() == ("First segment", False)

    # Mid-segment: cursor has drained part of the segment.
    now = 102.0
    assert app._current_playing_text() == ("First segment", False)

    # 3s after the anchor the cursor reaches the segment end exactly: finished.
    now = 103.0
    assert app._current_playing_text() == ("First segment", True)

    # The next segment starts where the first one ended, so the cursor is
    # inside it and the extrapolation un-freezes.
    app._record_segment(
        RealTimePlaybackSegmentText(1, "Second segment", 144000, 240000, 144000)
    )
    assert app._current_playing_text() == ("Second segment", False)

    # Underflow: the cursor freezes at the total added, past the last segment.
    now = 109.0
    assert app._current_playing_text() == ("Second segment", True)


def test_playing_text_resets_when_stream_is_reused_mid_run(monkeypatch) -> None:
    app = RealTimePlaybackApp(make_state(), [], None)
    now = 100.0
    monkeypatch.setattr(
        "tts_audiobook_tool.textual.real_time_playback_app.time.monotonic",
        lambda: now,
    )
    app._record_segment(
        RealTimePlaybackSegmentText(0, "First segment", 0, 240000, 0)
    )

    # The stream buffer was cleared mid-run: sample ranges restart from 0,
    # so the previous segment is stale and the map restarts fresh.
    app._record_segment(
        RealTimePlaybackSegmentText(1, "Fresh start", 0, 48000, 0)
    )

    assert app.spoken_segments == [("Fresh start", 0, 48000)]
    assert app._current_playing_text() == ("Fresh start", False)


def test_start_impl_emits_segment_text_with_sample_range(monkeypatch) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.total = 0
            self.shut_downs = 0

        def start(self) -> bool:
            return True

        def add_data(self, data: np.ndarray) -> tuple[int, int]:
            start = self.total
            self.total += len(data)
            return start, self.total

        @property
        def buffer_duration(self) -> float:
            return 1.0

        @property
        def played_samples(self) -> int:
            return 0

        def shut_down(self) -> None:
            self.shut_downs += 1

    streams: list[FakeStream] = []

    def fake_stream_factory() -> FakeStream:
        stream = FakeStream()
        streams.append(stream)
        return stream

    sound = SimpleNamespace(
        data=np.zeros(48000, dtype=np.float32),
        duration=1.0,
        sr=48000,
    )
    monkeypatch.setattr(real_time_playback, "SoundDeviceStream", fake_stream_factory)
    monkeypatch.setattr(
        real_time_playback,
        "generate_full_flow",
        lambda state, phrase_groups, index, has_runway,
        consecutive_model_errors=0, max_consecutive_model_errors=5: (
            sound, False, 0,
        ),
    )
    monkeypatch.setattr(
        real_time_playback.SoundPipeline,
        "prepare_generated_sound_for_playback",
        lambda sound, high_shelf, limit_silence_gaps, limit_silence_gaps_duration: sound,
    )
    monkeypatch.setattr(
        real_time_playback.ProjectBookUtil,
        "get_section_start_indices",
        lambda project: [],
    )
    monkeypatch.setattr(
        real_time_playback.ModelManager,
        "warm_up_models",
        lambda state: SimpleNamespace(
            should_stop=False, error=None, did_interrupt=False
        ),
    )
    monkeypatch.setattr(
        real_time_playback.readiness,
        "get_generate_blocker_text",
        lambda state, verbose=False, is_realtime_playback=False: None,
    )
    monkeypatch.setattr(
        real_time_playback.app_memory,
        "show_vram_memory_warning_if_necessary",
        lambda: False,
    )
    monkeypatch.setattr(real_time_playback.Tts, "clear_continuation", lambda: None)
    monkeypatch.setattr(
        real_time_playback.Tts, "reset_voice_selection_index", lambda: None
    )
    monkeypatch.setattr(
        real_time_playback.Tts,
        "get_instance",
        lambda: SimpleNamespace(get_warning_issues=lambda project: None),
    )
    monkeypatch.setattr(
        real_time_playback.MenuUtil,
        "print_heading",
        lambda state, text, dont_clear=False, non_menu=False, breadcrumb_text="": None,
    )
    heading_calls: list[tuple[list[int], bool]] = []
    monkeypatch.setattr(
        real_time_playback.GenerateUtil,
        "print_batch_heading",
        lambda indices, voice_index=None, show_divider=True: heading_calls.append(
            (indices, show_divider)
        ),
    )

    state = cast(
        State,
        SimpleNamespace(
            prefs=SimpleNamespace(stt_variant=SttVariant.DISABLED),
            project=SimpleNamespace(
                max_retries=0,
                limit_silence_gaps=False,
                limit_silence_gaps_duration=0.5,
                use_break_sound_effect=False,
                reason_pauses=SimpleNamespace(get_pause_for=lambda reason: 0.0),
                get_high_shelf=lambda: 0,
                realtime_save=False,
            ),
        ),
    )
    phrase_group = PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=0)
    second_phrase_group = PhraseGroup(
        [Phrase("Goodbye.", Reason.SENTENCE)], voice_index=0
    )
    Interrupts().clear()
    continue_event = threading.Event()
    continue_event.set()

    received: list = []
    with RealTimePlaybackEvents.using_sink(received.append):
        result = real_time_playback.start(
            state,
            [phrase_group, second_phrase_group],
            (1, 2),
            continue_event=continue_event,
        )

    assert result.status is real_time_playback.RealTimePlaybackRunStatus.COMPLETED
    assert heading_calls == [([0], False), ([1], True)]
    buffer_idx = next(
        i for i, e in enumerate(received) if isinstance(e, RealTimePlaybackBuffer)
    )
    segment_idx = next(
        i
        for i, e in enumerate(received)
        if isinstance(e, RealTimePlaybackSegmentText)
    )
    assert buffer_idx < segment_idx
    event = received[segment_idx]
    assert event.index == 0
    assert event.text == phrase_group.presentable_text
    assert "Hello." in event.text
    assert (event.start_sample, event.end_sample) == (0, 48000)
    assert event.played_samples == 0
    assert streams[0].shut_downs == 1
