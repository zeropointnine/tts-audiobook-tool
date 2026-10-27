import threading
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import sounddevice as sd
from textual.geometry import Size
from textual.widgets import Input, Rule, Static

from tts_audiobook_tool import real_time_playback
from tts_audiobook_tool.app_types import RoomReverb, RoomReverbStereo, RoomTone, SttVariant
from tts_audiobook_tool.app_types.validation_result import SkippedResult
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.generation_events import GenerationPhase, GenerationTimedOut, ModelUnhealthy
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    RealTimePlaybackCommand,
    RealTimePlaybackTerminalStatus,
    RealTimePlaybackUpdate,
    WorkerCommandFailed,
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
from tts_audiobook_tool.textual import real_time_playback_app as playback_app_module
from tts_audiobook_tool.textual.real_time_playback_app import (
    RealTimePlaybackApp,
    RealTimePlaybackModalResult,
    run_real_time_playback_modal,
)
from tts_audiobook_tool.textual.real_time_playback_header import (
    WAVEFORM_HALF_ROWS,
    RealTimePlaybackSourceText,
    RealTimePlaybackWaveform,
)
from tts_audiobook_tool.textual.worker_content import (
    WorkerLogContentArea,
    _SeparatorLine,
)
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.worker_reset import HardResetCause
from textual_editor_stubs import run


def make_state() -> State:
    # The shared compose reads the selected model's output filters from the
    # project, so the stub needs the one lookup it performs.
    project = SimpleNamespace(get_tts_model_type=lambda: TtsModelType.require_by_id("none"))
    return cast(State, SimpleNamespace(project=project))


def test_app_source_band_is_framed_below_the_shared_divider(fake_worker) -> None:
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
            waveform = app.query_one("#realtime-waveform", RealTimePlaybackWaveform)
            band_text_y = app.query_one("#realtime-source-text", Static).region.y
            band_rule_y = app.query_one("#realtime-source-divider", Rule).region.y
            log_y = app.query_one("#realtime-output-shell").region.y
            assert header_y < divider_y < waveform.region.y < band_text_y < band_rule_y < log_y
            # The waveform (2 * WAVEFORM_HALF_ROWS rows) sits directly above
            # the text band, and with no segment yet shows just the dim
            # midline, on the row above center, across its width.
            half = WAVEFORM_HALF_ROWS
            assert waveform.outer_size == Size(80, 2 * half)
            assert waveform.region.bottom == band_text_y
            rows = str(waveform.render()).split("\n")
            assert len(rows) == 2 * half
            assert rows[half - 1] == "▁" * 78
            assert all(row.strip() == "" for i, row in enumerate(rows) if i != half - 1)
            # Exactly two rule widgets: the shared divider and the band's own.
            assert len(app.query(Rule)) == 2
            # 2-line text area plus the 1-line closing rule.
            assert band.size == Size(80, 3)

    run(exercise())


def test_realtime_progress_promotes_batch_divider_to_full_width_rule(
    fake_worker,
) -> None:
    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test(size=(70, 16)) as pilot:
            await pilot.pause()
            # Prior output so the first segment heading is not the first
            # content in the log; the divider is then promoted.
            app._feed_console(["decoding audio"], "")
            app._handle_update(RealTimePlaybackProgress(0, 1, 0))
            # Realtime emits a deliberate blank line before its heading.
            app._feed_console(["", "-------", "Processing line 1"], "")
            await pilot.pause()

            log = app.query_one(WorkerLogContentArea).worker_log
            width = log.scrollable_content_region.width
            assert log._lines[0].text.plain == "decoding audio"
            assert log._lines[1].text.plain == ""
            assert log._lines[2].text.plain == "-------"
            assert log._lines[2].row_text(width, 0).plain == ("- " * ((width + 1) // 2))[:width]
            assert log._lines[3].text.plain == "Processing line 1"

    run(exercise())


def test_realtime_awaiting_continue_places_closing_divider_before_prompt(
    fake_worker,
) -> None:
    """The end of the batch loop inserts the closing rule ahead of the
    trailing continue prompt, and the terminal summary does not add another."""

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test(size=(70, 20)) as pilot:
            await pilot.pause()
            log = app.query_one(WorkerLogContentArea).worker_log

            app._handle_update(RealTimePlaybackStarted(1, 0, 0))
            app._handle_update(RealTimePlaybackProgress(0, 1, 0))
            app._feed_console(
                ["-------", "Processing line 1", "Saved: 00001.flac"], ""
            )
            await pilot.pause()

            app._handle_update(RealTimePlaybackAwaitingContinue(3.0, False))
            await pilot.pause()

            lines = [line.text.plain for line in log._lines]
            separator_index = next(
                index
                for index, line in enumerate(log._lines)
                if isinstance(line, _SeparatorLine)
            )
            assert lines[separator_index - 1] == "Saved: 00001.flac"
            assert "Press" in lines[separator_index + 2]
            assert "to finish" in lines[separator_index + 2]

            separators = [
                line for line in log._lines if isinstance(line, _SeparatorLine)
            ]
            assert len(separators) == 1

    run(exercise())


def test_realtime_awaiting_continue_releases_sleep_lock(fake_worker) -> None:
    """The lock ends when the batch loop does, not after the ENTER-to-finish
    wait: the terminal result only arrives once the user presses ENTER."""
    released: list[str] = []

    async def exercise() -> None:
        app = RealTimePlaybackApp(
            make_state(), [], None, on_job_end=lambda: released.append("released")
        )
        async with app.run_test(size=(70, 20)) as pilot:
            await pilot.pause()
            assert app.terminal_result is None

            app._handle_update(RealTimePlaybackAwaitingContinue(3.0, False))
            await pilot.pause()

            assert released == ["released"]
            # Still no terminal result: ENTER has not been pressed yet.
            assert app.terminal_result is None

    run(exercise())


@pytest.mark.parametrize("awaiting_continue", [False, True])
def test_p_toggles_pause_and_header_without_stealing_find_input(
    monkeypatch, fake_worker, awaiting_continue: bool
) -> None:
    pause_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        ModelWorker,
        "set_realtime_playback_paused",
        staticmethod(lambda operation_id, paused: pause_calls.append((operation_id, paused)) or True),
    )
    continue_calls: list[str] = []
    monkeypatch.setattr(
        ModelWorker,
        "continue_realtime_playback",
        staticmethod(lambda operation_id: continue_calls.append(operation_id) or True),
    )

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test() as pilot:
            await pilot.pause()
            if awaiting_continue:
                app._handle_update(RealTimePlaybackAwaitingContinue(3.0, False))
            prefix = (
                "Press [ENTER] to finish" if awaiting_continue
                else "Press [ESC] to interrupt"
            )
            hotkey = app.query_one("#realtime-hotkey", Static)
            assert str(hotkey.render()) == prefix + "  - [P] to pause"
            await pilot.press("p")
            assert app.is_paused
            assert str(hotkey.render()) == prefix + "  - [P] to unpause"
            await pilot.press("P")
            assert not app.is_paused
            assert str(hotkey.render()) == prefix + "  - [P] to pause"
            assert pause_calls == [("job", True), ("job", False)]

            await pilot.press("ctrl+f", "p")
            assert app.query_one("#find-input", Input).value == "p"
            assert not app.is_paused
            assert len(pause_calls) == 2
            await pilot.press("escape", "p")
            assert app.is_paused
            assert pause_calls[-1] == ("job", True)
            if awaiting_continue:
                # Finishing a paused stream still uses the normal handshake.
                await pilot.press("enter")
                assert continue_calls == ["job"]
                assert app.teardown_in_progress
                await pilot.press("p")
                assert len(pause_calls) == 3

    run(exercise())


@pytest.mark.parametrize(
    "blocked_attribute",
    ["find_active", "reset_in_progress", "finishing", "teardown_in_progress", "terminal_result", "operation_id"],
)
def test_pause_is_ignored_when_session_controls_are_inactive(monkeypatch, blocked_attribute) -> None:
    app = RealTimePlaybackApp(make_state(), [], None)
    app.operation_id = "job"
    value = (
        RealTimePlaybackModalResult(RealTimePlaybackTerminalStatus.COMPLETED)
        if blocked_attribute == "terminal_result"
        else None if blocked_attribute == "operation_id" else True
    )
    setattr(app, blocked_attribute, value)
    pause_calls = []
    monkeypatch.setattr(
        ModelWorker,
        "set_realtime_playback_paused",
        staticmethod(lambda *args: pause_calls.append(args) or True),
    )
    app.action_toggle_pause()
    assert not app.is_paused
    assert pause_calls == []


def test_rejected_pause_does_not_change_playback_estimates(monkeypatch) -> None:
    app = RealTimePlaybackApp(make_state(), [], None)
    app.operation_id = "job"
    app.buffer_seconds = 8.0
    app.buffer_updated_at = 100.0
    app.play_anchor = (100.0, 0, 480000)
    monkeypatch.setattr(ModelWorker, "set_realtime_playback_paused", staticmethod(lambda *_: False))
    app.action_toggle_pause()
    assert not app.is_paused
    assert (app.buffer_seconds, app.buffer_updated_at) == (8.0, 100.0)
    assert app.play_anchor == (100.0, 0, 480000)


def test_pause_freezes_buffer_and_playhead_and_resume_excludes_paused_time(monkeypatch) -> None:
    app = RealTimePlaybackApp(make_state(), [], None)
    app.operation_id = "job"
    # This test isolates estimates without mounting the Textual screen.
    monkeypatch.setattr(app, "_update_header", lambda: None)
    clock = [100.0]
    monkeypatch.setattr(playback_app_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ModelWorker, "set_realtime_playback_paused", staticmethod(lambda *_: True))
    app._record_buffer_duration(8.0)
    app._record_segment(RealTimePlaybackSegmentText(0, "First", 0, 480000, 0))
    clock[0] = 102.0
    app.action_toggle_pause()
    assert app.is_paused
    clock[0] = 200.0
    assert app.interpolated_buffer_seconds == 6.0
    assert app.interpolated_played_samples == 96000
    assert app._current_playing_text() == ("First", False)
    app.action_toggle_pause()
    assert not app.is_paused
    assert app.interpolated_buffer_seconds == 6.0
    assert app.interpolated_played_samples == 96000
    clock[0] = 201.0
    assert app.interpolated_buffer_seconds == 5.0
    assert app.interpolated_played_samples == 144000

    app.action_toggle_pause()
    # Generation can still append data while playback is paused. New worker
    # anchors must also stay frozen, then advance normally after unpausing.
    clock[0] = 250.0
    app._handle_update(RealTimePlaybackBuffer(12.0))
    app._handle_update(RealTimePlaybackSegmentText(1, "Second", 480000, 960000, 144000))
    clock[0] = 300.0
    assert app.interpolated_buffer_seconds == 12.0
    assert app.interpolated_played_samples == 144000
    app.action_toggle_pause()
    clock[0] = 301.0
    assert app.interpolated_buffer_seconds == 11.0
    assert app.interpolated_played_samples == 192000


def test_escape_requests_realtime_playback_cancellation(monkeypatch, fake_worker) -> None:
    """The first Escape cooperatively cancels running realtime playback."""

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
            assert "Press [ESC] to interrupt" in str(
                app.query_one("#realtime-hotkey", Static).render()
            )
            await pilot.press("escape")
            await pilot.pause()
            assert cancel_calls == ["job"]
            assert app.cancel_requested
            assert not app.find_active

    run(exercise())


def test_escape_in_sgl_omni_mode_does_not_offer_hard_reset(monkeypatch, fake_worker) -> None:
    """In SGL-Omni backend mode the worker holds no local TTS model memory,
    so the hard-reset offer is gated off: ESC only requests a cancel and
    further presses are no-ops."""
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)

    cancel_calls: list[str] = []
    monkeypatch.setattr(
        ModelWorker,
        "request_cancel",
        staticmethod(lambda operation_id: cancel_calls.append(operation_id) or True),
    )
    reset_calls: list[int] = []

    def fake_reset() -> str:
        reset_calls.append(1)
        return ""

    monkeypatch.setattr(ModelWorker, "reset", staticmethod(fake_reset))

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            assert cancel_calls == ["job"]
            assert app.cancel_requested
            await pilot.pause()
            # The cancellation notice no longer offers the hard reset.
            log = app.query_one(WorkerLogContentArea).worker_log
            document = "\n".join(line.text.plain for line in log._lines)
            assert "Cancellation requested" in document
            assert "hard-reset" not in document
            # The pending header prompt waits rather than advertising the
            # kill-process escape hatch.
            hotkey = str(app.query_one("#realtime-hotkey", Static).render())
            assert "kill process" not in hotkey
            assert "Waiting for the current generation to stop" in hotkey

            # Further ESC presses do nothing: no worker dump, no reset,
            # no terminal result.
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert reset_calls == []
            assert not app.reset_in_progress
            assert app.terminal_result is None

    run(exercise())


@pytest.mark.parametrize("finish_key", ["enter", "escape"])
def test_app_receives_structured_progress_buffer_and_waiting_events(monkeypatch, fake_worker, finish_key) -> None:
    cancelled: list[str] = []
    monkeypatch.setattr(
        ModelWorker, "request_cancel",
        staticmethod(lambda operation_id: cancelled.append(operation_id) or True),
    )
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

            await pilot.press("ctrl+c")
            assert continued == []
            await pilot.press(finish_key)
            app.phase = "Transient teardown status"
            app._update_header()
            assert str(app.query_one("#realtime-status", Static).render()) == rendered_status
            assert continued == ["job"]
            assert app.exit_after_terminal
            assert app.teardown_in_progress
            assert not app.waiting_for_continue
            # Repeated Escape during audio teardown must not cancel the job
            # or send a second continue handshake.
            await pilot.press("escape")
            assert continued == ["job"]
            assert cancelled == []

            lines_before_terminal = list(appended_lines)
            app._show_terminal_summary(
                RealTimePlaybackModalResult(
                    RealTimePlaybackTerminalStatus.COMPLETED
                )
            )
            assert appended_lines == lines_before_terminal

    run(exercise())


@pytest.mark.parametrize("buffer_seconds", [299.0, 300.0])
def test_buffer_throttle_returns_without_sleep_at_or_below_limit(monkeypatch, buffer_seconds) -> None:
    monkeypatch.setattr(Interrupts(), "_external_event", None)
    monkeypatch.setattr(Interrupts(), "_flag", False)
    sleeps = []
    monkeypatch.setattr(real_time_playback.time, "sleep", sleeps.append)
    stream = cast(real_time_playback.SoundDeviceStream, SimpleNamespace(buffer_duration=buffer_seconds))
    assert not real_time_playback.wait_for_buffer_runway(stream)
    assert sleeps == []


@pytest.mark.parametrize("pause_mid_wait", [False, True])
def test_buffer_throttle_waits_through_pause_until_samples_actually_drain(
    monkeypatch, pause_mid_wait
) -> None:
    stream_type = real_time_playback.SoundDeviceStream
    pause_event = threading.Event()
    if not pause_mid_wait:
        pause_event.set()
    stream = stream_type(sample_rate=10, pause_requested=pause_event.is_set)
    stream.add_data(np.ones(40, dtype=np.float32))  # Four seconds buffered.
    monkeypatch.setattr(real_time_playback, "REAL_TIME_BUFFER_MAX_SECONDS", 3.0)
    monkeypatch.setattr(Interrupts(), "_external_event", None)
    monkeypatch.setattr(Interrupts(), "_flag", False)
    sleeps = []

    def simulated_sleep(seconds):
        sleeps.append(seconds)
        assert len(sleeps) <= 4, "Throttle did not stop when the buffer drained"
        if pause_mid_wait and len(sleeps) == 2:
            pause_event.set()
        if len(sleeps) == 4:
            pause_event.clear()
        before = stream.buffer_duration
        frames = 5 if pause_mid_wait else 10
        out = np.full((frames, 1), np.nan, dtype=np.float32)
        stream._callback(
            out, frames,
            SimpleNamespace(outputBufferDacTime=float(len(sleeps))),
            cast(sd.CallbackFlags, SimpleNamespace(output_underflow=False)),
        )
        if pause_event.is_set():
            assert stream.buffer_duration == before
            assert np.all(out == 0)
        else:
            assert stream.buffer_duration == before - frames / stream.sample_rate

    monkeypatch.setattr(real_time_playback.time, "sleep", simulated_sleep)
    assert not real_time_playback.wait_for_buffer_runway(stream)
    assert sleeps == [0.1] * 4
    assert stream.buffer_duration == 3.0
    assert stream.played_samples == 10


@pytest.mark.parametrize("cancel_before_wait", [False, True])
def test_buffer_throttle_can_cancel_while_audio_remains_paused(monkeypatch, cancel_before_wait) -> None:
    cancel_event = threading.Event()
    pause_event = threading.Event()
    pause_event.set()
    stream = real_time_playback.SoundDeviceStream(sample_rate=10, pause_requested=pause_event.is_set)
    stream.add_data(np.ones(40, dtype=np.float32))
    monkeypatch.setattr(real_time_playback, "REAL_TIME_BUFFER_MAX_SECONDS", 3.0)
    monkeypatch.setattr(Interrupts(), "_external_event", cancel_event)
    monkeypatch.setattr(Interrupts(), "_flag", False)
    sleeps = []
    if cancel_before_wait:
        cancel_event.set()

    def simulated_sleep(seconds):
        sleeps.append(seconds)
        assert stream.buffer_duration == 4.0
        assert len(sleeps) <= 2, "Throttle ignored cancellation"
        if len(sleeps) == 2:
            cancel_event.set()

    monkeypatch.setattr(real_time_playback.time, "sleep", simulated_sleep)
    assert real_time_playback.wait_for_buffer_runway(stream)
    assert sleeps == ([] if cancel_before_wait else [0.1, 0.1])
    assert stream.played_samples == 0
    assert stream.buffer_duration == 4.0


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
    pause_event = threading.Event()
    pause_event.set()
    monkeypatch.setattr(ModelWorker, "start", classmethod(lambda cls: ""))
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(ModelWorker, "_command_queue", SimpleNamespace(put=commands.append))
    monkeypatch.setattr(ModelWorker, "_cancellation_event", cancellation_event)
    monkeypatch.setattr(ModelWorker, "_continue_event", continue_event)
    monkeypatch.setattr(ModelWorker, "_pause_event", pause_event)
    phrase_group = PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=2)
    state = SimpleNamespace(
        project=SimpleNamespace(dir_path="/project", tts_model_type="vibevoice_local"),
        prefs=Prefs(project_dir="/project"),
    )

    operation_id = ModelWorker.submit_realtime_playback(
        state=state,
        phrase_groups=[phrase_group],
        line_range=(2, 4),
    )

    assert not pause_event.is_set()
    assert ModelWorker.set_realtime_playback_paused(operation_id, True)
    assert pause_event.is_set()
    assert not ModelWorker.set_realtime_playback_paused("stale", False)
    assert pause_event.is_set()
    assert ModelWorker.set_realtime_playback_paused(operation_id, False)
    assert not pause_event.is_set()
    assert len(commands) == 1
    command = commands[0]
    assert isinstance(command, RealTimePlaybackCommand)
    assert command.operation_id == operation_id
    assert command.line_range == (2, 4)
    assert command.settings.tts_model_type_id == "vibevoice_local"
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

    def fail_impl(state, phrase_groups, line_range, continue_event, stream_holder, pause_requested):
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

    assert app.spoken_segments == [("Fresh start", 0, 48000, b"")]
    assert app._current_playing_text() == ("Fresh start", False)


def test_waveform_tracks_playhead_and_holds_through_envelope_less_tail(
    monkeypatch,
) -> None:
    app = RealTimePlaybackApp(make_state(), [], None)
    now = 100.0
    monkeypatch.setattr(
        "tts_audiobook_tool.textual.real_time_playback_app.time.monotonic",
        lambda: now,
    )
    # Before any segment: no envelope, nothing played.
    assert app._current_waveform() == (b"", 0.0)

    first, second = b"\x10\x20", b"\x30\x40"
    app._record_segment(RealTimePlaybackSegmentText(0, "A", 0, 48000, 0, first))
    app._record_segment(RealTimePlaybackSegmentText(1, "B", 48000, 96000, 0, second))
    assert app._current_waveform() == (first, 0.0)
    now = 100.25  # a quarter of the way into the first 1s segment
    assert app._current_waveform() == (first, pytest.approx(0.25))
    now = 101.5
    assert app._current_waveform() == (second, pytest.approx(0.5))

    # The room tail carries no envelope: the second waveform stays, full.
    app._record_segment(RealTimePlaybackSegmentText(1, "B", 96000, 120000, 72000))
    now = 102.2
    assert app._current_playing_text() == ("B", False)
    assert app._current_waveform() == (second, 1.0)
    # Past the end (underflow): still the last waveform, fully played.
    now = 110.0
    assert app._current_waveform() == (second, 1.0)


# Buffer thresholds matter; retries are independent because generation is stubbed.
@pytest.mark.parametrize("max_retries,buffer_seconds,expected_runway", [
    (0, 1.0, False), (2, 59.9, False), (2, 60.0, True),
    (2, 169.0, True), (2, 301.0, True), (5, 60.0, True),
])
def test_start_impl_emits_segment_text_and_checks_validation_runway(
    monkeypatch, max_retries: int, buffer_seconds: float, expected_runway: bool
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.total = 0
            self.shut_downs = 0
            self.remaining_buffer = buffer_seconds

        channels = 1

        def start(self) -> bool:
            return True

        def add_data(self, data: np.ndarray) -> tuple[int, int]:
            start = self.total
            self.total += len(data)
            self.remaining_buffer = buffer_seconds
            return start, self.total

        @property
        def buffer_duration(self) -> float:
            return self.remaining_buffer

        @property
        def played_samples(self) -> int:
            return 0

        def shut_down(self) -> None:
            self.shut_downs += 1

    streams: list[FakeStream] = []

    pause_event = threading.Event()
    pause_event.set()

    def fake_stream_factory(*, blocksize, latency, pause_requested, channels) -> FakeStream:
        assert channels == 1
        assert blocksize == 4096
        assert latency == "low"
        assert pause_requested is not None and pause_requested()
        pause_event.clear()
        assert not pause_requested()
        stream = FakeStream()
        streams.append(stream)
        return stream

    sound = SimpleNamespace(
        data=np.zeros(48000, dtype=np.float32),
        duration=1.0,
        sr=48000,
    )
    runway_checks: list[bool] = []
    throttle_calls: list[FakeStream] = []

    def fake_wait_for_runway(stream):
        # Generation cannot advance to the second segment until this wait
        # returns. Its final segment must not wait, even if still over limit.
        assert runway_checks == [False]
        assert stream.buffer_duration > real_time_playback.REAL_TIME_BUFFER_MAX_SECONDS
        throttle_calls.append(stream)
        stream.remaining_buffer = real_time_playback.REAL_TIME_BUFFER_MAX_SECONDS
        return False

    monkeypatch.setattr(real_time_playback, "wait_for_buffer_runway", fake_wait_for_runway)

    def fake_generate_full_flow(
        state, phrase_groups, index, has_runway,
        consecutive_model_errors=0, max_consecutive_model_errors=5,
        gen_timeout_tracker=None,
    ):
        runway_checks.append(has_runway)
        return SkippedResult(sound, message=""), False, 0

    monkeypatch.setattr(real_time_playback, "SoundDeviceStream", fake_stream_factory)
    monkeypatch.setattr(
        real_time_playback, "generate_full_flow", fake_generate_full_flow
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
        lambda state, verbose=False: None,
    )
    monkeypatch.setattr(
        real_time_playback.app_memory,
        "show_vram_memory_warning_if_necessary",
        lambda: False,
    )
    monkeypatch.setattr(real_time_playback.Tts, "clear_continuation", lambda: None)
    monkeypatch.setattr(
        real_time_playback.Tts,
        "get_instance",
        lambda: SimpleNamespace(get_warning_issues=lambda project: None),
    )
    # Run warnings (incl. the voice-variation note) read real project voice data
    monkeypatch.setattr(real_time_playback, "get_run_warnings", lambda *args: [])
    heading_calls: list[list[int]] = []
    monkeypatch.setattr(
        real_time_playback.GenerateUtil,
        "print_batch_heading",
        lambda indices, voice_index=None: heading_calls.append(indices),
    )

    state = cast(
        State,
        SimpleNamespace(
            prefs=SimpleNamespace(stt_variant=SttVariant.DISABLED),
            project=SimpleNamespace(
                max_retries=max_retries,
                limit_silence_gaps=False,
                limit_silence_gaps_duration=0.5,
                use_break_sound_effect=False,
                reason_pauses=SimpleNamespace(get_pause_for=lambda reason: 0.0),
                get_high_shelf=lambda: 0,
                realtime_save=False,
                get_tts_model_type=lambda: TtsModelType.require_by_id("none"),
                voice_references=[],
                room_tone=RoomTone.OFF,
                room_reverb_stereo=RoomReverbStereo.MONO,
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
            pause_requested=pause_event.is_set,
        )

    assert result.status is real_time_playback.RealTimePlaybackRunStatus.COMPLETED
    assert heading_calls == [[0], [1]]
    # No stream exists for the first segment. Later validation starts at 60s,
    # independent of the configured retry count (including zero retries).
    assert runway_checks == [False, expected_runway]
    assert len(throttle_calls) == (1 if buffer_seconds > 300.0 else 0)
    assert streams[0].buffer_duration == buffer_seconds
    wait_phases = [
        event for event in received
        if isinstance(event, GenerationPhase)
        and event.label == "Waiting for audio buffer to drain"
    ]
    assert len(wait_phases) == len(throttle_calls)
    assert isinstance(received[-1], RealTimePlaybackAwaitingContinue)
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
    # Two envelope bytes (positive, negative peak) per 10ms window of the
    # 1s segment (silent here).
    assert event.envelope == bytes(200)
    assert streams[0].shut_downs == 1


def test_start_impl_submits_completed_sound_when_cancelled_during_generation(
    monkeypatch,
) -> None:
    """A cooperative cancel (one ESC) arriving mid-generation stops the
    run, but the in-flight generation had completed: its sound is submitted
    to the output buffer (where it plays while the session waits for Enter)
    instead of being discarded."""

    class FakeStream:
        def __init__(self) -> None:
            self.total = 0
            self.shut_downs = 0

        channels = 1

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

    def fake_stream_factory(*, blocksize, latency, pause_requested, channels) -> FakeStream:
        assert channels == 1
        assert blocksize == 4096
        assert latency == "low"
        stream = FakeStream()
        streams.append(stream)
        return stream

    sound = SimpleNamespace(
        data=np.zeros(48000, dtype=np.float32),
        duration=1.0,
        sr=48000,
    )

    # The cancel request (as the worker signals a single ESC from the
    # parent) arrives while the second generation is in flight; that
    # generation still completes.
    cancel_event = threading.Event()

    def fake_generate_full_flow(
        state,
        phrase_groups,
        index,
        has_runway,
        consecutive_model_errors=0,
        max_consecutive_model_errors=5,
        gen_timeout_tracker=None,
    ):
        if index == 1:
            cancel_event.set()
        did_interrupt = Interrupts().did_interrupt
        # generate_full_flow() clears Interrupts at the end; the external
        # cancel event intentionally stays set until the next submit.
        Interrupts().clear()
        return SkippedResult(sound, message=""), did_interrupt, consecutive_model_errors

    tracker_calls: list[int] = []

    class FakeBreakEffectTracker:
        def __init__(self, section_start_indices, *, start_group_index=0) -> None:
            ...

        def next_break_effect(self, index, reason, effects_enabled, is_final):
            tracker_calls.append(index)
            return None

    monkeypatch.setattr(real_time_playback, "SoundDeviceStream", fake_stream_factory)
    monkeypatch.setattr(
        real_time_playback, "generate_full_flow", fake_generate_full_flow
    )
    monkeypatch.setattr(
        real_time_playback.SoundPipeline,
        "prepare_generated_sound_for_playback",
        lambda sound, high_shelf, limit_silence_gaps, limit_silence_gaps_duration: sound,
    )
    monkeypatch.setattr(real_time_playback, "BreakEffectTracker", FakeBreakEffectTracker)
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
        lambda state, verbose=False: None,
    )
    monkeypatch.setattr(
        real_time_playback.app_memory,
        "show_vram_memory_warning_if_necessary",
        lambda: False,
    )
    monkeypatch.setattr(real_time_playback.Tts, "clear_continuation", lambda: None)
    monkeypatch.setattr(
        real_time_playback.Tts,
        "get_instance",
        lambda: SimpleNamespace(get_warning_issues=lambda project: None),
    )
    # Run warnings (incl. the voice-variation note) read real project voice data
    monkeypatch.setattr(real_time_playback, "get_run_warnings", lambda *args: [])
    monkeypatch.setattr(
        real_time_playback.GenerateUtil,
        "print_batch_heading",
        lambda indices, voice_index=None: None,
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
                get_tts_model_type=lambda: TtsModelType.require_by_id("none"),
                voice_references=[],
                room_tone=RoomTone.OFF,
                room_reverb_stereo=RoomReverbStereo.MONO,
            ),
        ),
    )
    phrase_groups = [
        PhraseGroup([Phrase(f"Segment {i}.", Reason.SENTENCE)], voice_index=0)
        for i in range(3)
    ]
    Interrupts().clear()
    Interrupts().set_external_event(cancel_event)
    try:
        continue_event = threading.Event()
        continue_event.set()

        received: list = []
        with RealTimePlaybackEvents.using_sink(received.append):
            result = real_time_playback.start(
                state,
                phrase_groups,
                (1, 3),
                continue_event=continue_event,
            )
    finally:
        Interrupts().set_external_event(None)
        Interrupts().clear()

    # The run ends as a cancellation, with the interrupted segment's sound in
    # the buffer: both completed generations were submitted, the third was
    # never generated.
    assert result.status is real_time_playback.RealTimePlaybackRunStatus.CANCELLED
    assert len(streams) == 1
    assert streams[0].total == 2 * 48000
    assert streams[0].shut_downs == 1

    segments = [e for e in received if isinstance(e, RealTimePlaybackSegmentText)]
    assert [(e.index, e.start_sample, e.end_sample) for e in segments] == [
        (0, 0, 48000),
        (1, 48000, 96000),
    ]

    # The terminating segment got no trailing break effect or pause, and the
    # tracker was only advanced for the segment that did get one.
    assert tracker_calls == [0]

    # The run stopped before generating the third segment, so there is no
    # final Progress(total, total); it ends at AwaitingContinue(interrupted).
    progress = [e for e in received if isinstance(e, RealTimePlaybackProgress)]
    assert [(e.processed, e.total) for e in progress] == [(0, 3), (1, 3)]
    awaiting = [
        e for e in received if isinstance(e, RealTimePlaybackAwaitingContinue)
    ]
    assert len(awaiting) == 1
    assert awaiting[0].interrupted is True


def test_gen_timeout_update_hard_resets_worker_even_with_cancel_pending(
    monkeypatch, fake_worker,
) -> None:
    """A pending single-ESC cancel must not suppress the gen timeout."""
    reset_calls: list[int] = []
    cancel_calls: list[str] = []
    latch = {"deliver": False}
    events = [
        RealTimePlaybackUpdate("job", GenerationTimedOut(timeout_seconds=180.0))
    ]

    def fake_drain_events() -> list:
        if latch["deliver"] and events:
            return [events.pop(0)]
        return []

    def fake_reset() -> str:
        reset_calls.append(1)
        return ""

    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(fake_drain_events))
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))
    monkeypatch.setattr(
        ModelWorker,
        "request_cancel",
        staticmethod(lambda operation_id: cancel_calls.append(operation_id) or True),
    )
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(fake_reset))

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test() as pilot:
            await pilot.press("escape")
            assert cancel_calls == ["job"]
            assert app.cancel_requested

            # The timeout fires anyway and hard-resets the worker.
            latch["deliver"] = True
            await pilot.pause(0.3)

            assert reset_calls == [1]
            assert app.terminal_result is not None
            assert app.terminal_result.status is (
                RealTimePlaybackTerminalStatus.WORKER_RESET
            )
            assert app.terminal_result.hard_reset_cause is (
                HardResetCause.GENERATION_TIMEOUT
            )
            assert "GEN_TIMEOUT" in app.terminal_result.message
            assert "180" in app.terminal_result.message
            await pilot.press("enter")

    run(exercise())


def test_unhealthy_reset_ignores_queued_awaiting_continue_and_enter(
    monkeypatch, fake_worker,
) -> None:
    reset_started = threading.Event()
    release_reset = threading.Event()
    continue_calls: list[str] = []
    latch = {"deliver": False}
    events = [
        RealTimePlaybackUpdate("job", ModelUnhealthy("TTS model is unhealthy")),
        RealTimePlaybackUpdate(
            "job", RealTimePlaybackAwaitingContinue(2.5, interrupted=True)
        ),
        WorkerCommandFailed("job", "stale old-worker failure"),
    ]

    def fake_drain_events() -> list:
        if latch["deliver"] and events:
            drained = list(events)
            events.clear()
            return drained
        return []

    def blocking_reset() -> str:
        reset_started.set()
        release_reset.wait()
        return ""

    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(fake_drain_events))
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(blocking_reset))
    monkeypatch.setattr(
        ModelWorker,
        "continue_realtime_playback",
        staticmethod(lambda operation_id: continue_calls.append(operation_id) or True),
    )

    async def exercise() -> None:
        app = RealTimePlaybackApp(make_state(), [], None)
        async with app.run_test() as pilot:
            try:
                latch["deliver"] = True
                for _ in range(20):
                    if reset_started.is_set():
                        break
                    await pilot.pause(0.05)

                assert reset_started.is_set()
                assert app.reset_in_progress
                assert app.phase == "Hard-resetting model worker"
                assert not app.waiting_for_continue

                # Also defend against ENTER when a waiter predated the reset;
                # reset ownership must win regardless of event ordering.
                app.waiting_for_continue = True
                await pilot.press("enter", "escape")
                await pilot.pause()
                assert app.waiting_for_continue
                assert continue_calls == []
                assert not app.teardown_in_progress
            finally:
                release_reset.set()

            await pilot.pause(0.3)
            assert app.terminal_result is not None
            assert app.terminal_result.status is (
                RealTimePlaybackTerminalStatus.WORKER_RESET
            )
            assert app.terminal_result.hard_reset_cause is (
                HardResetCause.MODEL_UNHEALTHY
            )
            assert "stale old-worker failure" not in app.terminal_result.message
            await pilot.press("enter")

    run(exercise())


def test_interface_failure_cleanup_reports_worker_restart_failure(
    monkeypatch,
) -> None:
    reset_calls: list[None] = []

    def failing_run(app, **_kwargs):
        app.operation_id = "job"
        raise RuntimeError("interface exploded")

    monkeypatch.setattr(ModelWorker, "start", staticmethod(lambda: ""))
    monkeypatch.setattr(
        ModelWorker,
        "reset",
        staticmethod(
            lambda: reset_calls.append(None) or "replacement worker failed to start"
        ),
    )
    monkeypatch.setattr(RealTimePlaybackApp, "run", failing_run)
    monkeypatch.setattr(
        playback_app_module,
        "_present_console_result",
        lambda *_args: None,
    )

    result = run_real_time_playback_modal(make_state(), [], None)

    assert reset_calls == [None]
    assert result.status is RealTimePlaybackTerminalStatus.FAILED
    assert result.hard_reset_cause is HardResetCause.INTERFACE_FAILURE
    assert "RuntimeError: interface exploded" in result.message
    assert "replacement worker failed to start" in result.message


# Cover each spatial policy once, plus break effects in dry, wet, stereo and
# fallback playback. Voice slots/counts need representatives, not a full product.
@pytest.mark.parametrize(
    "room_reverb,room_reverb_stereo,device_stereo_ok,expected_channels,voice_count,voice_index,break_effect",
    [
        (RoomReverb.OFF, RoomReverbStereo.MONO, True, 1, 1, 1, False),
        (RoomReverb.MEDIUM_DRR21, RoomReverbStereo.MONO, True, 1, 2, 0, False),
        (RoomReverb.OFF, RoomReverbStereo.ALL_W100, True, 1, 2, 1, False),
        (RoomReverb.MEDIUM_DRR21, RoomReverbStereo.LATE_W100, True, 2, 2, 1, False),
        (RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, True, 2, 1, 1, False),
        (RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, False, 1, 2, 1, False),
        (RoomReverb.OFF, RoomReverbStereo.MONO, True, 1, 2, 1, True),
        (RoomReverb.MEDIUM_DRR21, RoomReverbStereo.MONO, True, 1, 2, 1, True),
        (RoomReverb.MEDIUM_DRR21, RoomReverbStereo.LATE_W100, True, 2, 2, 1, True),
        (RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, False, 1, 2, 1, True),
    ],
)
def test_realtime_save_writes_exactly_what_was_played(
    monkeypatch,
    room_reverb: RoomReverb,
    room_reverb_stereo: RoomReverbStereo,
    device_stereo_ok: bool,
    expected_channels: int,
    voice_count: int,
    voice_index: int,
    break_effect: bool,
    ending: str = "normal",
    other_voice_reverb: RoomReverb | None = None,
) -> None:
    # Saved realtime segments are free-standing artifacts of the playback: the
    # file must hold the post-processed audio (reverb, room tone bed, pause)
    # sent to the output stream, not the raw generation.
    # `room_reverb` is the speaking voice's; `other_voice_reverb`, when given,
    # is every other voice's (never fed, but it can make the run stereo).
    from tts_audiobook_tool.app_types import Sound

    class FakeStream:
        def __init__(self, channels: int) -> None:
            self.chunks: list[np.ndarray] = []
            self.total = 0
            self.channels = channels
            self.shut_downs = 0

        def start(self) -> bool:
            return self.channels == 1 or device_stereo_ok

        def add_data(self, data: np.ndarray) -> tuple[int, int]:
            self.chunks.append(np.copy(data))
            start = self.total
            self.total += len(data)
            return start, self.total

        buffer_duration = 1.0
        played_samples = 0

        def shut_down(self) -> None:
            self.shut_downs += 1

    streams: list[FakeStream] = []

    def fake_stream_factory(*, channels: int, **_) -> FakeStream:
        streams.append(FakeStream(channels))
        return streams[-1]

    printed: list[str] = []
    monkeypatch.setattr(real_time_playback, "printt", lambda s="", *args, **kwargs: printed.append(s))

    # Content in the last 0.1 s only, so a reverb tail would be loud in the pause
    raw_data = np.zeros(48000, dtype=np.float32)
    raw_data[-4800:] = 0.2 * np.sin(np.arange(4800, dtype=np.float32) * 0.1)
    raw = Sound(raw_data, 48000)
    saved: list[np.ndarray] = []
    spatial_processors = []
    speech_rooms = []
    spatial_type = real_time_playback.VoiceSpatialProcessor

    def make_spatial(**kwargs):
        processor = spatial_type(**kwargs)
        apply = processor.apply

        def apply_unfaded(*args, **options):
            # Final speech must retain its complete wet path; only draining
            # after the actual loop end may fade the remaining room.
            assert not options.get("is_last", False)
            assert options.get("pan", 0.0) == 0.0
            if "send_end" not in options:
                speech_rooms.append(options.get("room_reverb"))
            return apply(*args, **options)

        processor.apply = apply_unfaded
        spatial_processors.append(processor)
        return processor

    monkeypatch.setattr(real_time_playback, "VoiceSpatialProcessor", make_spatial)
    bed_calls = []
    noise_bed_type = real_time_playback.NoiseBed

    class TrackingNoiseBed(noise_bed_type):
        def apply(self, sound, is_last=False):
            bed_calls.append((len(sound.data), is_last))
            return super().apply(sound, is_last=is_last)

    monkeypatch.setattr(real_time_playback, "NoiseBed", TrackingNoiseBed)
    if break_effect:
        monkeypatch.setattr(real_time_playback.BreakEffectTracker, "next_break_effect", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            real_time_playback.SoundPipeline, "make_break_effect_sound",
            lambda *args: Sound(np.full(24000, .05, dtype=np.float32), 48000),
        )

    monkeypatch.setattr(real_time_playback, "SoundDeviceStream", fake_stream_factory)
    generated_indices = []

    def generate(state, groups, index, **kwargs):
        generated_indices.append(index)
        if index == 1 and ending in ("failed", "cancelled"):
            return None, ending == "cancelled", 0
        return SkippedResult(raw, message=""), ending == "cancelled_first", 0

    monkeypatch.setattr(real_time_playback, "generate_full_flow", generate)
    monkeypatch.setattr(
        real_time_playback, "save_played_sound",
        lambda state, phrase_group, index, validation_result, played: saved.append(played.data),
    )
    monkeypatch.setattr(
        real_time_playback.SoundPipeline, "prepare_generated_sound_for_playback",
        lambda sound, high_shelf, limit_silence_gaps, limit_silence_gaps_duration: sound,
    )
    monkeypatch.setattr(real_time_playback.ProjectBookUtil, "get_section_start_indices", lambda project: [])
    monkeypatch.setattr(
        real_time_playback.ModelManager, "warm_up_models",
        lambda state: SimpleNamespace(should_stop=False, error=None, did_interrupt=False),
    )
    monkeypatch.setattr(real_time_playback.readiness, "get_generate_blocker_text", lambda state, verbose=False: None)
    monkeypatch.setattr(real_time_playback.app_memory, "show_vram_memory_warning_if_necessary", lambda: False)
    monkeypatch.setattr(real_time_playback.Tts, "clear_continuation", lambda: None)
    monkeypatch.setattr(
        real_time_playback.Tts, "get_instance",
        lambda: SimpleNamespace(get_warning_issues=lambda project: None),
    )
    # Run warnings (incl. the voice-variation note) read real project voice data
    monkeypatch.setattr(real_time_playback, "get_run_warnings", lambda *args: [])
    monkeypatch.setattr(real_time_playback.GenerateUtil, "print_batch_heading", lambda indices, voice_index=None: None)

    state = cast(
        State,
        SimpleNamespace(
            prefs=SimpleNamespace(stt_variant=SttVariant.DISABLED),
            project=SimpleNamespace(
                max_retries=0,
                limit_silence_gaps=False,
                limit_silence_gaps_duration=0.5,
                use_break_sound_effect=False,
                reason_pauses=SimpleNamespace(get_pause_for=lambda reason: 0.5),
                get_high_shelf=lambda: 0,
                realtime_save=True,
                get_tts_model_type=lambda: TtsModelType.require_by_id("chatterbox_local"),
                voice_references=[
                    Voice(f"voice{i}.flac", room_reverb=(
                        room_reverb if other_voice_reverb is None or i == (voice_index if 0 <= voice_index < voice_count else 0)
                        else other_voice_reverb
                    ))
                    for i in range(voice_count)
                ],
                room_tone=RoomTone.HIGH,
                room_reverb_stereo=room_reverb_stereo,
            ),
        ),
    )
    phrase_groups = [
        PhraseGroup([Phrase(f"Segment {i}.", Reason.SENTENCE)], voice_index=voice_index) for i in range(2)
    ]
    if ending == "one":
        phrase_groups = phrase_groups[:1]
    elif ending == "empty":
        phrase_groups = []
    Interrupts().clear()
    continue_event = threading.Event()
    continue_event.set()
    received = []
    with RealTimePlaybackEvents.using_sink(received.append):
        result = real_time_playback.start(state, phrase_groups, None, continue_event=continue_event)
    if ending == "empty":
        assert not streams and not saved and not spatial_processors and not generated_indices
        assert result.status is real_time_playback.RealTimePlaybackRunStatus.COMPLETED
        return
    assert generated_indices == ([0] if ending in ("one", "cancelled_first") else [0, 1])
    assert result.status is (
        real_time_playback.RealTimePlaybackRunStatus.ABORTED
        if ending in ("cancelled", "cancelled_first")
        else real_time_playback.RealTimePlaybackRunStatus.COMPLETED
    )

    stream = streams[-1]
    assert stream.channels == expected_channels
    # Each segment's narration feeds the speaking voice's room
    assert speech_rooms and set(speech_rooms) == {room_reverb}
    assert len(spatial_processors) == 1
    assert spatial_processors[0].channels == expected_channels
    warnings = [i for i, line in enumerate(printed) if "Couldn't open the sound output in stereo" in line]
    if device_stereo_ok:
        assert len(streams) == 1 and not warnings
    else:
        # The failed stereo stream was released, then a mono one opened; the
        # warning is followed by a blank line
        assert [s.channels for s in streams] == [2, 1] and streams[0].shut_downs >= 1
        assert len(warnings) == 1 and printed[warnings[0] + 1] == ""

    played = stream.chunks
    has_reverb = room_reverb is not RoomReverb.OFF
    speech_chunks = 3 if ending == "normal" else (2 if ending in ("failed", "cancelled") else 1)
    assert len(played) == speech_chunks + int(has_reverb)
    assert len(saved) == (2 if ending == "normal" else 1)
    if ending == "normal":
        np.testing.assert_array_equal(saved[0], np.concatenate(played[:2]))
        np.testing.assert_array_equal(saved[1], np.concatenate(played[2:]))
    else:
        np.testing.assert_array_equal(saved[0], np.concatenate(played))
    ranges = [event for event in received if isinstance(event, RealTimePlaybackSegmentText)]
    assert ranges[0].start_sample == 0
    assert ranges[-1].end_sample == sum(map(len, played))
    assert all(a.end_sample == b.start_sample for a, b in zip(ranges, ranges[1:]))
    if has_reverb:
        assert len(played[-1]) > 0
        assert [is_last for _, is_last in bed_calls] == [False] * speech_chunks + [True]
        assert bed_calls[-1][0] == len(played[-1])
        assert np.any(played[-1] != 0)
        np.testing.assert_array_equal(played[-1][-1], np.zeros_like(played[-1][-1]))
        # Draining clears the processor; neither an ordinary boundary nor a
        # second drain can duplicate the final wet frames.
        assert spatial_processors[0].drain_tail() is None
    for chunk in played:
        assert chunk.ndim == (1 if expected_channels == 1 else 2)
        if expected_channels == 2:
            assert chunk.shape[1] == 2
    if ending != "normal":
        assert len(played[0]) == 48000
        if ending in ("failed", "cancelled"):
            assert len(played[1]) == 24000  # Preserve the already submitted pause.
        assert len(saved[-1]) == sum(map(len, played))
        return
    # Post-processed: the room tone bed was added to the silent start of the
    # raw generation
    assert np.any(saved[0][:4800] != 0)
    assert len(saved[0]) == 48000 + 24000
    # With reverb, the segment's tail carries into its pause, far above the bed
    pause_start = played[1][:2400].astype(np.float64)
    pause_rms_db = 10 * np.log10(np.mean(pause_start ** 2))
    if room_reverb is RoomReverb.OFF and not break_effect:
        assert pause_rms_db < -55
    else:
        assert pause_rms_db > -45
    if room_reverb is RoomReverb.OFF and break_effect:
        # An audible break effect remains dry and center-unity, not voice-panned.
        assert np.mean(played[1]) == pytest.approx(.05, abs=.0001)
    if expected_channels == 2:
        # Only active stereo reverb can make app playback stereo; voice slot 2
        # stays centered even in a multi-voice project.
        assert room_reverb_stereo.is_stereo
        # ...requested by some voice's room (the speaking voice's or another's)
        assert has_reverb or other_voice_reverb not in (None, RoomReverb.OFF)
        if has_reverb:
            assert not np.allclose(pause_start[:, 0], pause_start[:, 1])
        else:
            # A dry speaking voice stays centered, even in a stereo run
            np.testing.assert_array_equal(pause_start[:, 0], pause_start[:, 1])
    if warnings:
        assert "voice positioning" in printed[warnings[0]]


@pytest.mark.parametrize(
    ("speaking", "other", "stereo", "channels"),
    [
        # Another voice's room is never fed: the dry voice's pause stays dry
        (RoomReverb.OFF, RoomReverb.MEDIUM_DRR21, RoomReverbStereo.MONO, 1),
        (RoomReverb.MEDIUM_DRR21, RoomReverb.OFF, RoomReverbStereo.MONO, 1),
        # ...but any voice with a room makes the whole run stereo
        (RoomReverb.OFF, RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, 2),
        (RoomReverb.SMALL_DRR22, RoomReverb.MEDIUM_DRR21, RoomReverbStereo.LATE_W100, 2),
    ],
)
def test_realtime_uses_each_segments_voice_room(monkeypatch, speaking, other, stereo, channels) -> None:
    test_realtime_save_writes_exactly_what_was_played(
        monkeypatch, speaking, stereo, True, channels, 2, 1, False, other_voice_reverb=other,
    )


@pytest.mark.parametrize("ending", ["one", "failed", "cancelled", "cancelled_first", "empty"])
def test_realtime_drains_actual_final_segment(monkeypatch, ending: str) -> None:
    # Reuse the exact-playback fixture without multiplying the normal spatial
    # policy matrix. Request stereo but exercise actual mono device fallback.
    test_realtime_save_writes_exactly_what_was_played(
        monkeypatch,
        RoomReverb.MEDIUM_DRR21,
        RoomReverbStereo.ALL_W100,
        False, 1, 2, 1, False,
        ending=ending,
    )
