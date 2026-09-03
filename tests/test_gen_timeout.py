"""Tests for the GEN_TIMEOUT watchdog (gen_timeout_util) and its integration
into the two generation loops (GenerateUtil.generate_files and realtime
generate_full_flow), including the first-gen exemption."""

import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from tts_audiobook_tool import gen_timeout_util, real_time_playback
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationTimedOut
from tts_audiobook_tool.gen_timeout_util import (
    GenTimeoutTracker,
    gen_timeout_scope,
    make_gen_timeout_message,
)
from tts_audiobook_tool.state import State

from generate_files_test_support import StubValidationResult, generate_files_mock_stack


def make_generate_state(num_groups: int = 1, max_retries: int = 2) -> State:
    phrase_groups = [
        PhraseGroup([Phrase(f"Hello world {i}.", Reason.SENTENCE)])
        for i in range(num_groups)
    ]
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        max_retries=max_retries,
        phrase_groups=phrase_groups,
        sound_segments=sound_segments,
        generate_range_string="all",
        save=MagicMock(return_value=""),
    )
    return cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(stt_variant=None, stt_config=None, save_debug_files=False),
        ),
    )


def test_fast_call_does_not_time_out() -> None:
    events: list[GenerationTimedOut] = []

    with GenerationEvents.using_sink(events.append):
        with gen_timeout_scope(timeout_seconds=1.0) as guard:
            time.sleep(0.05)

    assert not guard.did_time_out
    assert events == []


def test_slow_call_times_out_reports_and_emits(capsys) -> None:
    events: list[GenerationTimedOut] = []

    with GenerationEvents.using_sink(events.append):
        with gen_timeout_scope(timeout_seconds=0.2) as guard:
            time.sleep(0.6)

    assert guard.did_time_out
    assert events == [GenerationTimedOut(timeout_seconds=0.2)]
    output = capsys.readouterr().out
    assert "GEN_TIMEOUT" in output
    assert "0.2s" in output
    assert "reset" in output


def test_gen_timeout_scope_reads_gen_timeout_at_call_time(monkeypatch) -> None:
    monkeypatch.setattr(gen_timeout_util, "GEN_TIMEOUT", 0.2)

    with gen_timeout_scope() as guard:
        time.sleep(0.6)

    assert guard.did_time_out


def test_make_gen_timeout_message_cites_the_cap_value() -> None:
    message = make_gen_timeout_message(180)
    assert "GEN_TIMEOUT" in message
    assert "180s" in message
    assert "reset" in message
    assert "was reset" not in message


def test_tracker_exempts_first_gen_only() -> None:
    """The run's first gen (warm-up/download) is untimed; later gens are not."""
    events: list[GenerationTimedOut] = []
    tracker = GenTimeoutTracker()

    with GenerationEvents.using_sink(events.append):
        with tracker.scope(timeout_seconds=0.2) as first_guard:
            time.sleep(0.5)  # would time out if the first gen were armed
        assert not first_guard.did_time_out

        with tracker.scope(timeout_seconds=0.2) as second_guard:
            time.sleep(0.5)

    assert second_guard.did_time_out
    assert events == [GenerationTimedOut(timeout_seconds=0.2)]


def test_generate_files_aborts_after_gen_timeout(monkeypatch, capsys) -> None:
    monkeypatch.setattr(gen_timeout_util, "GEN_TIMEOUT", 0.2)
    state = make_generate_state(num_groups=2)
    calls = {"n": 0}

    # Both gens are slow: the first is exempt (warm-up), the second times out.
    def slow_batch(**_: object) -> list:
        calls["n"] += 1
        time.sleep(0.6)
        return [StubValidationResult(True)]

    with generate_files_mock_stack(slow_batch):
        did_interrupt = GenerateUtil.generate_files(
            state, {0, 1}, batch_size=1, is_regen=False
        )

    assert did_interrupt is True
    assert calls["n"] == 2
    output = capsys.readouterr().out
    # Watchdog feedback fired exactly once (for the second gen), citing the cap...
    assert output.count("GEN_TIMEOUT") == 1
    assert "0.2s" in output
    # ...and the loop aborted rather than processing the late result.
    assert "Aborting generation run" in output


def test_realtime_generate_full_flow_aborts_after_gen_timeout(monkeypatch, capsys) -> None:
    monkeypatch.setattr(gen_timeout_util, "GEN_TIMEOUT", 0.2)
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    state = cast(
        State,
        SimpleNamespace(
            project=SimpleNamespace(max_retries=2, realtime_save=False),
            prefs=SimpleNamespace(stt_variant=None, stt_config=None),
        ),
    )

    clear_calls: list[bool] = []
    monkeypatch.setattr(
        real_time_playback.Tts,
        "clear_continuation",
        lambda: clear_calls.append(True),
    )

    # A warmed tracker (first-gen exemption already consumed elsewhere).
    tracker = GenTimeoutTracker()
    with tracker.scope():
        pass

    def slow_batch(**_: object) -> list:
        time.sleep(0.6)
        return [SimpleNamespace(is_fail=False, get_ui_message_with_extras=lambda: "Passed")]

    with patch.object(
        real_time_playback.GenerateUtil,
        "generate_and_validate_batch",
        side_effect=slow_batch,
    ):
        sound, did_interrupt, _consecutive = real_time_playback.generate_full_flow(
            state,
            [phrase_group],
            0,
            has_runway=False,
            gen_timeout_tracker=tracker,
        )

    assert sound is None
    assert did_interrupt is True
    assert clear_calls
    output = capsys.readouterr().out
    assert "GEN_TIMEOUT" in output
    assert "Aborting real-time playback" in output


def test_realtime_first_gen_is_exempt_from_gen_timeout(monkeypatch) -> None:
    monkeypatch.setattr(gen_timeout_util, "GEN_TIMEOUT", 0.2)
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    state = cast(
        State,
        SimpleNamespace(
            project=SimpleNamespace(max_retries=0, realtime_save=False),
            prefs=SimpleNamespace(stt_variant=None, stt_config=None),
        ),
    )

    sentinel = object()

    def slow_batch(**_: object) -> list:
        time.sleep(0.6)  # exceeds the cap, but this is the run's first gen
        return [
            SimpleNamespace(
                is_fail=False,
                get_ui_message_with_extras=lambda: "Passed",
                sound=sentinel,
            )
        ]

    with patch.object(
        real_time_playback.GenerateUtil,
        "generate_and_validate_batch",
        side_effect=slow_batch,
    ):
        sound, did_interrupt, _consecutive = real_time_playback.generate_full_flow(
            state,
            [phrase_group],
            0,
            has_runway=False,
            gen_timeout_tracker=GenTimeoutTracker(),
        )

    assert sound is sentinel
    assert did_interrupt is False
