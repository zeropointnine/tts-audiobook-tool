"""Tests for the per-batch ETA extrapolation in GenerateUtil.generate_files().

A fake clock replaces generate_util's `time` module so batch durations are
deterministic: the mocked generate_and_validate_batch side effect advances
the clock before returning. Captured GenerationProgress events (via the
GenerationEvents sink) expose what the header would render.
"""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from tts_audiobook_tool import generate_util
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import (
    BATCH_DURATION_HISTORY_SIZE,
    ETA_MIN_BATCH_DURATIONS,
    GenerateUtil,
    TtsModelError,
    make_batch_eta_seconds,
)
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationProgress
from tts_audiobook_tool.state import State

from generate_files_test_support import StubValidationResult, generate_files_mock_stack


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_generate_state(num_groups: int, max_retries: int) -> State:
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


def progress_emits(events: list) -> list[GenerationProgress]:
    return [event for event in events if isinstance(event, GenerationProgress)]


def run_generate_files(
    monkeypatch,
    state: State,
    batch_durations: list[float],
    results: list,
) -> list[GenerationProgress]:
    """Runs generate_files over one item per batch with scripted durations.

    `batch_durations[i]` is the clock advance for batch i+1; `results[i]` is
    its (single-item) generate_and_validate_batch return value.
    """
    clock = FakeClock()
    monkeypatch.setattr(generate_util, "time", clock)
    calls = {"n": 0}

    def generate_and_validate_batch(**_: object) -> list:
        i = calls["n"]
        calls["n"] += 1
        clock.advance(batch_durations[i])
        return [results[i]]

    events: list = []
    with (
        generate_files_mock_stack(generate_and_validate_batch),
        GenerationEvents.using_sink(events.append),
    ):
        did_interrupt = GenerateUtil.generate_files(
            state, set(range(len(state.project.phrase_groups))), batch_size=1, is_regen=False
        )

    assert not did_interrupt
    assert calls["n"] == len(batch_durations)
    return progress_emits(events)


def test_make_batch_eta_seconds_requires_minimum_history() -> None:
    assert BATCH_DURATION_HISTORY_SIZE == 20
    assert ETA_MIN_BATCH_DURATIONS == 10

    assert make_batch_eta_seconds([10.0] * (ETA_MIN_BATCH_DURATIONS - 1), 5) is None
    # Available at the minimum sample count; further samples only refine
    # the rolling average.
    assert make_batch_eta_seconds([10.0] * ETA_MIN_BATCH_DURATIONS, 5) == pytest.approx(50.0)
    assert make_batch_eta_seconds([10.0] * BATCH_DURATION_HISTORY_SIZE, 5) == pytest.approx(50.0)
    assert make_batch_eta_seconds([10.0] * ETA_MIN_BATCH_DURATIONS, 0) == pytest.approx(0.0)


def test_history_cap_drops_oldest_duration(monkeypatch) -> None:
    state = make_generate_state(num_groups=24, max_retries=0)
    # Batch 2 is slow (100s); once twenty-one batches have been recorded,
    # the cap drops it from the head of the history.
    durations = [300.0, 100.0] + [10.0] * 22
    results = [StubValidationResult(False)] * 24

    emits = run_generate_files(monkeypatch, state, durations, results)
    post = [e for e in emits if not e.current_indices]
    assert len(post) == 24

    # After batch 22 the history holds batches 3-22 (twenty 10s samples;
    # the 100s batch was capped out). Two batches remain queued: ETA = 20s.
    # Without the cap the average would still include the 100s sample.
    assert post[21].eta_seconds == pytest.approx(20.0)


def test_eta_appears_only_once_minimum_samples_recorded(monkeypatch) -> None:
    state = make_generate_state(num_groups=12, max_retries=0)
    # Batch 1 (300s warm-up) is skipped; batches 2-12 all take 10s.
    durations = [300.0] + [10.0] * 11
    results = [StubValidationResult(False)] * 12

    emits = run_generate_files(monkeypatch, state, durations, results)

    # One pre-batch and one post-batch emit per batch.
    pre = [e for e in emits if e.current_indices]
    post = [e for e in emits if not e.current_indices]
    assert len(pre) == 12
    assert len(post) == 12

    # The skipped warm-up batch leaves 9 samples after batch 10: no ETA yet...
    assert [e.eta_seconds for e in post[:10]] == [None] * 10
    # ...then the tenth sample is recorded (batches 2-11, all 10s; the 300s
    # never entered) and one batch remains queued: ETA = 10s x 1.
    assert post[10].eta_seconds == pytest.approx(10.0)
    assert post[11].eta_seconds == pytest.approx(0.0)

    # The pre-batch snapshot carries the previous batch's ETA so the header
    # does not blank the estimate during inference.
    assert pre[11].eta_seconds == pytest.approx(10.0)


def test_model_error_batch_is_excluded_from_history(monkeypatch) -> None:
    state = make_generate_state(num_groups=13, max_retries=0)
    # Batch 5 returns a model error after "taking" 999s; it must not skew
    # the rolling average.
    durations = [300.0, 10.0, 10.0, 10.0, 999.0] + [10.0] * 8
    results = (
        [StubValidationResult(False)] * 4
        + [TtsModelError("boom")]
        + [StubValidationResult(False)] * 8
    )

    emits = run_generate_files(monkeypatch, state, durations, results)
    post = [e for e in emits if not e.current_indices]
    assert len(post) == 13

    # After batch 12 the history holds ten 10s samples (batch 1 skipped as
    # warm-up, batch 5 excluded as a model error); one batch remains.
    assert post[11].eta_seconds == pytest.approx(10.0)
    assert post[12].eta_seconds == pytest.approx(0.0)
    # The errored item is tallied, not retried (max_retries=0).
    assert post[-1].errored == 1


def test_eta_counts_pending_retry_batches(monkeypatch) -> None:
    state = make_generate_state(num_groups=12, max_retries=1)
    # Batches 1-11 process items 1-11 (item 11 fails validation and is
    # re-added), batch 12 is the retry of item 11, batch 13 is item 12.
    durations = [300.0] + [10.0] * 12
    results = (
        [StubValidationResult(False)] * 10
        + [StubValidationResult(True)]
        + [StubValidationResult(False)] * 2
    )

    emits = run_generate_files(monkeypatch, state, durations, results)
    post = [e for e in emits if not e.current_indices]
    # 12 items + 1 retry batch.
    assert len(post) == 13

    # After the failing batch 11: the queued item-12 round plus one retry
    # batch remain, at a 10s average each.
    assert post[10].eta_seconds == pytest.approx(20.0)
    # After the retry batch: only the queued item-12 round remains.
    assert post[11].eta_seconds == pytest.approx(10.0)
    assert post[12].eta_seconds == pytest.approx(0.0)
    assert post[-1].retries == 1
