"""Tests for the ModelUnhealthy event: the generation loops emit it when they
abort because the TTS model appears unhealthy (too many consecutive model
errors or an OOM result), so the main process hard-resets the model worker."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool import real_time_playback
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil, TtsModelError
from tts_audiobook_tool.generation_events import GenerationEvents, ModelUnhealthy
from tts_audiobook_tool.state import State

from generate_files_test_support import generate_files_mock_stack


def make_generate_state(num_groups: int = 2, max_retries: int = 2) -> State:
    from unittest.mock import MagicMock

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


def make_realtime_state(max_retries: int = 0) -> State:
    return cast(
        State,
        SimpleNamespace(
            project=SimpleNamespace(max_retries=max_retries, realtime_save=False),
            prefs=SimpleNamespace(stt_variant=None, stt_config=None),
        ),
    )


def test_generate_files_emits_model_unhealthy_after_consecutive_model_errors(
    capsys,
) -> None:
    state = make_generate_state()
    events: list[object] = []

    def failing_batch(**_: object) -> list:
        return [TtsModelError("boom")]

    with generate_files_mock_stack(failing_batch):
        with GenerationEvents.using_sink(events.append):
            did_interrupt = GenerateUtil.generate_files(
                state, {0, 1}, batch_size=1, is_regen=False
            )

    assert did_interrupt is True
    unhealthy = [e for e in events if isinstance(e, ModelUnhealthy)]
    assert len(unhealthy) == 1
    assert isinstance(unhealthy[0], ModelUnhealthy)
    assert "consecutive TTS model errors" in unhealthy[0].reason
    assert "reset" in unhealthy[0].reason
    assert "was reset" not in unhealthy[0].reason
    output = capsys.readouterr().out
    assert "Too many consecutive TTS model errors" in output


def test_generate_files_emits_model_unhealthy_on_oom(capsys) -> None:
    state = make_generate_state(num_groups=1)
    events: list[object] = []

    def oom_batch(**_: object) -> list:
        return [TtsModelError("CUDA out of memory")]

    with generate_files_mock_stack(oom_batch):
        with GenerationEvents.using_sink(events.append):
            did_interrupt = GenerateUtil.generate_files(
                state, {0}, batch_size=1, is_regen=False
            )

    assert did_interrupt is True
    unhealthy = [e for e in events if isinstance(e, ModelUnhealthy)]
    assert len(unhealthy) == 1
    assert isinstance(unhealthy[0], ModelUnhealthy)
    assert "out of memory" in unhealthy[0].reason
    assert "reset" in unhealthy[0].reason
    assert "was reset" not in unhealthy[0].reason


def test_generate_files_no_model_unhealthy_on_user_interrupt_or_success() -> None:
    """A clean pass (and a max-retries item failure short of the consecutive
    cap) must not emit the event."""
    from generate_files_test_support import StubValidationResult

    state = make_generate_state(num_groups=1)
    events: list[object] = []

    def passing_batch(**_: object) -> list:
        return [StubValidationResult(True)]

    with generate_files_mock_stack(passing_batch):
        with GenerationEvents.using_sink(events.append):
            did_interrupt = GenerateUtil.generate_files(
                state, {0}, batch_size=1, is_regen=False
            )

    assert did_interrupt is False
    assert [e for e in events if isinstance(e, ModelUnhealthy)] == []


def test_realtime_generate_full_flow_emits_model_unhealthy_on_consecutive_errors(
    monkeypatch, capsys
) -> None:
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    state = make_realtime_state()
    events: list[object] = []

    monkeypatch.setattr(real_time_playback.Tts, "clear_continuation", lambda: None)

    def failing_batch(**_: object) -> list:
        return [TtsModelError("boom")]

    with patch.object(
        real_time_playback.GenerateUtil,
        "generate_and_validate_batch",
        side_effect=failing_batch,
    ):
        with GenerationEvents.using_sink(events.append):
            sound, did_interrupt, consecutive = real_time_playback.generate_full_flow(
                state,
                [phrase_group],
                0,
                has_runway=False,
                consecutive_model_errors=4,
                max_consecutive_model_errors=5,
            )

    assert sound is None
    assert did_interrupt is True
    assert consecutive == 5
    unhealthy = [e for e in events if isinstance(e, ModelUnhealthy)]
    assert len(unhealthy) == 1
    assert isinstance(unhealthy[0], ModelUnhealthy)
    assert "consecutive TTS model errors" in unhealthy[0].reason
    output = capsys.readouterr().out
    assert "Too many consecutive TTS model errors" in output


def test_realtime_generate_full_flow_emits_model_unhealthy_on_oom(
    monkeypatch, capsys
) -> None:
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    state = make_realtime_state()
    events: list[object] = []

    monkeypatch.setattr(real_time_playback.Tts, "clear_continuation", lambda: None)

    def oom_batch(**_: object) -> list:
        return [TtsModelError("CUDA out of memory")]

    with patch.object(
        real_time_playback.GenerateUtil,
        "generate_and_validate_batch",
        side_effect=oom_batch,
    ):
        with GenerationEvents.using_sink(events.append):
            sound, did_interrupt, _consecutive = real_time_playback.generate_full_flow(
                state,
                [phrase_group],
                0,
                has_runway=False,
            )

    assert sound is None
    assert did_interrupt is True
    unhealthy = [e for e in events if isinstance(e, ModelUnhealthy)]
    assert len(unhealthy) == 1
    assert isinstance(unhealthy[0], ModelUnhealthy)
    assert "out of memory" in unhealthy[0].reason
