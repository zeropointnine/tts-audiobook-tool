"""Generation requests a model worker reset on consecutive TTS errors or OOM."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from tts_audiobook_tool import real_time_playback
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil, Stt, Transcriber, TtsModelError
from tts_audiobook_tool.generation_events import GenerationEvents, ModelUnhealthy
from tts_audiobook_tool.state import State

from generate_files_test_support import generate_files_mock_stack, make_generate_state


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
    output = capsys.readouterr().out
    assert "Too many consecutive TTS model errors" in output


@pytest.mark.parametrize("realtime", [False, True], ids=["files", "realtime"])
@pytest.mark.parametrize(
    ("source", "expected_reason"),
    [
        ("tts", "TTS model ran out of memory while generating line 2"),
        ("whisper", "Whisper ran out of memory while validating line 2"),
        ("unknown", "Ran out of memory while processing line 2"),
    ],
)
def test_oom_reset_retains_source_line_and_error(
    monkeypatch, capsys, realtime: bool, source: str, expected_reason: str
) -> None:
    # Exercise the real validate wrapper: both callers must distinguish a
    # transcriber error string from an identical TTS or untyped error.
    state = make_generate_state(num_groups=2)
    events: list[object] = []
    message = "RuntimeError: CUDA out of memory (failed to allocate 128 MiB)"
    if source == "whisper":
        generated = (Sound(np.zeros(1600, dtype=np.float32), 16000), None, None, 0.1, None, "")
    elif source == "tts":
        generated = TtsModelError(message)
    else:
        generated = message

    monkeypatch.setattr(GenerateUtil, "generate", lambda **_: [generated])
    monkeypatch.setattr(Stt, "should_skip", lambda *_: "")
    monkeypatch.setattr(real_time_playback.Tts, "clear_continuation", lambda: None)

    with patch.object(Transcriber, "transcribe_to_words", return_value=message) as transcribe:
        with GenerationEvents.using_sink(events.append):
            if realtime:
                sound, did_interrupt, consecutive = real_time_playback.generate_full_flow(
                    state, state.project.phrase_groups, 1, has_runway=True,
                    consecutive_model_errors=4,
                )
                assert sound is None
                assert consecutive == (5 if source == "tts" else 0)
            else:
                with generate_files_mock_stack(GenerateUtil.generate_and_validate_batch):
                    did_interrupt = GenerateUtil.generate_files(
                        state, {1}, batch_size=1, is_regen=False
                    )

    assert did_interrupt is True
    assert [e.reason for e in events if isinstance(e, ModelUnhealthy)] == [
        f"{expected_reason}: {message}"
    ]
    output = capsys.readouterr().out
    if source == "whisper":
        transcribe.assert_called_once()
        assert "0 succeeded, 1 failed" in output
    else:
        transcribe.assert_not_called()
        assert "Transcription took" not in output
    assert "Transcribed audio in" not in output


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
