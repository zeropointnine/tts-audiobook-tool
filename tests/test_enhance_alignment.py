from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import ConcreteWord, SttVariant
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.enhance import enhance_alignment


def word(start: float, value: str) -> ConcreteWord:
    return ConcreteWord(start, start + 0.5, value, 1.0)


def test_alignment_progress_counts_orphans_across_sections(capsys) -> None:
    state = enhance_alignment.AlignmentState()
    words = [word(0, "alpha"), word(1, "beta"), word(2, "gamma")]
    first, state, interrupted = enhance_alignment.align_phrases_with_state(
        [Phrase("Alpha", Reason.SENTENCE), Phrase("", Reason.SENTENCE)],
        words, state, total_lines=3,
    )
    second, state, interrupted = enhance_alignment.align_phrases_with_state(
        [Phrase("Beta", Reason.SENTENCE)],
        words, state, line_offset=2, total_lines=3,
    )

    assert not interrupted
    assert len(first + second) == 3
    assert state.orphan_count == 1
    output = capsys.readouterr().out
    assert "Aligning 1/3 (orphans: 0)" in output
    assert "Aligning 2/3 (orphans: 1)" in output
    assert "Aligning 3/3 (orphans: 1)" in output


def test_alignment_progress_counts_unmatched_remainder(capsys) -> None:
    state = enhance_alignment.AlignmentState()
    timed, state, interrupted = enhance_alignment.align_phrases_with_state(
        [Phrase("Alpha", Reason.SENTENCE), Phrase("Two", Reason.SENTENCE), Phrase("Three", Reason.SENTENCE)],
        [word(0, "alpha")], state,
    )

    assert not interrupted
    assert len(timed) == 3
    assert state.orphan_count == 2
    assert "Aligning 3/3 (orphans: 2)" in capsys.readouterr().out


def test_overlapping_transcript_chunks_are_stitched_without_duplicate_tail() -> None:
    chunks = [
        [word(0, "a"), word(1, "b"), word(2, "c-old")],
        [word(1, "b-overlap"), word(2, "c"), word(3, "d")],
    ]

    result = enhance_alignment._stitch_transcripts(chunks)

    assert [item.word for item in result] == ["a", "b", "c", "d"]


@pytest.mark.parametrize(("language_code", "expected_language"), [("es", "es"), ("", None)])
@pytest.mark.parametrize(
    ("preferred_variant", "expected_id"),
    [
        (SttVariant.LARGE_V3_TURBO, SttVariant.LARGE_V3_TURBO.id),
        (SttVariant.LARGE_V3, SttVariant.LARGE_V3.id),
        (None, SttVariant.LARGE_V3.id),
    ],
)
def test_transcription_uses_selected_model_and_project_language(
    monkeypatch, preferred_variant, expected_id, language_code, expected_language
) -> None:
    monkeypatch.setattr(
        enhance_alignment, "_stream_audio_with_overlap",
        lambda **_kwargs: iter([np.zeros(16, dtype=np.float32)]),
    )
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil, "get_audio_duration", lambda _path: None
    )
    selected = []

    def transcribe(*_args, **kwargs):
        selected.append((kwargs["stt_variant_id"], kwargs["language"]))
        return SimpleNamespace(segments=[]), ""

    monkeypatch.setattr(
        enhance_alignment.ModelWorker, "transcribe_audio_blocking", transcribe
    )
    prefs = SimpleNamespace(stt_variant=preferred_variant)
    assert enhance_alignment._transcribe_stream_with_overlap("book.mp3", prefs, language_code) == [[]]  # type: ignore[arg-type]
    # The first call is the silent warm-up, the second is the real chunk.
    assert selected == [(expected_id, expected_language), (expected_id, expected_language)]


def test_disabled_stt_does_not_silently_select_large_v3(monkeypatch) -> None:
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil, "get_audio_duration",
        lambda _path: pytest.fail("decoded audio with STT disabled"),
    )
    with pytest.raises(ValueError, match="disabled in preferences"):
        enhance_alignment._transcribe_stream_with_overlap(
            "book.mp3", SimpleNamespace(stt_variant=SttVariant.DISABLED), "en"  # type: ignore[arg-type]
        )


def test_chunk_log_includes_elapsed_between_iterations(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        enhance_alignment, "_stream_audio_with_overlap",
        lambda **_kwargs: iter([np.zeros(16, dtype=np.float32)] * 2),
    )
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil, "get_audio_duration", lambda _path: None
    )
    monkeypatch.setattr(
        enhance_alignment.ModelWorker, "transcribe_audio_blocking",
        lambda *_args, **_kwargs: (SimpleNamespace(segments=[]), ""),
    )
    with caplog.at_level(logging.INFO, logger="tts-audiobook-tool"):
        enhance_alignment._transcribe_stream_with_overlap("book.mp3", object(), "en")  # type: ignore[arg-type]

    assert "chunk 0 finished in " in caplog.text
    assert "chunk 1 starting at audio 25.0s; elapsed since previous start=" in caplog.text
    assert "since previous finish=" in caplog.text
    assert "chunk 1 finished in " in caplog.text


def test_stalled_chunk_retries_once_without_losing_prior_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        enhance_alignment, "_stream_audio_with_overlap",
        lambda **_kwargs: iter([np.zeros(16, dtype=np.float32)] * 2),
    )
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil, "get_audio_duration", lambda _path: None
    )
    calls = []

    def transcribe(*_args, **kwargs):
        calls.append(kwargs["timeout_seconds"])
        # Call 1 is the best-effort silent warm-up; make the first real
        # chunk (call 3) stall once so the retry path is exercised.
        if len(calls) == 3:
            return None, enhance_alignment.STT_TRANSCRIPTION_TIMEOUT_ERROR
        return SimpleNamespace(segments=[]), ""

    monkeypatch.setattr(
        enhance_alignment.ModelWorker, "transcribe_audio_blocking", transcribe
    )
    result = enhance_alignment._transcribe_stream_with_overlap("book.mp3", object(), "en")  # type: ignore[arg-type]
    assert result == [[], []]
    # Warm-up, then chunk 0, then the timed-out chunk 1 retry that again
    # gets the full first-chunk budget, then chunk 1's regular budget retry.
    assert calls == [300.0, 300.0, 90.0, 300.0]


def test_stalled_chunk_fails_after_second_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        enhance_alignment, "_stream_audio_with_overlap",
        lambda **_kwargs: iter([np.zeros(16, dtype=np.float32)]),
    )
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil, "get_audio_duration", lambda _path: None
    )
    calls = []

    def timed_out(*_args, **kwargs):
        calls.append(kwargs["timeout_seconds"])
        return None, enhance_alignment.STT_TRANSCRIPTION_TIMEOUT_ERROR

    monkeypatch.setattr(
        enhance_alignment.ModelWorker, "transcribe_audio_blocking", timed_out
    )
    with pytest.raises(RuntimeError, match="at audio 0s: STT transcription timed out"):
        enhance_alignment._transcribe_stream_with_overlap("book.mp3", object(), "en")  # type: ignore[arg-type]
    # Warm-up (best-effort timeout), then chunk 0's two timed-out attempts.
    assert calls == [300.0, 300.0, 300.0]


def test_worker_error_clears_transcription_interrupt_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        enhance_alignment,
        "_stream_audio_with_overlap",
        lambda **_kwargs: iter([np.zeros(16, dtype=np.float32)]),
    )
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil,
        "get_audio_duration",
        lambda _path: None,
    )
    monkeypatch.setattr(
        enhance_alignment.ModelWorker,
        "transcribe_audio_blocking",
        lambda *_args, **_kwargs: (None, "worker failed"),
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        enhance_alignment._transcribe_stream_with_overlap("book.mp3", object(), "en")  # type: ignore[arg-type]

    assert Interrupts()._mode == ""


def test_interrupt_during_worker_transcription_returns_without_error(monkeypatch) -> None:
    monkeypatch.setattr(
        enhance_alignment,
        "_stream_audio_with_overlap",
        lambda **_kwargs: iter([np.zeros(16, dtype=np.float32)]),
    )
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil,
        "get_audio_duration",
        lambda _path: None,
    )

    def cancel_transcription(*_args, **kwargs):
        assert callable(kwargs["cancel_check"])
        Interrupts()._flag = True
        assert kwargs["cancel_check"]()
        return None, "Model worker operation was cancelled"

    monkeypatch.setattr(
        enhance_alignment.ModelWorker, "transcribe_audio_blocking", cancel_transcription
    )
    assert enhance_alignment._transcribe_stream_with_overlap("book.mp3", object(), "en") is None  # type: ignore[arg-type]
    assert Interrupts()._mode == ""
    assert not Interrupts().did_interrupt
