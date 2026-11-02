from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tts_audiobook_tool.app_support.interrupts import Interrupts
from audio_fixture_support import assert_no_handles, write_sine_wav
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


def test_cancellation_while_reading_next_chunk_discards_partial_transcript(monkeypatch) -> None:
    def chunks():
        yield np.zeros(16, dtype=np.float32)
        yield np.zeros(16, dtype=np.float32)
        # The stream itself notices the cancellation while reading, so the
        # chunk loop ends without ever reaching its interrupt check.
        Interrupts()._flag = True

    monkeypatch.setattr(
        enhance_alignment, "_stream_audio_with_overlap", lambda **_kwargs: chunks()
    )
    monkeypatch.setattr(
        enhance_alignment.AudioMetaUtil, "get_audio_duration", lambda _path: None
    )
    monkeypatch.setattr(
        enhance_alignment.ModelWorker,
        "transcribe_audio_blocking",
        lambda *_args, **_kwargs: (SimpleNamespace(segments=()), ""),
    )

    assert enhance_alignment._transcribe_stream_with_overlap("book.mp3", object(), "en") is None  # type: ignore[arg-type]
    assert Interrupts()._mode == ""


def _write_ramp_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Mono 16 kHz WAV whose sample values encode their own position; returns the samples as decoded."""
    import soundfile as sf

    count = round(duration_s * sample_rate)
    values = ((np.arange(count) % 20000) / 20000 - 0.5).astype(np.float32)
    sf.write(str(path), values, sample_rate, subtype="FLOAT")
    return values


@pytest.mark.no_ffmpeg_exe
def test_chunk_stream_windows_overlap_and_keep_their_offsets(tmp_path: Path) -> None:
    # Window i must start exactly i * (chunk - overlap) seconds into the file,
    # since transcribed word times are offset by that amount; the last window
    # holds the remainder and the bare overlap tail is not repeated.
    source = tmp_path / "Hörbuch.wav"
    values = _write_ramp_wav(source, 62.0)

    windows = list(enhance_alignment._stream_audio_with_overlap(str(source), chunk_duration=30, overlap_duration=5))

    stride = 25 * 16000
    assert [len(window) for window in windows] == [30 * 16000, 30 * 16000, 12 * 16000]
    for index, window in enumerate(windows):
        assert window.dtype == np.float32
        np.testing.assert_array_equal(window, values[index * stride:index * stride + len(window)])
    assert_no_handles(source)


@pytest.mark.no_ffmpeg_exe
def test_chunk_stream_ending_on_a_window_boundary_adds_no_overlap_only_window(tmp_path: Path) -> None:
    source = tmp_path / "book.wav"
    _write_ramp_wav(source, 55.0)
    windows = list(enhance_alignment._stream_audio_with_overlap(str(source)))
    assert [len(window) for window in windows] == [30 * 16000, 30 * 16000]


@pytest.mark.no_ffmpeg_exe
def test_chunk_stream_resamples_and_downmixes_compressed_input(tmp_path: Path) -> None:
    from audio_fixture_support import dominant_frequency, sine, write_encoded

    source = tmp_path / "book.mp3"
    tone = sine(500, 44100, 40.0)
    write_encoded(source, np.stack([tone, tone], axis=1), 44100, codec="libmp3lame")
    windows = list(enhance_alignment._stream_audio_with_overlap(str(source)))
    assert [len(window) for window in windows] == [30 * 16000, 15 * 16000]
    assert windows[0].ndim == 1
    assert dominant_frequency(windows[0], 16000) == pytest.approx(500, abs=1)


@pytest.mark.no_ffmpeg_exe
def test_abandoning_chunk_stream_releases_the_source_promptly(tmp_path: Path) -> None:
    """A cancelled transcription stops consuming chunks mid-file; closing the
    generator must release the decoder (and its file handle) right away."""
    source = tmp_path / "book.wav"
    write_sine_wav(source, frequency=440, sample_rate=16000, duration_s=180)

    stream = enhance_alignment._stream_audio_with_overlap(
        str(source), chunk_duration=30, overlap_duration=5
    )
    next(stream)
    next(stream)
    started_at = time.monotonic()
    stream.close()

    assert time.monotonic() - started_at < 2.0
    assert_no_handles(source)
    source.unlink()


@pytest.mark.no_ffmpeg_exe
def test_chunk_stream_cancel_check_stops_before_producing_a_chunk(tmp_path: Path) -> None:
    source = tmp_path / "book.wav"
    write_sine_wav(source, frequency=440, sample_rate=16000, duration_s=180)

    stream = enhance_alignment._stream_audio_with_overlap(
        str(source), cancel_check=lambda: True
    )

    assert next(stream, None) is None
    assert_no_handles(source)


@pytest.mark.no_ffmpeg_exe
def test_chunk_stream_cancelled_midway_yields_no_partial_window(tmp_path: Path) -> None:
    source = tmp_path / "book.wav"
    write_sine_wav(source, frequency=440, sample_rate=16000, duration_s=80)
    cancelled = False
    windows = []
    for window in enhance_alignment._stream_audio_with_overlap(str(source), cancel_check=lambda: cancelled):
        windows.append(window)
        cancelled = True
    # Only the complete first window; the partly read second one is dropped
    assert [len(window) for window in windows] == [30 * 16000]


@pytest.mark.no_ffmpeg_exe
def test_transcription_passes_waveform_arrays_to_the_worker(monkeypatch, tmp_path: Path) -> None:
    # STT input stays array-only: a path would let mlx-whisper shell out to ffmpeg
    source = tmp_path / "book.wav"
    write_sine_wav(source, frequency=440, sample_rate=16000, duration_s=40)
    monkeypatch.setattr(enhance_alignment.AudioMetaUtil, "get_audio_duration", lambda _path: None)
    received = []

    def transcribe(_prefs, audio, **_kwargs):
        received.append(audio)
        return SimpleNamespace(segments=[]), ""

    monkeypatch.setattr(enhance_alignment.ModelWorker, "transcribe_audio_blocking", transcribe)
    assert enhance_alignment._transcribe_stream_with_overlap(str(source), object(), "en") == [[], []]  # type: ignore[arg-type]
    assert len(received) == 3  # warm-up + two windows
    assert all(isinstance(audio, np.ndarray) and audio.dtype == np.float32 for audio in received)


def test_padded_progress_time_right_aligns_minutes_and_seconds() -> None:
    # Single-digit minutes/seconds each add one leading space, so every
    # readout under an hour is six characters wide and the columns line up.
    assert enhance_alignment._padded_progress_time(33 * 60 + 15) == "33m15s"
    assert enhance_alignment._padded_progress_time(33 * 60 + 5) == " 33m5s"
    assert enhance_alignment._padded_progress_time(3 * 60 + 15) == " 3m15s"
    assert enhance_alignment._padded_progress_time(3 * 60 + 5) == "  3m5s"
    assert enhance_alignment._padded_progress_time(9 * 3600 + 5 * 60 + 5) == "  9h5m5s"
    assert enhance_alignment._padded_progress_time(9 * 3600 + 19 * 60 + 15) == "9h19m15s"


def test_transcription_eta_waits_for_samples_and_rounds_to_five_seconds() -> None:
    # No estimate until 10 chunks have completed.
    assert enhance_alignment._transcription_eta(9, 20.0, 225.0, 1000.0) is None
    # 250s of audio took 20s; 750s remain -> 60s.
    assert enhance_alignment._transcription_eta(10, 20.0, 250.0, 1000.0) == 60
    # 750 * 21/250 = 63s -> rounds to 65s.
    assert enhance_alignment._transcription_eta(10, 21.0, 250.0, 1000.0) == 65
