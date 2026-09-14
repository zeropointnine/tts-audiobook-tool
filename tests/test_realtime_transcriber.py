import os
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tts_audiobook_tool.app_types import ConcreteSegment
from tts_audiobook_tool.conversation import realtime_transcriber as module
from tts_audiobook_tool.conversation.realtime_transcriber import (
    RealtimeTranscriber,
    block_sigint_during_stream_start,
)

BLOCKS = 1024  # RealtimeTranscriber.BLOCKSIZE


def loud_block() -> np.ndarray:
    return np.full(BLOCKS, 0.5, dtype=np.float32)


def silent_block() -> np.ndarray:
    return np.zeros(BLOCKS, dtype=np.float32)


def install_whisper(monkeypatch, segments: list) -> list:
    """Patch the worker STT client; returns transcribed arrays."""
    calls: list[np.ndarray] = []

    def transcribe(_prefs, data, **_kwargs):
        calls.append(data)
        return SimpleNamespace(segments=tuple(segments)), ""

    monkeypatch.setattr(
        module.ModelWorker,
        "transcribe_audio_blocking",
        transcribe,
    )
    monkeypatch.setattr(
        module, "Transcriber",
        SimpleNamespace(prepare_sound_for_whisper=lambda sound: sound),
    )
    return calls


def make_transcriber(on_transcription) -> RealtimeTranscriber:
    # Fast, deterministic VAD parameters so the scenarios finish quickly.
    return RealtimeTranscriber(
        SimpleNamespace(),
        on_transcription=on_transcription,
        silence_threshold=0.01,
        silence_duration_s=0.05,  # 800 samples: one silent block
        min_chunk_duration_s=0.01,
        max_chunk_duration_s=0.3,  # 4800 samples
        noise_window_s=0.2,
        pre_speech_pad_s=0.05,  # 800 samples
    )


def run_blocks(rt: RealtimeTranscriber, blocks: list) -> None:
    """Run _processing_loop in a thread fed by `blocks`, then stop it cleanly."""
    thread = threading.Thread(target=rt._processing_loop, daemon=True)
    rt._worker_thread = thread
    thread.start()
    for block in blocks:
        rt._audio_queue.put(block)
    # Wait for the worker to consume every block before stop(), so the stop
    # sentinel is processed after the full buffer is in hand (the loop exits
    # early once _stop_event is set, even with blocks still queued).
    deadline = time.monotonic() + 10.0
    while not rt._audio_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    rt.stop()  # sentinel + join
    assert not thread.is_alive(), "processing loop did not stop"


def test_speech_then_silence_dispatches_chunk(monkeypatch) -> None:
    transcribed = install_whisper(
        monkeypatch, [ConcreteSegment(0.0, 1.0, "hey", [])]
    )
    results: list = []
    rt = make_transcriber(lambda segments, audio: results.append((segments, audio)))

    # two loud blocks (speech), then enough silence to close the utterance
    run_blocks(rt, [loud_block(), loud_block()] + [silent_block()] * 4)

    assert len(transcribed) == 1
    assert len(results) == 1
    segments, audio = results[0]
    assert [s.text for s in segments] == ["hey"]
    # pre-speech pad trims the first block to 800 samples, then two full blocks
    assert len(audio) == 800 + 2 * BLOCKS


def test_silence_alone_never_dispatches(monkeypatch) -> None:
    transcribed = install_whisper(monkeypatch, [ConcreteSegment(0.0, 1.0, "x", [])])
    rt = make_transcriber(lambda segments, audio: None)

    run_blocks(rt, [silent_block()] * 8)

    assert transcribed == []


def test_continuous_speech_dispatches_on_max_chunk_duration(monkeypatch) -> None:
    transcribed = install_whisper(monkeypatch, [ConcreteSegment(0.0, 1.0, "x", [])])
    results: list = []
    rt = make_transcriber(lambda segments, audio: results.append(audio))

    # loud the whole time: buffer grows past max_chunk (4800 samples) at block 5;
    # ending right after keeps the trailing flush from adding a second dispatch
    run_blocks(rt, [loud_block()] * 5)

    assert len(transcribed) == 1
    assert len(results) == 1
    assert len(results[0]) >= 4800


def test_stop_flushes_trailing_speech(monkeypatch) -> None:
    transcribed = install_whisper(monkeypatch, [ConcreteSegment(0.0, 1.0, "hey", [])])
    results: list = []
    rt = make_transcriber(lambda segments, audio: results.append((segments, audio)))

    # speech detected, then stop() before any silence closed the utterance
    run_blocks(rt, [loud_block(), loud_block()])

    assert len(transcribed) == 1
    assert len(results) == 1
    assert results[0][0][0].text == "hey"


def _wait_queue_drained(rt: RealtimeTranscriber, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not rt._audio_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.005)


def test_pause_discards_audio_until_resume(monkeypatch) -> None:
    transcribed = install_whisper(monkeypatch, [ConcreteSegment(0.0, 1.0, "hey", [])])
    results: list = []
    rt = make_transcriber(lambda segments, audio: results.append((segments, audio)))

    thread = threading.Thread(target=rt._processing_loop, daemon=True)
    rt._worker_thread = thread
    thread.start()

    rt.pause()
    assert rt.is_paused

    # Audio captured while paused must be dropped, not transcribed.
    rt._audio_queue.put(loud_block())
    _wait_queue_drained(rt)

    rt.resume()
    assert not rt.is_paused
    _wait_queue_drained(rt)

    # A normal utterance after resume is transcribed.
    for block in [loud_block(), loud_block()] + [silent_block()] * 4:
        rt._audio_queue.put(block)
    _wait_queue_drained(rt)
    rt.stop()
    assert not thread.is_alive()

    assert len(transcribed) == 1
    assert len(results) == 1


def test_pause_times_out_and_raises_when_stt_never_finishes() -> None:
    rt = make_transcriber(lambda segments, audio: None)
    rt.PAUSE_POLL_INTERVAL_S = 0.01
    rt.PAUSE_TIMEOUT_S = 0.03
    rt._transcription_idle.clear()  # simulate an in-flight, wedged STT call

    with pytest.raises(RuntimeError, match="in-flight STT"):
        rt.pause()

    # The mic stays paused after the timeout: audio arriving meanwhile is
    # dropped rather than transcribed by the possibly-still-stuck worker.
    assert rt.is_paused


def test_audio_callback_stops_when_stop_requested(monkeypatch) -> None:
    import sounddevice as sd

    rt = make_transcriber(lambda segments, audio: None)
    rt._stop_event.set()

    with pytest.raises(sd.CallbackStop):
        rt._audio_callback(np.zeros((10, 2), dtype=np.float32), 10, None, None)


def test_audio_callback_ignores_audio_while_paused() -> None:
    rt = make_transcriber(lambda segments, audio: None)
    rt._paused_event.set()

    rt._audio_callback(np.zeros((10, 2), dtype=np.float32), 10, None, None)

    assert rt._audio_queue.empty()


def test_audio_callback_enqueues_first_channel_copy() -> None:
    rt = make_transcriber(lambda segments, audio: None)
    indata = np.arange(20, dtype=np.float32).reshape(10, 2)

    rt._audio_callback(indata, 10, None, None)

    (queued,) = [rt._audio_queue.get_nowait()]
    assert queued.shape == (10,)
    assert np.array_equal(queued, indata[:, 0])
    indata[0, 0] = 99.0  # the queued copy must not track the source
    assert queued[0] == 0.0
    assert rt._audio_queue.empty()


def test_audio_callback_publishes_clamped_rms_dbfs_and_flush_resets_it() -> None:
    rt = make_transcriber(lambda segments, audio: None)
    indata = np.full((BLOCKS, 1), 0.5, dtype=np.float32)

    rt._audio_callback(indata, BLOCKS, None, None)

    assert rt.latest_input_level_db == pytest.approx(-6.0206, abs=0.001)
    rt.flush()
    assert rt.latest_input_level_db is None

    rt._audio_callback(np.zeros((BLOCKS, 1), dtype=np.float32), BLOCKS, None, None)
    assert rt.latest_input_level_db == -99.0
    rt._audio_callback(np.full((BLOCKS, 1), 2.0, dtype=np.float32), BLOCKS, None, None)
    assert rt.latest_input_level_db == 0.0


def test_level_updates_while_processing_thread_is_blocked_in_stt(monkeypatch) -> None:
    entered_stt = threading.Event()
    release_stt = threading.Event()

    def transcribe(_prefs, _data, **_kwargs):
        entered_stt.set()
        assert release_stt.wait(5.0)
        return SimpleNamespace(segments=()), ""

    monkeypatch.setattr(module.ModelWorker, "transcribe_audio_blocking", transcribe)
    monkeypatch.setattr(
        module,
        "Transcriber",
        SimpleNamespace(prepare_sound_for_whisper=lambda sound: sound),
    )
    rt = make_transcriber(lambda segments, audio: None)
    thread = threading.Thread(target=rt._processing_loop, daemon=True)
    rt._worker_thread = thread
    thread.start()
    rt._audio_queue.put(loud_block())
    rt._audio_queue.put(silent_block())

    assert entered_stt.wait(5.0)
    rt._audio_callback(
        np.full((BLOCKS, 1), 0.25, dtype=np.float32), BLOCKS, None, None
    )
    assert rt.latest_input_level_db == pytest.approx(-12.0412, abs=0.001)

    release_stt.set()
    _wait_queue_drained(rt)
    rt.stop()
    assert not thread.is_alive()


@pytest.mark.skipif(os.name != "posix", reason="pthread_sigmask is POSIX-only")
def test_block_sigint_masks_during_and_restores_after() -> None:
    # pthread_sigmask has no pure peek mode: read the current mask via a
    # no-op set (set to empty, immediately restore) and return it.
    def peek_mask() -> set:
        current = signal.pthread_sigmask(signal.SIG_SETMASK, set())
        signal.pthread_sigmask(signal.SIG_SETMASK, current)
        return current

    baseline = peek_mask()

    with block_sigint_during_stream_start():
        masked = peek_mask()
        assert signal.SIGINT in masked, "SIGINT should be blocked inside the context"

    restored = peek_mask()
    assert restored == baseline
    assert signal.SIGINT not in restored