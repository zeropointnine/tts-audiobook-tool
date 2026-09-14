from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.conversation.response_session import ResponseSession
from tts_audiobook_tool.conversation.conversation_types import ChunkingConfig
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts


def make_session(
    project_payload: dict | None = None,
    phrase_stt_enabled: bool = False,
    echo_override: bool = False,
) -> ResponseSession:
    project = Project.model_validate(project_payload or {})
    state = SimpleNamespace(
        prefs=SimpleNamespace(chat_save=False, chat_save_mic=False),
        project=project,
    )
    sound_stream = MagicMock()
    sound_stream.play_position_samples = 0
    sound_stream.is_playback_complete = True
    session = ResponseSession(
        llm=LlmSession(
            api_endpoint_url="https://api.openai.com/v1/chat/completions",
            token="t",
            model="m",
        ),
        state=state,
        chunking_config=ChunkingConfig(language_code="en"),
        stt_variant=SimpleNamespace(),
        stt_config=SimpleNamespace(),
        sound_stream=sound_stream,
        phrase_stt_enabled=phrase_stt_enabled,
        echo_override=echo_override,
    )
    # Per-turn state is created in __init__, so on_chunk, snapshot, and the
    # save paths can be exercised without running a turn.
    return session


def test_snapshot_is_available_before_run_and_detached_from_mutable_state() -> None:
    session = make_session()
    initial = session.snapshot()

    assert initial.spoken_segments == ()
    assert initial.pending_sentences == ()
    assert initial.render_buffer == "..."
    assert initial.play_position_samples == 0
    assert not initial.playback_done
    assert not initial.llm_content_received
    assert not initial.interrupted

    with session.state_lock:
        session.spoken_segments.append(("Spoken.", 0, 100))
        session.pending_sentences.append("Pending.")
        session.render_buffer = "Partial"
        session.llm_content_received = True
    session.sound_stream.play_position_samples = 50
    current = session.snapshot()
    with session.state_lock:
        session.spoken_segments.clear()
        session.pending_sentences.clear()

    assert initial.spoken_segments == ()
    assert current.spoken_segments == (("Spoken.", 0, 100),)
    assert current.pending_sentences == ("Pending.",)
    assert current.render_buffer == "Partial"
    assert current.llm_content_received
    assert current.play_position_samples == 50


@pytest.mark.parametrize("interrupt_before_run", [False, True])
def test_snapshot_lifecycle_and_pre_run_interrupt(monkeypatch, interrupt_before_run) -> None:
    session = make_session()
    reset = MagicMock(return_value="")
    monkeypatch.setattr(ModelWorker, "reset_chat_session_blocking", reset)
    during = []

    def send(_assembled, **kwargs):
        during.append(session.snapshot())
        assert kwargs["interrupt_event"].is_set() == interrupt_before_run

    monkeypatch.setattr(session.llm, "send", send)
    if interrupt_before_run:
        session.request_interrupt()
        session.request_interrupt()
        assert session.snapshot().interrupted
        session.sound_stream.clear_buffer.assert_not_called()

    result = session.run("Hello")
    final = session.snapshot()

    assert len(during) == 1
    assert not during[0].playback_done
    assert during[0].interrupted == interrupt_before_run
    assert final.playback_done
    assert final.interrupted == interrupt_before_run
    assert result.interrupted == interrupt_before_run
    assert session.sound_stream.clear_buffer.called == interrupt_before_run
    assert reset.call_count == 2  # start and one terminal cleanup


def test_echo_override_routes_input_directly_into_response_pipeline(monkeypatch) -> None:
    session = make_session(echo_override=True)
    monkeypatch.setattr(
        ModelWorker, "reset_chat_session_blocking", MagicMock(return_value="")
    )
    llm = session.llm
    assert llm is not None
    send = MagicMock(side_effect=AssertionError("LLM request attempted"))
    monkeypatch.setattr(llm, "send", send)

    def consume_without_tts() -> None:
        while True:
            item = session.tts_q.get()
            if item is None:
                return
            text, _reason = item
            with session.state_lock:
                if session.pending_sentences and session.pending_sentences[0] == text:
                    session.pending_sentences.pop(0)
                session.spoken_segments.append((text, 0, 0))

    monkeypatch.setattr(session, "tts_worker", consume_without_tts)

    result = session.run("Echo this exactly.")

    assert result.text == "Echo this exactly."
    assert session.snapshot().llm_content_received
    assert session.snapshot().text == "Echo this exactly."
    send.assert_not_called()


@pytest.mark.parametrize("failure", ["llm", "base_exception", "startup", "save"])
def test_terminal_reset_is_engine_owned_on_failures(monkeypatch, failure) -> None:
    session = make_session()
    reset = MagicMock(return_value="")
    monkeypatch.setattr(ModelWorker, "reset_chat_session_blocking", reset)

    def fail(*_args, **_kwargs):
        if failure == "base_exception":
            raise KeyboardInterrupt()
        raise RuntimeError("test failure")

    monkeypatch.setattr(session.llm, "send", lambda *_args, **_kwargs: None)
    if failure in ("llm", "base_exception"):
        monkeypatch.setattr(session.llm, "send", fail)
    elif failure == "startup":
        monkeypatch.setattr("threading.Thread.start", fail)
    else:
        monkeypatch.setattr(session, "save_chat_output_if_needed", fail)

    if failure == "llm":
        session.run("Hello")  # ordinary LLM errors are reported, not raised
    else:
        expected = KeyboardInterrupt if failure == "base_exception" else RuntimeError
        with pytest.raises(expected):
            session.run("Hello")

    assert reset.call_count == 2
    assert session.snapshot().playback_done
    assert session.worker is None or not session.worker.is_alive()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["interrupt", "interrupt_exception", "recovery"])
def test_tts_reset_ordering(monkeypatch, mode, streaming) -> None:
    session = make_session()
    session.project.streaming_chat = streaming
    operations = []
    monkeypatch.setattr(Tts, "get_info", lambda: SimpleNamespace(can_stream=streaming))

    def reset(**_kwargs):
        operations.append("reset")
        return ""

    def send(*_args, **_kwargs):
        session.tts_q.put(("First.", Reason.SENTENCE))
        session.tts_q.put(("Second.", Reason.SENTENCE))

    def synthesize(*_args, **_kwargs):
        operations.append("synthesize")
        if mode.startswith("interrupt"):
            session.request_interrupt()
            if mode == "interrupt_exception":
                raise RuntimeError("interrupted")
            return (None, None, "interrupted") if streaming else (None, "interrupted")
        if streaming:
            return None, None, "recoverable failure"
        raise RuntimeError("recoverable failure")

    monkeypatch.setattr(ModelWorker, "reset_chat_session_blocking", reset)
    monkeypatch.setattr(ModelWorker, "synthesize_chat_blocking", synthesize)
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation_internals."
        "ConversationStreamingTts.generate_to_sound_stream",
        synthesize,
    )
    monkeypatch.setattr(session.llm, "send", send)
    session.run("Hello")

    if mode == "recovery":
        assert operations == ["reset", "synthesize", "reset", "synthesize", "reset", "reset"]
    else:
        assert operations == ["reset", "synthesize", "reset"]
    assert session.worker is not None and not session.worker.is_alive()


def test_settle_waits_for_worker_before_clearing_playback() -> None:
    session = make_session()
    worker = MagicMock()
    session.worker = worker
    order = []
    worker.join.side_effect = lambda: order.append("joined")
    session.sound_stream.clear_buffer.side_effect = lambda: order.append("cleared")

    session._settle_interrupted_turn()

    worker.join.assert_called_once_with()
    assert order == ["joined", "cleared"]


def test_make_display_text_normalizes_line_endings() -> None:
    assert ResponseSession.make_display_text("a\r\nb\rc") == "a\nb\nc"
    assert ResponseSession.make_display_text("a\nb") == "a\nb"


def test_extract_complete_chunks_single_sentence_emits_nothing() -> None:
    chunks, remainder = ResponseSession.extract_complete_chunks(
        "Hi there.", ChunkingConfig(language_code="en")
    )

    assert chunks == []
    assert remainder == "Hi there."


def test_extract_complete_chunks_splits_off_complete_sentences() -> None:
    chunks, remainder = ResponseSession.extract_complete_chunks(
        "Hello there my good friend. How are you doing today?",
        ChunkingConfig(language_code="en"),
    )

    assert len(chunks) == 1
    assert chunks[0][0].strip() == "Hello there my good friend."
    assert chunks[0][1] is Reason.SENTENCE
    assert remainder == "How are you doing today?"


def test_make_stable_chunk_preview_empty_for_single_chunk() -> None:
    assert (
        ResponseSession.make_stable_chunk_preview(
            "One short group of words.", ChunkingConfig(language_code="en")
        )
        == ""
    )


def test_make_stable_chunk_preview_returns_first_chunk_for_long_text() -> None:
    text = " ".join(f"word{i}" for i in range(80)) + "."
    config = ChunkingConfig(language_code="en")

    preview = ResponseSession.make_stable_chunk_preview(text, config)

    assert 0 < len(preview) < len(text)


def test_consume_tts_delta_emits_complete_sentence_keeps_remainder() -> None:
    to_send, next_buffer, render = ResponseSession.consume_tts_delta(
        tts_buffer="Hello there my good friend. ",
        delta="How are you doing?",
        config=ChunkingConfig(language_code="en"),
        has_pending_sentences=True,  # not the first chunk of the stream
        has_spoken_segments=True,
    )

    assert len(to_send) == 1
    assert to_send[0][0].strip() == "Hello there my good friend."
    assert next_buffer == "How are you doing?"
    assert render == ""


def test_consume_tts_delta_holds_short_first_chunk(monkeypatch) -> None:
    # The "toask" guard: an early boundary before 5 words is held back.
    to_send, next_buffer, render = ResponseSession.consume_tts_delta(
        tts_buffer="To ask.",
        delta="",
        config=ChunkingConfig(language_code="en"),
        has_pending_sentences=False,
        has_spoken_segments=False,
    )

    assert to_send == []
    assert next_buffer == "To ask."


def test_consume_tts_delta_holds_first_chunks_until_five_words() -> None:
    # "To ask." (2 words) is the only complete sentence; the trailing
    # "Can you" stays as the remainder. 2 < 5, so the boundary is held.
    to_send, next_buffer, _ = ResponseSession.consume_tts_delta(
        tts_buffer="To ask. Can you",
        delta="",
        config=ChunkingConfig(language_code="en"),
        has_pending_sentences=False,
        has_spoken_segments=False,
    )

    assert to_send == []
    assert next_buffer == "To ask. Can you"


def test_consume_tts_delta_first_chunk_emits_when_complete_sentences_reach_five_words() -> (
    None
):
    # "To ask." (2) + "Can you help me please." (5) = 7 words >= 5: emit.
    to_send, next_buffer, render = ResponseSession.consume_tts_delta(
        tts_buffer="To ask. Can you help me please.",
        delta=" Now",
        config=ChunkingConfig(language_code="en"),
        has_pending_sentences=False,
        has_spoken_segments=False,
    )

    assert len(to_send) == 1
    assert to_send[0][0].strip() == "To ask. Can you help me please."
    assert next_buffer == "Now"
    assert render == ""


def test_consume_tts_delta_first_chunk_latency_split_at_phrase_boundary() -> None:
    to_send, next_buffer, render = ResponseSession.consume_tts_delta(
        tts_buffer="Hi there, how are you, doing today?",
        delta="",
        config=ChunkingConfig(language_code="en"),
        has_pending_sentences=False,
        has_spoken_segments=False,
        allow_first_chunk_latency_split=True,
    )

    assert len(to_send) == 1
    assert to_send[0][0] == "Hi there, how are you, "
    assert isinstance(to_send[0][1], Reason)
    assert next_buffer == "doing today?"
    assert render == ""


def test_consume_tts_delta_no_latency_split_when_streaming_tts() -> None:
    to_send, next_buffer, _ = ResponseSession.consume_tts_delta(
        tts_buffer="Hi there, how are you, doing today?",
        delta="",
        config=ChunkingConfig(language_code="en"),
        has_pending_sentences=False,
        has_spoken_segments=False,
        allow_first_chunk_latency_split=False,
    )

    assert to_send == []
    assert next_buffer == "Hi there, how are you, doing today?"


def test_timed_phrases_map_to_sample_ranges_and_pin_last_segment_to_stream_end() -> (
    None
):
    dummy = object()
    timed = [
        TimedPhrase(text="A", time_start=0.0, time_end=0.05),
        TimedPhrase(text="B", time_start=0.05, time_end=0.1),
    ]

    segments = ResponseSession.timed_phrases_to_spoken_segments(
        dummy, timed, sample_rate=16000, stream_start=100, stream_end=1000
    )

    assert segments == [("A", 100, 900), ("B", 900, 1000)]


def test_timed_phrases_clamp_out_of_range_and_skip_degenerate_spans() -> None:
    dummy = object()
    timed = [
        TimedPhrase(text="A", time_start=0.0, time_end=2.0),  # ends beyond stream_end
        TimedPhrase(text="  ", time_start=0.0, time_end=0.05),  # blank: skipped
        TimedPhrase(text="C", time_start=0.0, time_end=0.0),  # zero length: skipped
    ]

    segments = ResponseSession.timed_phrases_to_spoken_segments(
        dummy, timed, sample_rate=16000, stream_start=0, stream_end=100
    )

    assert segments == [("A", 0, 100)]


def test_timed_phrases_empty_input() -> None:
    dummy = object()
    assert (
        ResponseSession.timed_phrases_to_spoken_segments(dummy, [], 16000, 0, 100) == []
    )


def test_make_spoken_segments_first_chunk_is_single_span_even_with_phrase_stt() -> None:
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = True

    segments = session.make_spoken_segments(
        "text", None, 0, 100, is_first_tts_chunk=True
    )

    assert segments == [("text", 0, 100)]


def test_make_spoken_segments_falls_back_to_single_span_when_phrase_stt_disabled() -> (
    None
):
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = False

    segments = session.make_spoken_segments(
        "text", None, 0, 100, is_first_tts_chunk=False
    )

    assert segments == [("text", 0, 100)]


def test_make_spoken_segments_uses_phrase_segments_when_available() -> None:
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = True
    session.make_phrase_spoken_segments = lambda text, sound, start, end: [
        ("A.", start, start + 50)
    ]

    segments = session.make_spoken_segments(
        "A.", None, 0, 100, is_first_tts_chunk=False
    )

    assert segments == [("A.", 0, 50)]


def test_make_spoken_segments_falls_back_when_phrase_segments_empty() -> None:
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = True
    session.make_phrase_spoken_segments = lambda text, sound, start, end: []

    segments = session.make_spoken_segments(
        "text", None, 0, 100, is_first_tts_chunk=False
    )

    assert segments == [("text", 0, 100)]


def test_on_chunk_queues_complete_sentence_to_tts_queue(monkeypatch) -> None:
    monkeypatch.setattr(Tts, "get_info", lambda: SimpleNamespace(can_stream=False))
    session = make_session()

    session.on_chunk("Hello there my good friend. Hi again.")

    chunk = session.tts_q.get_nowait()
    assert chunk[0].strip() == "Hello there my good friend."
    assert session.tts_q.empty()
    assert session.llm_content_received is True


def test_on_chunk_is_noop_when_interrupt_requested(monkeypatch) -> None:
    monkeypatch.setattr(Tts, "get_info", lambda: SimpleNamespace(can_stream=False))
    session = make_session()
    session.interrupt_requested.set()

    session.on_chunk("Hello there my good friend. Hi again.")

    assert session.tts_q.empty()
    assert session.llm_content_received is False


@pytest.mark.parametrize("synthesis_raises", [False, True])
def test_interrupted_active_tts_sentence_remains_in_response_snapshot(
    monkeypatch, synthesis_raises: bool
) -> None:
    monkeypatch.setattr(Tts, "get_info", lambda: SimpleNamespace(can_stream=False))
    monkeypatch.setattr(
        ModelWorker,
        "reset_chat_session_blocking",
        lambda **_kwargs: "",
    )
    session = make_session()
    text = "This sentence was visible while its audio was being generated."
    session.pending_sentences = [text]
    session.tts_q.put((text, Reason.SENTENCE))
    session.tts_q.put(None)

    def interrupt_during_synthesis(*_args, **_kwargs):
        session.interrupt_requested.set()
        if synthesis_raises:
            raise RuntimeError("cancelled synthesis")
        return None, ""

    monkeypatch.setattr(
        ModelWorker,
        "synthesize_chat_blocking",
        interrupt_during_synthesis,
    )

    session.tts_worker()

    assert session.pending_sentences == [text]
    assert session.snapshot().text == text


def test_interrupt_signal_keeps_buffer_settle_clears_it() -> None:
    # The signal alone must not cut playback: a turn with TTS inference in
    # flight cannot converge until that inference finishes, so the session
    # would wait in silence. Queued audio keeps playing until the engine
    # settles the turn, and the clear happens then.
    session = make_session()

    session.request_interrupt()

    assert session.interrupt_requested.is_set()
    session.sound_stream.clear_buffer.assert_not_called()

    session._settle_interrupted_turn()
    assert session.sound_stream.clear_buffer.call_count == 1

    # Both phases are idempotent.
    session.request_interrupt()
    session._settle_interrupted_turn()
    assert session.sound_stream.clear_buffer.call_count == 2


def test_interrupt_state_exists_before_run() -> None:
    # The per-turn interrupt state is created at construction, so an
    # interrupt requested before run() persists into the turn: the event is
    # set and the poison pill is queued for the not-yet-started worker. The
    # engine's run() must not discard that queue.
    session = make_session()

    session.request_interrupt()

    assert session.interrupt_requested.is_set()
    assert session.tts_q.get_nowait() is None


def test_save_chat_output_skipped_when_disabled(tmp_path) -> None:
    session = make_session()
    session.saved_turn_sounds = [Sound(np.zeros(10, dtype=np.float32), 16000)]

    # chat_save disabled (and no dir_path): returns without saving anything.
    session.save_chat_output_if_needed()


def test_save_chat_output_saves_concatenated_sounds(tmp_path, monkeypatch) -> None:
    import tts_audiobook_tool.conversation.conversation_internals as internals

    session = make_session(project_payload={"dir_path": str(tmp_path)})
    session.prefs.chat_save = True
    session.saved_turn_sounds = [
        Sound(np.zeros(10, dtype=np.float32), 16000),
        Sound(np.zeros(20, dtype=np.float32), 16000),
    ]
    saved: list[tuple] = []

    def fake_save_flac(sound, path):
        saved.append((sound, path))
        return ""

    monkeypatch.setattr(internals.SoundFileUtil, "save_flac", fake_save_flac)
    monkeypatch.setattr(Tts, "get_info", lambda: SimpleNamespace(file_tag="testmodel"))
    monkeypatch.setattr(
        Tts,
        "get_class",
        lambda: SimpleNamespace(get_voice_tag=lambda project: "voice1"),
    )

    session.save_chat_output_if_needed()

    assert len(saved) == 1
    sound, path = saved[0]
    assert sound.data.size == 30  # both turn sounds concatenated
    assert path.endswith(".flac")
    assert "[chat]" in path
    assert "[testmodel]" in path
    assert "[voice1]" in path
