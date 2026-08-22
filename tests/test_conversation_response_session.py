import io
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.conversation.conversation_internals import ResponseSession
from tts_audiobook_tool.conversation.conversation_types import ChunkingConfig
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts


class RecordingUi:
    def __init__(self) -> None:
        self.renders: list[str] = []
        self.prints: list[str] = []

    def render(self, display: str) -> None:
        self.renders.append(display)

    def println(self, text: str = "") -> None:
        self.prints.append(text)

    def clear(self) -> None:
        pass

    def commit_render(self, extra_blank_lines: int = 0) -> None:
        pass

    def wait_idle(self) -> None:
        pass


def make_session(
        ui=None,
        project_payload: dict | None = None,
        phrase_stt_enabled: bool = False,
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
        ui=ui or RecordingUi(),
        llm=LlmSession(
            api_endpoint_url="https://api.openai.com/v1/chat/completions",
            token="t",
            model="m",
        ),
        state=state,
        chunking_config=ChunkingConfig(language_code="en"),
        stt_variant=SimpleNamespace(),
        stt_config=SimpleNamespace(),
        real_stderr=io.StringIO(),
        fd2_redirect_lock=threading.Lock(),
        ctrl_c_requested=threading.Event(),
        sound_stream=sound_stream,
        phrase_stt_enabled=phrase_stt_enabled,
    )
    # Per-turn state normally created in run(); initialize it here so
    # on_chunk and the save paths can be exercised without running a turn.
    session.tts_q = queue.Queue()
    session.tts_buffer = ""
    session.render_buffer = ResponseSession.RESPONSE_PLACEHOLDER
    session.spoken_segments = []
    session.saved_turn_sounds = []
    session.pending_sentences = []
    session.state_lock = threading.Lock()
    session.render_lock = threading.Lock()
    session.last_render_key = None
    session.playback_done = False
    session.worker = None
    session.render_stop = threading.Event()
    session.interrupt_requested = threading.Event()
    session.response_aborted = threading.Event()
    session.llm_content_received = False
    session.first_audio_latency_logged = False
    session.output_turn_tts_started_at = None
    session.output_turn_tts_mode = None
    session.output_turn_tts_preview_text = ""
    return session


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
    assert ResponseSession.make_stable_chunk_preview(
        "One short group of words.", ChunkingConfig(language_code="en")
    ) == ""


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


def test_consume_tts_delta_first_chunk_emits_when_complete_sentences_reach_five_words() -> None:
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


def test_timed_phrases_map_to_sample_ranges_and_pin_last_segment_to_stream_end() -> None:
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
    assert ResponseSession.timed_phrases_to_spoken_segments(
        dummy, [], 16000, 0, 100
    ) == []


def test_make_spoken_segments_first_chunk_is_single_span_even_with_phrase_stt() -> None:
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = True

    segments = session.make_spoken_segments("text", None, 0, 100, is_first_tts_chunk=True)

    assert segments == [("text", 0, 100)]


def test_make_spoken_segments_falls_back_to_single_span_when_phrase_stt_disabled() -> None:
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = False

    segments = session.make_spoken_segments("text", None, 0, 100, is_first_tts_chunk=False)

    assert segments == [("text", 0, 100)]


def test_make_spoken_segments_uses_phrase_segments_when_available() -> None:
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = True
    session.make_phrase_spoken_segments = (
        lambda text, sound, start, end: [("A.", start, start + 50)]
    )

    segments = session.make_spoken_segments("A.", None, 0, 100, is_first_tts_chunk=False)

    assert segments == [("A.", 0, 50)]


def test_make_spoken_segments_falls_back_when_phrase_segments_empty() -> None:
    session = object.__new__(ResponseSession)
    session.phrase_stt_enabled = True
    session.make_phrase_spoken_segments = lambda text, sound, start, end: []

    segments = session.make_spoken_segments("text", None, 0, 100, is_first_tts_chunk=False)

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


def test_rollback_interrupted_llm_user_turn_pops_only_matching_message() -> None:
    session = make_session()
    session.llm.history = [{"role": "user", "content": "interrupted message"}]

    session.rollback_interrupted_llm_user_turn("different message")
    assert len(session.llm.history) == 1

    session.rollback_interrupted_llm_user_turn("interrupted message")
    assert session.llm.history == []


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
        Tts, "get_class", lambda: SimpleNamespace(get_voice_tag=lambda project: "voice1")
    )

    session.save_chat_output_if_needed()

    assert len(saved) == 1
    sound, path = saved[0]
    assert sound.data.size == 30  # both turn sounds concatenated
    assert path.endswith(".flac")
    assert "[chat]" in path
    assert "[testmodel]" in path
    assert "[voice1]" in path