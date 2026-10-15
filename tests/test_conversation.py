import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.conversation.conversation import ConversationRuntime
from tts_audiobook_tool.conversation.conversation_types import (
    ChatInputMode,
    ChunkingConfig,
    ResponseResult,
)
from tts_audiobook_tool.conversation.prompt_draft import PromptSubmission
from tts_audiobook_tool.conversation.response_session import ConversationStreamingTts
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class FakeSoundStream:
    def __init__(self, sample_rate: int, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._offset = 0
        self.added: list[np.ndarray] = []

    def add_data(self, data: np.ndarray) -> tuple[int, int]:
        start = self._offset
        self.added.append(data)
        self._offset += len(data)
        return start, self._offset


class FakeCloseResource:
    def __init__(self) -> None:
        self.calls = 0

    def request_interrupt(self) -> None:
        self.calls += 1

    def stop(self) -> None:
        self.calls += 1

    def shut_down(self) -> None:
        self.calls += 1

    def clear(self) -> None:
        self.calls += 1


def test_cancel_initialization_marks_closing_hard_resets_and_cleans_resources(
    monkeypatch,
) -> None:
    state = SimpleNamespace(
        prefs=SimpleNamespace(chat_input_mode=ChatInputMode.TEXT),
        project=SimpleNamespace(),
    )
    runtime = ConversationRuntime(state)  # type: ignore[arg-type]
    response = FakeCloseResource()
    transcriber = FakeCloseResource()
    sound_stream = FakeCloseResource()
    llm = FakeCloseResource()
    runtime.active_response = response  # type: ignore[assignment]
    runtime.transcriber = transcriber  # type: ignore[assignment]
    runtime.sound_stream = sound_stream  # type: ignore[assignment]
    runtime.llm = llm  # type: ignore[assignment]
    runtime.initialized = True
    reset_saw_closing: list[bool] = []

    def fake_reset() -> str:
        reset_saw_closing.append(runtime.closing)
        return "replacement failed"

    monkeypatch.setattr(ModelWorker, "reset", staticmethod(fake_reset))
    monkeypatch.setattr(
        ModelWorker,
        "reset_chat_session_blocking",
        staticmethod(
            lambda **_: (_ for _ in ()).throw(
                AssertionError("ordinary close reset ran")
            )
        ),
    )

    assert runtime.cancel_initialization() == "replacement failed"
    assert reset_saw_closing == [True]
    assert not runtime.initialized
    assert runtime.active_response is None
    assert runtime.transcriber is None
    assert runtime.sound_stream is None
    assert runtime.llm is None
    assert response.calls == 1
    assert transcriber.calls == 1
    assert sound_stream.calls == 1
    assert llm.calls == 1

    # Cancellation and the later unmount close are both idempotent.
    assert runtime.cancel_initialization() == ""
    runtime.close()
    assert reset_saw_closing == [True]


def test_create_response_uses_project_max_words(monkeypatch) -> None:
    import tts_audiobook_tool.conversation.conversation as conversation_module

    project = Project.model_validate({"max_words": 73})
    state = SimpleNamespace(
        prefs=SimpleNamespace(
            chat_input_mode=ChatInputMode.TEXT,
            chat_echo_override=True,
            stt_variant="test",
            stt_config={},
        ),
        project=project,
    )
    runtime = ConversationRuntime(state)  # type: ignore[arg-type]
    runtime.initialized = True
    runtime.sound_stream = object()  # type: ignore[assignment]
    captured: dict[str, object] = {}

    def fake_response_session(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(conversation_module, "ResponseSession", fake_response_session)

    runtime.create_response()

    chunking_config = captured["chunking_config"]
    assert isinstance(chunking_config, ChunkingConfig)
    assert chunking_config.max_words == 73
    # Generated assistant audio must not be sent back through Whisper by
    # default. Text chat does not warm STT, and the redundant output alignment
    # can otherwise load it after the first TTS chunk and exhaust VRAM later.
    assert captured["phrase_stt_enabled"] is False


@pytest.mark.parametrize(
    ("tts_type", "expected"),
    [(TtsModelType.require_by_id("glm_local"), False), (TtsModelType.require_by_id("pocket_local"), True)],
)
def test_create_response_disables_redundant_output_stt_only_for_glm_mic_chat(
    monkeypatch, tts_type: TtsModelType, expected: bool
) -> None:
    import tts_audiobook_tool.conversation.conversation as conversation_module

    project = Project.model_validate({})
    state = SimpleNamespace(
        prefs=SimpleNamespace(
            chat_input_mode=ChatInputMode.MIC_IMMEDIATE,
            chat_echo_override=True,
            stt_variant="test",
            stt_config={},
        ),
        project=project,
    )
    runtime = ConversationRuntime(state)  # type: ignore[arg-type]
    runtime.initialized = True
    runtime.sound_stream = object()  # type: ignore[assignment]
    captured: dict[str, object] = {}

    project.tts_model_type = tts_type.id
    monkeypatch.setattr(
        conversation_module,
        "ResponseSession",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    runtime.create_response()

    assert captured["phrase_stt_enabled"] is expected


class FakeMicTranscriber:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def pause(self) -> None:
        self.log.append("pause")

    def flush(self) -> None:
        self.log.append("flush")

    def resume(self) -> None:
        self.log.append("resume")


class FakeDrainingStream:
    """Output stream whose cleared tail falls silent after N polls."""

    def __init__(self, log: list[str], complete_after_polls: int) -> None:
        self.log = log
        self.polls = 0
        self.complete_after_polls = complete_after_polls
        self.output_latency = 0.0

    @property
    def is_playback_complete(self) -> bool:
        self.polls += 1
        complete = self.polls > self.complete_after_polls
        if complete and "drain_done" not in self.log:
            self.log.append("drain_done")
        return complete


class FakeTurnResponse:
    def __init__(self, interrupted: bool) -> None:
        self.interrupted = interrupted

    def run(self, text: str, user_input_sound: object = None) -> ResponseResult:
        return ResponseResult("partial reply", interrupted=self.interrupted)


def make_mic_runtime(log: list[str], stream: FakeDrainingStream) -> ConversationRuntime:
    state = SimpleNamespace(
        prefs=SimpleNamespace(chat_input_mode=ChatInputMode.MIC_IMMEDIATE),
        project=SimpleNamespace(),
    )
    runtime = ConversationRuntime(state)  # type: ignore[arg-type]
    runtime.transcriber = FakeMicTranscriber(log)  # type: ignore[assignment]
    runtime.sound_stream = stream  # type: ignore[assignment]
    return runtime


def test_interrupted_response_waits_for_output_tail_before_mic_resume(
    monkeypatch,
) -> None:
    import tts_audiobook_tool.conversation.conversation as conversation_module

    monkeypatch.setattr(conversation_module, "OUTPUT_TAIL_SETTLE_MARGIN_S", 0.0)
    log: list[str] = []
    stream = FakeDrainingStream(log, complete_after_polls=1)
    runtime = make_mic_runtime(log, stream)

    result = runtime.run_response(
        FakeTurnResponse(interrupted=True),  # type: ignore[arg-type]
        PromptSubmission("hello"),
    )

    assert result.interrupted is True
    # The mic only reopens after the interrupted playback's tail has drained.
    assert log == ["pause", "drain_done", "flush", "resume"]
    assert stream.polls == 2


def test_completed_response_resumes_mic_without_tail_wait() -> None:
    log: list[str] = []
    stream = FakeDrainingStream(log, complete_after_polls=0)
    runtime = make_mic_runtime(log, stream)

    result = runtime.run_response(
        FakeTurnResponse(interrupted=False),  # type: ignore[arg-type]
        PromptSubmission("hello"),
    )

    assert result.interrupted is False
    # The engine already waited for playback; the existing double flush is
    # kept and no tail drain is added.
    assert log == ["pause", "flush", "flush", "resume"]
    assert stream.polls == 0


def test_stuck_transcriber_pause_fails_turn_but_still_resumes_mic(
    monkeypatch,
) -> None:
    import tts_audiobook_tool.conversation.conversation as conversation_module

    class StuckPauseTranscriber(FakeMicTranscriber):
        def pause(self) -> None:
            self.log.append("pause")
            raise RuntimeError(
                "Transcriber stuck: in-flight STT did not finish within 60s"
            )

    monkeypatch.setattr(conversation_module, "OUTPUT_TAIL_SETTLE_MARGIN_S", 0.0)
    log: list[str] = []
    stream = FakeDrainingStream(log, complete_after_polls=0)
    runtime = make_mic_runtime(log, stream)
    runtime.transcriber = StuckPauseTranscriber(log)  # type: ignore[assignment]
    response = FakeTurnResponse(interrupted=True)
    runtime.active_response = response  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Transcriber stuck"):
        runtime.run_response(
            response,  # type: ignore[arg-type]
            PromptSubmission("hello"),
        )

    # The turn fails loudly, but the response slot is freed and the mic is
    # flushed and resumed, so the session stays usable for the next turn.
    assert log == ["pause", "drain_done", "flush", "resume"]
    assert runtime.active_response is None


def _patch_tts_for_streaming(monkeypatch, sample_rate: int, generate_side_effect):
    spies = SimpleNamespace(synthesis_calls=0)

    def fake_synthesize(_state, _text, _reason, *, on_chunk=None, **_kwargs):
        spies.synthesis_calls += 1
        result = generate_side_effect(on_chunk, lambda: None)
        return (None, result) if isinstance(result, str) else (None, "")

    fake_tts_class = SimpleNamespace(
        get_output_sample_rate=lambda project, instance=None: sample_rate
    )
    monkeypatch.setattr(Tts, "get_model_support", staticmethod(lambda project: fake_tts_class))
    monkeypatch.setattr(ModelWorker, "synthesize_chat_blocking", fake_synthesize)
    return spies


def test_streaming_tts_reports_sample_rate_mismatch(monkeypatch) -> None:
    fake_tts_class = SimpleNamespace(
        get_output_sample_rate=lambda project, instance=None: 48000
    )
    monkeypatch.setattr(Tts, "get_model_support", staticmethod(lambda project: fake_tts_class))
    project = Project.model_validate({})

    stream_range, saved_sound, error = (
        ConversationStreamingTts.generate_to_sound_stream(
            state=SimpleNamespace(project=project),
            text="hello",
            reason=Reason.SENTENCE,
            sound_stream=FakeSoundStream(24000),
            interrupt_requested=threading.Event(),
        )
    )

    assert stream_range is None and saved_sound is None
    assert error is not None and "48000" in error and "24000" in error


def test_streaming_tts_streams_chunks_and_saves_combined_sound(monkeypatch) -> None:
    spies = _patch_tts_for_streaming(
        monkeypatch,
        24000,
        lambda on_chunk, on_end: (
            on_chunk(np.full(100, 0.5, dtype=np.float32)),
            on_chunk(np.full(50, 0.25, dtype=np.float32)),
            on_end(),
            ("streamed",),
        )[-1],
    )
    project = Project.model_validate({})

    stream_range, saved_sound, error = (
        ConversationStreamingTts.generate_to_sound_stream(
            state=SimpleNamespace(project=project),
            text="hello",
            reason=Reason.SENTENCE,
            sound_stream=FakeSoundStream(24000),
            interrupt_requested=threading.Event(),
        )
    )

    assert error is None
    assert stream_range is not None and stream_range[0] == 0
    assert stream_range[1] >= 150
    assert saved_sound is not None and saved_sound.data.size >= 150
    assert spies.synthesis_calls == 1


def test_streaming_tts_propagates_generation_error(monkeypatch) -> None:
    spies = _patch_tts_for_streaming(
        monkeypatch, 24000, lambda on_chunk, on_end: "model exploded"
    )
    project = Project.model_validate({})

    stream_range, saved_sound, error = (
        ConversationStreamingTts.generate_to_sound_stream(
            state=SimpleNamespace(project=project),
            text="hello",
            reason=Reason.SENTENCE,
            sound_stream=FakeSoundStream(24000),
            interrupt_requested=threading.Event(),
        )
    )

    assert (stream_range, saved_sound) == (None, None)
    assert error == "model exploded"
    assert spies.synthesis_calls == 1


def test_streaming_tts_reports_when_no_audio_was_streamed(monkeypatch) -> None:
    spies = _patch_tts_for_streaming(
        monkeypatch, 24000, lambda on_chunk, on_end: ("idle",)
    )
    project = Project.model_validate({})

    stream_range, saved_sound, error = (
        ConversationStreamingTts.generate_to_sound_stream(
            state=SimpleNamespace(project=project),
            text="hello",
            reason=Reason.SENTENCE,
            sound_stream=FakeSoundStream(24000),
            interrupt_requested=threading.Event(),
        )
    )

    assert (stream_range, saved_sound) == (None, None)
    assert error == "No streamed audio output"
    assert spies.synthesis_calls == 1


def test_streaming_tts_applies_room_reverb_in_stereo(monkeypatch) -> None:
    from tts_audiobook_tool.sound.room_reverb import RoomReverbProcessor

    rng = np.random.default_rng(0)
    first = (rng.standard_normal(2400) * 0.1).astype(np.float32)
    second = (rng.standard_normal(1200) * 0.1).astype(np.float32)
    _patch_tts_for_streaming(
        monkeypatch,
        24000,
        lambda on_chunk, on_end: (on_chunk(first), on_chunk(second), ("streamed",))[-1],
    )
    project = Project.model_validate({})
    stream = FakeSoundStream(24000, channels=2)
    reverb = RoomReverbProcessor(24000, "small", 17.0, width=1.0)

    stream_range, saved_sound, error = ConversationStreamingTts.generate_to_sound_stream(
        state=SimpleNamespace(project=project),
        text="hello",
        reason=Reason.SENTENCE,
        sound_stream=stream,  # type: ignore[arg-type]
        interrupt_requested=threading.Event(),
        room_reverb=reverb,
    )

    assert error is None
    # Ranges stay in frames: the speech, then the pause
    assert stream_range == (0, 3600)
    assert [chunk.shape[1] for chunk in stream.added] == [2, 2, 2]
    # Voice centered, reverb differs between the ears; the pause carries the tail
    speech = np.concatenate(stream.added[:2])
    assert not np.allclose(speech[:, 0], speech[:, 1])
    pause = stream.added[2]
    assert np.any(pause != 0) and not np.allclose(pause[:, 0], pause[:, 1])
    assert saved_sound is not None and saved_sound.data.shape == (3600 + len(pause), 2)


def test_streaming_tts_without_reverb_is_unchanged(monkeypatch) -> None:
    data = np.full(100, 0.5, dtype=np.float32)
    _patch_tts_for_streaming(
        monkeypatch, 24000, lambda on_chunk, on_end: (on_chunk(data), ("streamed",))[-1]
    )
    stream = FakeSoundStream(24000)

    ConversationStreamingTts.generate_to_sound_stream(
        state=SimpleNamespace(project=Project.model_validate({})),
        text="hello",
        reason=Reason.SENTENCE,
        sound_stream=stream,  # type: ignore[arg-type]
        interrupt_requested=threading.Event(),
    )

    np.testing.assert_array_equal(stream.added[0], data)
    assert not np.any(stream.added[1])


@pytest.mark.parametrize(
    ("channels", "stereo_ok", "mono_ok", "expected"),
    [
        (2, True, True, [2]),
        (1, True, True, [1]),
        (2, False, True, [2, 1]),
        (2, False, False, [2, 1]),
        (1, True, False, [1]),
    ],
)
def test_open_chat_output_stream_falls_back_to_mono(
    monkeypatch, channels: int, stereo_ok: bool, mono_ok: bool, expected: list[int]
) -> None:
    from tts_audiobook_tool.conversation import conversation

    made = []

    class FakeStream:
        def __init__(self, sample_rate: int, channels: int = 1) -> None:
            self.sample_rate = sample_rate
            self.channels = channels
            self.shut_downs = 0
            made.append(self)

        def start(self) -> bool:
            return stereo_ok if self.channels == 2 else mono_ok

        def shut_down(self) -> None:
            self.shut_downs += 1

    monkeypatch.setattr(conversation, "SoundDeviceStream", FakeStream)

    stream, warning = conversation.open_chat_output_stream(48000, channels)

    assert [s.channels for s in made] == expected
    started = made[-1].start()
    assert (stream is made[-1]) == started and (stream is None) == (not started)
    # Streams that didn't start are released
    assert all(s.shut_downs == (0 if s is stream else 1) for s in made)
    assert (warning == conversation.STEREO_FALLBACK_WARNING) == (expected == [2, 1])
