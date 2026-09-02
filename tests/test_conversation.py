import io
import os
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tts_audiobook_tool import app_support, readiness
import tts_audiobook_tool.conversation.conversation as conversation_module
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.conversation.conversation import Conversation
from tts_audiobook_tool.conversation.conversation_internals import (
    ConversationStreamingTts,
    MuteCurrentThreadOutput,
)
from tts_audiobook_tool.conversation.conversation_types import QueuedStream
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts


def make_conversation(
        chat_input_mode: str,
        stt_immediate: bool | None = None,
) -> Conversation:
    state = SimpleNamespace(
        prefs=SimpleNamespace(chat_input_mode=chat_input_mode),
        project=SimpleNamespace(language_code="en"),
    )
    return Conversation(state, stt_immediate=stt_immediate)


def test_init_text_input_mode() -> None:
    conv = make_conversation("text")

    assert conv.is_text_input is True
    assert conv.stt_immediate is False


def test_init_mic_modes_derive_stt_immediate() -> None:
    assert make_conversation("mic_immediate").stt_immediate is True
    assert make_conversation("mic_enter").stt_immediate is False


def test_init_explicit_stt_immediate_overrides_mode() -> None:
    conv = make_conversation("text", stt_immediate=True)

    assert conv.stt_immediate is True


def test_preflight_fails_and_reports_blockers(monkeypatch) -> None:
    feedback: list[str] = []
    monkeypatch.setattr(
        readiness, "get_chat_blockers",
        lambda state: [SimpleNamespace(verbose="need a model", brief="no model")],
    )
    monkeypatch.setattr(conversation_module.ask, "ask_error", feedback.append)

    state = SimpleNamespace(prefs=SimpleNamespace(), project=SimpleNamespace())
    assert Conversation._run_preflight_checks(state) is False
    assert feedback == ["need a model"]


def test_preflight_fails_when_pre_inference_hints_block(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "get_chat_blockers", lambda state: [])
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation.app_hint_util.show_pre_inference_hints",
        lambda prefs, project: False,
    )
    warm_up_spy = []
    monkeypatch.setattr(
        ModelManager, "warm_up_models",
        lambda state, skip_yamnet=False: warm_up_spy.append(True)
        or SimpleNamespace(should_stop=False),
    )

    state = SimpleNamespace(prefs=SimpleNamespace(), project=SimpleNamespace())
    assert Conversation._run_preflight_checks(state) is False
    assert warm_up_spy == []  # no warm-up after hint block


def test_preflight_stops_on_worker_model_error(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "get_chat_blockers", lambda state: [])
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation.app_hint_util.show_pre_inference_hints",
        lambda prefs, project: True,
    )
    monkeypatch.setattr(
        ModelWorker,
        "inspect_tts_blocking",
        lambda state: (None, "oom"),
    )
    feedback: list = []
    monkeypatch.setattr(conversation_module.ask, "ask_error", feedback.append)
    state = SimpleNamespace(prefs=SimpleNamespace(), project=SimpleNamespace())
    assert Conversation._run_preflight_checks(state) is False
    assert feedback == ["oom"]


def test_preflight_fails_on_model_blockers_after_warm_up(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "get_chat_blockers", lambda state: [])
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation.app_hint_util.show_pre_inference_hints",
        lambda prefs, project: True,
    )
    monkeypatch.setattr(
        ModelWorker,
        "inspect_tts_blocking",
        lambda state: (SimpleNamespace(blocking_issues=("bad model detail",)), ""),
    )
    feedback: list = []
    monkeypatch.setattr(conversation_module.ask, "ask_error", feedback.append)

    state = SimpleNamespace(prefs=SimpleNamespace(), project=SimpleNamespace())
    assert Conversation._run_preflight_checks(state) is False
    assert feedback == ["bad model detail"]


def test_preflight_success(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "get_chat_blockers", lambda state: [])
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation.app_hint_util.show_pre_inference_hints",
        lambda prefs, project: True,
    )
    monkeypatch.setattr(
        ModelWorker,
        "inspect_tts_blocking",
        lambda state: (SimpleNamespace(blocking_issues=()), ""),
    )

    state = SimpleNamespace(prefs=SimpleNamespace(), project=SimpleNamespace())
    assert Conversation._run_preflight_checks(state) is True


def _make_started_conversation(chat_input_mode: str = "mic_immediate") -> Conversation:
    conv = make_conversation(chat_input_mode)
    conv.ctrl_c_requested = threading.Event()
    conv.in_response = False
    conv.exiting = False
    return conv


def test_on_sigint_sets_event_while_idle_in_mic_mode() -> None:
    conv = _make_started_conversation("mic_immediate")

    conv.on_sigint()

    assert conv.ctrl_c_requested.is_set()
    assert conv.exiting is False


def test_on_sigint_raises_keyboard_interrupt_during_response() -> None:
    conv = _make_started_conversation("mic_immediate")
    conv.in_response = True

    with pytest.raises(KeyboardInterrupt):
        conv.on_sigint()


def test_on_sigint_raises_keyboard_interrupt_in_text_mode() -> None:
    conv = _make_started_conversation("text")

    with pytest.raises(KeyboardInterrupt):
        conv.on_sigint()


def test_on_sigint_is_noop_when_exiting() -> None:
    conv = _make_started_conversation("text")
    conv.exiting = True

    conv.on_sigint()  # must not raise even in text mode

    assert conv.ctrl_c_requested.is_set()


def test_build_prompt_dispatches_by_input_mode() -> None:
    conv = make_conversation("text")
    conv.build_text_prompt = lambda: "typed prompt"
    assert conv.build_prompt() == "typed prompt"

    conv_mic = make_conversation("mic_immediate")
    conv_mic.prompt_builder = None
    with pytest.raises(RuntimeError, match="prompt builder"):
        conv_mic.build_prompt()

    conv_mic.prompt_builder = SimpleNamespace(build=lambda: "built prompt")
    assert conv_mic.build_prompt() == "built prompt"


def test_build_text_prompt_reads_strips_and_manages_cursor(monkeypatch) -> None:
    conv = make_conversation("text")
    conv.ui = SimpleNamespace(wait_idle=lambda: None)
    conv.real_stdout = io.StringIO()
    monkeypatch.setattr("builtins.input", lambda: "  hi there  \n")

    assert conv.build_text_prompt() == "hi there"
    assert "> " in conv.real_stdout.getvalue()


def test_build_text_prompt_eof_raises_keyboard_interrupt(monkeypatch) -> None:
    conv = make_conversation("text")
    conv.ui = SimpleNamespace(wait_idle=lambda: None)
    conv.real_stdout = io.StringIO()

    def raise_eof() -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    with pytest.raises(KeyboardInterrupt):
        conv.build_text_prompt()


class FakeSoundStream:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._offset = 0
        self.added: list[np.ndarray] = []

    def add_data(self, data: np.ndarray) -> tuple[int, int]:
        start = self._offset
        self.added.append(data)
        self._offset += len(data)
        return start, self._offset


def _patch_tts_for_streaming(monkeypatch, sample_rate: int, generate_side_effect) -> SimpleNamespace:
    spies = SimpleNamespace(synthesis_calls=0)

    def fake_synthesize(_state, _text, _reason, *, on_chunk=None, **_kwargs):
        spies.synthesis_calls += 1
        result = generate_side_effect(on_chunk, lambda: None)
        return (None, result) if isinstance(result, str) else (None, "")

    fake_tts_class = SimpleNamespace(
        get_output_sample_rate=lambda project, instance=None: sample_rate
    )
    monkeypatch.setattr(Tts, "get_class", staticmethod(lambda: fake_tts_class))
    monkeypatch.setattr(ModelWorker, "synthesize_chat_blocking", fake_synthesize)
    return spies


def test_streaming_tts_reports_sample_rate_mismatch(monkeypatch) -> None:
    fake_tts_class = SimpleNamespace(
        get_output_sample_rate=lambda project, instance=None: 48000
    )
    monkeypatch.setattr(Tts, "get_class", staticmethod(lambda: fake_tts_class))
    project = Project.model_validate({})

    def must_not_generate(*args, **kwargs):
        raise AssertionError("generate should not run on sample rate mismatch")

    monkeypatch.setattr(Tts, "generate_using_project", must_not_generate)

    result = ConversationStreamingTts.generate_to_sound_stream(
        state=SimpleNamespace(project=project),
        text="hello",
        reason=Reason.SENTENCE,
        sound_stream=FakeSoundStream(24000),
        interrupt_requested=threading.Event(),
        response_aborted=threading.Event(),
    )

    stream_range, saved_sound, err = result
    assert stream_range is None and saved_sound is None
    assert "48000" in err and "24000" in err  # type: ignore[union-attr]


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

    stream_range, saved_sound, err = ConversationStreamingTts.generate_to_sound_stream(
        state=SimpleNamespace(project=project),
        text="hello",
        reason=Reason.SENTENCE,
        sound_stream=FakeSoundStream(24000),
        interrupt_requested=threading.Event(),
        response_aborted=threading.Event(),
    )

    assert err is None
    assert stream_range is not None
    assert stream_range[0] == 0
    assert stream_range[1] >= 150  # speech plus any appended reason pause
    assert saved_sound is not None
    assert saved_sound.data.size >= 150
    assert spies.synthesis_calls == 1


def test_streaming_tts_propagates_generation_error(monkeypatch) -> None:
    spies = _patch_tts_for_streaming(
        monkeypatch, 24000, lambda on_chunk, on_end: "model exploded"
    )
    project = Project.model_validate({})

    stream_range, saved_sound, err = ConversationStreamingTts.generate_to_sound_stream(
        state=SimpleNamespace(project=project),
        text="hello",
        reason=Reason.SENTENCE,
        sound_stream=FakeSoundStream(24000),
        interrupt_requested=threading.Event(),
        response_aborted=threading.Event(),
    )

    assert (stream_range, saved_sound) == (None, None)
    assert err == "model exploded"
    assert spies.synthesis_calls == 1


def test_streaming_tts_reports_when_no_audio_was_streamed(monkeypatch) -> None:
    spies = _patch_tts_for_streaming(
        monkeypatch, 24000, lambda on_chunk, on_end: ("idle",)
    )
    project = Project.model_validate({})

    stream_range, saved_sound, err = ConversationStreamingTts.generate_to_sound_stream(
        state=SimpleNamespace(project=project),
        text="hello",
        reason=Reason.SENTENCE,
        sound_stream=FakeSoundStream(24000),
        interrupt_requested=threading.Event(),
        response_aborted=threading.Event(),
    )

    assert (stream_range, saved_sound) == (None, None)
    assert err == "No streamed audio output"
    assert spies.synthesis_calls == 1


@pytest.mark.skipif(os.name != "posix", reason="fd redirection is POSIX-only")
def test_mute_current_thread_output_mutes_streams_and_restores_fd2() -> None:
    import sys

    inner = io.StringIO()
    q: "queue.Queue" = queue.Queue()
    real_stdout = QueuedStream(inner, q)
    real_stderr = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = real_stdout, real_stderr
    try:
        fd_stat_before = os.fstat(2)  # whatever pytest points fd 2 at
        lock = threading.Lock()
        with MuteCurrentThreadOutput(real_stderr, lock):
            print("hidden line")
            assert q.empty()  # muted: nothing enqueued
            # fd 2 is redirected: it must no longer be the original descriptor
            assert (os.fstat(2).st_dev, os.fstat(2).st_ino) != (
                fd_stat_before.st_dev, fd_stat_before.st_ino
            )
        assert not lock.locked()

        # fd 2 is restored to the exact descriptor it had before
        assert (os.fstat(2).st_dev, os.fstat(2).st_ino) == (
            fd_stat_before.st_dev, fd_stat_before.st_ino
        )

        print("visible line")
        ops = []
        while True:
            try:
                ops.append(q.get_nowait())
            except queue.Empty:
                break
        assert [op.text for op in ops] == ["visible line"]
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr