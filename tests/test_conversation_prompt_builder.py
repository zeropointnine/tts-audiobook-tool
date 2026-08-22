import queue
import threading

import numpy as np
import pytest

from tts_audiobook_tool.app_types import ConcreteSegment
from tts_audiobook_tool.constants import COL_DIM, COL_MEDIUM
from tts_audiobook_tool.conversation.conversation_internals import PromptBuilder
from tts_audiobook_tool.conversation.console_session import (
    KEY_CTRL_C,
    KEY_DEL,
    KEY_DEL2,
    KEY_ENTER,
    KEY_FWDDEL,
    KEY_LEFT,
    KEY_RIGHT,
)
from tts_audiobook_tool.system_support.ansi import Ansi


class FakeConsole:
    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self.keys_read = 0

    def read_key(self) -> str | None:
        self.keys_read += 1
        return self.keys.pop(0) if self.keys else None


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


def make_builder(
        keys: list[str],
        stt_immediate: bool = False,
) -> tuple[PromptBuilder, FakeConsole, RecordingUi, threading.Event]:
    ui = RecordingUi()
    console = FakeConsole(keys)
    ctrl_c = threading.Event()
    return PromptBuilder(ui, console, ctrl_c, stt_immediate=stt_immediate), console, ui, ctrl_c


def seg(text: str) -> list:
    return [ConcreteSegment(0.0, 1.0, text, [])]


def test_on_transcription_buffers_chunk_and_copies_audio() -> None:
    builder, console, ui, _ = make_builder([])
    audio = np.array([0.1, 0.2], dtype=np.float32)

    builder.on_transcription(seg("Hello"))

    text, chunk_audio = builder.chunk_queue.get_nowait()
    assert text == "Hello"
    assert chunk_audio is None

    builder.on_transcription(seg("world"), audio)
    text2, audio2 = builder.chunk_queue.get_nowait()
    assert text2 == "world"
    assert audio2 is not None
    assert audio2.dtype == np.float32
    audio[0] = 9.9  # the queued copy must be independent of the original
    assert audio2[0] == 0.1


def test_on_transcription_is_ignored_while_mic_paused() -> None:
    builder, _, _, _ = make_builder([])
    builder.mic_paused = True

    builder.on_transcription(seg("ignored"))

    assert builder.chunk_queue.empty()


def test_on_transcription_ignores_blank_segments() -> None:
    builder, _, _, _ = make_builder([])
    segments = [ConcreteSegment(0.0, 1.0, "   ", [])]

    builder.on_transcription(segments)

    assert builder.chunk_queue.empty()


def test_build_enter_assembles_commits_and_resets_state() -> None:
    builder, console, ui, ctrl_c = make_builder([KEY_ENTER])
    builder.on_transcription(seg("Hello"))
    builder.on_transcription(seg("world"))

    assembled = builder.build()

    assert assembled == "Hello world"
    assert builder.mic_paused is True
    assert builder.prompt_chunks == []
    assert builder.selected_idx is None
    assert builder.chunk_queue.empty()
    assert any("Hello world" in line for line in ui.prints)
    assert not ctrl_c.is_set()


def test_build_ctrl_c_key_sets_event_and_raises() -> None:
    builder, console, ui, ctrl_c = make_builder([KEY_CTRL_C])

    with pytest.raises(KeyboardInterrupt):
        builder.build()

    assert ctrl_c.is_set()


def test_build_raises_immediately_when_ctrl_c_already_requested() -> None:
    builder, console, _, ctrl_c = make_builder([KEY_ENTER])
    ctrl_c.set()

    with pytest.raises(KeyboardInterrupt):
        builder.build()

    assert console.keys_read == 0


def test_build_left_right_arrow_move_selection_without_reordering() -> None:
    builder, console, ui, _ = make_builder([KEY_LEFT, KEY_RIGHT, KEY_ENTER])
    builder.on_transcription(seg("chunk one"))
    builder.on_transcription(seg("chunk two"))

    assembled = builder.build()

    assert assembled == "chunk one chunk two"  # selection never reorders
    # After LEFT the selection lands on the first chunk: it renders italic
    # while the unselected second chunk renders dimmed.
    assert any(
        f"{Ansi.ITALICS}chunk one{Ansi.RESET}" in render
        and f"{COL_DIM}chunk two{Ansi.RESET}" in render
        for render in ui.renders
    )


def test_build_delete_keys_remove_selected_chunk() -> None:
    for key in (KEY_DEL, KEY_DEL2, KEY_FWDDEL):
        builder, console, _, _ = make_builder([key, KEY_ENTER])
        builder.on_transcription(seg("keep me"))
        builder.on_transcription(seg("drop me"))

        assert builder.build() == "keep me"


def test_build_deleting_last_chunk_clears_selection_then_ctrl_c_exits() -> None:
    builder, console, _, _ = make_builder([KEY_DEL, KEY_ENTER, KEY_CTRL_C])
    builder.on_transcription(seg("only one"))

    with pytest.raises(KeyboardInterrupt):
        builder.build()

    assert builder.prompt_chunks == []
    assert builder.selected_idx is None


def test_stt_immediate_commits_asap_without_reading_keys() -> None:
    builder, console, ui, _ = make_builder([])
    builder.stt_immediate = True
    builder.on_transcription(seg("quick reply"))

    assert builder.build() == "quick reply"
    assert console.keys_read == 0
    assert any("quick reply" in line for line in ui.prints)


def test_render_uses_placeholder_when_no_chunks() -> None:
    builder, console, ui, _ = make_builder([])

    builder.render()

    assert len(ui.renders) == 1
    assert "Speak into the mic" in ui.renders[0]


def test_commit_finalized_prompt_concatenates_and_normalizes_audio() -> None:
    builder, _, ui, _ = make_builder([])
    builder.prompt_chunks = ["a", "b"]
    builder.prompt_chunk_audios = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.3, 0.0], dtype=np.float32),
    ]

    builder.commit_finalized_prompt("a b")

    audio = builder.take_finalized_mic_audio()
    assert audio is not None
    assert audio.size == 4
    assert np.isclose(np.max(np.abs(audio)), 1.0)  # peak-normalized
    assert builder.take_finalized_mic_audio() is None  # consumed
    assert builder.mic_paused is True
    assert any("a b" in line for line in ui.prints)


def test_commit_finalized_prompt_without_audio_sets_none() -> None:
    builder, _, _, _ = make_builder([])
    builder.prompt_chunks = ["a"]
    builder.prompt_chunk_audios = [None, np.array([], dtype=np.float32)]

    builder.commit_finalized_prompt("a")

    assert builder.take_finalized_mic_audio() is None