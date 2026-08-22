import queue
import threading
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.conversation.conversation_internals import Ui
from tts_audiobook_tool.conversation.conversation_types import QueuedStream, UiOp


class RecordingStdout:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: str) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    @property
    def data(self) -> str:
        return "".join(self.writes)


def drain_ops(q: "queue.Queue[UiOp]") -> list[UiOp]:
    ops = []
    while True:
        try:
            ops.append(q.get_nowait())
        except queue.Empty:
            return ops


def test_queued_stream_buffers_partial_lines_until_newline() -> None:
    q: queue.Queue[UiOp] = queue.Queue()
    stream = QueuedStream(RecordingStdout(), q)

    stream.write("hel")
    assert drain_ops(q) == []

    stream.write("lo\nwor")
    assert [op.text for op in drain_ops(q)] == ["hello"]

    stream.write("ld\n")
    ops = drain_ops(q)
    assert [op.text for op in ops] == ["world"]
    assert all(op.kind == "println" for op in ops)


def test_queued_stream_empty_write_is_noop() -> None:
    q: queue.Queue[UiOp] = queue.Queue()
    stream = QueuedStream(RecordingStdout(), q)

    assert stream.write("") == 0
    assert q.empty()


def test_queued_stream_mute_discards_writes_for_current_thread_only() -> None:
    q: queue.Queue[UiOp] = queue.Queue()
    stream = QueuedStream(RecordingStdout(), q)

    stream.mute()
    assert stream.write("hidden\n") == len("hidden\n")
    assert q.empty()

    stream.unmute()
    stream.write("visible\n")
    assert [op.text for op in drain_ops(q)] == ["visible"]


def test_queued_stream_mute_does_not_affect_other_threads() -> None:
    q: queue.Queue[UiOp] = queue.Queue()
    stream = QueuedStream(RecordingStdout(), q)
    stream.mute()
    stream.write("main-thread-hidden\n")

    result = {}

    def worker() -> None:
        result["written"] = stream.write("other-thread\n")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2.0)

    assert result["written"] == len("other-thread\n")
    ops = drain_ops(q)
    assert [op.text for op in ops] == ["other-thread"]


def test_queued_stream_forwards_unknown_attributes_to_real_stream() -> None:
    real = SimpleNamespace(isclosed=lambda: True, some_attr="sentinel")
    stream = QueuedStream(real, queue.Queue())

    assert stream.some_attr == "sentinel"
    assert stream.isclosed() is True


@pytest.mark.parametrize(
    ("num_lines", "expected"),
    [
        (0, ""),
        (1, "\r\033[J"),
        (3, "\033[2A\r\033[J"),
        (10, "\033[9A\r\033[J"),
    ],
)
def test_make_clear_seq(num_lines: int, expected: str) -> None:
    assert Ui.make_clear_seq(num_lines) == expected


def test_display_width_counts_wide_chars_double_and_skips_marks() -> None:
    assert Ui.display_width("abc") == 3
    assert Ui.display_width("日本語") == 6
    assert Ui.display_width("abé") == 3  # e + combining acute
    assert Ui.display_width("a\x01b") == 2  # control chars are 0
    assert Ui.display_width("") == 0


def test_count_display_lines_wraps_on_terminal_width(monkeypatch) -> None:
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation_internals.get_terminal_width",
        lambda: 10,
    )

    assert Ui.count_display_lines("abc") == 1
    assert Ui.count_display_lines("a" * 10) == 1
    assert Ui.count_display_lines("a" * 11) == 2
    assert Ui.count_display_lines("abc\ndef") == 2
    assert Ui.count_display_lines("") == 1


def test_ui_worker_coalesces_consecutive_renders_to_latest() -> None:
    ui = Ui(RecordingStdout())
    ui.start()
    try:
        ui.render("first")
        ui.render("second")
        ui.render("third")
        ui.wait_idle()
        assert "third" in ui.real_stdout.data  # type: ignore[attr-defined]
        assert "first" not in ui.real_stdout.data  # type: ignore[attr-defined]
    finally:
        ui.stop()


def test_ui_rapid_identical_renders_are_deduped_on_producer_side() -> None:
    ui = Ui(RecordingStdout())

    ui.render("same text")
    ui.render("same text")

    assert ui.queue.qsize() == 1


def test_ui_empty_render_clears_previous_render(monkeypatch) -> None:
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation_internals.get_terminal_width",
        lambda: 80,
    )
    ui = Ui(RecordingStdout())
    ui.start()
    try:
        ui.render("abc")
        ui.wait_idle()
        before = ui.real_stdout.data  # type: ignore[attr-defined]
        assert "\033[J" not in before  # single line: clear seq is just \r...

        ui.render("")
        ui.wait_idle()
        after = ui.real_stdout.data  # type: ignore[attr-defined]
        assert "\033[J" in after[len(before):]
    finally:
        ui.stop()


def test_ui_println_and_commit_render(monkeypatch) -> None:
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.conversation_internals.get_terminal_width",
        lambda: 80,
    )
    ui = Ui(RecordingStdout())
    ui.start()
    try:
        ui.render("line one")
        ui.wait_idle()
        # Committing right after a render advances past the rendered line
        # (one newline) plus two extra blanks.
        ui.commit_render(2)
        ui.wait_idle()
        ui.println("after commit")
        ui.wait_idle()

        data = ui.real_stdout.data  # type: ignore[attr-defined]
        assert data.startswith("line one\r")
        middle = data[len("line one\r"):data.index("after commit")]
        # one advance newline for the rendered line + two extra blanks;
        # println resets the line count, so its own clear seq is empty
        assert middle == "\n\n\n"
    finally:
        ui.stop()