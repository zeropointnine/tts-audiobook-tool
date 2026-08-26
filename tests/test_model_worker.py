import os
import queue
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book, BookSection, SttVariant
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.l import L
from tts_audiobook_tool.model_worker import (
    ModelWorker,
    _OperationTracker,
    _QueueTextStream,
    _WorkerOutputCapture,
)
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleFlush,
    ConsoleOutput,
    GenerationFinished,
    GenerationTerminalStatus,
    InspectTtsCommand,
    TtsInspected,
    WorkerExited,
    WorkerStatus,
)
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.state import State


def _drain_until_terminal(operation_id: str, timeout: float = 20.0):
    """Drain worker events until a terminal event for the operation arrives."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in ModelWorker.drain_events():
            if (
                getattr(event, "operation_id", None) == operation_id
                and isinstance(
                    event, (GenerationFinished, WorkerExited)
                )
            ):
                return event
        time.sleep(0.05)
    return None


@pytest.fixture(autouse=True)
def stop_model_worker():
    ModelWorker.shutdown()
    yield
    ModelWorker.shutdown()
    Interrupts().set_external_event(None)
    Interrupts().clear()


def test_spawned_model_worker_starts_and_shuts_down() -> None:
    assert ModelWorker.start() == ""
    assert ModelWorker.is_alive()

    ModelWorker.shutdown()

    assert not ModelWorker.is_alive()


def test_spawned_model_worker_can_hard_reset() -> None:
    assert ModelWorker.start() == ""
    first_process = ModelWorker._process

    assert ModelWorker.reset() == ""

    assert ModelWorker.is_alive()
    assert ModelWorker._process is not first_process


def test_sigterm_exits_worker_gracefully() -> None:
    """SIGTERM must unwind the worker so its atexit finalizers run.

    The hard reset escalates from ``terminate()`` (SIGTERM) to ``kill()``
    (SIGKILL) only when the worker ignores the first signal.  The worker
    installs a SIGTERM handler that raises SystemExit, so a responsive
    worker exits cleanly and unregisters the multiprocessing semaphores a
    model library may have created while loading the model; SIGKILL would
    skip those finalizers and leave them for ``resource_tracker``.
    """
    assert ModelWorker.start() == ""
    process = ModelWorker._process
    os.kill(process.pid, signal.SIGTERM)

    process.join(timeout=5.0)

    assert not process.is_alive()


def test_worker_status_tracks_start_and_shutdown() -> None:
    assert ModelWorker.status() is WorkerStatus.ABSENT
    assert not ModelWorker.is_alive()

    assert ModelWorker.start() == ""
    assert ModelWorker.status() is WorkerStatus.RUNNING
    assert ModelWorker.is_alive()
    assert not ModelWorker.is_busy()

    ModelWorker.shutdown()
    assert ModelWorker.status() is WorkerStatus.ABSENT
    assert not ModelWorker.is_alive()


def test_unload_models_exits_worker_and_restarts_lazily() -> None:
    assert ModelWorker.start() == ""
    first_process = ModelWorker._process
    assert first_process is not None

    assert ModelWorker.unload_models_blocking() == ""

    assert ModelWorker._process is None
    assert ModelWorker.status() is WorkerStatus.ABSENT
    assert not ModelWorker.is_alive()

    assert ModelWorker.start() == ""
    assert ModelWorker._process is not first_process
    assert ModelWorker.is_alive()


def test_worker_exited_is_synthesized_once_on_death() -> None:
    assert ModelWorker.start() == ""
    assert ModelWorker.status() is WorkerStatus.RUNNING
    process = ModelWorker._process

    # Simulate a crash of the worker process tree.
    ModelWorker._force_stop_process(process)
    events = ModelWorker.drain_events()

    exited = [event for event in events if isinstance(event, WorkerExited)]
    assert len(exited) == 1
    assert exited[0].operation_id == ""
    # The synthesized message points at the worker's own log file.
    assert "Worker log" in exited[0].message
    assert ModelWorker.status() is WorkerStatus.DEAD

    # Death is reported exactly once: later drains stay quiet.
    assert ModelWorker.drain_events() == []
    assert ModelWorker.get_event(timeout=0.05) is None


def test_submit_generation_resurrects_dead_worker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(L, "d", lambda *_: None)
    phrase_group = PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])
    project = Project(
        dir_path=str(tmp_path),
        book=Book(sections=[BookSection(phrase_groups=[phrase_group])]),
    )
    assert project.save() == ""
    assert ProjectTextIOUtil.save_book(project) == ""
    state = SimpleNamespace(
        project=project,
        prefs=Prefs(project_dir=str(tmp_path), stt_variant=SttVariant.DISABLED),
    )

    try:
        first_id = ModelWorker.submit_generation(
            state=state,
            indices={0},
            batch_size=1,
            is_regen=False,
        )
        terminal = _drain_until_terminal(first_id)
        assert terminal is not None
        first_process = ModelWorker._process
        assert ModelWorker.status() is WorkerStatus.RUNNING

        # Simulate a mid-job crash.
        ModelWorker._force_stop_process(first_process)
        exited = [
            event
            for event in ModelWorker.drain_events()
            if isinstance(event, WorkerExited)
        ]
        assert len(exited) == 1
        assert ModelWorker.status() is WorkerStatus.DEAD

        # A new generation resurrects the worker instead of failing.
        second_id = ModelWorker.submit_generation(
            state=state,
            indices={0},
            batch_size=1,
            is_regen=False,
        )
        assert second_id != first_id
        assert ModelWorker.status() is WorkerStatus.RUNNING
        assert ModelWorker._process is not first_process

        terminal = _drain_until_terminal(second_id)
        assert terminal is not None
    finally:
        project.kill()


def test_clear_models_blocking_reports_synthesized_worker_exited(monkeypatch) -> None:
    assert ModelWorker.start() == ""
    # Pretend start() is a no-op so the killed worker is not resurrected.
    monkeypatch.setattr(ModelWorker, "start", staticmethod(lambda: ""))
    ModelWorker._force_stop_process(ModelWorker._process)
    assert ModelWorker.status() is WorkerStatus.RUNNING

    error = ModelWorker.clear_models_blocking()

    assert "exited" in error.lower()
    assert "Worker log" in error
    assert ModelWorker.status() is WorkerStatus.DEAD
    assert not ModelWorker.is_busy()


def test_spawned_worker_runs_generation_command_and_returns_terminal_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(L, "d", lambda *_: None)
    phrase_group = PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])
    project = Project(
        dir_path=str(tmp_path),
        book=Book(sections=[BookSection(phrase_groups=[phrase_group])]),
    )
    assert project.save() == ""
    assert ProjectTextIOUtil.save_book(project) == ""
    state = SimpleNamespace(
        project=project,
        prefs=Prefs(project_dir=str(tmp_path), stt_variant=SttVariant.DISABLED),
    )

    try:
        operation_id = ModelWorker.submit_generation(
            state=state,
            indices={0},
            batch_size=1,
            is_regen=False,
        )
        deadline = time.monotonic() + 15.0
        terminal = None
        while time.monotonic() < deadline and terminal is None:
            event = ModelWorker.get_event(timeout=0.1)
            if isinstance(event, GenerationFinished) and event.operation_id == operation_id:
                terminal = event

        assert terminal is not None
        # venv-base intentionally has no concrete TTS engine. Reaching the
        # normal warm-up abort proves project reconstruction and IPC dispatch.
        assert terminal.status == GenerationTerminalStatus.ABORTED
    finally:
        project.kill()


def test_external_interrupt_event_is_additional_cancel_source() -> None:
    interrupt_event = threading.Event()
    interrupts = Interrupts()
    interrupts.set_external_event(interrupt_event)
    interrupts.set("generating")

    assert not interrupts.did_interrupt
    interrupt_event.set()
    assert interrupts.did_interrupt

    # Changing the descriptive mode must not erase a process-safe request.
    interrupts.set("model init")
    assert interrupts.did_interrupt

    interrupt_event.clear()
    assert not interrupts.did_interrupt


def test_queue_stream_preserves_partial_writes_and_flushes() -> None:
    event_queue: queue.Queue[object] = queue.Queue()
    tracker = _OperationTracker()
    tracker.set("job-1")

    class Original:
        encoding = "utf-8"
        errors = "replace"

        def isatty(self):
            return True

        def fileno(self):
            return 1

    stream = _QueueTextStream(Original(), "stdout", event_queue, tracker, True)  # type: ignore[arg-type]

    assert stream.write("partial") == len("partial")
    stream.flush()

    assert event_queue.get_nowait() == ConsoleOutput("job-1", "stdout", "partial")
    assert event_queue.get_nowait() == ConsoleFlush("job-1", "stdout")


def test_queue_stream_reports_lost_capture_once() -> None:
    tracker = _OperationTracker()
    tracker.set("job-1")
    sentinels: list[str] = []

    class Original:
        encoding = "utf-8"
        errors = "replace"

        def isatty(self):
            return True

        def fileno(self):
            return 1

    class BrokenQueue:
        def put(self, _item: object) -> None:
            raise OSError("event queue closed")

    stream = _QueueTextStream(  # type: ignore[arg-type]
        Original(),
        "stdout",
        BrokenQueue(),  # type: ignore[arg-type]
        tracker,
        True,
        on_capture_lost=sentinels.append,
    )

    assert stream.write("partial") == len("partial")
    stream.write("more")
    stream.flush()

    # One sentinel per stream, no matter how many writes failed.
    assert sentinels == ["stdout"]


def test_capture_lost_sentinel_is_one_shot() -> None:
    event_queue: queue.Queue[object] = queue.Queue()
    tracker = _OperationTracker()
    tracker.set("job-1")
    capture = _WorkerOutputCapture(event_queue, tracker)

    capture._send_capture_lost_sentinel("stdout")
    capture._send_capture_lost_sentinel("stdout")
    capture._send_capture_lost_sentinel("stderr")

    first = event_queue.get_nowait()
    assert isinstance(first, ConsoleOutput)
    assert first.operation_id == "job-1"
    assert first.stream == "stdout"
    assert "capture lost" in first.text
    # The guard is shared across streams: only one marker is shipped.
    assert event_queue.empty()


def test_discard_process_state_closes_ipc_resources_deterministically() -> None:
    calls: list[str] = []

    class QueueStub:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(f"{self.name}.close")

        def join_thread(self) -> None:
            calls.append(f"{self.name}.join_thread")

    class ProcessStub:
        def is_alive(self) -> bool:
            return False

        def join(self) -> None:
            calls.append("process.join")

        def close(self) -> None:
            calls.append("process.close")

    ModelWorker._process = ProcessStub()
    ModelWorker._command_queue = QueueStub("command")
    ModelWorker._event_queue = QueueStub("event")

    ModelWorker._discard_process_state()

    assert calls == [
        "command.close",
        "event.close",
        "command.join_thread",
        "event.join_thread",
        "process.join",
        "process.close",
    ]
    assert ModelWorker._process is None
    assert ModelWorker._command_queue is None
    assert ModelWorker._event_queue is None
    assert ModelWorker._cancellation_event is None
    assert ModelWorker._continue_event is None

    # Cleanup can be called again after a partially failed lifecycle path.
    ModelWorker._discard_process_state()
    assert len(calls) == 6


def test_state_for_worker_mirrors_init_attribute_set() -> None:
    state = State.for_worker(
        Prefs(project_dir="", stt_variant=SttVariant.DISABLED)
    )

    # Must match exactly the instance attributes set by State.__init__, so a
    # worker State behaves like the main process' State for property access.
    assert set(vars(state)) == {
        "_project",
        "real_time",
        "_prefs",
        "dont_show_scan_message",
        "has_shown_main_menu",
    }
    assert state.project is None
    assert state.dont_show_scan_message is False

def test_clear_models_if_running_does_not_spawn_worker(monkeypatch) -> None:
    monkeypatch.setattr(
        ModelWorker,
        "start",
        classmethod(lambda cls: pytest.fail("must not start worker")),
    )

    assert ModelWorker.clear_models_if_running_blocking() == ""


def test_worker_reports_its_own_empty_model_inventory() -> None:
    snapshot, error = ModelWorker.get_model_state_blocking()

    assert error == ""
    assert snapshot is not None
    assert not snapshot.any_loaded
    assert snapshot.tts_loaded is False
    assert snapshot.stt_loaded is False
    assert snapshot.yamnet_loaded is False
    assert snapshot.lava_sr_loaded is False


def test_inspect_tts_queues_unsaved_model_params(tmp_path, monkeypatch) -> None:
    """Validation must inspect live edits, not only project.json on disk."""
    monkeypatch.setattr(L, "d", lambda *_: None)
    project = Project(dir_path=str(tmp_path))
    assert project.save() == ""
    project.vibevoice_lora_target = "vibevoice-community/unsaved-adapter"
    state = SimpleNamespace(
        project=project,
        prefs=Prefs(project_dir=str(tmp_path), stt_variant=SttVariant.DISABLED),
    )
    commands: list[object] = []

    class CommandQueue:
        def put(self, command: object) -> None:
            commands.append(command)

    def wait_for_result(cls, operation_id, _expected_type):
        cls._active_operation_id = None
        return TtsInspected(operation_id, "vibevoice")

    monkeypatch.setattr(ModelWorker, "start", classmethod(lambda cls: ""))
    monkeypatch.setattr(ModelWorker, "_command_queue", CommandQueue())
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(
        ModelWorker,
        "_wait_for_blocking_result",
        classmethod(wait_for_result),
    )

    inspection, error = ModelWorker.inspect_tts_blocking(state)

    assert error == ""
    assert inspection is not None
    assert len(commands) == 1
    command = commands[0]
    assert isinstance(command, InspectTtsCommand)
    assert command.model_params["vibevoice_lora_path"] == project.vibevoice_lora_target

