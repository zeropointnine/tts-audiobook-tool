import os
import queue
from dataclasses import fields, replace
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import tts_audiobook_tool.model_worker as model_worker_module
from tts_audiobook_tool import gen_timeout_util
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book, BookSection, Sound, SttVariant
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationTimedOut
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
    GenerationSettings,
    GenerationTerminalStatus,
    GenerationUpdate,
    InspectTtsCommand,
    TtsInspected,
    TtsPreviewCommand,
    TtsPreviewFinished,
    WorkerCommandCancelled,
    WorkerExited,
    WorkerStatus,
)
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from project_settings_test_support import get_setting, set_setting
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts


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


def test_local_hard_reset_reports_catalog_change_without_menu_restart_loop(monkeypatch, capsys) -> None:
    assert not Tts.is_remote_mode()
    assert ModelWorker.start() == ""
    # Simulate the main process retaining a fingerprint that no longer matches
    # the catalog a freshly spawned worker reads (including local metadata).
    monkeypatch.setattr(Tts, "_config_fingerprint", "0" * 64)

    error = ModelWorker.reset()

    assert error == model_worker_module.MODEL_CATALOG_MISMATCH_ERROR
    assert "restart the application" in error
    assert "local and remote" in error
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    assert ModelWorker.status() is WorkerStatus.ABSENT
    assert not ModelWorker.is_alive()
    monkeypatch.setattr(
        ModelWorker, "start",
        classmethod(lambda cls: pytest.fail("menu status must not retry failed startup")),
    )
    for _ in range(3):
        snapshot, state_error = ModelWorker.get_model_state_blocking()
        assert state_error == ""
        assert snapshot is not None and not snapshot.any_loaded


def test_blocking_wait_stops_when_hard_reset_invalidates_operation(
    monkeypatch,
) -> None:
    operation_id = "initializing"
    entered_get_event = threading.Event()
    release_get_event = threading.Event()
    get_event_calls = 0
    result: list[object] = []

    def fake_get_event(timeout: float = 0.1):
        nonlocal get_event_calls
        get_event_calls += 1
        entered_get_event.set()
        release_get_event.wait(1.0)
        return None

    monkeypatch.setattr(ModelWorker, "get_event", staticmethod(fake_get_event))
    ModelWorker._active_operation_id = operation_id
    waiter = threading.Thread(
        target=lambda: result.append(
            ModelWorker._wait_for_blocking_result(operation_id, TtsInspected)
        )
    )
    waiter.start()
    assert entered_get_event.wait(1.0)

    # reset() clears the operation before starting its replacement. The old
    # waiter must return without polling that replacement worker's queue.
    with ModelWorker._lock:
        ModelWorker._active_operation_id = None
    release_get_event.set()
    waiter.join(1.0)

    assert not waiter.is_alive()
    assert result == ["Model worker operation was cancelled"]
    assert get_event_calls == 1


class _FakeCancellationEvent:
    def __init__(self) -> None:
        self.signalled = False

    def set(self) -> None:
        self.signalled = True

    def clear(self) -> None:
        self.signalled = False

    def is_set(self) -> bool:
        return self.signalled


def test_blocking_transcription_cancel_without_ack_uses_the_stall_timeout(
    monkeypatch
) -> None:
    stopped = []
    monkeypatch.setattr(ModelWorker, "shutdown", lambda: stopped.append(True))
    polls = {"n": 0}

    def fake_get_event(**_kwargs):
        # Stands in for a worker inside a decode step it cannot interrupt:
        # nothing ever comes back, so the wait ends at the stall timeout that
        # already covers a hung chunk.
        polls["n"] += 1
        return None

    monkeypatch.setattr(ModelWorker, "get_event", staticmethod(fake_get_event))
    cancellation = _FakeCancellationEvent()
    ModelWorker._cancellation_event = cancellation
    ModelWorker._active_operation_id = "transcribing"
    try:
        result = ModelWorker._wait_for_blocking_result(
            "transcribing",
            TtsInspected,
            cancel_check=lambda: True,
            timeout_seconds=0.2,
        )
    finally:
        ModelWorker._active_operation_id = None
        ModelWorker._cancellation_event = None
    # Cancellation was requested and the wait stayed on the worker instead of
    # reaping it on the first press. The request stays up: the worker that
    # never got there is being reaped, not reused.
    assert cancellation.signalled is True
    assert polls["n"] > 0
    assert stopped == [True]
    assert result == model_worker_module.STT_TRANSCRIPTION_TIMEOUT_ERROR


def test_cancel_lets_worker_acknowledge_and_stay_alive(monkeypatch) -> None:
    stopped: list[bool] = []
    monkeypatch.setattr(ModelWorker, "shutdown", lambda: stopped.append(True))
    cancellation = _FakeCancellationEvent()
    ModelWorker._cancellation_event = cancellation
    ModelWorker._active_operation_id = "transcribing"
    monkeypatch.setattr(
        ModelWorker,
        "get_event",
        staticmethod(lambda timeout=0.1: WorkerCommandCancelled("transcribing")),
    )
    try:
        result = ModelWorker._wait_for_blocking_result(
            "transcribing", TtsInspected, cancel_check=lambda: True
        )
    finally:
        ModelWorker._active_operation_id = None
        ModelWorker._cancellation_event = None
    # The worker stopped the command itself and is back at its command queue,
    # so the process and its loaded model survive for the next run.
    assert result == "Model worker operation was cancelled"
    assert stopped == []
    assert cancellation.signalled is False


def test_segment_drain_stops_at_cancellation_and_unwinds() -> None:
    cancellation = _FakeCancellationEvent()
    unwound: list[bool] = []

    def segments():
        try:
            yield "a"
            # Stands in for the parent asking for cancellation mid-stream.
            cancellation.set()
            yield "b"
        finally:
            unwound.append(True)

    collected, cancelled = model_worker_module._collect_segments_until_cancelled(
        segments(), cancellation
    )
    assert cancelled is True
    assert collected == ["a"]
    assert unwound == [True]


def test_segment_drain_collects_everything_when_not_cancelled() -> None:
    collected, cancelled = model_worker_module._collect_segments_until_cancelled(
        iter(["a", "b"]), _FakeCancellationEvent()
    )
    assert cancelled is False
    assert collected == ["a", "b"]


def test_blocking_transcription_timeout_stops_worker(monkeypatch, caplog) -> None:
    stopped = []
    monkeypatch.setattr(ModelWorker, "shutdown", lambda: stopped.append(True))
    monkeypatch.setattr(
        ModelWorker, "get_event", lambda **_kwargs: pytest.fail("polled after timeout")
    )
    ModelWorker._active_operation_id = "transcribing"
    try:
        result = ModelWorker._wait_for_blocking_result(
            "transcribing", TtsInspected, timeout_seconds=0.0
        )
        assert result == model_worker_module.STT_TRANSCRIPTION_TIMEOUT_ERROR
        assert stopped == [True]
        assert "[stt watchdog] TIMEOUT: operation=transcribing" in caplog.text
        assert "[stt watchdog] worker stopped after timeout" in caplog.text
    finally:
        ModelWorker._active_operation_id = None


def test_stt_diagnostic_threshold_logs_stack_location(monkeypatch, caplog) -> None:
    with monkeypatch.context() as patch:
        clock = iter((0.0, 46.0, 46.0))
        patch.setattr(model_worker_module.time, "monotonic", lambda: next(clock))
        patch.setattr(
            ModelWorker, "get_event", lambda **_kwargs: TtsInspected("transcribing", "test")
        )
        ModelWorker._active_operation_id = "transcribing"
        try:
            ModelWorker._wait_for_blocking_result(
                "transcribing", TtsInspected, timeout_seconds=90.0
            )
        finally:
            ModelWorker._active_operation_id = None
    assert "[stt watchdog] 45s diagnostic threshold reached" in caplog.text
    assert model_worker_module.STT_WORKER_STACK_LOG_PATH in caplog.text


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

    # Simulate a crash of the worker process tree. An inventory query must not
    # resurrect it, even before the event drainer observes its death.
    ModelWorker._force_stop_process(process)
    snapshot, error = ModelWorker.get_model_state_blocking()
    assert error == ""
    assert snapshot is not None and not snapshot.any_loaded
    assert ModelWorker._process is process
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
        "_prefs",
        "dont_show_scan_message",
        "has_shown_main_menu",
        "pocket_voice_clone_access_validated",
        "pending_project_load_checks",
        "pending_tts_model_change",
    }
    assert state.project is None
    assert state.dont_show_scan_message is False
    assert state.pocket_voice_clone_access_validated is False

def test_clear_models_if_running_does_not_spawn_worker(monkeypatch) -> None:
    monkeypatch.setattr(
        ModelWorker,
        "start",
        classmethod(lambda cls: pytest.fail("must not start worker")),
    )

    assert ModelWorker.clear_models_if_running_blocking() == ""


@pytest.mark.parametrize("status", [WorkerStatus.ABSENT, WorkerStatus.DEAD])
def test_model_inventory_does_not_spawn_absent_or_dead_worker(monkeypatch, status) -> None:
    monkeypatch.setattr(ModelWorker, "_status", status)
    monkeypatch.setattr(
        ModelWorker, "start",
        classmethod(lambda cls: pytest.fail("inventory must not start worker")),
    )

    for _ in range(3):
        snapshot, error = ModelWorker.get_model_state_blocking()
        assert error == ""
        assert snapshot == model_worker_module.ModelStateSnapshot()
    assert ModelWorker.status() is status


def test_model_inventory_does_not_query_starting_worker(monkeypatch) -> None:
    monkeypatch.setattr(ModelWorker, "_status", WorkerStatus.STARTING)

    snapshot, error = ModelWorker.get_model_state_blocking()

    assert snapshot is None
    assert error == "Model worker is starting"
    assert not ModelWorker.is_busy()


def test_worker_reports_its_own_empty_model_inventory() -> None:
    assert ModelWorker.start() == ""
    snapshot, error = ModelWorker.get_model_state_blocking()

    assert error == ""
    assert snapshot is not None
    assert not snapshot.any_loaded
    assert snapshot.tts_loaded is False
    assert snapshot.stt_loaded is False
    assert snapshot.yamnet_loaded is False
    assert snapshot.lava_sr_loaded is False


def test_submit_tts_preview_queues_prompt_project_and_settings(
    tmp_path, monkeypatch
) -> None:
    project = Project(dir_path=str(tmp_path))
    prefs = Prefs(project_dir=str(tmp_path), stt_variant=SttVariant.DISABLED)
    state = SimpleNamespace(project=project, prefs=prefs)
    commands: list[object] = []

    class CommandQueue:
        def put(self, command: object) -> None:
            commands.append(command)

    class CancellationEvent:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1

    cancellation_event = CancellationEvent()
    monkeypatch.setattr(ModelWorker, "start", classmethod(lambda cls: ""))
    monkeypatch.setattr(ModelWorker, "_command_queue", CommandQueue())
    monkeypatch.setattr(ModelWorker, "_cancellation_event", cancellation_event)
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)

    operation_id = ModelWorker.submit_tts_preview(
        state=state,
        prompt="Original word: Ariekei. Substitute word: AriaKay",
        apply_word_substitutions=False,
    )

    assert len(commands) == 1
    command = commands[0]
    assert isinstance(command, TtsPreviewCommand)
    assert command.operation_id == operation_id
    assert command.project_dir == str(tmp_path)
    assert command.prompt == (
        "Original word: Ariekei. Substitute word: AriaKay"
    )
    assert command.apply_word_substitutions is False
    assert command.settings.stt_variant_id == SttVariant.DISABLED.id
    assert cancellation_event.clear_calls == 1
    assert ModelWorker.is_busy()


def test_tts_preview_finished_releases_worker_busy_state(monkeypatch) -> None:
    sound = Sound(np.zeros(8, dtype=np.float32), 24_000)
    monkeypatch.setattr(ModelWorker, "_active_operation_id", "preview-job")

    ModelWorker._observe_event(
        TtsPreviewFinished(
            "preview-job",
            GenerationTerminalStatus.COMPLETED,
            sound=sound,
        )
    )

    assert not ModelWorker.is_busy()


def test_run_tts_preview_returns_processed_sound_without_word_substitutions(
    monkeypatch,
) -> None:
    sound = Sound(np.zeros(8, dtype=np.float32), 24_000)
    generated: list[tuple[object, list[str], bool, bool]] = []
    events: list[object] = []
    project = SimpleNamespace(kill=lambda: None)
    state = SimpleNamespace(project=project)

    class CancellationEvent:
        def is_set(self) -> bool:
            return False

        def clear(self) -> None:
            pass

    class EventQueue:
        def put(self, event: object) -> None:
            events.append(event)

    command = TtsPreviewCommand(
        operation_id="preview-job",
        project_dir="/project",
        settings=GenerationSettings("disabled", "cpu_int8_float32", False, "none", "", False),
        prompt="Original word: foo. Substitute word: bar",
        apply_word_substitutions=False,
    )
    monkeypatch.setattr(
        model_worker_module, "_make_worker_state", lambda _command: state
    )
    monkeypatch.setattr(Tts, "clear_continuation", staticmethod(lambda: None))
    monkeypatch.setattr(
        Tts, "reset_voice_selection_index", staticmethod(lambda: None)
    )
    # This test covers the unwatched path (a call that may still load the
    # model); the watchdog itself is covered separately.
    monkeypatch.setattr(
        model_worker_module, "_should_watch_preview_inference", lambda: False
    )
    monkeypatch.setattr(
        SoundPipeline,
        "generate_processed_using_project",
        staticmethod(
            lambda passed_project, prompts, force_random_seed=False,
            apply_word_substitutions=True: (
                generated.append(
                    (
                        passed_project,
                        prompts,
                        force_random_seed,
                        apply_word_substitutions,
                    )
                )
                or [sound]
            )
        ),
    )

    model_worker_module._run_tts_preview_command(
        command, EventQueue(), CancellationEvent()
    )

    assert generated == [
        (
            project,
            ["Original word: foo. Substitute word: bar"],
            True,
            False,
        )
    ]
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, TtsPreviewFinished)
    assert event.status is GenerationTerminalStatus.COMPLETED
    assert event.sound is sound


def test_run_tts_preview_rejects_nan_output(monkeypatch) -> None:
    """NaN audio must fail the preview instead of reaching the listener."""
    data = np.zeros(8, dtype=np.float32)
    data[3] = np.nan
    sound = Sound(data, 24_000)
    project = SimpleNamespace(kill=lambda: None)
    state = SimpleNamespace(project=project)

    class CancellationEvent:
        def is_set(self) -> bool:
            return False

        def clear(self) -> None:
            pass

    class EventQueue:
        def __init__(self) -> None:
            self.events: list[object] = []

        def put(self, event: object) -> None:
            self.events.append(event)

    command = TtsPreviewCommand(
        operation_id="preview-job",
        project_dir="/project",
        settings=GenerationSettings("disabled", "cpu_int8_float32", False, "none", "", False),
        prompt="Original word: foo. Substitute word: bar",
        apply_word_substitutions=False,
    )
    monkeypatch.setattr(
        model_worker_module, "_make_worker_state", lambda _command: state
    )
    monkeypatch.setattr(Tts, "clear_continuation", staticmethod(lambda: None))
    monkeypatch.setattr(
        Tts, "reset_voice_selection_index", staticmethod(lambda: None)
    )
    monkeypatch.setattr(
        model_worker_module, "_should_watch_preview_inference", lambda: False
    )
    monkeypatch.setattr(
        SoundPipeline,
        "generate_processed_using_project",
        staticmethod(lambda *_args, **_kwargs: [sound]),
    )

    event_queue = EventQueue()
    model_worker_module._run_tts_preview_command(
        command, event_queue, CancellationEvent()
    )

    event = event_queue.events[0]
    assert isinstance(event, TtsPreviewFinished)
    assert event.status is GenerationTerminalStatus.FAILED
    assert event.sound is None
    assert event.message == "Model outputted NaN"


def _make_preview_command() -> TtsPreviewCommand:
    return TtsPreviewCommand(
        operation_id="preview-job",
        project_dir="/project",
        settings=GenerationSettings("disabled", "cpu_int8_float32", False, "none", "", False),
        prompt="Original word: foo. Substitute word: bar",
        apply_word_substitutions=False,
    )


class _PreviewCancellationEvent:
    def is_set(self) -> bool:
        return False

    def clear(self) -> None:
        pass


class _PreviewEventQueue:
    def __init__(self) -> None:
        self.events: list[object] = []

    def put(self, event: object) -> None:
        self.events.append(event)


def _install_preview_state(monkeypatch) -> None:
    project = SimpleNamespace(kill=lambda: None)
    monkeypatch.setattr(
        model_worker_module,
        "_make_worker_state",
        lambda _command: SimpleNamespace(project=project),
    )
    monkeypatch.setattr(Tts, "clear_continuation", staticmethod(lambda: None))
    monkeypatch.setattr(
        Tts, "reset_voice_selection_index", staticmethod(lambda: None)
    )


@pytest.mark.parametrize("cancel_during_notice", [False, True])
def test_audio_cpp_preview_reports_init_before_inference(monkeypatch, cancel_during_notice: bool) -> None:
    from tts_audiobook_tool.model_manager import ModelManager
    from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport
    from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

    _install_preview_state(monkeypatch)
    definition = load_audio_cpp_definitions().models["breeze_tts_2_audiocpp"]
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "exact-server-id")
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id(definition.spec.id))
    monkeypatch.setattr(Tts, "get_instance", lambda: adapter)
    monkeypatch.setattr(model_worker_module, "_should_watch_preview_inference", lambda: False)
    monkeypatch.setattr(model_worker_module, "_preview_inferences_this_process", 0)
    cancellation_event = threading.Event()
    calls: list[str] = []
    sound = Sound(np.zeros(8, dtype=np.float32), 24_000)

    def notice() -> None:
        calls.append("notice")
        if cancel_during_notice:
            cancellation_event.set()

    def generate(*_args, **_kwargs):
        calls.append("generate")
        return [sound]

    monkeypatch.setattr(adapter, "print_init_if_unloaded", notice)
    monkeypatch.setattr(SoundPipeline, "generate_processed_using_project", generate)
    monkeypatch.setattr(ModelManager, "warm_up_models", lambda *_args, **_kwargs: pytest.fail("Preview must not warm STT/YAMNet"))
    event_queue = _PreviewEventQueue()
    model_worker_module._run_tts_preview_command(
        _make_preview_command(), event_queue, cancellation_event
    )

    assert calls == (["notice"] if cancel_during_notice else ["notice", "generate"])
    assert len(event_queue.events) == 1
    event = event_queue.events[0]
    assert isinstance(event, TtsPreviewFinished)
    assert event.status is (GenerationTerminalStatus.CANCELLED if cancel_during_notice else GenerationTerminalStatus.COMPLETED)
    assert event.sound is (None if cancel_during_notice else sound)


def test_should_watch_preview_inference_exempts_model_setup_calls(monkeypatch) -> None:
    """Only a call that cannot be doing model setup is watched."""
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(model_worker_module, "_preview_inferences_this_process", 0)
    monkeypatch.setattr(
        Tts, "get_instance_if_exists", staticmethod(lambda: object())
    )

    # First preview of the worker process: may load/compile/download the model.
    assert model_worker_module._should_watch_preview_inference() is False

    monkeypatch.setattr(model_worker_module, "_preview_inferences_this_process", 3)
    # A resident model means the upcoming call is pure inference.
    assert model_worker_module._should_watch_preview_inference() is True

    # No resident model (eg after `Options > Unload models`): this call has to
    # load it, so it keeps the exemption.
    monkeypatch.setattr(Tts, "get_instance_if_exists", staticmethod(lambda: None))
    assert model_worker_module._should_watch_preview_inference() is False


def test_run_tts_preview_relays_generation_events_as_generation_updates(
    monkeypatch,
) -> None:
    """The watchdog's events must reach the parent as GenerationUpdate."""
    _install_preview_state(monkeypatch)
    monkeypatch.setattr(
        model_worker_module, "_should_watch_preview_inference", lambda: False
    )
    sound = Sound(np.zeros(8, dtype=np.float32), 24_000)

    def fake_generate(*_args: object, **_kwargs: object) -> list[Sound]:
        # Stands in for the watchdog thread: the sink installed by the command
        # is what carries the event to the parent.
        GenerationEvents.emit(GenerationTimedOut(timeout_seconds=180.0))
        return [sound]

    monkeypatch.setattr(
        SoundPipeline, "generate_processed_using_project", staticmethod(fake_generate)
    )

    event_queue = _PreviewEventQueue()
    model_worker_module._run_tts_preview_command(
        _make_preview_command(), event_queue, _PreviewCancellationEvent()
    )

    assert event_queue.events[0] == GenerationUpdate(
        "preview-job", GenerationTimedOut(timeout_seconds=180.0)
    )
    assert isinstance(event_queue.events[1], TtsPreviewFinished)


def test_run_tts_preview_reports_a_watchdog_timeout(monkeypatch, capsys) -> None:
    """A watched preview that outlives GEN_TIMEOUT fails and asks for a reset."""
    _install_preview_state(monkeypatch)
    monkeypatch.setattr(
        model_worker_module, "_should_watch_preview_inference", lambda: True
    )
    monkeypatch.setattr(gen_timeout_util, "GEN_TIMEOUT", 0.2)
    monkeypatch.setattr(Tts, "is_remote_mode", staticmethod(lambda: False))

    def slow_generate(*_args: object, **_kwargs: object) -> list[Sound]:
        time.sleep(0.6)
        return [Sound(np.zeros(8, dtype=np.float32), 24_000)]

    monkeypatch.setattr(
        SoundPipeline, "generate_processed_using_project", staticmethod(slow_generate)
    )

    event_queue = _PreviewEventQueue()
    model_worker_module._run_tts_preview_command(
        _make_preview_command(), event_queue, _PreviewCancellationEvent()
    )

    updates = [
        event
        for event in event_queue.events
        if isinstance(event, GenerationUpdate)
    ]
    assert updates == [GenerationUpdate("preview-job", GenerationTimedOut(0.2))]
    finished = event_queue.events[-1]
    assert isinstance(finished, TtsPreviewFinished)
    assert finished.status is GenerationTerminalStatus.FAILED
    assert finished.sound is None
    assert "GEN_TIMEOUT" in finished.message
    assert "reset" in finished.message
    assert "GEN_TIMEOUT" in capsys.readouterr().out


def test_run_tts_preview_counts_inferences_for_the_watchdog(monkeypatch) -> None:
    """Each completed preview advances the process-local inference count."""
    _install_preview_state(monkeypatch)
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(model_worker_module, "_preview_inferences_this_process", 0)
    monkeypatch.setattr(
        model_worker_module, "_should_watch_preview_inference", lambda: False
    )
    monkeypatch.setattr(
        SoundPipeline,
        "generate_processed_using_project",
        staticmethod(
            lambda *_args, **_kwargs: [Sound(np.zeros(8, dtype=np.float32), 24_000)]
        ),
    )

    for _ in range(2):
        model_worker_module._run_tts_preview_command(
            _make_preview_command(), _PreviewEventQueue(), _PreviewCancellationEvent()
        )

    assert model_worker_module._preview_inferences_this_process == 2


def test_inspect_tts_queues_unsaved_model_params(tmp_path, monkeypatch) -> None:
    """Validation must inspect live edits, not only project.json on disk."""
    monkeypatch.setattr(L, "d", lambda *_: None)
    project = Project(dir_path=str(tmp_path))
    assert project.save() == ""
    project.tts_model_type = "vibevoice_local"
    set_setting(project, "vibevoice_lora_target", "vibevoice-community/unsaved-adapter")
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
        return TtsInspected(operation_id, "vibevoice_local")

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
    assert command.model_params["vibevoice_lora_path"] == get_setting(project, "vibevoice_lora_target")
    assert command.settings.tts_model_type_id == "vibevoice_local"
    assert command.warm_models is False
    assert command.warm_stt is False


def test_inspect_tts_chat_warm_up_flags_are_carried(tmp_path, monkeypatch) -> None:
    """Mic-input chat asks the worker to run the shared warm-up and to warm STT."""
    monkeypatch.setattr(L, "d", lambda *_: None)
    project = Project(dir_path=str(tmp_path))
    assert project.save() == ""
    state = SimpleNamespace(
        project=project,
        prefs=Prefs(project_dir=str(tmp_path), stt_variant=SttVariant.LARGE_V3),
    )
    commands: list[object] = []

    class CommandQueue:
        def put(self, command: object) -> None:
            commands.append(command)

    def wait_for_result(cls, operation_id, _expected_type):
        cls._active_operation_id = None
        return TtsInspected(operation_id, "vibevoice_local")

    monkeypatch.setattr(ModelWorker, "start", classmethod(lambda cls: ""))
    monkeypatch.setattr(ModelWorker, "_command_queue", CommandQueue())
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(
        ModelWorker,
        "_wait_for_blocking_result",
        classmethod(wait_for_result),
    )

    _ = ModelWorker.inspect_tts_blocking(state, warm_models=True, warm_stt=True)

    command = commands[0]
    assert isinstance(command, InspectTtsCommand)
    assert command.warm_models is True
    assert command.warm_stt is True


@pytest.mark.parametrize("kind", ["generation", "preview", "realtime", "chat", "inspection"])
def test_command_producers_snapshot_live_project_selection(kind, tmp_path, monkeypatch) -> None:
    """All command paths use unsaved selection, not global prefs or disk state."""
    project = Project(dir_path=str(tmp_path))
    assert project.save() == ""
    project.tts_model_type = "higgs_v3_audiocpp"
    prefs = Prefs(
        project_dir=str(tmp_path), stt_variant=SttVariant.DISABLED,
        remote_tts_url="http://example.test:9009", tts_force_cpu=True,
        save_debug_files=True,
    )
    state = SimpleNamespace(project=project, prefs=prefs)
    commands = queue.Queue()
    monkeypatch.setattr(ModelWorker, "start", classmethod(lambda cls: ""))
    monkeypatch.setattr(ModelWorker, "_start_with_console_handler", classmethod(lambda cls, handler: ""))
    monkeypatch.setattr(ModelWorker, "_command_queue", commands)
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(ModelWorker, "_cancellation_event", threading.Event())
    monkeypatch.setattr(ModelWorker, "_continue_event", threading.Event())
    monkeypatch.setattr(
        ModelWorker, "_wait_for_blocking_result_with_handler",
        classmethod(lambda cls, operation_id, expected, handler: TtsInspected(operation_id, "bound-model")),
    )
    monkeypatch.setattr(
        ModelWorker, "get_event",
        classmethod(lambda cls, timeout: model_worker_module.ChatSynthesisFinished(cls._active_operation_id)),
    )

    if kind == "generation":
        ModelWorker.submit_generation(state=state, indices={1}, batch_size=1, is_regen=False)
    elif kind == "preview":
        ModelWorker.submit_tts_preview(state=state, prompt="Hello")
    elif kind == "realtime":
        ModelWorker.submit_realtime_playback(state=state, phrase_groups=[], line_range=None)
    elif kind == "chat":
        ModelWorker.synthesize_chat_blocking(state, "Hello", None, streaming=False)
    else:
        ModelWorker.inspect_tts_blocking(state)

    command = commands.get_nowait()
    assert command.settings.tts_model_type_id == "higgs_v3_audiocpp"
    assert command.settings.remote_tts_url == "http://example.test:9009"
    assert command.settings.stt_variant_id == SttVariant.DISABLED.id
    assert command.settings.tts_force_cpu is True
    assert command.settings.save_debug_files is True
    assert {field.name for field in fields(command.settings)} == {
        "stt_variant_id", "stt_config_id", "tts_force_cpu", "tts_model_type_id",
        "remote_tts_url", "save_debug_files",
    }
    # A queued snapshot remains independent of further live edits.
    project.tts_model_type = "vibevoice_local"
    assert command.settings.tts_model_type_id == "higgs_v3_audiocpp"


def test_worker_state_overrides_disk_selection_before_binding(tmp_path, monkeypatch) -> None:
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.tts import TtsRuntimeMode

    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    project = Project(dir_path=str(tmp_path), tts_model_type="vibevoice_local")
    project._sound_segments = SimpleNamespace(dont_show_scan_message=False)
    calls = []
    monkeypatch.setattr(
        ProjectLoadUtil, "load_using_dir_path",
        staticmethod(lambda *args, **kwargs: calls.append("load") or project),
    )
    monkeypatch.setattr(
        Project, "save", lambda self: pytest.fail("worker snapshots must not save project settings"),
    )
    monkeypatch.setattr(
        Tts, "bind_project",
        staticmethod(lambda selected, **kwargs: calls.append(("bind", selected.tts_model_type))),
    )
    monkeypatch.setattr(
        RemoteTtsDiscovery, "refresh",
        classmethod(lambda cls, force=False: calls.append(("refresh", force))),
    )
    command = TtsPreviewCommand(
        "snapshot", str(tmp_path),
        GenerationSettings("disabled", "cpu_int8_float32", False, "higgs_v3_audiocpp", "http://example.test", False),
        "Hello",
    )

    state = model_worker_module._make_worker_state(command)

    assert state.project is project
    assert calls == [("refresh", True), "load", ("bind", "higgs_v3_audiocpp")]
    assert state.prefs.remote_tts_url == "http://example.test"
    assert not hasattr(state.prefs, "remote_tts_type")
    assert not hasattr(state.prefs, "remote_tts_model_id")


def test_local_worker_state_ignores_saved_remote_url(tmp_path, monkeypatch) -> None:
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.tts import TtsRuntimeMode
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

    model_type = TtsModelType.require_by_id("chatterbox_local")
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_available_local_models", (model_type,))
    project = Project(dir_path=str(tmp_path), tts_model_type="vibevoice_local")
    project._sound_segments = SimpleNamespace(dont_show_scan_message=False)
    monkeypatch.setattr(
        ProjectLoadUtil, "load_using_dir_path", staticmethod(lambda *args, **kwargs: project),
    )
    monkeypatch.setattr(
        Project, "save", lambda self: pytest.fail("worker snapshots must not save project settings"),
    )
    monkeypatch.setattr(
        RemoteTtsDiscovery, "refresh",
        classmethod(lambda cls, force=False: pytest.fail("local workers must not probe remote endpoints")),
    )
    command = TtsPreviewCommand(
        "local-snapshot", str(tmp_path),
        GenerationSettings("disabled", "cpu_int8_float32", False, model_type.id, "http://offline.example", False),
        "Hello",
    )

    state = model_worker_module._make_worker_state(command)

    assert state.project is project
    assert state.project.tts_model_type == model_type.id
    assert state.prefs.remote_tts_url == "http://offline.example"
    assert Tts.get_active_type() is model_type
    assert Tts._binding_issue is None


def test_chat_cached_project_rebinds_each_command_selection_without_forced_probe(tmp_path, monkeypatch) -> None:
    import tts_audiobook_tool.tts as tts_module
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
    from tts_audiobook_tool.model_runtime import ModelRuntimeRole
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

    (tmp_path / "project.json").write_text("{}", encoding="utf-8")
    project = Project(dir_path=str(tmp_path), tts_model_type="vibevoice_local")
    project._sound_segments = SimpleNamespace(dont_show_scan_message=False)
    loads = []
    bound_ids = []
    monkeypatch.setattr(model_worker_module, "_chat_project_cache", None)
    monkeypatch.setattr(
        ProjectLoadUtil, "load_using_dir_path",
        staticmethod(lambda *args, **kwargs: loads.append(True) or project),
    )
    monkeypatch.setattr(
        Project, "save", lambda self: pytest.fail("chat snapshots must not save project settings"),
    )
    model_types = [TtsModelType.require_by_id("higgs_v3_audiocpp"), TtsModelType.require_by_id("chatterbox_audiocpp")]
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        candidates=tuple(zip(model_types, (
            "entry-server_higgs_v3_audio_cpp", "entry-server_chatterbox_audio_cpp_v3",
        ))),
    )
    cleared_types = []
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    monkeypatch.setattr(Tts, "_selected_server_model_id", "")
    monkeypatch.setattr(Tts, "_binding_issue", None)
    monkeypatch.setattr(Tts, "_remote_issue", "")
    monkeypatch.setattr(tts_module.SglOmniUtil, "_model_id", "")
    monkeypatch.setattr(tts_module, "current_role", lambda: ModelRuntimeRole.MODEL_WORKER)
    monkeypatch.setattr(Tts, "is_remote_mode", staticmethod(lambda: True))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", classmethod(lambda cls: snapshot))
    monkeypatch.setattr(Tts, "clear_tts_model", staticmethod(lambda: cleared_types.append(Tts.get_active_type())))
    monkeypatch.setattr(
        Tts, "set_model_params_using_project",
        staticmethod(lambda selected: bound_ids.append(selected.tts_model_type)),
    )
    refresh_modes = []
    def cached_refresh(cls, force=False):
        refresh_modes.append(force)
        assert not force, "chat sentences must not force discovery"
        return snapshot
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", classmethod(cached_refresh))
    settings = GenerationSettings("disabled", "cpu_int8_float32", False, "higgs_v3_audiocpp", "http://example.test", False)
    command = model_worker_module.SynthesizeChatCommand("first", str(tmp_path), settings, "Hello", False, None)

    first = model_worker_module._make_chat_worker_state(command)
    assert Tts.get_active_type() is TtsModelType.require_by_id("higgs_v3_audiocpp")
    unchanged = model_worker_module._make_chat_worker_state(replace(command, operation_id="unchanged"))
    assert cleared_types == [TtsModelType.require_by_id("none")]
    second = model_worker_module._make_chat_worker_state(replace(
        command, operation_id="second",
        settings=replace(settings, tts_model_type_id="chatterbox_audiocpp"),
    ))

    assert first.project is unchanged.project is second.project is project
    assert loads == [True]
    assert bound_ids == ["higgs_v3_audiocpp", "higgs_v3_audiocpp", "chatterbox_audiocpp"]
    assert project.tts_model_type == "chatterbox_audiocpp"
    assert Tts.get_active_type() is TtsModelType.require_by_id("chatterbox_audiocpp")
    assert Tts._selected_server_model_id == "entry-server_chatterbox_audio_cpp_v3"
    assert cleared_types == [TtsModelType.require_by_id("none"), TtsModelType.require_by_id("higgs_v3_audiocpp")]
    assert refresh_modes == [False, False, False]


def test_worker_inspection_preserves_unsaved_params_after_selection_binding(tmp_path, monkeypatch) -> None:
    import tts_audiobook_tool.app_support as app_support
    import tts_audiobook_tool.model_runtime as model_runtime
    from tts_audiobook_tool.model_manager import ModelManager
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

    project = Project(dir_path=str(tmp_path), tts_model_type="none")
    project._sound_segments = SimpleNamespace(dont_show_scan_message=False, observer=SimpleNamespace(stop=lambda: None))
    calls = []
    instance = SimpleNamespace(
        get_device_type=lambda: None,
        get_warning_issues=lambda selected: [],
        has_lora=lambda: True,
    )
    monkeypatch.setattr(model_runtime, "mark_model_worker", lambda: None)
    monkeypatch.setattr(model_worker_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(model_worker_module._WorkerOutputCapture, "install", lambda self: None)
    monkeypatch.setattr(app_support, "init_logging", lambda *args: None)
    monkeypatch.setattr(Tts, "init_local_model_type", staticmethod(lambda: None))
    monkeypatch.setattr(Tts, "_config_fingerprint", "")
    monkeypatch.setattr(
        ProjectLoadUtil, "load_using_dir_path", staticmethod(lambda *args, **kwargs: project),
    )
    monkeypatch.setattr(
        Tts, "bind_project", staticmethod(lambda selected, **kwargs: calls.append(("bind", selected.tts_model_type))),
    )
    monkeypatch.setattr(Tts, "set_model_params", staticmethod(lambda params: calls.append(("params", params))))
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: calls.append("instance") or instance))
    monkeypatch.setattr(Tts, "get_active_type", staticmethod(lambda: TtsModelType.require_by_id("vibevoice_local")))
    monkeypatch.setattr(
        Tts, "get_model_support",
        staticmethod(lambda selected: SimpleNamespace(get_blocking_issues=lambda p, i: [])),
    )
    monkeypatch.setattr(ModelManager, "clear_all_models", staticmethod(lambda: None))
    command = InspectTtsCommand(
        "inspection", str(tmp_path),
        GenerationSettings("disabled", "cpu_int8_float32", False, "vibevoice_local", "", False),
        {"vibevoice_lora_path": "unsaved-adapter"},
    )
    commands = queue.Queue()
    commands.put(command)
    commands.put(model_worker_module.ShutdownCommand("shutdown"))
    events = queue.Queue()

    model_worker_module._model_worker_main(commands, events, threading.Event(), threading.Event())

    assert calls == [("bind", "vibevoice_local"), ("params", command.model_params), "instance"]
    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    inspections = [event for event in emitted if isinstance(event, TtsInspected)]
    assert len(inspections) == 1
    assert inspections[0].tts_type_id == "vibevoice_local"
    assert inspections[0].metadata["has_lora"] is True


def test_release_chat_synthesis_state_clears_callbacks_and_transient_memory(
    monkeypatch,
) -> None:
    calls: list[str] = []
    model = SimpleNamespace(
        clear_stream_state=lambda: calls.append("stream"),
        release_inference_memory=lambda: calls.append("memory"),
    )
    monkeypatch.setattr(Tts, "get_instance_if_exists", staticmethod(lambda: model))

    model_worker_module._release_chat_synthesis_state()

    assert calls == ["stream", "memory"]


def test_load_chat_models_delegates_to_shared_warm_up(monkeypatch) -> None:
    """Worker chat warm-up routes through ModelManager.warm_up_models, so it
    emits the generation flow's init messaging: the "Warming up models..."
    banner whenever the (silent) TTS load is needed - including a cold
    text-input chat - and the STT init line only when STT is actually
    created for microphone input."""
    import tts_audiobook_tool.model_manager as model_manager_module
    import tts_audiobook_tool.stt as stt_module

    state = SimpleNamespace(project=None, prefs=None)

    def run_case(
        *,
        tts_resident: bool,
        stt_resident: bool,
        warm_stt: bool,
        skip_reason: str,
    ) -> tuple[list[str], list[str]]:
        messages: list[str] = []
        calls: list[str] = []
        monkeypatch.setattr(
            Tts, "instance_exists", staticmethod(lambda: tts_resident)
        )
        monkeypatch.setattr(
            Tts, "get_instance", staticmethod(lambda: "tts-instance")
        )
        monkeypatch.setattr(
            stt_module.Stt, "has_instance", staticmethod(lambda: stt_resident)
        )
        monkeypatch.setattr(
            stt_module.Stt, "should_skip", staticmethod(lambda _state: skip_reason)
        )
        monkeypatch.setattr(
            stt_module.Stt,
            "eager_warm_up_for_inference",
            staticmethod(lambda: calls.append("stt-warm") or None),
        )
        # warm_up_models imports print_init from util into its own namespace.
        monkeypatch.setattr(
            model_manager_module, "print_init", lambda s: messages.append(s)
        )
        instance = model_worker_module._load_chat_models(state, warm_stt=warm_stt)
        assert instance == "tts-instance"
        return messages, calls

    # Cold mic-mode chat: TTS and STT both load -> banner, then STT warm-up.
    messages, calls = run_case(
        tts_resident=False, stt_resident=False, warm_stt=True, skip_reason=""
    )
    assert messages == ["Warming up models..."]
    assert calls == ["stt-warm"]

    # TTS already resident: no banner, STT still warms up (its own line).
    messages, calls = run_case(
        tts_resident=True, stt_resident=False, warm_stt=True, skip_reason=""
    )
    assert messages == []
    assert calls == ["stt-warm"]

    # Both resident: nothing happens.
    messages, calls = run_case(
        tts_resident=True, stt_resident=True, warm_stt=True, skip_reason=""
    )
    assert messages == []
    assert calls == []

    # Cold text-mode chat: STT is never touched, but the silent TTS load is
    # still announced.
    messages, calls = run_case(
        tts_resident=False, stt_resident=False, warm_stt=False, skip_reason=""
    )
    assert messages == ["Warming up models..."]
    assert calls == []

    # Text-mode chat with TTS already resident: nothing to announce.
    messages, calls = run_case(
        tts_resident=True, stt_resident=False, warm_stt=False, skip_reason=""
    )
    assert messages == []
    assert calls == []

    # Mic mode but STT skipped by config: TTS banner only, no STT warm-up.
    messages, calls = run_case(
        tts_resident=False,
        stt_resident=False,
        warm_stt=True,
        skip_reason="Whisper disabled",
    )
    assert messages == ["Warming up models..."]
    assert calls == []


def test_load_chat_models_reports_warm_up_failure(monkeypatch) -> None:
    """A failed shared warm-up surfaces as a command failure, not a silent
    partial load."""
    import tts_audiobook_tool.model_manager as model_manager_module
    from tts_audiobook_tool.app_types import ModelWarmUpResult

    state = SimpleNamespace(project=None, prefs=None)
    monkeypatch.setattr(
        model_manager_module.ModelManager,
        "warm_up_models",
        staticmethod(lambda *args, **kwargs: ModelWarmUpResult(error="boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        model_worker_module._load_chat_models(state, warm_stt=True)


def test_blocking_wait_routes_output_and_flush_to_handler(monkeypatch) -> None:
    events = iter(
        [
            ConsoleOutput("chat-op", "stdout", "loading\r"),
            ConsoleFlush("chat-op", "stdout"),
            TtsInspected("chat-op", "tts"),
        ]
    )
    monkeypatch.setattr(
        ModelWorker,
        "get_event",
        classmethod(lambda cls, timeout=None: next(events)),
    )
    handled: list[ConsoleOutput | ConsoleFlush] = []

    result = ModelWorker._wait_for_blocking_result(
        "chat-op", TtsInspected, handled.append
    )

    assert isinstance(result, TtsInspected)
    assert handled == [
        ConsoleOutput("chat-op", "stdout", "loading\r"),
        ConsoleFlush("chat-op", "stdout"),
    ]

