from __future__ import annotations

import codecs
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from multiprocessing.context import BaseContext
from typing import Any, Callable, TextIO

from tts_audiobook_tool.model_worker_protocol import (
    AudioFileUpsampled,
    AudioTranscribed,
    ChatAudioChunk,
    ChatSessionReset,
    ChatSynthesisFinished,
    ClearModelsCommand,
    ConsoleFlush,
    ConsoleOutput,
    CreateOuteSpeakerCommand,
    GenerateCommand,
    GenerationFinished,
    GenerationSettings,
    GenerationTerminalStatus,
    GenerationUpdate,
    InspectTtsCommand,
    GetModelStateCommand,
    LavaSrProbed,
    ModelsCleared,
    ModelStateReported,
    ModelStateSnapshot,
    OuteSpeakerCreated,
    TtsInspected,
    ModelWorkerCommand,
    ModelWorkerEvent,
    RealTimePlaybackCommand,
    RealTimePlaybackFinished,
    RealTimePlaybackTerminalStatus,
    RealTimePlaybackUpdate,
    ProbeLavaSrCommand,
    ResetChatSessionCommand,
    ShutdownCommand,
    SynthesizeChatCommand,
    TranscribeAudioCommand,
    UpsampleFileCommand,
    WorkerCommandFailed,
    WorkerExited,
    WorkerReady,
    WorkerStatus,
    WorkerStopped,
)

WORKER_START_TIMEOUT_SECONDS = 20.0
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class ModelWorkerUnavailable(RuntimeError):
    pass


class _OperationTracker:
    def __init__(self) -> None:
        self._operation_id = ""
        self._lock = threading.Lock()

    def set(self, operation_id: str) -> None:
        with self._lock:
            self._operation_id = operation_id

    def get(self) -> str:
        with self._lock:
            return self._operation_id


class _QueueTextStream:
    """Route Python-level stream writes directly to the worker event queue."""

    def __init__(
        self,
        original: TextIO,
        stream_name: str,
        event_queue: Any,
        tracker: _OperationTracker,
        was_tty: bool,
        on_capture_lost: Callable[[str], None] | None = None,
    ) -> None:
        self._original = original
        self._stream_name = stream_name
        self._event_queue = event_queue
        self._tracker = tracker
        self._was_tty = was_tty
        self._on_capture_lost = on_capture_lost
        self._capture_lost_reported = False

    @property
    def encoding(self) -> str:
        return self._original.encoding or "utf-8"

    @property
    def errors(self) -> str:
        return self._original.errors or "replace"

    def _capture_lost(self) -> None:
        # Best-effort one-shot marker so the transcript/UI shows where worker
        # console output was lost.
        if self._capture_lost_reported or self._on_capture_lost is None:
            return
        self._capture_lost_reported = True
        self._on_capture_lost(self._stream_name)

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            self._event_queue.put(
                ConsoleOutput(
                    operation_id=self._tracker.get(),
                    stream=self._stream_name,
                    text=data,
                )
            )
        except Exception:
            self._capture_lost()
        return len(data)

    def flush(self) -> None:
        try:
            self._event_queue.put(
                ConsoleFlush(
                    operation_id=self._tracker.get(),
                    stream=self._stream_name,
                )
            )
        except Exception:
            self._capture_lost()

    def isatty(self) -> bool:
        return self._was_tty

    def fileno(self) -> int:
        # The capture replaces fds 1/2 via os.dup2, so the original fd
        # numbers still refer to the capture pipes. Returning them means raw
        # os.write(fileno()) consumers are captured instead of writing
        # straight to the parent process's terminal.
        return self._original.fileno()

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


class _WorkerOutputCapture:
    """Capture Python streams and fd-level writes without touching the TUI tty."""

    def __init__(self, event_queue: Any, tracker: _OperationTracker) -> None:
        self.event_queue = event_queue
        self.tracker = tracker
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._sentinel_lock = threading.Lock()
        self._sentinel_sent = False

    def install(self) -> None:
        stdout_tty = self.original_stdout.isatty()
        stderr_tty = self.original_stderr.isatty()
        self._redirect_fd(1, "stdout", self.original_stdout.encoding or "utf-8")
        self._redirect_fd(2, "stderr", self.original_stderr.encoding or "utf-8")
        sys.stdout = _QueueTextStream(
            self.original_stdout,
            "stdout",
            self.event_queue,
            self.tracker,
            stdout_tty,
            self._send_capture_lost_sentinel,
        )  # type: ignore[assignment]
        sys.stderr = _QueueTextStream(
            self.original_stderr,
            "stderr",
            self.event_queue,
            self.tracker,
            stderr_tty,
            self._send_capture_lost_sentinel,
        )  # type: ignore[assignment]

    def _send_capture_lost_sentinel(self, stream_name: str) -> None:
        """Ship one best-effort marker when console capture is interrupted.

        Both the stream wrappers and the fd reader threads funnel through
        here, so a worker whose event queue died (or whose capture pipe
        broke) still tells the main process where output stopped. If the
        queue is unusable as well the marker is dropped silently: the main
        process' ``WorkerExited`` diagnostics identify the same boundary.
        """
        with self._sentinel_lock:
            if self._sentinel_sent:
                return
            self._sentinel_sent = True
        try:
            self.event_queue.put(
                ConsoleOutput(
                    operation_id=self.tracker.get(),
                    stream=stream_name,
                    text=f"[worker console capture lost: {stream_name}]\n",
                )
            )
        except Exception:
            pass

    def _redirect_fd(self, fd: int, stream_name: str, encoding: str) -> None:
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)
        thread = threading.Thread(
            target=self._read_fd,
            args=(read_fd, stream_name, encoding),
            name=f"model-worker-{stream_name}",
            daemon=True,
        )
        thread.start()

    def _read_fd(self, read_fd: int, stream_name: str, encoding: str) -> None:
        decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
        try:
            while True:
                chunk = os.read(read_fd, 8192)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    self.event_queue.put(
                        ConsoleOutput(
                            operation_id=self.tracker.get(),
                            stream=stream_name,
                            text=text,
                        )
                    )
            tail = decoder.decode(b"", final=True)
            if tail:
                self.event_queue.put(
                    ConsoleOutput(
                        operation_id=self.tracker.get(),
                        stream=stream_name,
                        text=tail,
                    )
                )
        except Exception:
            self._send_capture_lost_sentinel(stream_name)
        finally:
            os.close(read_fd)


def _make_worker_state(
    command: GenerateCommand | RealTimePlaybackCommand | InspectTtsCommand | CreateOuteSpeakerCommand | SynthesizeChatCommand,
) -> Any:
    from tts_audiobook_tool.app_types import SttConfig, SttVariant
    from tts_audiobook_tool.prefs import Prefs
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.state import State
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

    stt_variant = SttVariant.get_by_id(command.settings.stt_variant_id)
    stt_config = SttConfig.from_id(command.settings.stt_config_id)
    if stt_variant is None or stt_config is None:
        raise ValueError("Generation settings contain an unsupported STT configuration")
    sgl_type = (
        None
        if command.settings.sgl_omni_type_id is None
        else TtsModelType.get_by_id(command.settings.sgl_omni_type_id)
    )
    prefs = Prefs(
        project_dir=command.project_dir,
        stt_variant=stt_variant,
        stt_config=stt_config,
        tts_force_cpu=command.settings.tts_force_cpu,
        sgl_omni_type=sgl_type,
        sgl_omni_url=command.settings.sgl_omni_url,
        save_debug_files=command.settings.save_debug_files,
    )

    # The worker builds a process-local State without invoking the interactive
    # startup path (which loads a project from mutable global prefs). Property
    # setters are still used, so process-local model configuration is
    # synchronized exactly as it is in the conventional application.
    state = State.for_worker(prefs)
    project = ProjectLoadUtil.load_using_dir_path(
        command.project_dir,
        prompt_on_warnings=False,
    )
    if isinstance(project, str):
        raise RuntimeError(project)
    try:
        state.project = project
    except BaseException:
        project.kill()
        raise
    return state


def _run_generate_command(
    command: GenerateCommand,
    event_queue: Any,
    cancellation_event: Any,
) -> None:
    from tts_audiobook_tool.app_support.interrupts import Interrupts
    from tts_audiobook_tool.generate_util import GenerateUtil
    from tts_audiobook_tool.generation_events import GenerationEvents

    state = None
    interrupts = Interrupts()
    interrupts.set_external_event(cancellation_event)
    try:
        state = _make_worker_state(command)
        with GenerationEvents.using_sink(
            lambda update: event_queue.put(
                GenerationUpdate(command.operation_id, update)
            )
        ):
            did_stop = GenerateUtil.generate_files(
                state=state,
                indices_set=set(command.indices),
                batch_size=command.batch_size,
                is_regen=command.is_regen,
            )
        if did_stop:
            status = (
                GenerationTerminalStatus.CANCELLED
                if cancellation_event.is_set()
                else GenerationTerminalStatus.ABORTED
            )
        else:
            status = GenerationTerminalStatus.COMPLETED
        event_queue.put(
            GenerationFinished(
                operation_id=command.operation_id,
                status=status,
                remaining_range_string=state.project.generate_range_string,
            )
        )
    except Exception as exception:
        traceback.print_exc()
        remaining_range = (
            state.project.generate_range_string if state is not None else ""
        )
        event_queue.put(
            GenerationFinished(
                operation_id=command.operation_id,
                status=GenerationTerminalStatus.FAILED,
                remaining_range_string=remaining_range,
                message=f"{type(exception).__name__}: {exception}",
            )
        )
    finally:
        interrupts.clear()
        interrupts.set_external_event(None)
        if state is not None:
            state.project.kill()


def _run_realtime_playback_command(
    command: RealTimePlaybackCommand,
    event_queue: Any,
    cancellation_event: Any,
    continue_event: Any,
) -> None:
    from tts_audiobook_tool.app_support.interrupts import Interrupts
    from tts_audiobook_tool.app_types.phrase import PhraseGroup
    from tts_audiobook_tool.generation_events import (
        GenerationEvents,
        GenerationPhase,
        GenerationTimedOut,
    )
    from tts_audiobook_tool.real_time_playback import (
        RealTimePlaybackRunStatus,
        start,
    )
    from tts_audiobook_tool.real_time_playback_events import RealTimePlaybackEvents

    state = None
    interrupts = Interrupts()
    interrupts.set_external_event(cancellation_event)

    def relay(update: object) -> None:
        if isinstance(update, (GenerationPhase, GenerationTimedOut)):
            event_queue.put(RealTimePlaybackUpdate(command.operation_id, update))

    try:
        state = _make_worker_state(command)
        phrase_groups = PhraseGroup.phrase_groups_from_json_list(
            list(command.phrase_groups_json)
        )
        if isinstance(phrase_groups, str):
            raise ValueError(phrase_groups)
        with GenerationEvents.using_sink(relay), RealTimePlaybackEvents.using_sink(
            lambda update: event_queue.put(
                RealTimePlaybackUpdate(command.operation_id, update)
            )
        ):
            run_result = start(
                state=state,
                phrase_groups=phrase_groups,
                line_range=command.line_range,
                continue_event=continue_event,
            )
        status_by_result = {
            RealTimePlaybackRunStatus.COMPLETED: RealTimePlaybackTerminalStatus.COMPLETED,
            RealTimePlaybackRunStatus.CANCELLED: RealTimePlaybackTerminalStatus.CANCELLED,
            RealTimePlaybackRunStatus.ABORTED: RealTimePlaybackTerminalStatus.ABORTED,
            RealTimePlaybackRunStatus.FAILED: RealTimePlaybackTerminalStatus.FAILED,
        }
        event_queue.put(
            RealTimePlaybackFinished(
                operation_id=command.operation_id,
                status=status_by_result[run_result.status],
                message=run_result.message,
            )
        )
    except Exception as exception:
        traceback.print_exc()
        event_queue.put(
            RealTimePlaybackFinished(
                operation_id=command.operation_id,
                status=RealTimePlaybackTerminalStatus.FAILED,
                message=f"{type(exception).__name__}: {exception}",
            )
        )
    finally:
        interrupts.clear()
        interrupts.set_external_event(None)
        if state is not None:
            state.project.kill()


def _terminate_worker(signum: int, frame: Any) -> None:
    """Handle the parent's pre-SIGKILL ``SIGTERM`` escalation gracefully.

    The default ``SIGTERM`` disposition terminates the process without running
    Python's atexit finalizers.  That leaks any multiprocessing semaphores the
    worker created while loading a model (the model library, not the shared
    queues/events, owns those).  Raising ``SystemExit`` instead unwinds the
    worker through its ``finally`` blocks and runs the atexit finalizers, so
    those semaphores are unregistered before the process goes away.
    """
    raise SystemExit(0)


def _model_worker_main(
    command_queue: Any,
    event_queue: Any,
    cancellation_event: Any,
    continue_event: Any,
) -> None:
    from tts_audiobook_tool.model_runtime import mark_model_worker

    mark_model_worker()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _terminate_worker)
    tracker = _OperationTracker()
    _WorkerOutputCapture(event_queue, tracker).install()

    try:
        from tts_audiobook_tool import app_support
        from tts_audiobook_tool.constants import APP_NAME
        from tts_audiobook_tool.tts import Tts

        app_support.init_logging(f"{APP_NAME}-worker")
        Tts.init_local_model_type()
        event_queue.put(WorkerReady(os.getpid()))
    except Exception as exception:
        traceback.print_exc()
        event_queue.put(WorkerCommandFailed("", f"Worker startup failed: {exception}"))
        return

    while True:
        try:
            command: ModelWorkerCommand = command_queue.get()
        except (EOFError, OSError):
            return

        tracker.set(command.operation_id)
        if isinstance(command, ShutdownCommand):
            try:
                from tts_audiobook_tool.model_manager import ModelManager

                ModelManager.clear_all_models()
            except Exception:
                traceback.print_exc()
            event_queue.put(WorkerStopped(command.operation_id))
            return
        if isinstance(command, GenerateCommand):
            _run_generate_command(command, event_queue, cancellation_event)
            tracker.set("")
            continue
        if isinstance(command, RealTimePlaybackCommand):
            _run_realtime_playback_command(
                command,
                event_queue,
                cancellation_event,
                continue_event,
            )
            tracker.set("")
            continue
        if isinstance(command, ResetChatSessionCommand):
            try:
                from tts_audiobook_tool.tts import Tts

                Tts.clear_continuation()
                if command.reset_voice_selection:
                    Tts.reset_voice_selection_index()
                model = Tts.get_instance_if_exists()
                if model is not None:
                    model.clear_stream_state()
                event_queue.put(ChatSessionReset(command.operation_id))
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            tracker.set("")
            continue
        if isinstance(command, SynthesizeChatCommand):
            state = None
            try:
                from tts_audiobook_tool.app_support.interrupts import Interrupts
                from tts_audiobook_tool.app_types.phrase import Reason
                from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
                from tts_audiobook_tool.tts import Tts

                state = _make_worker_state(command)
                interrupts = Interrupts()
                interrupts.set_external_event(cancellation_event)
                if command.streaming:
                    result = Tts.generate_using_project(
                        state.project,
                        [command.text],
                        on_stream_chunk=lambda data: event_queue.put(
                            ChatAudioChunk(command.operation_id, data)
                        ),
                        on_stream_end=lambda: None,
                    )
                    sound = None
                else:
                    result = SoundPipeline.generate_processed_using_project(
                        state.project, [command.text]
                    )
                    sound = None if isinstance(result, str) else result
                if isinstance(result, str):
                    Tts.clear_continuation()
                    event_queue.put(
                        ChatSynthesisFinished(command.operation_id, error=result)
                    )
                else:
                    if isinstance(command.reason, Reason):
                        Tts.clear_continuation_if_reason(command.reason)
                    event_queue.put(
                        ChatSynthesisFinished(command.operation_id, sound=sound)
                    )
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    ChatSynthesisFinished(
                        command.operation_id,
                        error=f"{type(exception).__name__}: {exception}",
                    )
                )
            finally:
                try:
                    from tts_audiobook_tool.app_support.interrupts import Interrupts
                    from tts_audiobook_tool.tts import Tts

                    model = Tts.get_instance_if_exists()
                    if model is not None:
                        model.clear_stream_state()
                    Interrupts().clear()
                    Interrupts().set_external_event(None)
                except Exception:
                    traceback.print_exc()
                if state is not None:
                    state.project.kill()
                cancellation_event.clear()
            tracker.set("")
            continue
        if isinstance(command, TranscribeAudioCommand):
            try:
                from tts_audiobook_tool.app_types import (
                    ConcreteSegment,
                    ConcreteWord,
                    SttConfig,
                    SttVariant,
                )
                from tts_audiobook_tool.stt import Stt

                variant = SttVariant.get_by_id(command.stt_variant_id)
                config = SttConfig.from_id(command.stt_config_id)
                if variant is None or config is None:
                    raise ValueError("Unsupported STT configuration")
                Stt.set_variant(variant)
                Stt.set_config(config)
                whisper = Stt.get_whisper()
                language_supported = (
                    command.language is None
                    or command.language in whisper.supported_languages
                )
                if not language_supported:
                    event_queue.put(
                        AudioTranscribed(command.operation_id, (), False)
                    )
                else:
                    with Stt.inference_lock:
                        raw_segments, _ = whisper.transcribe(
                            command.audio,
                            word_timestamps=command.word_timestamps,
                            language=command.language,
                        )
                        raw_segments = list(raw_segments)
                    segments = tuple(
                        ConcreteSegment(
                            start=float(segment.start),
                            end=float(segment.end),
                            text=str(segment.text),
                            words=[
                                ConcreteWord(
                                    start=float(word.start),
                                    end=float(word.end),
                                    word=str(word.word),
                                    probability=float(word.probability),
                                )
                                for word in (getattr(segment, "words", None) or [])
                            ],
                        )
                        for segment in raw_segments
                    )
                    event_queue.put(
                        AudioTranscribed(command.operation_id, segments, True)
                    )
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            tracker.set("")
            continue
        if isinstance(command, CreateOuteSpeakerCommand):
            state = None
            try:
                from tts_audiobook_tool.tts import Tts

                state = _make_worker_state(command)
                result = Tts.get_oute().create_speaker(command.source_path)
                if isinstance(result, str):
                    raise RuntimeError(result)
                event_queue.put(OuteSpeakerCreated(command.operation_id, result))
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            finally:
                if state is not None:
                    state.project.kill()
            tracker.set("")
            continue
        if isinstance(command, InspectTtsCommand):
            state = None
            try:
                from tts_audiobook_tool.tts import Tts

                state = _make_worker_state(command)
                # The project on disk can intentionally lag the interactive
                # state while a custom model or adapter is being validated.
                Tts.set_model_params(command.model_params)
                instance = Tts.get_instance()
                device_type = instance.get_device_type()
                device = device_type.value if device_type is not None else ""
                blocking_issues = tuple(
                    issue.verbose
                    for issue in Tts.get_class().get_blocking_issues(
                        state.project, instance
                    )
                )
                warnings = tuple(instance.get_warning_issues(state.project))
                metadata: dict[str, object] = {}
                for attribute in (
                    "model_type",
                    "supported_languages",
                    "supported_speakers",
                    "generate_defaults",
                    "is_model_type_supported",
                    "has_lora",
                ):
                    if hasattr(instance, attribute):
                        value = getattr(instance, attribute)
                        metadata[attribute] = value() if callable(value) else value
                supported_languages_multi = getattr(
                    instance, "supported_languages_multi", None
                )
                if callable(supported_languages_multi):
                    metadata["supported_languages_multi"] = (
                        supported_languages_multi()
                    )
                event_queue.put(
                    TtsInspected(
                        command.operation_id,
                        Tts.get_type().value.id,
                        device,
                        blocking_issues,
                        warnings,
                        metadata,
                    )
                )
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            finally:
                if state is not None:
                    state.project.kill()
            tracker.set("")
            continue
        if isinstance(command, ClearModelsCommand):
            try:
                from tts_audiobook_tool.model_manager import ModelManager

                ModelManager.clear_all_models()
                event_queue.put(ModelsCleared(command.operation_id))
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            tracker.set("")
            continue
        if isinstance(command, GetModelStateCommand):
            try:
                from tts_audiobook_tool.model_manager import ModelManager
                from tts_audiobook_tool.stt import Stt
                from tts_audiobook_tool.tts import Tts

                tts_instance = Tts.get_instance_if_exists()
                tts_device = ""
                if tts_instance is not None:
                    device_type = tts_instance.get_device_type()
                    if device_type is not None:
                        tts_device = device_type.value
                event_queue.put(
                    ModelStateReported(
                        command.operation_id,
                        ModelStateSnapshot(
                            tts_loaded=tts_instance is not None,
                            tts_type_id=Tts.get_type().value.id,
                            tts_device=tts_device,
                            stt_loaded=Stt.has_instance(),
                            stt_variant_id=Stt.get_variant().id,
                            stt_device=Stt.get_config().device,
                            yamnet_loaded=ModelManager.has_yamnet_detector(),
                            lava_sr_loaded=ModelManager.lava_sr_upsampler is not None,
                        ),
                    )
                )
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            tracker.set("")
            continue
        if isinstance(command, ProbeLavaSrCommand):
            try:
                from tts_audiobook_tool.sound.lava_sr_util import LavaSrUtil

                event_queue.put(
                    LavaSrProbed(command.operation_id, LavaSrUtil.has_lava_sr())
                )
            except Exception as exception:
                traceback.print_exc()
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            tracker.set("")
            continue
        if isinstance(command, UpsampleFileCommand):
            try:
                from tts_audiobook_tool.model_manager import ModelManager
                from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

                # LavaSR has exclusive use of worker model memory during concat.
                ModelManager.clear_all_models(except_lava_sr=True)
                upsampler = ModelManager.get_lava_sr_upsampler(
                    isolate_cuda=False
                )
                if upsampler is None:
                    raise RuntimeError("LavaSR v2 upsampler is not installed")
                sound = SoundFileUtil.load(command.source_path)
                if isinstance(sound, str):
                    raise RuntimeError(sound)
                result = upsampler.process(sound, denoise=command.denoise)
                if isinstance(result, str):
                    raise RuntimeError(result)
                error = SoundFileUtil.save_flac(result, command.destination_path)
                if error:
                    raise RuntimeError(error)
                event_queue.put(
                    AudioFileUpsampled(command.operation_id, command.destination_path)
                )
            except Exception as exception:
                traceback.print_exc()
                try:
                    if os.path.exists(command.destination_path):
                        os.remove(command.destination_path)
                except OSError:
                    pass
                event_queue.put(
                    WorkerCommandFailed(
                        command.operation_id,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
            tracker.set("")


class ModelWorker:
    """Static owner/client for the application's long-lived model process.

    The worker lifecycle is an explicit ``WorkerStatus`` state machine whose
    transitions happen only under ``cls._lock``:

    - ``ABSENT``: no worker process (initial state, after reset/shutdown, or a
      failed start);
    - ``STARTING``: a process was spawned and has not reported readiness;
    - ``RUNNING``: the worker reported readiness;
    - ``DEAD``: a client drainer detected the process exited.

    The drainers (``get_event``/``drain_events``) are the single place that
    detects worker death — a closed event queue or a process that is no
    longer alive. On detection they transition to ``DEAD`` exactly once and
    synthesize one ``WorkerExited`` terminal event into the pending events,
    so consumers handle a single event type instead of polling
    ``ModelWorker.is_alive()``, and busy-ness is answerable via
    ``is_busy()``/``status()`` without polling. A dead worker is
    resurrected by the next ``start()`` call (``submit_generation`` and
    ``clear_models_blocking`` both start first), so a crashed worker never
    wedges the application.
    """

    _context: BaseContext | None = None
    _process: Any | None = None
    _command_queue: Any | None = None
    _event_queue: Any | None = None
    _cancellation_event: Any | None = None
    _continue_event: Any | None = None
    _pending_events: deque[ModelWorkerEvent] = deque()
    _active_operation_id: str | None = None
    _status: WorkerStatus = WorkerStatus.ABSENT
    _lock = threading.RLock()

    @classmethod
    def start(cls) -> str:
        with cls._lock:
            if cls.is_alive():
                return ""
            cls._discard_process_state()
            context = multiprocessing.get_context("spawn")
            command_queue = context.Queue()
            event_queue = context.Queue()
            cancellation_event = context.Event()
            continue_event = context.Event()
            process = context.Process(
                target=_model_worker_main,
                args=(command_queue, event_queue, cancellation_event, continue_event),
                name="model-worker",
                daemon=False,
            )
            cls._context = context
            cls._command_queue = command_queue
            cls._event_queue = event_queue
            cls._cancellation_event = cancellation_event
            cls._continue_event = continue_event
            cls._process = process
            process.start()
            cls._status = WorkerStatus.STARTING

        deadline = time.monotonic() + WORKER_START_TIMEOUT_SECONDS
        startup_output: list[ConsoleOutput] = []
        while time.monotonic() < deadline:
            if not process.is_alive():
                try:
                    if event_queue.empty():
                        break
                except (EOFError, OSError, ValueError):
                    break
            try:
                event = event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                break
            if isinstance(event, WorkerReady):
                with cls._lock:
                    cls._status = WorkerStatus.RUNNING
                for output in startup_output:
                    cls._write_console_output(output)
                return ""
            if isinstance(event, ConsoleOutput):
                startup_output.append(event)
                continue
            if isinstance(event, WorkerCommandFailed):
                for output in startup_output:
                    cls._write_console_output(output)
                with cls._lock:
                    cls._force_stop_process(process)
                    cls._discard_process_state()
                return event.message
            cls._pending_events.append(event)

        for output in startup_output:
            cls._write_console_output(output)
        process_exited = not process.is_alive()
        with cls._lock:
            cls._force_stop_process(process)
            cls._discard_process_state()
        if process_exited:
            return cls._with_log_hint("Model worker process exited during startup.")
        return "Model worker did not become ready"

    @classmethod
    def is_alive(cls) -> bool:
        with cls._lock:
            if cls._status not in (WorkerStatus.STARTING, WorkerStatus.RUNNING):
                return False
            process = cls._process
        return process is not None and process.is_alive()

    @classmethod
    def status(cls) -> WorkerStatus:
        """Current worker lifecycle state, without polling the process."""
        with cls._lock:
            return cls._status

    @classmethod
    def is_busy(cls) -> bool:
        """True while a command is in flight, without polling the process."""
        with cls._lock:
            return cls._active_operation_id is not None

    @classmethod
    def submit_generation(
        cls,
        *,
        state: Any,
        indices: set[int],
        batch_size: int,
        is_regen: bool,
    ) -> str:
        error = cls.start()
        if error:
            raise ModelWorkerUnavailable(error)
        with cls._lock:
            if cls._active_operation_id is not None:
                raise RuntimeError("Model worker is already processing a command")
            operation_id = uuid.uuid4().hex
            prefs = state.prefs
            sgl_type = prefs.sgl_omni_type
            settings = GenerationSettings(
                stt_variant_id=prefs.stt_variant.id,
                stt_config_id=prefs.stt_config.id,
                tts_force_cpu=prefs.tts_force_cpu,
                sgl_omni_type_id=(None if sgl_type is None else sgl_type.value.id),
                sgl_omni_url=prefs.sgl_omni_url,
                save_debug_files=prefs.save_debug_files,
            )
            command = GenerateCommand(
                operation_id=operation_id,
                project_dir=state.project.dir_path,
                indices=tuple(sorted(indices)),
                batch_size=batch_size,
                is_regen=is_regen,
                settings=settings,
            )
            cancellation_event = cls._cancellation_event
            command_queue = cls._command_queue
            assert cancellation_event is not None and command_queue is not None
            cancellation_event.clear()
            cls._active_operation_id = operation_id
            command_queue.put(command)
            return operation_id

    @classmethod
    def submit_realtime_playback(
        cls,
        *,
        state: Any,
        phrase_groups: list[Any],
        line_range: tuple[int, int] | None,
    ) -> str:
        error = cls.start()
        if error:
            raise ModelWorkerUnavailable(error)
        with cls._lock:
            if cls._active_operation_id is not None:
                raise RuntimeError("Model worker is already processing a command")
            operation_id = uuid.uuid4().hex
            prefs = state.prefs
            sgl_type = prefs.sgl_omni_type
            settings = GenerationSettings(
                stt_variant_id=prefs.stt_variant.id,
                stt_config_id=prefs.stt_config.id,
                tts_force_cpu=prefs.tts_force_cpu,
                sgl_omni_type_id=(None if sgl_type is None else sgl_type.value.id),
                sgl_omni_url=prefs.sgl_omni_url,
                save_debug_files=prefs.save_debug_files,
            )
            command = RealTimePlaybackCommand(
                operation_id=operation_id,
                project_dir=state.project.dir_path,
                phrase_groups_json=tuple(
                    phrase_group.to_json_dict() for phrase_group in phrase_groups
                ),
                line_range=line_range,
                settings=settings,
            )
            cancellation_event = cls._cancellation_event
            continue_event = cls._continue_event
            command_queue = cls._command_queue
            assert (
                cancellation_event is not None
                and continue_event is not None
                and command_queue is not None
            )
            cancellation_event.clear()
            continue_event.clear()
            cls._active_operation_id = operation_id
            command_queue.put(command)
            return operation_id

    @classmethod
    def request_cancel(cls, operation_id: str) -> bool:
        with cls._lock:
            if cls._active_operation_id != operation_id:
                return False
            cancellation_event = cls._cancellation_event
            if cancellation_event is None:
                return False
            cancellation_event.set()
            return True

    @classmethod
    def continue_realtime_playback(cls, operation_id: str) -> bool:
        with cls._lock:
            if cls._active_operation_id != operation_id:
                return False
            continue_event = cls._continue_event
            if continue_event is None:
                return False
            continue_event.set()
            return True

    @classmethod
    def drain_events(cls, max_events: int = 1000) -> list[ModelWorkerEvent]:
        """Return up to max_events pending events and detect worker death."""
        events: list[ModelWorkerEvent] = []
        with cls._lock:
            while cls._pending_events and len(events) < max_events:
                event = cls._pending_events.popleft()
                cls._observe_event(event)
                events.append(event)
            event_queue = cls._event_queue
        if event_queue is None:
            return events
        while len(events) < max_events:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                with cls._lock:
                    if cls._note_worker_death(queue_failed=False):
                        events.extend(cls._take_pending_events(max_events - len(events)))
                break
            except (EOFError, OSError, ValueError):
                with cls._lock:
                    cls._note_worker_death(queue_failed=True)
                    events.extend(cls._take_pending_events(max_events - len(events)))
                break
            cls._observe_event(event)
            events.append(event)
        return events

    @classmethod
    def get_event(cls, timeout: float = 0.1) -> ModelWorkerEvent | None:
        """Return one pending event, waiting up to timeout seconds."""
        with cls._lock:
            if cls._pending_events:
                event = cls._pending_events.popleft()
                cls._observe_event(event)
                return event
            event_queue = cls._event_queue
        if event_queue is None:
            return None
        try:
            event = event_queue.get(timeout=timeout)
        except queue.Empty:
            with cls._lock:
                if cls._note_worker_death(queue_failed=False):
                    return cls._take_pending_event()
            return None
        except (EOFError, OSError, ValueError):
            with cls._lock:
                cls._note_worker_death(queue_failed=True)
                return cls._take_pending_event()
        cls._observe_event(event)
        return event

    @classmethod
    def unload_models_blocking(cls) -> str:
        """Stop the model worker so all accelerator resources are released.

        Clearing Python references inside a long-lived CUDA process is only
        best-effort: library-owned references and the CUDA allocator/context can
        keep VRAM resident. Process exit is the isolation boundary that makes
        the user-facing unload operation deterministic. The next model command
        starts a fresh worker lazily.
        """
        cls.shutdown()
        return "" if not cls.is_alive() else "Couldn't stop model worker"

    @classmethod
    def clear_models_blocking(cls) -> str:
        error = cls.start()
        if error:
            return error
        return cls._clear_models_on_running_worker_blocking()

    @classmethod
    def clear_models_if_running_blocking(cls) -> str:
        """Clear worker-owned models without creating a new worker."""
        if not cls.is_alive():
            return ""
        return cls._clear_models_on_running_worker_blocking()

    @classmethod
    def _clear_models_on_running_worker_blocking(cls) -> str:
        with cls._lock:
            if cls._active_operation_id is not None:
                return "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return "Model worker is unavailable"
            cls._command_queue.put(ClearModelsCommand(operation_id))
        result = cls._wait_for_blocking_result(operation_id, ModelsCleared)
        return result if isinstance(result, str) else ""

    @classmethod
    def reset_chat_session_blocking(
        cls, *, reset_voice_selection: bool = True
    ) -> str:
        error = cls.start()
        if error:
            return error
        with cls._lock:
            if cls._active_operation_id is not None:
                return "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return "Model worker is unavailable"
            cls._command_queue.put(
                ResetChatSessionCommand(operation_id, reset_voice_selection)
            )
        result = cls._wait_for_blocking_result(operation_id, ChatSessionReset)
        return result if isinstance(result, str) else ""

    @classmethod
    def synthesize_chat_blocking(
        cls,
        state: Any,
        text: str,
        reason: object,
        *,
        streaming: bool,
        on_chunk: Callable[[Any], None] | None = None,
        interrupt_event: threading.Event | None = None,
    ) -> tuple[Any | None, str]:
        error = cls.start()
        if error:
            return None, error
        prefs = state.prefs
        sgl_type = prefs.sgl_omni_type
        settings = GenerationSettings(
            stt_variant_id=prefs.stt_variant.id,
            stt_config_id=prefs.stt_config.id,
            tts_force_cpu=prefs.tts_force_cpu,
            sgl_omni_type_id=(None if sgl_type is None else sgl_type.value.id),
            sgl_omni_url=prefs.sgl_omni_url,
            save_debug_files=prefs.save_debug_files,
        )
        with cls._lock:
            if cls._active_operation_id is not None:
                return None, "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return None, "Model worker is unavailable"
            cancellation_event = cls._cancellation_event
            if cancellation_event is not None:
                cancellation_event.clear()
            cls._command_queue.put(
                SynthesizeChatCommand(
                    operation_id,
                    state.project.dir_path,
                    settings,
                    text,
                    streaming,
                    reason,
                )
            )

        deferred: list[ModelWorkerEvent] = []
        try:
            while True:
                if interrupt_event is not None and interrupt_event.is_set():
                    cls.request_cancel(operation_id)
                event = cls.get_event(timeout=0.1)
                if event is None:
                    continue
                if getattr(event, "operation_id", None) != operation_id:
                    deferred.append(event)
                    continue
                if isinstance(event, ConsoleOutput):
                    cls._write_console_output(event)
                elif isinstance(event, ChatAudioChunk):
                    if on_chunk is not None:
                        on_chunk(event.data)
                elif isinstance(event, ChatSynthesisFinished):
                    return event.sound, event.error
                elif isinstance(event, WorkerCommandFailed):
                    return None, event.message
                elif isinstance(event, WorkerExited):
                    return None, event.message or "Model worker exited during chat synthesis"
        finally:
            with cls._lock:
                cls._pending_events.extendleft(reversed(deferred))

    @classmethod
    def transcribe_audio_blocking(
        cls,
        state: Any,
        audio: object,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        stt_variant_id: str | None = None,
        stt_config_id: str | None = None,
    ) -> tuple[AudioTranscribed | None, str]:
        error = cls.start()
        if error:
            return None, error
        prefs = getattr(state, "prefs", state)
        with cls._lock:
            if cls._active_operation_id is not None:
                return None, "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return None, "Model worker is unavailable"
            cls._command_queue.put(
                TranscribeAudioCommand(
                    operation_id,
                    audio,
                    stt_variant_id or prefs.stt_variant.id,
                    stt_config_id or prefs.stt_config.id,
                    language,
                    word_timestamps,
                )
            )
        result = cls._wait_for_blocking_result(operation_id, AudioTranscribed)
        if isinstance(result, AudioTranscribed):
            return result, ""
        return None, result if isinstance(result, str) else "Unexpected transcription response"

    @classmethod
    def create_oute_speaker_blocking(
        cls,
        state: Any,
        source_path: str,
    ) -> tuple[dict[str, object] | None, str]:
        error = cls.start()
        if error:
            return None, error
        prefs = state.prefs
        sgl_type = prefs.sgl_omni_type
        settings = GenerationSettings(
            stt_variant_id=prefs.stt_variant.id,
            stt_config_id=prefs.stt_config.id,
            tts_force_cpu=prefs.tts_force_cpu,
            sgl_omni_type_id=(None if sgl_type is None else sgl_type.value.id),
            sgl_omni_url=prefs.sgl_omni_url,
            save_debug_files=prefs.save_debug_files,
        )
        with cls._lock:
            if cls._active_operation_id is not None:
                return None, "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return None, "Model worker is unavailable"
            cls._command_queue.put(
                CreateOuteSpeakerCommand(
                    operation_id,
                    state.project.dir_path,
                    settings,
                    source_path,
                )
            )
        result = cls._wait_for_blocking_result(operation_id, OuteSpeakerCreated)
        if isinstance(result, OuteSpeakerCreated):
            return result.voice, ""
        return None, result if isinstance(result, str) else "Unexpected Oute response"

    @classmethod
    def inspect_tts_blocking(cls, state: Any) -> tuple[TtsInspected | None, str]:
        from tts_audiobook_tool.tts import Tts

        error = cls.start()
        if error:
            return None, error
        prefs = state.prefs
        sgl_type = prefs.sgl_omni_type
        settings = GenerationSettings(
            stt_variant_id=prefs.stt_variant.id,
            stt_config_id=prefs.stt_config.id,
            tts_force_cpu=prefs.tts_force_cpu,
            sgl_omni_type_id=(None if sgl_type is None else sgl_type.value.id),
            sgl_omni_url=prefs.sgl_omni_url,
            save_debug_files=prefs.save_debug_files,
        )
        with cls._lock:
            if cls._active_operation_id is not None:
                return None, "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return None, "Model worker is unavailable"
            cls._command_queue.put(
                InspectTtsCommand(
                    operation_id,
                    state.project.dir_path,
                    settings,
                    Tts.get_model_params_using_project(state.project),
                )
            )
        result = cls._wait_for_blocking_result(operation_id, TtsInspected)
        if isinstance(result, TtsInspected):
            return result, ""
        return None, result if isinstance(result, str) else "Unexpected TTS inspection response"

    @classmethod
    def get_model_state_blocking(cls) -> tuple[ModelStateSnapshot | None, str]:
        """Return the worker model inventory, never parent static state."""
        error = cls.start()
        if error:
            return None, error
        with cls._lock:
            if cls._active_operation_id is not None:
                return None, "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return None, "Model worker is unavailable"
            cls._command_queue.put(GetModelStateCommand(operation_id))
        result = cls._wait_for_blocking_result(operation_id, ModelStateReported)
        if isinstance(result, ModelStateReported):
            return result.state, ""
        if isinstance(result, str):
            return None, result
        return None, "Unexpected model-state response"

    @classmethod
    def probe_lava_sr_blocking(cls) -> tuple[bool, str]:
        error = cls.start()
        if error:
            return False, error
        operation_id, error = cls._begin_blocking_command(ProbeLavaSrCommand)
        if error:
            return False, error
        result = cls._wait_for_blocking_result(operation_id, LavaSrProbed)
        if isinstance(result, LavaSrProbed):
            return result.available, ""
        return False, result if isinstance(result, str) else "Unexpected LavaSR probe response"

    @classmethod
    def upsample_file_blocking(
        cls,
        source_path: str,
        destination_path: str,
        *,
        denoise: bool = False,
    ) -> str:
        error = cls.start()
        if error:
            return error
        with cls._lock:
            if cls._active_operation_id is not None:
                return "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return "Model worker is unavailable"
            cls._command_queue.put(
                UpsampleFileCommand(
                    operation_id,
                    source_path,
                    destination_path,
                    denoise,
                )
            )
        result = cls._wait_for_blocking_result(operation_id, AudioFileUpsampled)
        return result if isinstance(result, str) else ""

    @classmethod
    def _begin_blocking_command(cls, command_type: Any) -> tuple[str, str]:
        with cls._lock:
            if cls._active_operation_id is not None:
                return "", "Model worker is busy"
            operation_id = uuid.uuid4().hex
            cls._active_operation_id = operation_id
            if cls._command_queue is None:
                cls._active_operation_id = None
                return "", "Model worker is unavailable"
            cls._command_queue.put(command_type(operation_id))
            return operation_id, ""

    @classmethod
    def _wait_for_blocking_result(
        cls,
        operation_id: str,
        success_type: type[ModelsCleared]
            | type[ChatSessionReset]
            | type[OuteSpeakerCreated]
            | type[AudioTranscribed]
            | type[TtsInspected]
            | type[ModelStateReported]
            | type[LavaSrProbed]
            | type[AudioFileUpsampled],
    ) -> ModelsCleared | ChatSessionReset | OuteSpeakerCreated | AudioTranscribed | TtsInspected | ModelStateReported | LavaSrProbed | AudioFileUpsampled | str:
        deferred: list[ModelWorkerEvent] = []
        try:
            while True:
                event = cls.get_event(timeout=0.1)
                if event is None:
                    continue
                if getattr(event, "operation_id", None) != operation_id:
                    deferred.append(event)
                    continue
                if isinstance(event, ConsoleOutput):
                    cls._write_console_output(event)
                elif isinstance(event, success_type):
                    return event
                elif isinstance(event, WorkerCommandFailed):
                    return event.message
                elif isinstance(event, WorkerExited):
                    return event.message or "Model worker exited during blocking command"
        finally:
            with cls._lock:
                cls._pending_events.extendleft(reversed(deferred))

    @classmethod
    def reset(cls) -> str:
        with cls._lock:
            process = cls._process
            cancellation_event = cls._cancellation_event
            if cancellation_event is not None:
                cancellation_event.set()
        if process is not None:
            cls._force_stop_process(process)
        with cls._lock:
            cls._discard_process_state()
        return cls.start()

    @classmethod
    def shutdown(cls) -> None:
        with cls._lock:
            process = cls._process
            if process is None:
                return
            if cls._cancellation_event is not None:
                cls._cancellation_event.set()
            if cls._continue_event is not None:
                cls._continue_event.set()
            if process.is_alive() and cls._command_queue is not None:
                operation_id = uuid.uuid4().hex
                try:
                    cls._command_queue.put(ShutdownCommand(operation_id))
                except Exception:
                    pass
        process.join(WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        if process.is_alive():
            cls._force_stop_process(process)
        with cls._lock:
            cls._discard_process_state()

    @classmethod
    def _observe_event(cls, event: ModelWorkerEvent) -> None:
        operation_id = getattr(event, "operation_id", None)
        if operation_id != cls._active_operation_id:
            return
        if isinstance(
            event,
            (
                GenerationFinished,
                RealTimePlaybackFinished,
                ModelsCleared,
                ChatSessionReset,
                OuteSpeakerCreated,
                AudioTranscribed,
                TtsInspected,
                ModelStateReported,
                LavaSrProbed,
                AudioFileUpsampled,
                ChatSynthesisFinished,
                WorkerCommandFailed,
                WorkerStopped,
                WorkerExited,
            ),
        ):
            cls._active_operation_id = None

    @classmethod
    def _note_worker_death(cls, *, queue_failed: bool) -> bool:
        """Record a dead worker once and synthesize its terminal event.

        Returns True when a ``WorkerExited`` event was synthesized into the
        pending events. Must be called with ``cls._lock`` held.

        ``queue_failed`` distinguishes a broken event queue (the writer is
        gone; trust it even if process liveness has not caught up) from an
        empty queue (the process must be confirmed dead, since a live
        worker can simply have no events pending).
        """
        if cls._status is not WorkerStatus.RUNNING:
            return False
        if not queue_failed:
            process = cls._process
            if process is not None and process.is_alive():
                return False
        cls._status = WorkerStatus.DEAD
        cls._pending_events.append(
            WorkerExited(
                operation_id=cls._active_operation_id or "",
                message=cls._with_log_hint(
                    "Model worker process exited unexpectedly."
                ),
            )
        )
        cls._active_operation_id = None
        return True

    @classmethod
    def _take_pending_event(cls) -> ModelWorkerEvent | None:
        if not cls._pending_events:
            return None
        event = cls._pending_events.popleft()
        cls._observe_event(event)
        return event

    @classmethod
    def _take_pending_events(cls, max_events: int) -> list[ModelWorkerEvent]:
        events: list[ModelWorkerEvent] = []
        while cls._pending_events and len(events) < max_events:
            event = cls._take_pending_event()
            if event is None:
                break
            events.append(event)
        return events

    @staticmethod
    def _worker_log_hint() -> str:
        try:
            from tts_audiobook_tool import app_support

            return app_support.make_worker_log_file_path()
        except Exception:
            return ""

    @classmethod
    def _with_log_hint(cls, message: str) -> str:
        log_path = cls._worker_log_hint()
        if not log_path:
            return message
        return f"{message} Worker log: {log_path}"

    @staticmethod
    def _write_console_output(event: ConsoleOutput) -> None:
        stream = sys.stderr if event.stream == "stderr" else sys.stdout
        stream.write(event.text)
        stream.flush()

    @staticmethod
    def _force_stop_process(process: Any) -> None:
        # The non-daemon model worker may own nested helpers such as the LavaSR
        # CUDA worker. Capture them before terminating their parent so a hard
        # reset does not leave GPU processes orphaned.
        descendants: list[Any] = []
        try:
            import psutil

            descendants = psutil.Process(process.pid).children(recursive=True)
            for descendant in descendants:
                try:
                    descendant.terminate()
                except psutil.Error:
                    pass
        except ImportError:
            descendants = []
        except (AttributeError, OSError, psutil.Error):  # pyright: ignore[reportPossiblyUnboundVariable]
            descendants = []

        # SIGTERM reaches the worker's `_terminate_worker` handler, which raises
        # SystemExit so the worker unwinds through its atexit finalizers before
        # exiting.  That unregisters any multiprocessing semaphores a model
        # library created in the worker; escalating straight to kill() would
        # skip those finalizers and leave them for resource_tracker.
        if process.is_alive():
            process.terminate()
            process.join(WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join()

        if descendants:
            try:
                import psutil

                _, remaining = psutil.wait_procs(
                    descendants,
                    timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS,
                )
                killed: list[Any] = []
                for descendant in remaining:
                    try:
                        descendant.kill()
                        killed.append(descendant)
                    except psutil.Error:
                        pass
                if killed:
                    psutil.wait_procs(
                        killed,
                        timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS,
                    )
            except (ImportError, psutil.Error):  # pyright: ignore[reportPossiblyUnboundVariable]
                pass

    @classmethod
    def _discard_process_state(cls) -> None:
        process = cls._process
        ipc_queues = (cls._command_queue, cls._event_queue)

        # Queue.close() only tells its feeder thread to stop after flushing its
        # buffer.  Waiting for that thread is what releases the queue's locks
        # and semaphores deterministically; otherwise a hard-reset followed by
        # immediate application exit can leave them for resource_tracker.
        for ipc_queue in ipc_queues:
            if ipc_queue is not None:
                try:
                    ipc_queue.close()
                except Exception:
                    pass
        for ipc_queue in ipc_queues:
            if ipc_queue is not None:
                try:
                    ipc_queue.join_thread()
                except Exception:
                    pass

        # Reap and close a stopped Process explicitly rather than relying on
        # interpreter-shutdown finalizers.  All callers stop the process first,
        # or arrive here after observing that it has already died.
        if process is not None:
            try:
                if not process.is_alive():
                    process.join()
                    process.close()
            except (AssertionError, OSError, ValueError):
                pass

        cls._context = None
        cls._process = None
        cls._command_queue = None
        cls._event_queue = None
        cls._cancellation_event = None
        cls._continue_event = None
        cls._pending_events.clear()
        cls._active_operation_id = None
        cls._status = WorkerStatus.ABSENT
