from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tts_audiobook_tool.generation_events import GenerationEvent
from tts_audiobook_tool.real_time_playback_events import RealTimePlaybackEvent


class GenerationTerminalStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    FAILED = "failed"
    WORKER_RESET = "worker_reset"


class RealTimePlaybackTerminalStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    FAILED = "failed"
    WORKER_RESET = "worker_reset"


class WorkerStatus(str, Enum):
    """Main-process view of the worker process lifecycle.

    Transitions happen only under ``ModelWorker._lock``:

    - ``ABSENT``: no worker process (initial state, after reset/shutdown, or a
      failed start);
    - ``STARTING``: a process has been spawned but has not reported readiness;
    - ``RUNNING``: the worker reported readiness;
    - ``DEAD``: a client drainer detected the process exited.
    """

    ABSENT = "absent"
    STARTING = "starting"
    RUNNING = "running"
    DEAD = "dead"


@dataclass(frozen=True)
class GenerationSettings:
    """Small, explicitly serializable preference snapshot for one generation."""

    stt_variant_id: str
    stt_config_id: str
    tts_force_cpu: bool
    sgl_omni_type_id: str | None
    sgl_omni_url: str
    save_debug_files: bool


@dataclass(frozen=True)
class GenerateCommand:
    operation_id: str
    project_dir: str
    indices: tuple[int, ...]
    batch_size: int
    is_regen: bool
    settings: GenerationSettings


@dataclass(frozen=True)
class RealTimePlaybackCommand:
    operation_id: str
    project_dir: str
    phrase_groups_json: tuple[dict[str, object], ...]
    line_range: tuple[int, int] | None
    settings: GenerationSettings


@dataclass(frozen=True)
class ClearModelsCommand:
    operation_id: str


@dataclass(frozen=True)
class ResetChatSessionCommand:
    operation_id: str
    reset_voice_selection: bool = True


@dataclass(frozen=True)
class SynthesizeChatCommand:
    operation_id: str
    project_dir: str
    settings: GenerationSettings
    text: str
    streaming: bool
    reason: object


@dataclass(frozen=True)
class TranscribeAudioCommand:
    operation_id: str
    audio: object
    stt_variant_id: str
    stt_config_id: str
    language: str | None = None
    word_timestamps: bool = False


@dataclass(frozen=True)
class CreateOuteSpeakerCommand:
    operation_id: str
    project_dir: str
    settings: GenerationSettings
    source_path: str


@dataclass(frozen=True)
class InspectTtsCommand:
    operation_id: str
    project_dir: str
    settings: GenerationSettings
    model_params: dict[str, object]


@dataclass(frozen=True)
class GetModelStateCommand:
    operation_id: str


@dataclass(frozen=True)
class ProbeLavaSrCommand:
    operation_id: str


@dataclass(frozen=True)
class UpsampleFileCommand:
    operation_id: str
    source_path: str
    destination_path: str
    denoise: bool = False


@dataclass(frozen=True)
class ModelStateSnapshot:
    tts_loaded: bool = False
    tts_type_id: str = ""
    tts_device: str = ""
    stt_loaded: bool = False
    stt_variant_id: str = ""
    stt_device: str = ""
    yamnet_loaded: bool = False
    lava_sr_loaded: bool = False

    @property
    def any_loaded(self) -> bool:
        return self.tts_loaded or self.stt_loaded or self.yamnet_loaded or self.lava_sr_loaded


@dataclass(frozen=True)
class ShutdownCommand:
    operation_id: str


ModelWorkerCommand = (
    GenerateCommand
    | RealTimePlaybackCommand
    | ClearModelsCommand
    | ResetChatSessionCommand
    | SynthesizeChatCommand
    | TranscribeAudioCommand
    | CreateOuteSpeakerCommand
    | InspectTtsCommand
    | GetModelStateCommand
    | ProbeLavaSrCommand
    | UpsampleFileCommand
    | ShutdownCommand
)


@dataclass(frozen=True)
class WorkerReady:
    process_id: int


@dataclass(frozen=True)
class ConsoleOutput:
    operation_id: str
    stream: str
    text: str


@dataclass(frozen=True)
class ConsoleFlush:
    operation_id: str
    stream: str


@dataclass(frozen=True)
class GenerationUpdate:
    """Relay of one structured generation event emitted inside the worker.

    ``update`` is always one of the ``GenerationEvent`` types defined in
    ``generation_events.py``; consumers dispatch on it with ``isinstance``
    instead of parsing console text.
    """

    operation_id: str
    update: GenerationEvent


@dataclass(frozen=True)
class GenerationFinished:
    operation_id: str
    status: GenerationTerminalStatus
    remaining_range_string: str
    message: str = ""


@dataclass(frozen=True)
class RealTimePlaybackUpdate:
    operation_id: str
    update: RealTimePlaybackEvent


@dataclass(frozen=True)
class RealTimePlaybackFinished:
    operation_id: str
    status: RealTimePlaybackTerminalStatus
    message: str = ""


@dataclass(frozen=True)
class ModelsCleared:
    operation_id: str


@dataclass(frozen=True)
class ModelStateReported:
    operation_id: str
    state: ModelStateSnapshot


@dataclass(frozen=True)
class ChatSessionReset:
    operation_id: str


@dataclass(frozen=True)
class ChatAudioChunk:
    operation_id: str
    data: object


@dataclass(frozen=True)
class ChatSynthesisFinished:
    operation_id: str
    sound: object | None = None
    error: str = ""


@dataclass(frozen=True)
class OuteSpeakerCreated:
    operation_id: str
    voice: dict[str, object]


@dataclass(frozen=True)
class AudioTranscribed:
    operation_id: str
    segments: tuple[object, ...]
    language_supported: bool


@dataclass(frozen=True)
class TtsInspected:
    operation_id: str
    tts_type_id: str
    device: str = ""
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class LavaSrProbed:
    operation_id: str
    available: bool


@dataclass(frozen=True)
class AudioFileUpsampled:
    operation_id: str
    destination_path: str


@dataclass(frozen=True)
class WorkerCommandFailed:
    operation_id: str
    message: str


@dataclass(frozen=True)
class WorkerStopped:
    operation_id: str


@dataclass(frozen=True)
class WorkerExited:
    """Terminal event synthesized by the MAIN process (never sent by the
    worker) when a client drainer detects the worker process has died: the
    event queue is closed or the process is no longer alive. It carries the
    active operation id (empty when no operation was in flight) and a
    diagnostic message naming the worker log file, so both UI modes can
    finalize on one event type instead of polling process liveness.
    """

    operation_id: str
    message: str = ""


ModelWorkerEvent = (
    WorkerReady
    | ConsoleOutput
    | ConsoleFlush
    | GenerationUpdate
    | GenerationFinished
    | RealTimePlaybackUpdate
    | RealTimePlaybackFinished
    | ModelsCleared
    | ModelStateReported
    | ChatSessionReset
    | ChatAudioChunk
    | ChatSynthesisFinished
    | OuteSpeakerCreated
    | AudioTranscribed
    | TtsInspected
    | LavaSrProbed
    | AudioFileUpsampled
    | WorkerCommandFailed
    | WorkerStopped
    | WorkerExited
)
