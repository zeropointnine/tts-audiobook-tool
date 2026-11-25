"""Temporary reference transcription must release STT before its IPC handoff."""

import queue
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

import tts_audiobook_tool.model_worker as worker_module
from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    AudioTranscribed,
    ClearModelsCommand,
    ModelsCleared,
    ShutdownCommand,
    TranscribeAudioCommand,
    WorkerCommandCancelled,
    WorkerCommandFailed,
    WorkerStopped,
)
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts


def test_lifecycle_command_flags_default_to_retaining_existing_behavior() -> None:
    command = TranscribeAudioCommand(
        "transcribe", np.zeros(8, dtype=np.float32),
        SttVariant.LARGE_V3_TURBO.id, SttConfig.CPU_INT8FLOAT32.id,
    )

    assert command.unload_after is False
    assert ClearModelsCommand("clear").stt_only is False


@pytest.mark.parametrize(
    ("preference", "override", "unload_after"),
    [
        (SttVariant.DISABLED, SttVariant.LARGE_V3_TURBO.id, True),
        (SttVariant.DISABLED, None, True),
        (SttVariant.LARGE_V3, None, False),
        (SttVariant.LARGE_V3, SttVariant.LARGE_V3_TURBO.id, False),
    ],
)
def test_parent_transcription_uses_preference_for_temporary_lifetime(
    monkeypatch, preference, override, unload_after,
) -> None:
    # An explicit sample-transcription model must not enable persistent STT.
    commands = queue.Queue()
    cancellation = threading.Event()
    cancellation.set()
    prefs = SimpleNamespace(stt_variant=preference, stt_config=SttConfig.CPU_INT8FLOAT32)
    audio = np.zeros(8, dtype=np.float32)
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(ModelWorker, "_command_queue", commands)
    monkeypatch.setattr(ModelWorker, "_cancellation_event", cancellation)
    monkeypatch.setattr(
        ModelWorker, "_start_with_console_handler", classmethod(lambda cls, handler: ""),
    )

    def wait(cls, operation_id, expected, handler):
        assert expected is AudioTranscribed
        return AudioTranscribed(operation_id, (), True)

    monkeypatch.setattr(ModelWorker, "_wait_for_blocking_result_with_handler", classmethod(wait))

    result, error = ModelWorker.transcribe_audio_blocking(
        SimpleNamespace(prefs=prefs), audio, stt_variant_id=override,
    )

    command = commands.get_nowait()
    assert isinstance(command, TranscribeAudioCommand)
    assert command.audio is audio
    assert command.stt_variant_id == (override or preference.id)
    assert command.stt_config_id == prefs.stt_config.id
    assert command.unload_after is unload_after
    assert not cancellation.is_set()
    assert error == ""
    assert result == AudioTranscribed(command.operation_id, (), True)


def test_transcriber_language_retry_preserves_explicit_temporary_configuration(monkeypatch) -> None:
    # Unsupported-language retry must not fall back to the disabled preferences.
    import tts_audiobook_tool.model_runtime as model_runtime
    from tts_audiobook_tool.app_types import Sound
    from tts_audiobook_tool.transcriber import Transcriber

    monkeypatch.setattr(model_runtime, "_role", model_runtime.ModelRuntimeRole.INTERACTIVE_MAIN)
    prefs = SimpleNamespace(stt_variant=SttVariant.DISABLED, stt_config=SttConfig.CUDA_FLOAT16)
    state = SimpleNamespace(prefs=prefs)
    sound = Sound(np.zeros(8, dtype=np.float32), 16000)
    calls = []
    monkeypatch.setattr(Transcriber, "prepare_sound_for_whisper", staticmethod(lambda value: value))

    def transcribe(cls, received_state, audio, **kwargs):
        calls.append((received_state, audio, kwargs))
        return AudioTranscribed(f"transcribe-{len(calls)}", (), len(calls) > 1), ""

    monkeypatch.setattr(ModelWorker, "transcribe_audio_blocking", classmethod(transcribe))

    words = Transcriber.transcribe_to_words(
        sound, "en-US", stt_variant=SttVariant.LARGE_V3_TURBO,
        stt_config=SttConfig.CPU_INT8FLOAT32, state=state,
    )

    assert words == []
    assert len(calls) == 2
    assert [call[2]["language"] for call in calls] == ["en", None]
    for received_state, audio, kwargs in calls:
        assert received_state is state
        assert audio is sound.data
        assert kwargs["stt_variant_id"] == SttVariant.LARGE_V3_TURBO.id
        assert kwargs["stt_config_id"] == SttConfig.CPU_INT8FLOAT32.id
        assert kwargs["word_timestamps"] is True
    assert prefs.stt_variant is SttVariant.DISABLED


@pytest.mark.parametrize("stt_only", [False, True])
def test_parent_clear_models_forwards_scope_to_running_worker(monkeypatch, stt_only) -> None:
    # Clearing an existing worker is a queued operation, not a restart.
    commands = queue.Queue()
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(ModelWorker, "_command_queue", commands)
    monkeypatch.setattr(ModelWorker, "is_alive", classmethod(lambda cls: True))
    monkeypatch.setattr(
        ModelWorker, "start", classmethod(lambda cls: pytest.fail("must not start worker")),
    )

    def wait(cls, operation_id, expected):
        assert expected is ModelsCleared
        return ModelsCleared(operation_id)

    monkeypatch.setattr(ModelWorker, "_wait_for_blocking_result", classmethod(wait))

    assert ModelWorker.clear_models_if_running_blocking(stt_only=stt_only) == ""

    command = commands.get_nowait()
    assert isinstance(command, ClearModelsCommand)
    assert command.stt_only is stt_only
    assert command.operation_id == ModelWorker._active_operation_id
    assert commands.empty()


def test_parent_selective_clear_does_not_spawn_absent_worker(monkeypatch) -> None:
    # Disabling STT when there is no worker is a genuine no-op.
    commands = queue.Queue()
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(ModelWorker, "_command_queue", commands)
    monkeypatch.setattr(ModelWorker, "is_alive", classmethod(lambda cls: False))
    monkeypatch.setattr(
        ModelWorker, "start", classmethod(lambda cls: pytest.fail("must not start worker")),
    )
    monkeypatch.setattr(
        ModelWorker, "_wait_for_blocking_result",
        classmethod(lambda *args, **kwargs: pytest.fail("must not await an absent worker")),
    )

    assert ModelWorker.clear_models_if_running_blocking(stt_only=True) == ""
    assert commands.empty()
    assert ModelWorker._active_operation_id is None


@pytest.fixture
def worker_loop(monkeypatch, tmp_path):
    """Exercise real dispatch while replacing only process/native bootstrap."""
    import tts_audiobook_tool.app_support as app_support
    import tts_audiobook_tool.model_runtime as model_runtime
    from tts_audiobook_tool.app_support import app_memory
    from tts_audiobook_tool.model_manager import ModelManager
    from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import FfmpegLibMonitor

    monkeypatch.setattr(model_runtime, "_role", model_runtime.ModelRuntimeRole.MODEL_WORKER)
    monkeypatch.setattr(model_runtime, "mark_model_worker", lambda: None)
    monkeypatch.setattr(worker_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker_module._WorkerOutputCapture, "install", lambda self: None)
    monkeypatch.setattr(app_support, "init_logging", lambda *args: None)
    monkeypatch.setattr(Tts, "init_local_model_type", staticmethod(lambda: None))
    monkeypatch.setattr(FfmpegLibMonitor, "check", lambda *args: None)
    monkeypatch.setattr(app_memory, "gc_ram_vram", lambda: None)
    monkeypatch.setattr(worker_module, "STT_WORKER_STACK_LOG_PATH", str(tmp_path / "stt-stacks.log"))
    monkeypatch.setattr(worker_module.faulthandler, "dump_traceback_later", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module.faulthandler, "cancel_dump_traceback_later", lambda: None)
    monkeypatch.setattr(Stt, "_variant", SttVariant.DISABLED)
    monkeypatch.setattr(Stt, "_config", SttConfig.CPU_INT8FLOAT32)
    monkeypatch.setattr(Stt, "_whisper", None)
    monkeypatch.setattr(Stt, "_did_eager_warm_up_mlx", False)
    monkeypatch.setattr(Stt, "_did_eager_warm_up_inference", False)
    clear_all_calls = []
    monkeypatch.setattr(
        ModelManager, "clear_all_models",
        staticmethod(lambda: clear_all_calls.append("all")),
    )

    def run(commands, cancellation=None, observe=None):
        command_queue = queue.Queue()
        for command in commands:
            command_queue.put(command)
        command_queue.put(ShutdownCommand("shutdown"))
        events = []

        class EventQueue:
            def put(self, event):
                if observe is not None:
                    observe(event)
                events.append(event)

        worker_module._model_worker_main(
            command_queue, EventQueue(), cancellation or threading.Event(), threading.Event(),
        )
        assert WorkerStopped("shutdown") in events
        return events

    return SimpleNamespace(run=run, clear_all_calls=clear_all_calls)


@pytest.mark.parametrize(
    ("outcome", "unload_after"),
    [
        ("success", True),
        ("unsupported_language", True),
        ("cancel_before_inference", True),
        ("cancel_during_segments", True),
        ("bad_config", True),
        ("warmup_error", True),
        ("transcribe_error", True),
        ("segment_error", True),
        ("success", False),
        ("transcribe_error", False),
    ],
)
def test_worker_transcription_lifetime_is_settled_before_terminal_event(
    monkeypatch, worker_loop, outcome, unload_after,
) -> None:
    # Snapshot inside put(): assertions after shutdown alone hide late cleanup.
    cancellation = threading.Event()
    backend_refs = []
    calls = []
    snapshots = []
    segment = SimpleNamespace(
        start=0.0, end=0.5, text="hello",
        words=[SimpleNamespace(start=0.0, end=0.5, word="hello", probability=0.9)],
    )

    class Backend:
        supported_languages = ("en",)

        def transcribe(self, audio, **kwargs):
            calls.append("transcribe")
            assert isinstance(audio, np.ndarray)
            assert kwargs == {"word_timestamps": True, "language": "en"}
            if outcome == "transcribe_error":
                raise RuntimeError("fake inference failure")

            def segments():
                try:
                    if outcome == "segment_error":
                        raise RuntimeError("fake segment failure")
                    if outcome == "cancel_during_segments":
                        cancellation.set()
                    yield segment
                finally:
                    calls.append("segments_closed")

            return segments(), {}

    def warm_up():
        calls.append("warmup")
        Stt._whisper = Backend()
        backend_refs.append(weakref.ref(Stt._whisper))
        Stt._did_eager_warm_up_mlx = True
        Stt._did_eager_warm_up_inference = True
        if outcome == "warmup_error":
            raise RuntimeError("fake warmup failure")
        if outcome == "cancel_before_inference":
            cancellation.set()

    monkeypatch.setattr(Stt, "eager_warm_up_for_inference", staticmethod(warm_up))
    # Never fall through to a real model constructor, even on a regression.
    monkeypatch.setattr(Stt, "get_whisper", staticmethod(lambda: Stt._whisper))
    if outcome == "bad_config":
        # Failed validation must also clear an already resident backend.
        Stt._whisper = Backend()
        backend_refs.append(weakref.ref(Stt._whisper))

    command = TranscribeAudioCommand(
        "transcribe", np.zeros(8, dtype=np.float32),
        SttVariant.LARGE_V3_TURBO.id,
        "not-a-config" if outcome == "bad_config" else SttConfig.CPU_INT8FLOAT32.id,
        language="unsupported" if outcome == "unsupported_language" else "en",
        word_timestamps=True, unload_after=unload_after,
    )

    def observe(event):
        if isinstance(event, (AudioTranscribed, WorkerCommandCancelled, WorkerCommandFailed)):
            snapshots.append((
                Stt.get_variant(), Stt._whisper is not None,
                Stt._did_eager_warm_up_inference, Stt._did_eager_warm_up_mlx,
                all(reference() is None for reference in backend_refs),
                list(worker_loop.clear_all_calls),
            ))

    events = worker_loop.run([command], cancellation, observe)
    terminal = [event for event in events if getattr(event, "operation_id", None) == "transcribe"]

    assert len(terminal) == 1
    assert len(snapshots) == 1
    variant, loaded, warmed, mlx_warmed, freed, cleared_other_models = snapshots[0]
    assert cleared_other_models == []
    if unload_after:
        assert variant is SttVariant.DISABLED
        assert not loaded
        assert not warmed
        assert not mlx_warmed
        if outcome in {"success", "unsupported_language", "cancel_before_inference", "cancel_during_segments"}:
            # Dropping the class cache is insufficient if dispatch retains whisper.
            assert freed
    else:
        assert variant is SttVariant.LARGE_V3_TURBO
        assert loaded and warmed and mlx_warmed
        assert not freed

    if outcome == "success":
        assert isinstance(terminal[0], AudioTranscribed)
        assert terminal[0].language_supported is True
        assert len(terminal[0].segments) == 1
        assert terminal[0].segments[0].text == "hello"
        assert terminal[0].segments[0].words[0].word == "hello"
    elif outcome == "unsupported_language":
        assert terminal == [AudioTranscribed("transcribe", (), False)]
        assert "transcribe" not in calls
    elif outcome.startswith("cancel_"):
        assert terminal == [WorkerCommandCancelled("transcribe")]
        if outcome == "cancel_before_inference":
            assert "transcribe" not in calls
        else:
            assert "segments_closed" in calls
    else:
        assert isinstance(terminal[0], WorkerCommandFailed)
        expected = {
            "bad_config": "Unsupported STT configuration",
            "warmup_error": "fake warmup failure",
            "transcribe_error": "fake inference failure",
            "segment_error": "fake segment failure",
        }[outcome]
        assert expected in terminal[0].message


@pytest.mark.parametrize("initial_variant", [SttVariant.LARGE_V3_TURBO, SttVariant.DISABLED])
def test_worker_selective_clear_is_idempotent_and_preserves_other_models(
    monkeypatch, worker_loop, initial_variant,
) -> None:
    # Disabled configuration can still have a stale cache; repeated clears are safe.
    from tts_audiobook_tool.model_manager import ModelManager

    class Backend:
        pass

    Stt._variant = initial_variant
    Stt._whisper = Backend()
    backend_ref = weakref.ref(Stt._whisper)
    Stt._did_eager_warm_up_mlx = True
    Stt._did_eager_warm_up_inference = True
    tts_backend, yamnet, lava_sr = object(), object(), object()
    monkeypatch.setattr(Tts, "get_instance_if_exists", staticmethod(lambda: tts_backend))
    monkeypatch.setattr(ModelManager, "yamnet_detector", yamnet)
    monkeypatch.setattr(ModelManager, "lava_sr_upsampler", lava_sr)
    touched_other_models = []
    monkeypatch.setattr(
        Tts, "clear_tts_model", staticmethod(lambda: touched_other_models.append("tts")),
    )
    monkeypatch.setattr(
        ModelManager, "clear_yamnet_detector", staticmethod(lambda: touched_other_models.append("yamnet")),
    )
    monkeypatch.setattr(
        ModelManager, "clear_lava_sr_upsampler", staticmethod(lambda: touched_other_models.append("lava_sr")),
    )
    snapshots = []

    def observe(event):
        if isinstance(event, ModelsCleared):
            snapshots.append((
                Stt.get_variant(), Stt._whisper is None, backend_ref() is None,
                Stt._did_eager_warm_up_mlx, Stt._did_eager_warm_up_inference,
                Tts.get_instance_if_exists() is tts_backend,
                ModelManager.yamnet_detector is yamnet,
                ModelManager.lava_sr_upsampler is lava_sr,
                list(worker_loop.clear_all_calls), list(touched_other_models),
            ))

    events = worker_loop.run([
        ClearModelsCommand("clear-first", stt_only=True),
        ClearModelsCommand("clear-again", stt_only=True),
    ], observe=observe)

    assert [event for event in events if isinstance(event, ModelsCleared)] == [
        ModelsCleared("clear-first"), ModelsCleared("clear-again"),
    ]
    assert not any(isinstance(event, WorkerCommandFailed) for event in events)
    expected = (SttVariant.DISABLED, True, True, False, False, True, True, True, [], [])
    assert snapshots == [expected, expected]
    # The one all-model clear belongs solely to the explicit shutdown command.
    assert worker_loop.clear_all_calls == ["all"]
    assert touched_other_models == []
