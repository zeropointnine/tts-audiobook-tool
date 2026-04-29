import platform
import threading
from typing import Any, Protocol, Sequence, cast

import numpy as np
import torch

from tts_audiobook_tool.app_types import ConcreteSegment, ConcreteWord, SttConfig, SttVariant
from tts_audiobook_tool.util import *


class WhisperBackend(Protocol):
    @property
    def supported_languages(self) -> Sequence[str]:
        ...

    def transcribe(
        self,
        audio: Any,
        *,
        word_timestamps: bool = False,
        language: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ...


class MlxWhisperAdapter:
    """
    Small adapter that exposes a faster-whisper-like surface over mlx-whisper.
    """

    MODEL_MAP = {
        SttVariant.LARGE_V3.id: "mlx-community/whisper-large-v3-mlx",
        SttVariant.LARGE_V3_TURBO.id: "mlx-community/whisper-large-v3-turbo",
    }

    def __init__(self, model: str):
        import mlx_whisper
        from mlx_whisper.tokenizer import LANGUAGES

        self._mlx_whisper = mlx_whisper
        self._model = self.MODEL_MAP.get(model, model)
        self.supported_languages = list(LANGUAGES.keys())

    def transcribe(
        self,
        audio: Any,
        *,
        word_timestamps: bool = False,
        language: str | None = None,
        **kwargs: Any,
    ) -> Any:
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
            word_timestamps=word_timestamps,
            language=language,
            verbose=None,
            **kwargs,
        )
        raw_segments = [item for item in result.get("segments", []) if isinstance(item, dict)]
        segments = [self._convert_segment(cast(dict[str, Any], item)) for item in raw_segments]
        return segments, result

    @staticmethod
    def _convert_segment(item: dict[str, Any]) -> ConcreteSegment:
        words = [
            ConcreteWord(
                start=float(word.get("start", item.get("start", 0.0))),
                end=float(word.get("end", item.get("end", 0.0))),
                word=str(word.get("word", "")),
                probability=float(word.get("probability", 1.0)),
            )
            for word in (item.get("words") or [])
        ]
        return ConcreteSegment(
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
            text=str(item.get("text", "")),
            words=words,
        )


class FasterWhisperAdapter:
    """
    Small adapter that exposes the narrowed WhisperBackend protocol over
    faster-whisper's WhisperModel.
    """

    def __init__(self, model: str, device: str, compute_type: str, cpu_threads: int):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    @property
    def supported_languages(self) -> Sequence[str]:
        return self._model.supported_languages

    def transcribe(
        self,
        audio: Any,
        *,
        word_timestamps: bool = False,
        language: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._model.transcribe(
            audio,
            word_timestamps=word_timestamps,
            language=language,
            **kwargs,
        )


class Stt:
    """
    Static class for accessing the STT (transcription) model.
    """

    _whisper: WhisperBackend | None = None
    _variant = SttVariant.get_default()
    _config = SttConfig.CUDA_FLOAT16

    # Serializes all faster-whisper transcribe() calls. The underlying
    # CTranslate2 model is not safe for concurrent access from multiple
    # threads on a single instance — concurrent calls crash natively
    # (segfault, no Python traceback). Every call site must take this lock
    # for the duration of transcribe() AND the iteration that materializes
    # the returned generator.
    inference_lock = threading.Lock()
    _did_eager_warm_up_mlx = False

    @staticmethod
    def should_use_mlx_whisper() -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @staticmethod
    def get_variant() -> SttVariant:
        return Stt._variant

    @staticmethod
    def set_variant(value: SttVariant) -> None:
        if value != Stt._variant:
            Stt._variant = value
            # Clear model, will get lazy re-inited as needed
            Stt.clear_stt_model()

    @staticmethod
    def get_config() -> SttConfig:
        return Stt._config

    @staticmethod
    def set_config(value: SttConfig) -> None:
        if value != Stt._config:
            Stt._config = value
            # Clear model, will get lazy re-inited as needed
            Stt.clear_stt_model()
    
    @staticmethod
    def get_whisper() -> WhisperBackend:
        """
        Returns lazy-initialized whisper backend instance.
        """
        if Stt._variant == SttVariant.DISABLED:
            raise ValueError(f"Bad variant: {Stt._variant}")

        if Stt._whisper is None:

            model = Stt._variant.id

            if Stt.should_use_mlx_whisper():
                print_init(f"Initializing mlx-whisper model ({model})...")
                Stt._whisper = MlxWhisperAdapter(model)
                Stt.eager_warm_up_mlx_whisper()
                return cast(WhisperBackend, Stt._whisper)

            dq = Stt._config
            if dq.device == "cuda" and not torch.cuda.is_available():
                dq = SttConfig.CPU_INT8FLOAT32 # fallback
            device = dq.device
            compute_type = dq.compute_type

            if device == "cpu":
                try:
                    import psutil
                    cpu_threads = psutil.cpu_count(logical=False) or 0
                except Exception:
                    cpu_threads = 0 # ie, fall back to default (4)
            else:
                cpu_threads = 0
            cpu_threads_string = f", {cpu_threads} threads" if cpu_threads else ""

            print_init(f"Initializing faster-whisper model ({model}, {device}, {compute_type}{cpu_threads_string})...")
            Stt._whisper = FasterWhisperAdapter(
                model,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )

        assert Stt._whisper is not None
        return cast(WhisperBackend, Stt._whisper)

    @staticmethod
    def eager_warm_up_mlx_whisper() -> None:
        if not Stt.should_use_mlx_whisper() or Stt._did_eager_warm_up_mlx:
            return

        whisper = Stt._whisper
        if whisper is None:
            return

        # mlx-whisper defers the real model download/load until the first
        # transcribe() call. Force a tiny silent transcription here so the
        # startup behavior matches the eager faster-whisper path and any
        # progress output appears during model warmup rather than after the
        # conversation UI prompt is already on screen.
        silent_audio = np.zeros(1600, dtype=np.float32)
        with Stt.inference_lock:
            segments, _ = whisper.transcribe(
                silent_audio,
                word_timestamps=False,
                language=None,
            )
            _ = list(segments)
        Stt._did_eager_warm_up_mlx = True

    @staticmethod
    def short_description() -> str:
        if Stt.should_use_mlx_whisper():
            return f"{Stt._variant.id}, mlx"

        config = Stt._config
        if config.device == "cuda" and not torch.cuda.is_available():
            config = SttConfig.CPU_INT8FLOAT32
        return f"{Stt._variant.id}, {config.device}"

    @staticmethod
    def has_instance() -> bool:
        return Stt._whisper is not None

    @staticmethod
    def clear_stt_model() -> None:
        if Stt._whisper:
            Stt._whisper = None
            Stt._did_eager_warm_up_mlx = False
            from tts_audiobook_tool.memory_util import MemoryUtil
            MemoryUtil.gc_ram_vram()

    @staticmethod
    def should_skip(state, is_real_time_buffer_too_short: bool=False) -> str:
        """
        Given app `state`, returns UI-message reason why STT/validation should be skipped, or empty string        
        """
        from tts_audiobook_tool.state import State
        from tts_audiobook_tool.validate_util import ValidateUtil
        assert(isinstance(state, State))
        if state.prefs.stt_variant == SttVariant.DISABLED:
            return "Whisper disabled"
        if ValidateUtil.is_unsupported_language_code(state.project.language_code):
            return "Unsupported language"
        if is_real_time_buffer_too_short:
            return "Sound buffer duration too short"
        return ""
