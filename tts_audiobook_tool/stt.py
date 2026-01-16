from faster_whisper import WhisperModel
import torch

from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.util import *


class Stt:
    """
    Static class for accessing the STT (transcription) model.
    """

    _whisper: WhisperModel | None = None
    _variant = list(SttVariant)[0]
    _config = SttConfig.CUDA_FLOAT16

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
    def get_whisper() -> WhisperModel:
        """
        Returns lazy-initialized WhisperModel
        """
        if Stt._variant == SttVariant.DISABLED:
            raise ValueError(f"Bad variant: {Stt._variant}")

        if Stt._whisper is None:

            model = Stt._variant.id

            dq = Stt._config
            if dq.device == "cuda" and not torch.cuda.is_available():
                dq = SttConfig.CPU_INT8FLOAT32 # fallback
            device = dq.device
            compute_type = dq.compute_type

            if device == "cpu":
                try:
                    import psutil
                    cpu_threads = psutil.cpu_count(logical=False) or 0
                except:
                    cpu_threads = 0 # ie, fall back to default (4)
            else:
                cpu_threads = 0
            cpu_threads_string = f", {cpu_threads} threads" if cpu_threads else ""

            print_init(f"Initializing faster-whisper model ({model}, {device}, {compute_type}{cpu_threads_string})...")
            Stt._whisper = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

        return Stt._whisper

    @staticmethod
    def short_description() -> str:
        config = Stt._config
        if config.device == "cuda" and not torch.cuda.is_available():
            config = SttConfig.CPU_INT8FLOAT32
        return f"{Stt._variant.id}, {config.device}"

    @staticmethod
    def has_whisper() -> bool:
        return Stt._whisper is not None

    @staticmethod
    def clear_stt_model() -> None:
        if Stt._whisper:
            Stt._whisper = None
            from tts_audiobook_tool.memory_util import MemoryUtil
            MemoryUtil.gc_ram_vram()
