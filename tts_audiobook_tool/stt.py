from faster_whisper import WhisperModel
import torch

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.util import *


class Stt:
    """
    Static class for accessing the STT (transcription) model.
    """

    _whisper: WhisperModel | None = None
    _variant = list(SttVariant)[0]

    @staticmethod
    def get_variant() -> SttVariant:
        return Stt._variant

    @staticmethod
    def set_variant_using_project(project: Any) -> None:

        from tts_audiobook_tool.project import Project
        assert(isinstance(project, Project))

        if project.stt_variant == Stt._variant:
            return

        Stt._variant = project.stt_variant
        Stt.clear_stt_model()

    @staticmethod
    def get_whisper() -> WhisperModel:

        if Stt._whisper is None:
            model = Stt._variant.value
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            printt(f"{Ansi.ITALICS}Initializing whisper model ({model}, {device}, {compute_type})...")
            printt()
            Stt._whisper = WhisperModel(model, device=device, compute_type=compute_type)

        return Stt._whisper

    @staticmethod
    def has_whisper() -> bool:
        return Stt._whisper is not None

    @staticmethod
    def clear_stt_model() -> None:

        if Stt._whisper:
            Stt._whisper = None

            from tts_audiobook_tool.app_util import AppUtil
            AppUtil.gc_ram_vram()


