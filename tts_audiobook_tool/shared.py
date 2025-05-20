from typing import Any
import torch
import whisper
from whisper.model import Whisper

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.util import *

class Shared:

    # Oute TTS model object

    _oute_interface: Any = None

    @staticmethod
    def get_oute_interface() -> Any:
        if not Shared._oute_interface:
            printt("Initializing Oute TTS model...")
            printt()

            # Lazy import of oute machinery
            import outetts
            from outetts.version.interface import InterfaceHF

            from tts_audiobook_tool.model_config import MODEL_CONFIG
            try:
                from .model_config_dev import MODEL_CONFIG # type: ignore
            except ImportError:
                pass

            Shared._oute_interface = outetts.Interface(config=MODEL_CONFIG)

        return Shared._oute_interface

    @staticmethod
    def clear_oute_interface() -> None:
        if not Shared._oute_interface:
            return
        printt("Unloading Oute TTS model...")
        printt()
        del Shared._oute_interface
        AppUtil.gc_ram_vram()

    # Whisper

    _whisper: Whisper | None = None

    @staticmethod
    def get_whisper() -> Whisper:
        if not Shared._whisper:
            printt("Initializing whisper model...")
            printt()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Shared._whisper = whisper.load_model("turbo", device=device)
        return Shared._whisper

    @staticmethod
    def clear_whisper() -> None:
        if not Shared._whisper:
            return
        printt("Unloading whisper...")
        printt()
        del Shared._whisper
        AppUtil.gc_ram_vram()


    # Cheesy control-c capture flag variables

    mode = ""
    stop_flag = False