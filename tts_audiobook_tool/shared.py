from typing import Any
import torch
import whisper
from whisper.model import Whisper

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.util import *

class Shared:

    _oute_interface: Any = None
    _whisper: Whisper | None = None

    @staticmethod
    def get_oute_interface() -> Any:
        if not Shared._oute_interface:
            printt("Initializing Oute TTS model...")
            printt()

            # Lazy import of oute machinery
            import outetts
            from outetts.version.interface import InterfaceHF

            from tts_audiobook_tool.tts_config import MODEL_CONFIG
            try:
                # Overwrite with dev version if exists
                from .tts_config_dev import MODEL_CONFIG
            except ImportError:
                pass

            # Not catching any exception here (let app crash if incorrect)
            Shared._oute_interface = outetts.Interface(config=MODEL_CONFIG)

        return Shared._oute_interface

    @staticmethod
    def clear_oute_interface() -> None:
        if not Shared._oute_interface:
            return
        printt("Unloading Oute TTS model...")
        printt()
        Shared._oute_interface = None
        AppUtil.gc_ram_vram()

    # Whisper

    @staticmethod
    def get_whisper() -> Whisper:
        if Shared._whisper is None:
            printt("Initializing whisper model...")
            printt()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Shared._whisper = whisper.load_model("turbo", device=device)
        return Shared._whisper

    @staticmethod
    def clear_whisper() -> None:
        if Shared._whisper is None:
            return
        printt("Unloading whisper...")
        printt()
        Shared._whisper = None
        AppUtil.gc_ram_vram()


    # Cheesy control-c capture flag variables

    mode = ""
    stop_flag = False