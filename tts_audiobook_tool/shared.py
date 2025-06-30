from typing import Any
import torch
import whisper
from whisper.model import Whisper

from tts_audiobook_tool.util import *

class Shared:
    """
    TODO: most of this should be moved to smth like "ModelShared"
    """

    _whisper: Whisper | None = None
    _oute_interface: Any = None
    _chatterbox: Any = None

    _model_type: str = ""
    _MODEL_TYPES = ["oute", "chatterbox"] # TODO enum

    # Cheesy control-c capture flag variables
    mode = ""
    stop_flag = False


    @staticmethod
    def set_model_type(typ: str) -> None:
        if not typ in Shared._MODEL_TYPES:
            raise Exception(f"Bad model type; must be in {Shared._MODEL_TYPES}")
        Shared._model_type = typ

    @staticmethod
    def get_model_type() -> str:
        return Shared._model_type

    @staticmethod
    def get_model_label() -> str:
        typ = Shared.get_model_type()
        if not typ:
            return "None"
        match typ:
            case "oute":
                return "Oute"
            case "chatterbox":
                return "Chatterbox"
            case _:
                raise ValueError("Bad type")

    @staticmethod
    def get_model_samplerate() -> int:
        if Shared.is_oute():
            return 44100
        elif Shared.is_chatterbox():
            return 24000
        else:
            return 0

    @staticmethod
    def is_oute() -> bool:
        return Shared._model_type == "oute"

    @staticmethod
    def is_chatterbox() -> bool:
        return Shared._model_type == "chatterbox"

    @staticmethod
    def get_chatterbox() -> Any:
        if not Shared._chatterbox:
            device = Shared.get_torch_device()
            printt(f"Initializing Chatterbox TTS model ({device})...")
            printt()
            from chatterbox.tts import ChatterboxTTS  # type: ignore
            Shared._chatterbox = ChatterboxTTS.from_pretrained(device=device)
        return Shared._chatterbox

    @staticmethod
    def get_oute() -> Any:
        if not Shared._oute_interface:
            printt("Initializing Oute TTS model...")
            printt()

            # Lazy import
            import outetts # type: ignore

            from tts_audiobook_tool.config_oute import MODEL_CONFIG
            try:
                # Overwrite with dev version if exists
                from .config_oute_dev import MODEL_CONFIG
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
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

    # Whisper

    @staticmethod
    def get_whisper() -> Whisper:
        if Shared._whisper is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            printt(f"Initializing whisper model ({device})...")
            printt()
            Shared._whisper = whisper.load_model("turbo", device=device)
        return Shared._whisper

    @staticmethod
    def clear_whisper() -> None:
        """
        In general, do not hold onto a reference to whisper from outside.
        If you do, delete the reference before calling this
        """
        if Shared._whisper is None:
            return
        printt()
        printt("Unloading whisper...")
        printt()
        Shared._whisper = None
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

    @staticmethod
    def get_torch_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

