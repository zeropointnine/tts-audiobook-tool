from __future__ import annotations

from importlib import util
from typing import Any, cast
import torch
import whisper
from whisper.model import Whisper

from tts_audiobook_tool.fish_generator import FishGenerator
from tts_audiobook_tool.higgs_generator import HiggsGenerator
from tts_audiobook_tool.tts_info import TtsType
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model (and also Whisper)
    """

    # TODO: create tts interface to replace hardcoded logic; this also applies to Project and GenerateUtil ideally

    _whisper: Whisper | None = None

    _oute: Any = None
    _chatterbox: Any = None
    _fish: Any = None
    _higgs: Any = None

    _type: TtsType

    @staticmethod
    def init_active_model() -> str:
        """
        Sets the tts model type by reflecting on the existing modules in the environment.
        Returns error string on fail, else empty string on success
        """
        num_models = 0
        tts_type = None
        for item in TtsType:
            exists = util.find_spec(item.value.module_test) is not None
            if exists:
                num_models += 1
                tts_type = item
        if num_models == 0:
            s = "None of the supported TTS models are currently installed.\n"
            s += "Please follow the install instructions in the README."
            return s
        elif num_models > 1:
            s = "More than one of the supported TTS model libraries is currently installed.\n"
            s = "This is not recommended.\n"
            s += "Please re-install python environment, following the instructions in the README."
            return s

        Tts.set_type(cast(TtsType, tts_type))
        return ""

    @staticmethod
    def set_type(typ: TtsType) -> None:
        Tts._type = typ

    @staticmethod
    def get_type() -> TtsType:
        return Tts._type

    @staticmethod
    def warm_up_models() -> None:
        """ Instantiates tts and stt models if not already, as a convenience """

        no_tts = (Tts._type == TtsType.OUTE and not Tts._oute) or \
            (Tts._type == TtsType.CHATTERBOX and not Tts._chatterbox) or \
            (Tts._type == TtsType.FISH and not Tts._fish)
        if no_tts and not Tts._whisper:
            # Going to warm up two models
            print("Warming up models...")
            printt()

        if Tts._type == TtsType.OUTE and not Tts._oute:
            _ = Tts.get_oute()
        if Tts._type == TtsType.CHATTERBOX and not Tts._chatterbox:
            _ = Tts.get_chatterbox()
        if Tts._type == TtsType.FISH and not Tts._fish:
            _ = Tts.get_fish()

        if not Tts._whisper:
            _ = Tts.get_whisper()

    @staticmethod
    def get_oute() -> Any:
        if not Tts._oute:
            printt("Initializing Oute TTS model...")
            printt()
            import outetts # type: ignore
            from tts_audiobook_tool.config_oute import MODEL_CONFIG
            try:
                # Overwrite with dev version if exists
                from .config_oute_dev import MODEL_CONFIG
            except ImportError:
                pass
            # Not catching any exception here (let app crash if incorrect)
            Tts._oute = outetts.Interface(config=MODEL_CONFIG)
        return Tts._oute

    @staticmethod
    def get_chatterbox() -> Any:
        if not Tts._chatterbox:
            device = Tts.get_best_torch_device()
            printt(f"Initializing Chatterbox TTS model ({device})...")
            printt()
            from chatterbox.tts import ChatterboxTTS  # type: ignore
            Tts._chatterbox = ChatterboxTTS.from_pretrained(device=device)
        return Tts._chatterbox

    @staticmethod
    def get_fish() -> FishGenerator:
        if not Tts._fish:
            device = Tts.get_best_torch_device()
            printt(f"Initializing Fish OpenAudio S1-mini TTS model ({device})...")
            printt()
            Tts._fish = FishGenerator(device)
        return Tts._fish

    @staticmethod
    def get_higgs() -> HiggsGenerator:
        if not Tts._higgs:
            device = Tts.get_best_torch_device() # TODO
            printt(f"Initializing Higgs V2 TTS model ({device})...")
            printt()
            Tts._higgs = HiggsGenerator(device)
        return Tts._higgs

    @staticmethod
    def clear_oute() -> None:
        if not Tts._oute:
            return
        printt("Unloading Oute TTS model...")
        printt()
        Tts._oute = None
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

    @staticmethod
    def clear_fish() -> None:
        if not Tts._fish:
            return
        printt("Unloading Fish model...")
        printt()
        Tts._fish = None
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

    @staticmethod
    def get_whisper() -> Whisper:
        if Tts._whisper is None:
            device = "cuda" if torch.cuda.is_available() else "cpu" # todo mps? did i test this earlier?
            printt(f"Initializing whisper model ({device})...")
            printt()
            Tts._whisper = whisper.load_model("turbo", device=device)
        return Tts._whisper

    @staticmethod
    def clear_whisper() -> None:
        """
        In general, do not hold onto a reference to whisper from outside (prefer using "get_whisper())
        If you do, delete the reference before calling this, or else it will not be GC'ed
        """
        if Tts._whisper is None:
            return
        printt()
        printt("Unloading whisper...")
        printt()
        Tts._whisper = None
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

    @staticmethod
    def get_best_torch_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
