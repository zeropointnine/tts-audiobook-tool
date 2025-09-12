from __future__ import annotations

from importlib import util
import sys
from typing import Any, cast
import torch

from faster_whisper import WhisperModel

from tts_audiobook_tool.tts_info import TtsType
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model. And also whisper model.
    """

    # TODO: create tts interface to replace hardcoded logic; this also applies to Project and GenerateUtil ideally

    _oute: Any = None
    _chatterbox: Any = None
    _fish: Any = None
    _higgs: Any = None
    _vibevoice: Any = None

    _type: TtsType

    _whisper: WhisperModel | None = None

    _align_model = None
    _align_meta = None
    _align_device: str = ""

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
    def has_tts() -> bool:
        return Tts._oute or Tts._chatterbox or Tts._fish or Tts._higgs or Tts._vibevoice

    @staticmethod
    def warm_up_models() -> None:
        """
        If not already, instantiates tts and stt models as a convenience, and prints that it is doing so
        """

        if not Tts.has_tts() or not Tts._whisper:
            printt(f"{Ansi.ITALICS}Warming up models...")
            printt()

        match Tts._type:
            case TtsType.OUTE:
                if not Tts._oute:
                    _ = Tts.get_oute()
            case TtsType.CHATTERBOX:
                if not Tts._chatterbox:
                    _ = Tts.get_chatterbox()
            case TtsType.FISH:
                if not Tts._fish:
                    _ = Tts.get_fish()
            case TtsType.HIGGS:
                if not Tts._higgs:
                    _ = Tts.get_higgs()
            case TtsType.VIBEVOICE:
                if not Tts._vibevoice:
                    _ = Tts.get_vibevoice()

        if not Tts._whisper:
            _ = Tts.get_whisper()

        # TODO: later
        if False and not Tts._align_model:
            _ = Tts.get_align_model_and_meta_and_device()

    @staticmethod
    def get_oute() -> Any:
        if not Tts._oute:
            printt(f"{Ansi.ITALICS}Initializing Oute TTS model...")
            printt()
            import outetts # type: ignore
            from tts_audiobook_tool.config_oute import MODEL_CONFIG
            try:
                # Overwrite with dev version if exists
                from .config_oute_dev import MODEL_CONFIG
            except ImportError:
                pass
            # Not catching any exception here (let app crash if incorrect):
            Tts._oute = outetts.Interface(config=MODEL_CONFIG)
        return Tts._oute

    @staticmethod
    def get_chatterbox() -> Any:
        if not Tts._chatterbox:
            device = Tts.get_best_torch_device()
            printt(f"{Ansi.ITALICS}Initializing Chatterbox TTS model ({device})...")
            printt()
            from chatterbox.tts import ChatterboxTTS  # type: ignore
            Tts._chatterbox = ChatterboxTTS.from_pretrained(device=device)
        return Tts._chatterbox

    @staticmethod
    def get_fish() -> Any:
        from tts_audiobook_tool.fish_generator import FishGenerator

        if not Tts._fish:
            device = Tts.get_best_torch_device()
            printt(f"{Ansi.ITALICS}Initializing Fish OpenAudio S1-mini TTS model ({device})...")
            printt()
            Tts._fish = FishGenerator(device)

            # Customize logging/logger levels now that fish is initialized
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr, level="WARNING", filter="fish_speech")
        return Tts._fish

    @staticmethod
    def get_higgs() -> Any:
        from tts_audiobook_tool.higgs_generator import HiggsGenerator
        if not Tts._higgs:
            device = Tts.get_best_torch_device() # TODO
            printt(f"{Ansi.ITALICS}Initializing Higgs V2 TTS model ({device})...")
            printt()
            Tts._higgs = HiggsGenerator(device)
        return Tts._higgs

    @staticmethod
    def get_vibevoice() -> Any:
        from tts_audiobook_tool.vibevoice_generator import VibeVoiceGenerator
        if not Tts._vibevoice:
            device = Tts.get_best_torch_device() # TODO verify mps
            printt(f"{Ansi.ITALICS}Initializing VibeVoice TTS model ({device})...")
            printt()
            Tts._vibevoice = VibeVoiceGenerator(device, max_new_tokens=MAX_TOKENS_VIBE_VOICE)
        return Tts._vibevoice

    @staticmethod
    def get_whisper() -> WhisperModel:
        if Tts._whisper is None:
            model = "large-v3"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            printt(f"{Ansi.ITALICS}Initializing whisper model ({model}, {device}, {compute_type})...")
            printt()
            Tts._whisper = WhisperModel(model, device=device, compute_type=compute_type)
        return Tts._whisper

    @staticmethod
    def get_align_model_and_meta_and_device() -> tuple[Any, Any, str]: # obnoxious
        import whisperx # type: ignore

        if not Tts._align_model:
            Tts._align_device = Tts.get_best_torch_device()
            printt(f"{Ansi.ITALICS}Initializing align model ({Tts._align_device})...")
            printt()
            Tts._align_model, Tts._align_meta = whisperx.load_align_model(language_code="en", device=Tts._align_device)
        return Tts._align_model, Tts._align_meta, Tts._align_device

    @staticmethod
    def clear_stt_models() -> None:
        """
        In general, do not hold onto a reference to whisper from outside (prefer using "get_whisper())
        If you do, delete the reference before calling this, or else it will not be GC'ed
        """
        if Tts._whisper is None and Tts._align_model is None:
            return
        printt()
        printt("{Ansi.ITALICS}Unloading whisper and align model...")
        printt()
        Tts._whisper = None
        Tts._align_model = None
        Tts._align_meta = None
        Tts._align_device = ""
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

    @staticmethod
    def clear_all_models() -> None:

        all_models = [Tts._oute, Tts._chatterbox, Tts._fish, Tts._higgs, Tts._vibevoice, Tts._whisper, Tts._align_model, Tts._align_meta]
        has = False
        for item in all_models:
            if item:
                has = True
                break

        if not has:
            printt("No models currently loaded")
            printt()

        else:
            printt(f"{Ansi.ITALICS}Unloading all models...")
            printt()

            Tts._oute = None
            Tts._chatterbox = None
            if Tts._fish:
                Tts._fish.kill()
            Tts._fish = None
            if Tts._higgs:
                Tts._higgs.kill()
            Tts._higgs = None
            if Tts._vibevoice:
                Tts._vibevoice.kill()
            Tts._vibevoice = None

            Tts._whisper = None
            Tts._align_model = None
            Tts._align_meta = None
            Tts._align_device = ""

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
