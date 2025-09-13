from __future__ import annotations

from importlib import util
from typing import Any
import torch

from faster_whisper import WhisperModel

from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.tts_model import ChatterboxModelProtocol, FishModelProtocol, HiggsModelProtocol, OuteModelProtocol, TtsModel, VibeVoiceModelProtocol
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model. And also whisper model.
    """

    _oute: OuteModelProtocol | None = None
    _chatterbox: ChatterboxModelProtocol | None = None
    _fish: FishModelProtocol | None = None
    _higgs: HiggsModelProtocol | None = None
    _vibevoice: VibeVoiceModelProtocol | None = None

    _type: TtsModelInfos

    _whisper: WhisperModel | None = None

    _align_model = None
    _align_meta = None
    _align_device: str = ""

    @staticmethod
    def init_model_type() -> str:
        """
        Sets the tts model type by reflecting on the existing modules in the environment.
        Returns error string on fail, else empty string on success
        """
        num_models = 0
        tts_type = None
        for item in TtsModelInfos:
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

        assert(tts_type is not None)
        Tts._type = tts_type
        return ""

    @staticmethod
    def get_type() -> TtsModelInfos:
        return Tts._type

    @staticmethod
    def has_tts() -> bool:
        items = [Tts._oute, Tts._chatterbox, Tts._fish, Tts._higgs, Tts._vibevoice]
        for item in items:
            if item is not None:
                return True
        return False

    @staticmethod
    def warm_up_models() -> None:
        """
        Instantiates tts and stt models (if not already), as a convenience,
        and prints that it is doing so.
        """

        if not Tts.has_tts() and not Tts._whisper:
            printt(f"{Ansi.ITALICS}Warming up models...")
            printt()

        _ = Tts.get_tts_model()
        _ = Tts.get_whisper()

        # TODO: later
        if False and not Tts._align_model:
            _ = Tts.get_align_model_and_meta_and_device()

    @staticmethod
    def get_tts_model() -> TtsModel:

        match Tts._type:
            #z
            case TtsModelInfos.OUTE:
                return Tts.get_oute()
            case TtsModelInfos.CHATTERBOX:
                return Tts.get_chatterbox()
            case TtsModelInfos.FISH:
                return Tts.get_fish()
            case TtsModelInfos.HIGGS:
                return Tts.get_higgs()
            case TtsModelInfos.VIBEVOICE:
                return Tts.get_vibevoice()
            case _:
                raise Exception("TODO") #z

    @staticmethod
    def get_oute() -> OuteModelProtocol:

        if not Tts._oute:
            printt(f"{Ansi.ITALICS}Initializing Oute TTS model...")
            printt()

            from tts_audiobook_tool.oute_model import OuteModel
            Tts._oute = OuteModel()

        return Tts._oute

    @staticmethod
    def get_chatterbox() -> ChatterboxModelProtocol:

        if not Tts._chatterbox:
            device = Tts.get_best_torch_device()
            printt(f"{Ansi.ITALICS}Initializing Chatterbox TTS model ({device})...")
            printt()
            from tts_audiobook_tool.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(device)

        return Tts._chatterbox

    @staticmethod
    def get_fish() -> FishModelProtocol:

        if not Tts._fish:
            device = Tts.get_best_torch_device()
            printt(f"{Ansi.ITALICS}Initializing Fish OpenAudio S1-mini TTS model ({device})...")
            printt()
            from tts_audiobook_tool.fish_model import FishModel
            Tts._fish = FishModel(device)

        return Tts._fish

    @staticmethod
    def get_higgs() -> HiggsModelProtocol:

        if not Tts._higgs:
            device = Tts.get_best_torch_device() # TODO verify mps
            printt(f"{Ansi.ITALICS}Initializing Higgs V2 TTS model ({device})...")
            printt()
            from tts_audiobook_tool.higgs_model import HiggsModel
            Tts._higgs = HiggsModel(device)

        return Tts._higgs

    @staticmethod
    def get_vibevoice() -> VibeVoiceModelProtocol:

        if not Tts._vibevoice:

            device = Tts.get_best_torch_device() # TODO verify mps
            printt(f"{Ansi.ITALICS}Initializing VibeVoice 1.5B TTS model ({device})...")
            printt()

            from tts_audiobook_tool.vibe_voice_model import VibeVoiceModel
            Tts._vibevoice = VibeVoiceModel(device, max_new_tokens=VIBEVOICE_MAX_TOKENS)

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
        printt(f"{Ansi.ITALICS}Unloading whisper...")
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

            tts_models: list[TtsModel | None] = [Tts._oute, Tts._chatterbox, Tts._fish, Tts._higgs, Tts._vibevoice]
            for item in tts_models:
                if item:
                    item.kill()

            Tts._oute = None
            Tts._chatterbox = None
            Tts._fish = None
            Tts._higgs = None
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
