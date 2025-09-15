from __future__ import annotations

from importlib import util
import torch

from faster_whisper import WhisperModel

from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.tts_model import ChatterboxModelProtocol, FishModelProtocol, HiggsModelProtocol, OuteModelProtocol, TtsModel, VibeVoiceModelProtocol, VibeVoiceProtocol
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

    _whisper: WhisperModel | None = None

    _type: TtsModelInfos

    _model_params: dict = {}

    @staticmethod
    def init_model_type() -> str:
        """
        Sets the tts model type by checking the installed modules in the python environment.
        It does not instantiate the TtsModel as such.

        Returns error string on fail, else empty string on success.
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
            s += "Please follow the install instructions in project repo's README."
            return s
        elif num_models > 1:
            s = "More than one of the supported TTS model libraries is currently installed.\n"
            s = "This is not recommended.\n"
            s += "Please re-install python environment, following the instructions in project repo's README."
            return s

        assert(tts_type is not None)
        Tts._type = tts_type
        return ""

    @staticmethod
    def get_type() -> TtsModelInfos:
        return Tts._type

    @staticmethod
    def set_model_params_using_project(project) -> None:
        model_params = { }
        if project.vibevoice_model_path:
            model_params["vibevoice_model_path"] = project.vibevoice_model_path
        Tts.set_model_params(model_params)

    @staticmethod
    def set_model_params(new_params: dict) -> None:
        old_params = Tts._model_params
        Tts._model_params = new_params

        invalidate = False
        invalidate |= new_params.get("vibevoice_model_path", "") != old_params.get("vibevoice_model_path", "")
        if invalidate:
            Tts.clear_tts_model()

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
        Initializes tts and stt models (if not already), as a convenience,
        and prints that it is doing so.
        """

        has_both = Tts.has_tts() and Tts._whisper
        if has_both:
            return

        has_neither = not Tts.has_tts() and not Tts._whisper
        if has_neither:
            printt(f"{Ansi.ITALICS}Warming up models...")
            printt()

        _ = Tts.get_tts_model()

        if has_neither:
            printt() # yes rly

        _ = Tts.get_whisper()


    @staticmethod
    def get_tts_model() -> TtsModel:
        match Tts._type:
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
                # TODO: Revisit
                raise Exception("No tts model type set")

    @staticmethod
    def get_tts_model_if_exists() -> TtsModel | None:
        map = {
            TtsModelInfos.OUTE: Tts._oute,
            TtsModelInfos.CHATTERBOX: Tts._chatterbox,
            TtsModelInfos.FISH: Tts._fish,
            TtsModelInfos.HIGGS: Tts._higgs,
            TtsModelInfos.VIBEVOICE: Tts._vibevoice,
        }
        return map[ Tts._type ]

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
            device = Tts.get_best_torch_device()
            printt(f"{Ansi.ITALICS}Initializing Higgs V2 TTS model ({device})...")
            printt()
            from tts_audiobook_tool.higgs_model import HiggsModel
            Tts._higgs = HiggsModel(device)

        return Tts._higgs

    @staticmethod
    def get_vibevoice() -> VibeVoiceModelProtocol:

        if not Tts._vibevoice:

            device = Tts.get_best_torch_device()
            model_path = Tts._model_params.get("vibevoice_model_path", "")
            name = model_path or VibeVoiceProtocol.DEFAULT_MODEL_NAME
            printt(f"{Ansi.ITALICS}Initializing VibeVoice TTS model ({name}) ({device})...")
            printt()

            from tts_audiobook_tool.vibe_voice_model import VibeVoiceModel
            Tts._vibevoice = VibeVoiceModel(
                device_map=device,
                model_path=model_path,
                max_new_tokens=VibeVoiceProtocol.MAX_TOKENS
            )

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
    def has_whisper() -> bool:
        return Tts._whisper is not None

    @staticmethod
    def clear_stt_model() -> None:

        if Tts._whisper:
            Tts._whisper = None

            from tts_audiobook_tool.app_util import AppUtil
            AppUtil.gc_ram_vram()

    @staticmethod
    def clear_tts_model() -> None:

        model = Tts.get_tts_model_if_exists()
        if model:
            model.kill()

            Tts._oute = None
            Tts._chatterbox = None
            Tts._fish = None
            Tts._higgs = None
            Tts._vibevoice = None

            from tts_audiobook_tool.app_util import AppUtil
            AppUtil.gc_ram_vram()

    @staticmethod
    def clear_all_models() -> None:

        vram_before = get_torch_allocated_vram()

        Tts.clear_stt_model()
        Tts.clear_tts_model()

        # And for good measure
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

        vram_after = get_torch_allocated_vram()

        if vram_before > -1:
            printt(f"Allocated VRAM before: {make_gb_string(vram_before)}")
            printt(f"Allocated VRAM after: {make_gb_string(vram_after)}")
            printt()

    @staticmethod
    def get_best_torch_device() -> str:

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
