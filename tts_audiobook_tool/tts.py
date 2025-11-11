from __future__ import annotations

from importlib import util
from typing import Callable
import torch

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.tts_model import ChatterboxModelProtocol, FishModelProtocol, HiggsModelProtocol, IndexTts2ModelProtocol, OuteModelProtocol, TtsModel, VibeVoiceModelProtocol, VibeVoiceProtocol
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model.
    """

    _oute: OuteModelProtocol | None = None
    _chatterbox: ChatterboxModelProtocol | None = None
    _fish: FishModelProtocol | None = None
    _higgs: HiggsModelProtocol | None = None
    _vibevoice: VibeVoiceModelProtocol | None = None
    _indextts2: IndexTts2ModelProtocol | None = None

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
            s = "Make sure your virtual environment is activated.\n"
            s += "Otherwise, follow the install instructions in the project repo's README."
            return s
        elif num_models > 1:
            s = "More than one of the supported TTS model libraries is currently installed.\n"
            s = "This is not recommended.\n"
            s += "Please re-install python environment, following the instructions in the project repo's README."
            return s

        assert(tts_type is not None)
        Tts._type = tts_type
        return ""

    @staticmethod
    def get_type() -> TtsModelInfos:
        return Tts._type

    @staticmethod
    def set_model_params_using_project(project) -> None:

        from tts_audiobook_tool.project import Project
        assert(isinstance(project, Project))

        model_params = { }
        model_params["vibevoice_model_path"] = project.vibevoice_model_path
        model_params["indextts2_use_fp16"] = project.indextts2_use_fp16

        Tts.set_model_params(model_params)

    @staticmethod
    def set_model_params(new_params: dict) -> None:
        """
        Sets any customizable values required for the instantiation of the the TTS model
        Changed model param values trigger a re-instantiation as needed
        """
        old_params = Tts._model_params
        Tts._model_params = new_params

        invalidate = False
        invalidate |= new_params.get("vibevoice_model_path", "") != old_params.get("vibevoice_model_path", "")
        invalidate |= new_params.get("indextts2_use_fp16", False) != old_params.get("indextts2_use_fp16", False)
        if invalidate:
            Tts.clear_tts_model()

    @staticmethod
    def has_instance() -> bool:
        items = [Tts._oute, Tts._chatterbox, Tts._fish, Tts._higgs, Tts._vibevoice, Tts._indextts2]
        for item in items:
            if item is not None:
                return True
        return False

    @staticmethod
    def warm_up_models() -> None:
        """
        Instantiates tts model and stt model - if not already - as a convenience,
        and prints that it is doing so.
        """

        should_instantiate_tts = not Tts.has_instance()
        should_instantiate_whisper = (Stt.get_variant() != SttVariant.DISABLED) and not Stt._whisper

        should_neither = (not should_instantiate_whisper and not should_instantiate_whisper)
        if should_neither:
            return

        should_both = (should_instantiate_tts and should_instantiate_whisper)

        if should_both:
            print_model_init("Warming up models...")
            printt()

        if should_instantiate_tts:
            _ = Tts.get_instance()

        if should_both:
            printt() # yes rly

        if should_instantiate_whisper:
            _ = Stt.get_whisper()



    @staticmethod
    def get_instance() -> TtsModel:

        MAP: dict[TtsModelInfos, Callable] = {
            TtsModelInfos.OUTE: Tts.get_oute,
            TtsModelInfos.CHATTERBOX: Tts.get_chatterbox,
            TtsModelInfos.FISH: Tts.get_fish,
            TtsModelInfos.HIGGS: Tts.get_higgs,
            TtsModelInfos.VIBEVOICE: Tts.get_vibevoice,
            TtsModelInfos.INDEXTTS2: Tts.get_indextts2
        }
        factory_function = MAP.get(Tts._type, None)
        if not factory_function:
            raise Exception(f"Lookup failed for {Tts._type}")

        instance = factory_function()
        return instance

    @staticmethod
    def get_instance_if_exists() -> TtsModel | None:
        MAP = {
            TtsModelInfos.OUTE: Tts._oute,
            TtsModelInfos.CHATTERBOX: Tts._chatterbox,
            TtsModelInfos.FISH: Tts._fish,
            TtsModelInfos.HIGGS: Tts._higgs,
            TtsModelInfos.VIBEVOICE: Tts._vibevoice,
            TtsModelInfos.INDEXTTS2: Tts._indextts2
        }
        return MAP.get(Tts._type, None)

    @staticmethod
    def get_oute() -> OuteModelProtocol:

        if not Tts._oute:
            print_model_init("Initializing Oute TTS model...")
            printt()

            from tts_audiobook_tool.oute_model import OuteModel
            Tts._oute = OuteModel()

        return Tts._oute

    @staticmethod
    def get_chatterbox() -> ChatterboxModelProtocol:

        if not Tts._chatterbox:
            device = Tts.get_best_torch_device()
            print_model_init("Initializing Chatterbox TTS model ({device})...")
            printt()
            from tts_audiobook_tool.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(device)

        return Tts._chatterbox

    @staticmethod
    def get_fish() -> FishModelProtocol:

        if not Tts._fish:
            device = Tts.get_best_torch_device()
            print_model_init("Initializing Fish OpenAudio S1-mini TTS model ({device})...")
            printt()
            from tts_audiobook_tool.fish_model import FishModel
            Tts._fish = FishModel(device)

        return Tts._fish

    @staticmethod
    def get_higgs() -> HiggsModelProtocol:

        if not Tts._higgs:
            device = Tts.get_best_torch_device()
            print_model_init("Initializing Higgs V2 TTS model ({device})...")
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
            print_model_init("Initializing VibeVoice TTS model ({name}) ({device})...")
            printt()

            from tts_audiobook_tool.vibe_voice_model import VibeVoiceModel
            Tts._vibevoice = VibeVoiceModel(
                device_map=device,
                model_path=model_path,
                max_new_tokens=VibeVoiceProtocol.MAX_TOKENS
            )

        return Tts._vibevoice

    @staticmethod
    def get_indextts2() -> IndexTts2ModelProtocol:

        if not Tts._indextts2:

            device = Tts.get_best_torch_device() # "mps" does not seem to make a difference fyi
            use_fp16 = Tts._model_params.get("indextts2_use_fp16", False)

            print_model_init("Initializing IndexTTS2 model ({device}, use_fp16: {use_fp16})")
            printt()

            from tts_audiobook_tool.indextts2_model import IndexTts2Model
            Tts._indextts2 = IndexTts2Model(use_fp16=use_fp16) # model will use cuda if available

        return Tts._indextts2


    @staticmethod
    def clear_tts_model() -> None:

        model = Tts.get_instance_if_exists()
        if model:
            model.kill()

            Tts._oute = None
            Tts._chatterbox = None
            Tts._fish = None
            Tts._higgs = None
            Tts._vibevoice = None
            Tts._indextts2 = None

            from tts_audiobook_tool.app_util import AppUtil
            AppUtil.gc_ram_vram()

    @staticmethod
    def clear_all_models() -> tuple[int, int] | None:
        """ Returns before/after VRAM usage if is nvidia, else None """

        from tts_audiobook_tool.app_util import AppUtil

        vram_before = AppUtil.get_nv_vram()

        Stt.clear_stt_model()
        Tts.clear_tts_model()
        AppUtil.gc_ram_vram() # for good measure

        vram_after = AppUtil.get_nv_vram()

        if vram_before and vram_after:
            return vram_before[0], vram_after[0]
        else:
            return None

    @staticmethod
    def get_best_torch_device() -> str:

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
