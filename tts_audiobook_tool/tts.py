from __future__ import annotations

from importlib import util
from typing import Callable

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.tts_model import ChatterboxModelProtocol, FishModelProtocol, GlmModelProtocol, HiggsModelProtocol, IndexTts2ModelProtocol, OuteModelProtocol, TtsModel, VibeVoiceModelProtocol, VibeVoiceProtocol
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
    _glm: GlmModelProtocol | None = None

    _type: TtsModelInfos

    _model_params: dict = {}
    _force_cpu: bool = False

    @staticmethod
    def init_model_type() -> str:
        """
        Sets the tts model type by checking the installed modules in the python environment.
        It does not instantiate the TtsModel as such.
        Must be run first.

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
    def set_model_params(new_params: dict) -> None:
        """
        Sets any customizable values required for the instantiation of the the TTS model
        Changed model param values trigger a re-instantiation as needed
        """
        old_params = Tts._model_params
        Tts._model_params = new_params

        dirty = False
        dirty |= new_params.get("vibevoice_model_path", "") != old_params.get("vibevoice_model_path", "")
        dirty |= new_params.get("indextts2_use_fp16", False) != old_params.get("indextts2_use_fp16", False)
        dirty |= new_params.get("glm_sr", 0) != old_params.get("glm_sr", 0)
        if dirty:
            Tts.clear_tts_model()

    @staticmethod
    def set_model_params_using_project(project) -> None:

        from tts_audiobook_tool.project import Project
        assert(isinstance(project, Project))

        model_params = { }
        model_params["vibevoice_model_path"] = project.vibevoice_model_path
        model_params["indextts2_use_fp16"] = project.indextts2_use_fp16
        model_params["glm_sr"] = project.glm_sr

        Tts.set_model_params(model_params)

    @staticmethod
    def set_force_cpu(value: bool) -> None:
        if Tts._force_cpu != value:
            Tts._force_cpu = value
            # Clear model, will get lazy re-inited as needed
            Tts.clear_tts_model()

    @staticmethod
    def instance_exists() -> bool:
        items = [Tts._oute, Tts._chatterbox, Tts._fish, Tts._higgs, Tts._vibevoice, Tts._indextts2, Tts._glm]
        for item in items:
            if item is not None:
                return True
        return False

    @staticmethod
    def warm_up_models(force_no_stt: bool) -> bool:
        """
        Instantiates tts model and stt model - if not already - as a convenience,
        and prints that it is doing so.

        Unlike the 'direct' model getters, it also checks for keyboard interrupt
        and return True if control-c was pressed

        TODO: `force_no_stt` should be handled elsewhere
        """

        should_instantiate_tts = not Tts.instance_exists()        
        should_instantiate_whisper = (Stt.get_variant() != SttVariant.DISABLED) and \
            not force_no_stt and not Stt._whisper
        
        should_neither = (not should_instantiate_tts and not should_instantiate_whisper)
        if should_neither:
            return False

        SigIntHandler().set("model init")

        should_both = (should_instantiate_tts and should_instantiate_whisper)

        if should_both:
            print_init("Warming up models...")

        if not should_instantiate_whisper:
            # "Lazy unload", useful for user flows like: 
            # Do inference, choose unsupported validation language, do inference
            Stt.clear_stt_model()

        if should_instantiate_tts:
            _ = Tts.get_instance()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True

        if should_both:
            printt() # yes rly

        if should_instantiate_whisper:
            _ = Stt.get_whisper()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True
        
        SigIntHandler().clear()
        return False


    @staticmethod
    def get_instance() -> TtsModel:
        # Returns existing or newly instantiated instance
        MAP: dict[TtsModelInfos, Callable] = {
            TtsModelInfos.OUTE: Tts.get_oute,
            TtsModelInfos.CHATTERBOX: Tts.get_chatterbox,
            TtsModelInfos.FISH: Tts.get_fish,
            TtsModelInfos.HIGGS: Tts.get_higgs,
            TtsModelInfos.VIBEVOICE: Tts.get_vibevoice,
            TtsModelInfos.INDEXTTS2: Tts.get_indextts2,
            TtsModelInfos.GLM: Tts.get_glm,
        }
        factory_function = MAP.get(Tts._type, None)
        if not factory_function:
            raise Exception(f"Lookup failed for {Tts._type}")
        instance = factory_function()
        return instance

    @staticmethod
    def get_instance_if_exists() -> TtsModel | None:
        # Returns instance only if it already exists, else none
        MAP = {
            TtsModelInfos.OUTE: Tts._oute,
            TtsModelInfos.CHATTERBOX: Tts._chatterbox,
            TtsModelInfos.FISH: Tts._fish,
            TtsModelInfos.HIGGS: Tts._higgs,
            TtsModelInfos.VIBEVOICE: Tts._vibevoice,
            TtsModelInfos.INDEXTTS2: Tts._indextts2,
            TtsModelInfos.GLM: Tts._glm,
        }
        return MAP.get(Tts._type, None)

    @staticmethod
    def get_oute() -> OuteModelProtocol:
        if not Tts._oute:
            print_model_init()
            from tts_audiobook_tool.oute_model import OuteModel
            Tts._oute = OuteModel()
        return Tts._oute

    @staticmethod
    def get_chatterbox() -> ChatterboxModelProtocol:
        if not Tts._chatterbox:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(device)
            from tts_audiobook_tool.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(device)
        return Tts._chatterbox

    @staticmethod
    def get_fish() -> FishModelProtocol:
        if not Tts._fish:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(device)
            from tts_audiobook_tool.fish_model import FishModel
            Tts._fish = FishModel(device)
        return Tts._fish

    @staticmethod
    def get_higgs() -> HiggsModelProtocol:
        if not Tts._higgs:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(device)
            from tts_audiobook_tool.higgs_model import HiggsModel
            Tts._higgs = HiggsModel(device)
        return Tts._higgs

    @staticmethod
    def get_vibevoice() -> VibeVoiceModelProtocol:
        if not Tts._vibevoice:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            model_path = Tts._model_params.get("vibevoice_model_path", "")
            model_name = model_path or VibeVoiceProtocol.DEFAULT_MODEL_NAME
            print_model_init(f"{model_name}) ({device}")

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
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            use_fp16 = Tts._model_params.get("indextts2_use_fp16", False)
            print_model_init(f"{device}, fp16: {use_fp16}")

            from tts_audiobook_tool.indextts2_model import IndexTts2Model
            Tts._indextts2 = IndexTts2Model(use_fp16=use_fp16) # model will use cuda if available
        return Tts._indextts2

    @staticmethod
    def get_glm() -> GlmModelProtocol:
        if not Tts._glm:
            device = "cuda" # cpu not currently supported
            sr = Tts._model_params["glm_sr"]
            print_model_init(f"{device}, {sr}hz")

            from tts_audiobook_tool.glm_model import GlmModel
            Tts._glm = GlmModel(device, sr)
        return Tts._glm

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
            Tts._glm = None

            from tts_audiobook_tool.app_util import AppUtil
            AppUtil.gc_ram_vram()

    @staticmethod
    def clear_all_models() -> None:
        Stt.clear_stt_model()
        Tts.clear_tts_model()
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram() # for good measure

    @staticmethod
    def get_resolved_torch_device() -> str:
        """
        Gets the best torch device available which is supported by the current Tts model
        Can return empty string (eg, Oute. or a model that supports only cuda on an environment that does not have cuda)
        """
        import torch
        available_devices = []
        if torch.cuda.is_available():
            available_devices.append("cuda")
        if torch.backends.mps.is_available():
            available_devices.append("mps")
        available_devices.append("cpu")
        supported_devices = Tts.get_type().value.torch_devices
        intersection = [item for item in available_devices if item in supported_devices]
        return intersection[0] if intersection else ""
    
    @staticmethod
    def validate_language_code(language_code: str) -> str:
        """ Returns error string or empty string """
        is_valid = True
        extra = ""

        if Tts.get_type() == TtsModelInfos.CHATTERBOX:        
            from tts_audiobook_tool.chatterbox_model import ChatterboxModel
            if not language_code in ChatterboxModel.supported_languages():
                is_valid = False
                extra = f"Chatterbox requires one of the following: {ChatterboxModel.supported_languages()}\n"
        
        if is_valid:
            return ""
        
        err = f"{COL_ERROR}Invalid language code\n{COL_DEFAULT}" + \
                "Navigate to `Project > Language code` to change it"
        if extra:
            err += "\n\n" + extra
        return err

# ---

def print_model_init(properties_string: str = "") -> None:
    model_name = Tts.get_type().value.ui["proper_name"]
    s = f"Initializing {model_name} model"
    if properties_string:
        s += f" {COL_DIM}({properties_string})"
    s += "..."
    print_init(s)


