from __future__ import annotations

from importlib import util
import sys
from typing import Callable

from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.tts_model import (
    ChatterboxModelProtocol,
    ChatterboxType,
    FishModelProtocol,
    GlmModelProtocol,
    HiggsModelProtocol,
    IndexTts2ModelProtocol,
    MiraModelProtocol,
    OuteModelProtocol,
    Qwen3ModelProtocol,
    Qwen3Protocol,
    TtsModel,
    VibeVoiceModelProtocol,
    VibeVoiceProtocol,
)
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model.

    The model type is derived from the state of the virtual environment 
    and remains unchanged during the app's runtime.
    """

    _oute: OuteModelProtocol | None = None
    _chatterbox: ChatterboxModelProtocol | None = None
    _fish: FishModelProtocol | None = None
    _higgs: HiggsModelProtocol | None = None
    _vibevoice: VibeVoiceModelProtocol | None = None
    _indextts2: IndexTts2ModelProtocol | None = None
    _glm: GlmModelProtocol | None = None
    _mira: MiraModelProtocol | None = None
    _qwen3: Qwen3ModelProtocol | None = None

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
            s += "Make sure your virtual environment is activated.\n"
            s += "Otherwise, follow the install instructions in the project repo's README."
            return s
        elif num_models > 1:
            s = "More than one of the supported TTS model libraries is currently installed.\n"
            s += "This is not recommended.\n"
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
        model_params["chatterbox_type"] = project.chatterbox_type
        model_params["vibevoice_model_path"] = project.vibevoice_model_path
        model_params["vibevoice_lora_path"] = project.vibevoice_lora_path
        model_params["indextts2_use_fp16"] = project.indextts2_use_fp16
        model_params["glm_sr"] = project.glm_sr
        model_params["qwen3_path_or_id"] = project.qwen3_path_or_id

        Tts.set_model_params(model_params)

    @staticmethod
    def set_model_params(new_params: dict) -> None:
        """
        Sets any customizable values required for the instantiation of the the TTS model
        Changed model param values trigger a re-instantiation as needed
        """
        old_params = Tts._model_params
        Tts._model_params = new_params

        dirty = False
        dirty |= new_params.get("chatterbox_type", "") != old_params.get("chatterbox_type", "")
        dirty |= new_params.get("vibevoice_model_path", "") != old_params.get("vibevoice_model_path", "")
        dirty |= new_params.get("vibevoice_lora_path", "") != old_params.get("vibevoice_lora_path", "")
        dirty |= new_params.get("indextts2_use_fp16", False) != old_params.get("indextts2_use_fp16", False)
        dirty |= new_params.get("glm_sr", 0) != old_params.get("glm_sr", 0)
        dirty |= new_params.get("qwen_path_or_id", "") != old_params.get("qwen_path_or_id", "")
        if dirty:
            Tts.clear_tts_model()

    @staticmethod
    def set_force_cpu(value: bool) -> None:
        if Tts._force_cpu != value:
            Tts._force_cpu = value
            # Clear model, will get lazy re-inited as needed
            Tts.clear_tts_model()

    @staticmethod
    def instance_exists() -> bool:
        items = [
            Tts._oute, 
            Tts._chatterbox, 
            Tts._fish, 
            Tts._higgs, 
            Tts._vibevoice, 
            Tts._indextts2, 
            Tts._glm, 
            Tts._mira, 
            Tts._qwen3
        ]
        for item in items:
            if item is not None:
                return True
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
            TtsModelInfos.MIRA: Tts.get_mira,
            TtsModelInfos.QWEN3TTS: Tts.get_qwen3,
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
            TtsModelInfos.MIRA: Tts._mira,
            TtsModelInfos.QWEN3TTS: Tts._qwen3
        }
        return MAP.get(Tts._type, None)

    @staticmethod
    def get_oute() -> OuteModelProtocol:
        if not Tts._oute:
            print_model_init()
            from tts_audiobook_tool.tts_model.oute_model import OuteModel
            Tts._oute = OuteModel()
            printt()
        return Tts._oute

    @staticmethod
    def get_chatterbox() -> ChatterboxModelProtocol:
        if not Tts._chatterbox:
            typ = Tts._model_params.get("chatterbox_type")
            assert isinstance(typ, ChatterboxType), "chatterbox_type not set"
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            s = f"{typ.label}, {device}"            
            print_model_init(s)
            from tts_audiobook_tool.tts_model.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(typ, device)
            printt()
        return Tts._chatterbox

    @staticmethod
    def get_fish() -> FishModelProtocol:
        if not Tts._fish:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(device)
            from tts_audiobook_tool.tts_model.fish_model import FishModel
            Tts._fish = FishModel(device)
            printt()
        return Tts._fish

    @staticmethod
    def get_higgs() -> HiggsModelProtocol:
        if not Tts._higgs:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(device)
            from tts_audiobook_tool.tts_model.higgs_model import HiggsModel
            Tts._higgs = HiggsModel(device)
            printt()
        return Tts._higgs

    @staticmethod
    def get_vibevoice() -> VibeVoiceModelProtocol:
        if not Tts._vibevoice:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            model_path = Tts._model_params.get("vibevoice_model_path", "")
            model_desc = model_path or VibeVoiceProtocol.DEFAULT_MODEL_NAME
            lora_path = Tts._model_params.get("vibevoice_lora_path", "")
            lora_desc = f"LoRA: {lora_path}, " if lora_path else ""
            desc = f"{lora_desc}{device}"
            print_model_init(desc, model_desc)

            from tts_audiobook_tool.tts_model.vibe_voice_model import VibeVoiceModel
            Tts._vibevoice = VibeVoiceModel(
                device_map=device,
                model_path=model_path,
                lora_path=lora_path,
                max_new_tokens=VibeVoiceProtocol.MAX_TOKENS
            )
            printt()
        return Tts._vibevoice

    @staticmethod
    def get_indextts2() -> IndexTts2ModelProtocol:
        if not Tts._indextts2:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            use_fp16 = Tts._model_params.get("indextts2_use_fp16", False)
            print_model_init(f"{device}, fp16: {use_fp16}")

            from tts_audiobook_tool.tts_model.indextts2_model import IndexTts2Model
            Tts._indextts2 = IndexTts2Model(use_fp16=use_fp16) # model will use cuda if available
            printt()
        return Tts._indextts2

    @staticmethod
    def get_glm() -> GlmModelProtocol:
        if not Tts._glm:
            device = "cuda" # cpu not currently supported
            sr = Tts._model_params["glm_sr"]
            print_model_init(f"{device}, {sr}hz")

            from tts_audiobook_tool.tts_model.glm_model import GlmModel
            Tts._glm = GlmModel(device, sr)
            printt()
        return Tts._glm

    @staticmethod
    def get_mira() -> MiraModelProtocol:
        if not Tts._mira:
            print_model_init(f"cuda")
            from tts_audiobook_tool.tts_model.mira_model import MiraModel
            Tts._mira = MiraModel()
            printt()
        return Tts._mira

    @staticmethod
    def get_qwen3() -> Qwen3ModelProtocol:
        if not Tts._qwen3:
            path_or_id = Tts._model_params["qwen3_path_or_id"]
            if not path_or_id:
                path_or_id = Qwen3Protocol.REPO_ID_BASE_DEFAULT
            display_path_or_id = Qwen3Protocol.get_display_path_or_id(path_or_id)
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(f"{display_path_or_id}, {device}")

            from tts_audiobook_tool.tts_model.qwen3_model import Qwen3Model
            Tts._qwen3 = Qwen3Model(path_or_id, device)
            printt()
        return Tts._qwen3

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
            Tts._mira = None
            Tts._qwen3 = None

        from tts_audiobook_tool.memory_util import MemoryUtil
        MemoryUtil.gc_ram_vram()

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
    def check_valid_language_code(project) -> str:
        """ Returns error string if current TTS model does not support given language code """

        from tts_audiobook_tool.project import Project
        assert(isinstance(project, Project))

        is_valid = True
        extra = ""

        if Tts.get_type() == TtsModelInfos.CHATTERBOX and project.chatterbox_type == ChatterboxType.MULTILINGUAL: 
            if not 'tts_audiobook_tool.chatterbox_model' in sys.modules:
                print_init("Initializing...")
            from tts_audiobook_tool.tts_model.chatterbox_model import ChatterboxModel
            if not project.language_code in ChatterboxModel.supported_languages_multi():
                is_valid = False
                extra = f"Chatterbox-Multilingual requires one of the following: {ChatterboxModel.supported_languages_multi()}\n"
        
        if is_valid:
            return ""
        
        err = f"{COL_ERROR}Invalid language code for the current TTS model configuration\n{COL_DEFAULT}" + \
                "Navigate to `Project > Language code` to change it"
        if extra:
            err += "\n\n" + extra
        return err
