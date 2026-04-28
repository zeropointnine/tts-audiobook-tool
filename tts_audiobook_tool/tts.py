from __future__ import annotations

from importlib import util
import os
import sys
from typing import Callable

from tts_audiobook_tool.tts_model.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
from tts_audiobook_tool.tts_model.fish_s1_base_model import FishS1BaseModel
from tts_audiobook_tool.tts_model.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_model.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_model.higgs_base_model import HiggsBaseModel
from tts_audiobook_tool.tts_model.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_model.mira_base_model import MiraBaseModel
from tts_audiobook_tool.tts_model.none_base_model import NoneBaseModel
from tts_audiobook_tool.tts_model.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_model.oute_base_model import OuteBaseModel
from tts_audiobook_tool.tts_model.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfo, TtsModelInfos
from tts_audiobook_tool.tts_model.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model.

    The model type is derived from the state of the virtual environment 
    and remains unchanged during the app's runtime.
    """

    _oute: OuteBaseModel | None = None
    _chatterbox: ChatterboxBaseModel | None = None
    _fish: FishS1BaseModel | None = None
    _fish_s2: FishS2BaseModel | None = None
    _higgs: HiggsBaseModel | None = None
    _vibevoice: VibeVoiceBaseModel | None = None
    _indextts2: IndexTts2BaseModel | None = None
    _glm: GlmBaseModel | None = None
    _mira: MiraBaseModel | None = None
    _qwen3: Qwen3BaseModel | None = None
    _pocket: PocketBaseModel | None = None

    _type: TtsModelInfos

    _model_params: dict = {}
    _force_cpu: bool = False

    @staticmethod
    def init_model_type() -> tuple[TtsModelInfos, int]:
        """
        Sets the tts model type by checking the installed modules in the python environment.
        Does not instantiate the TtsModel as such.
        Must be run on startup.

        Returns the model type that was set, and num matches (should be either 1 or 0).
        """

        def get_matches() -> list[TtsModelInfos]:
            model_infos = []
            for model_info in TtsModelInfos:
                exists = False
                try:
                    exists = util.find_spec(model_info.value.module_test) is not None
                except:
                    ...
                if exists:
                    model_infos.append(model_info)
            return model_infos
        
        matches = get_matches()
        
        # Fish S2 special case
        if TtsModelInfos.FISH_S1 in matches and TtsModelInfos.FISH_S2 in matches:
            matches = [TtsModelInfos.FISH_S2]

        match len(matches):
            case 0:
                # No match
                Tts._type = TtsModelInfos.NONE
                return Tts._type, 0
            case 1:
                # Happy path
                Tts._type = matches[0]
                return Tts._type, 1
            case _: # > 1
                # Not cool
                Tts._type = matches[0]
                return Tts._type, len(matches)

    @staticmethod
    def get_type() -> TtsModelInfos:
        return Tts._type

    @staticmethod
    def set_model_params_using_project(project) -> None:

        from tts_audiobook_tool.project import Project
        assert(isinstance(project, Project))

        model_params = { }
        model_params["chatterbox_type"] = project.chatterbox_type
        model_params["vibevoice_target"] = project.vibevoice_target
        model_params["vibevoice_lora_path"] = project.vibevoice_lora_target
        model_params["indextts2_use_fp16"] = project.indextts2_use_fp16
        model_params["glm_sr"] = project.glm_sr
        model_params["qwen3_target"] = project.qwen3_target
        model_params["fish_s1_compile_enabled"] = project.fish_s1_compile_enabled
        model_params["fish_s2_compile_enabled"] = project.fish_s2_compile_enabled
        model_params["pocket_model_code"] = project.pocket_model_code

        Tts.set_model_params(model_params)

    @staticmethod
    def set_model_params(new_params: dict) -> None:
        """
        Sets any customizable values required for the instantiation of the the TTS model
        Changed values trigger invalidation of existing instance
        """
        old_params = Tts._model_params
        Tts._model_params = new_params

        dirty = False
        dirty |= new_params.get("chatterbox_type", "") != old_params.get("chatterbox_type", "")
        dirty |= new_params.get("vibevoice_target", "") != old_params.get("vibevoice_target", "")
        dirty |= new_params.get("vibevoice_lora_path", "") != old_params.get("vibevoice_lora_path", "")
        dirty |= new_params.get("indextts2_use_fp16", False) != old_params.get("indextts2_use_fp16", False)
        dirty |= new_params.get("glm_sr", 0) != old_params.get("glm_sr", 0)
        dirty |= new_params.get("qwen3_target", "") != old_params.get("qwen3_target", "")
        dirty |= new_params.get("fish_s1_compile_enabled", False) != old_params.get("fish_s1_compile_enabled", False)
        dirty |= new_params.get("fish_s2_compile_enabled", False) != old_params.get("fish_s2_compile_enabled", False)
        dirty |= new_params.get("pocket_model_code", "") != old_params.get("pocket_model_code", "")
        if dirty:
            Tts.clear_tts_model()

    @staticmethod
    def set_force_cpu(value: bool) -> None:
        if Tts._force_cpu != value:
            Tts._force_cpu = value
            # Clear model, will get lazy re-inited as needed
            Tts.clear_tts_model()

    @staticmethod
    def get_class() -> type[TtsBaseModel]:
        """
        Gets the current tts model's class, used for accessing static methods.
        """
        MAP = {
            TtsModelInfos.NONE: NoneBaseModel,
            TtsModelInfos.OUTE: OuteBaseModel,
            TtsModelInfos.CHATTERBOX: ChatterboxBaseModel,
            TtsModelInfos.FISH_S1: FishS1BaseModel,
            TtsModelInfos.FISH_S2: FishS2BaseModel,
            TtsModelInfos.HIGGS: HiggsBaseModel,
            TtsModelInfos.VIBEVOICE: VibeVoiceBaseModel,
            TtsModelInfos.INDEXTTS2: IndexTts2BaseModel,
            TtsModelInfos.GLM: GlmBaseModel,
            TtsModelInfos.MIRA: MiraBaseModel,
            TtsModelInfos.QWEN3TTS: Qwen3BaseModel,
            TtsModelInfos.POCKET: PocketBaseModel,
        }
        cls = MAP.get(Tts._type, None)
        if cls is None:
            raise Exception("Not supported")
        return cls
    
    @staticmethod
    def get_info() -> TtsModelInfo:
        return Tts.get_class().INFO

    @staticmethod
    def instance_exists() -> bool:
        items = [
            Tts._oute,
            Tts._chatterbox,
            Tts._fish,
            Tts._fish_s2,
            Tts._higgs,
            Tts._vibevoice,
            Tts._indextts2,
            Tts._glm,
            Tts._mira,
            Tts._qwen3,
            Tts._pocket,
        ]
        for item in items:
            if item is not None:
                return True
        return False

    @staticmethod
    def get_instance() -> TtsBaseModel:
        # Returns existing or newly instantiated instance
        MAP: dict[TtsModelInfos, Callable] = {
            TtsModelInfos.OUTE: Tts.get_oute,
            TtsModelInfos.CHATTERBOX: Tts.get_chatterbox,
            TtsModelInfos.FISH_S1: Tts.get_fish,
            TtsModelInfos.FISH_S2: Tts.get_fish_s2,
            TtsModelInfos.HIGGS: Tts.get_higgs,
            TtsModelInfos.VIBEVOICE: Tts.get_vibevoice,
            TtsModelInfos.INDEXTTS2: Tts.get_indextts2,
            TtsModelInfos.GLM: Tts.get_glm,
            TtsModelInfos.MIRA: Tts.get_mira,
            TtsModelInfos.QWEN3TTS: Tts.get_qwen3,
            TtsModelInfos.POCKET: Tts.get_pocket,
        }
        factory_function = MAP.get(Tts._type, None)
        if not factory_function:
            raise Exception(f"Lookup failed for {Tts._type}")
        instance = factory_function()
        return instance

    @staticmethod
    def get_instance_if_exists() -> TtsBaseModel | None:
        # Returns instance only if it already exists, else none
        MAP = {
            TtsModelInfos.OUTE: Tts._oute,
            TtsModelInfos.CHATTERBOX: Tts._chatterbox,
            TtsModelInfos.FISH_S1: Tts._fish,
            TtsModelInfos.FISH_S2: Tts._fish_s2,
            TtsModelInfos.HIGGS: Tts._higgs,
            TtsModelInfos.VIBEVOICE: Tts._vibevoice,
            TtsModelInfos.INDEXTTS2: Tts._indextts2,
            TtsModelInfos.GLM: Tts._glm,
            TtsModelInfos.MIRA: Tts._mira,
            TtsModelInfos.QWEN3TTS: Tts._qwen3,
            TtsModelInfos.POCKET: Tts._pocket,
        }
        return MAP.get(Tts._type, None)

    @staticmethod
    def get_oute() -> OuteBaseModel:
        if not Tts._oute:
            print_model_init()
            from tts_audiobook_tool.tts_model.oute_model import OuteModel
            Tts._oute = OuteModel()
            printt()
        return Tts._oute

    @staticmethod
    def get_chatterbox() -> ChatterboxBaseModel:
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
    def get_fish() -> FishS1BaseModel:
        if not Tts._fish:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            compile_enabled = Tts._model_params.get("fish_compile_enabled", True)
            s = f"{device}"
            if device == "cuda":
                s += f", compile: {compile_enabled}"
            print_model_init(s)
            from tts_audiobook_tool.tts_model.fish_s1_model import FishS1Model
            Tts._fish = FishS1Model(device, compile_enabled)
            printt()
        return Tts._fish

    @staticmethod
    def get_fish_s2() -> FishS2BaseModel:
        if not Tts._fish_s2:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            compile_enabled = Tts._model_params.get("fish_s2_compile_enabled", True)
            s = f"{device}"
            if device == "cuda":
                s += f", compile: {compile_enabled}"
            print_model_init(s)
            from tts_audiobook_tool.tts_model.fish_s2_model import FishS2Model
            Tts._fish_s2 = FishS2Model(device, compile_enabled)
            printt()
        return Tts._fish_s2

    @staticmethod
    def get_higgs() -> HiggsBaseModel:
        if not Tts._higgs:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(device)
            from tts_audiobook_tool.tts_model.higgs_model import HiggsModel
            Tts._higgs = HiggsModel(device)
            printt()
        return Tts._higgs

    @staticmethod
    def get_vibevoice() -> VibeVoiceBaseModel:

        if not Tts._vibevoice:

            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()

            target = Tts._model_params.get("vibevoice_target", "") or VibeVoiceBaseModel.DEFAULT_REPO_ID
            trunc_target = ellipsize_path_middle(target)
            lora_path = Tts._model_params.get("vibevoice_lora_path", "")
            lora_desc = f"LoRA: {lora_path}, " if lora_path else ""
            desc = f"{lora_desc}{device}"
            print_model_init(desc, trunc_target)

            from tts_audiobook_tool.tts_model.vibe_voice_model import VibeVoiceModel
            Tts._vibevoice = VibeVoiceModel(
                device_map=device,
                model_target=target,
                lora_path=lora_path,
                max_new_tokens=VibeVoiceBaseModel.MAX_TOKENS
            )
            printt()

        return Tts._vibevoice

    @staticmethod
    def get_indextts2() -> IndexTts2BaseModel:
        if not Tts._indextts2:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            use_fp16 = Tts._model_params.get("indextts2_use_fp16", False)
            print_model_init(f"{device}, fp16: {use_fp16}")

            from tts_audiobook_tool.tts_model.indextts2_model import IndexTts2Model
            Tts._indextts2 = IndexTts2Model(use_fp16=use_fp16) # model will use cuda if available
            printt()
        return Tts._indextts2

    @staticmethod
    def get_glm() -> GlmBaseModel:
        if not Tts._glm:
            device = "cuda" # cpu not currently supported
            sr = Tts._model_params["glm_sr"]
            print_model_init(f"{device}, {sr}hz")

            from tts_audiobook_tool.tts_model.glm_model import GlmModel
            Tts._glm = GlmModel(device, sr)
            printt()
        return Tts._glm

    @staticmethod
    def get_mira() -> MiraBaseModel:
        if not Tts._mira:
            print_model_init(f"cuda")
            from tts_audiobook_tool.tts_model.mira_model import MiraModel
            Tts._mira = MiraModel()
            printt()
        return Tts._mira

    @staticmethod
    def get_qwen3() -> Qwen3BaseModel:
        
        if not Tts._qwen3:

            target = Tts._model_params["qwen3_target"] or Qwen3BaseModel.DEFAULT_REPO_ID            
            s = ellipsize_path_middle(target)
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(f"{s}, {device}")

            looks_like_path = os.path.isabs(target) or target.startswith(("./", "../")) or "\\" in target
            if looks_like_path and not os.path.exists(target):
                raise ValueError(f"Qwen3 model path not found: '{target}'")

            from tts_audiobook_tool.tts_model.qwen3_model import Qwen3Model
            try:
                Tts._qwen3 = Qwen3Model(target, device)
            except Exception as e:
                Tts._qwen3 = None
                raise RuntimeError(f"Failed to load Qwen3 model from '{target}': {e}") from e
            printt()

        return Tts._qwen3

    @staticmethod
    def get_pocket() -> PocketBaseModel:
        if not Tts._pocket:
            language = Tts._model_params.get("pocket_model_code", "")
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            print_model_init(device)
            from tts_audiobook_tool.tts_model.pocket_model import PocketModel
            Tts._pocket = PocketModel(device=device, language=language)
            printt()
        return Tts._pocket

    @staticmethod
    def clear_tts_model() -> None:
        model = Tts.get_instance_if_exists()
        if model:
            model.kill()

            Tts._oute = None
            Tts._chatterbox = None
            Tts._fish = None
            Tts._fish_s2 = None
            Tts._higgs = None
            Tts._vibevoice = None
            Tts._indextts2 = None
            Tts._glm = None
            Tts._mira = None
            Tts._qwen3 = None
            Tts._pocket = None

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
