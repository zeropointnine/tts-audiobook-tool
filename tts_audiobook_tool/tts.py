from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from importlib import util
import os
from typing import Callable

from tts_audiobook_tool.app_types import StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.app_types.phrase import Reason

from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
from tts_audiobook_tool.tts_models.fish_s1_base_model import FishS1BaseModel
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.fish_s2_server_base_model import FishS2ServerBaseModel
from tts_audiobook_tool.tts_models.higgs_v3_server_base_model import HiggsV3ServerBaseModel
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.higgs_v2_base_model import HiggsV2BaseModel
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_models.mira_base_model import MiraBaseModel
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.moss_server_base_model import MossServerBaseModel
from tts_audiobook_tool.tts_models.moss_server_model import MossServerModel
from tts_audiobook_tool.tts_models.none_base_model import NoneBaseModel
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_models.oute_base_model import OuteBaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.qwen3_server_base_model import Qwen3ServerBaseModel
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelSpec, TtsModelType
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model.

    The model type is derived from the state of the virtual environment 
    and remains unchanged during the app's runtime.
    """

    _type: TtsModelType

    _chatterbox: ChatterboxBaseModel | None = None
    _fish_s1: FishS1BaseModel | None = None
    _fish_s2: FishS2BaseModel | None = None
    _fish_s2_server: FishS2ServerBaseModel | None = None
    _glm: GlmBaseModel | None = None
    _higgs_v2: HiggsV2BaseModel | None = None
    _higgs_v3: HiggsV3ServerBaseModel | None = None
    _indextts2: IndexTts2BaseModel | None = None
    _mira: MiraBaseModel | None = None
    _moss: MossBaseModel | None = None
    _moss_server: MossServerBaseModel | None = None
    _omnivoice: OmniVoiceBaseModel | None = None
    _oute: OuteBaseModel | None = None
    _pocket: PocketBaseModel | None = None
    _qwen3: Qwen3BaseModel | None = None
    _qwen3tts_server: Qwen3ServerBaseModel | None = None
    _vibevoice: VibeVoiceBaseModel | None = None

    _sgl_omni_type: TtsModelType | None = None

    # Salient details of the current instance for display
    _instance_display_info: InstanceDisplayInfo | None = None

    _model_params: dict = {}
    _force_cpu: bool = False

    @staticmethod
    def init_local_model_type() -> tuple[TtsModelType, int]:
        """
        Sets the tts model type by checking the installed modules in the python environment.
        Does not instantiate the TtsModel as such.
        Must be run on startup.

        Returns the model type that was set, and num matches (should be either 1 or 0).
        """

        def get_matches() -> list[TtsModelType]:
            model_infos = []
            for model_info in TtsModelType:
                exists = False
                try:
                    module_test = model_info.value.local_module_test
                    if model_info.value.is_sgl_omni or not module_test:
                        continue
                    if module_test.startswith("dist:"):
                        dist_test = module_test.removeprefix("dist:").strip()
                        if "==" in dist_test:
                            dist_name, expected_version = [part.strip() for part in dist_test.split("==", 1)]
                            exists = metadata.version(dist_name) == expected_version
                        else:
                            metadata.version(dist_test)
                            exists = True
                    else:
                        exists = util.find_spec(module_test) is not None
                except:
                    ...
                if exists:
                    model_infos.append(model_info)
            return model_infos
        
        matches = get_matches()
        
        match len(matches):
            case 0:
                # No match
                Tts._type = TtsModelType.NONE
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
    def get_type() -> TtsModelType:
        return Tts._type

    @staticmethod
    def set_type(value: TtsModelType) -> None:
        if Tts._type != value:
            Tts.clear_tts_model()
        Tts._type = value

    @staticmethod
    def set_sgl_omni_type(value: TtsModelType | None) -> None:
        if value is not None and (value == TtsModelType.NONE or not value.value.is_sgl_omni):
            value = None
        Tts._sgl_omni_type = value
        Tts.update_tts_type()
   
    @staticmethod
    def is_local_model() -> bool:
        return Tts._type != TtsModelType.NONE and not Tts._type.value.is_sgl_omni

    @staticmethod
    def is_sgl_mode() -> bool:
        """
        Internal nomenclature
        TODO: Ideally, this should be reflected by a formal construction of some kind, but yea
        """
        return not Tts.is_local_model()

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
        model_params["moss_target"] = project.moss_target
        model_params["qwen3_target"] = project.qwen3_target
        model_params["fish_s1_compile_enabled"] = project.fish_s1_compile_enabled
        model_params["fish_s2_compile_enabled"] = project.fish_s2_compile_enabled
        model_params["pocket_model_code"] = project.pocket_model_code
        model_params["omnivoice_target"] = project.omnivoice_target

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
        dirty |= new_params.get("moss_target", "") != old_params.get("moss_target", "")
        dirty |= new_params.get("qwen3_target", "") != old_params.get("qwen3_target", "")
        dirty |= new_params.get("fish_s1_compile_enabled", False) != old_params.get("fish_s1_compile_enabled", False)
        dirty |= new_params.get("fish_s2_compile_enabled", False) != old_params.get("fish_s2_compile_enabled", False)
        dirty |= new_params.get("pocket_model_code", "") != old_params.get("pocket_model_code", "")
        dirty |= new_params.get("omnivoice_target", "") != old_params.get("omnivoice_target", "")
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
            TtsModelType.NONE: NoneBaseModel,
            TtsModelType.CHATTERBOX: ChatterboxBaseModel,
            TtsModelType.FISH_S1: FishS1BaseModel,
            TtsModelType.FISH_S2: FishS2BaseModel,
            TtsModelType.FISH_S2_SERVER: FishS2ServerBaseModel,
            TtsModelType.GLM: GlmBaseModel,
            TtsModelType.HIGGS_V2: HiggsV2BaseModel,
            TtsModelType.HIGGS_V3_SERVER: HiggsV3ServerBaseModel,
            TtsModelType.INDEXTTS2: IndexTts2BaseModel,
            TtsModelType.MIRA: MiraBaseModel,
            TtsModelType.MOSS: MossBaseModel,
            TtsModelType.MOSS_SERVER: MossServerModel,
            TtsModelType.OMNIVOICE: OmniVoiceBaseModel,
            TtsModelType.OUTE: OuteBaseModel,
            TtsModelType.POCKET: PocketBaseModel,
            TtsModelType.QWEN3TTS: Qwen3BaseModel,
            TtsModelType.QWEN3TTS_SERVER: Qwen3ServerBaseModel,
            TtsModelType.VIBEVOICE: VibeVoiceBaseModel,            
        }
        cls = MAP.get(Tts._type, None)
        if cls is None:
            raise Exception(f"Not implemented: {Tts._type}")
        return cls
    
    @staticmethod
    def get_info() -> TtsModelSpec:
        return Tts.get_class().INFO

    @staticmethod
    def instance_exists() -> bool:
        items = [
            Tts._chatterbox,
            Tts._fish_s1,
            Tts._fish_s2,
            Tts._fish_s2_server,
            Tts._glm,
            Tts._higgs_v2,
            Tts._higgs_v3,
            Tts._indextts2,
            Tts._mira,
            Tts._moss,
            Tts._moss_server,
            Tts._omnivoice,
            Tts._oute,
            Tts._pocket,
            Tts._qwen3,
            Tts._qwen3tts_server,
            Tts._vibevoice,
        ]
        for item in items:
            if item is not None:
                return True
        return False

    @staticmethod
    def get_instance() -> TtsBaseModel:
        # Returns existing or newly instantiated instance
        MAP: dict[TtsModelType, Callable] = {
            TtsModelType.CHATTERBOX: Tts.get_chatterbox,
            TtsModelType.FISH_S1: Tts.get_fish_s1,
            TtsModelType.FISH_S2: Tts.get_fish_s2,
            TtsModelType.FISH_S2_SERVER: Tts.get_fish_s2_server,
            TtsModelType.GLM: Tts.get_glm,
            TtsModelType.HIGGS_V2: Tts.get_higgs,
            TtsModelType.HIGGS_V3_SERVER: Tts.get_higgs_v3,
            TtsModelType.INDEXTTS2: Tts.get_indextts2,
            TtsModelType.MIRA: Tts.get_mira,
            TtsModelType.MOSS: Tts.get_moss,
            TtsModelType.MOSS_SERVER: Tts.get_moss_server,
            TtsModelType.OMNIVOICE: Tts.get_omnivoice,
            TtsModelType.OUTE: Tts.get_oute,
            TtsModelType.POCKET: Tts.get_pocket,
            TtsModelType.QWEN3TTS: Tts.get_qwen3,
            TtsModelType.QWEN3TTS_SERVER: Tts.get_qwen3tts_server,
            TtsModelType.VIBEVOICE: Tts.get_vibevoice
        }
        factory_function = MAP.get(Tts._type, None)
        if not factory_function:
            raise Exception(f"Lookup failed for {Tts._type}")
        instance = factory_function()
        return instance

    @staticmethod
    def generate_using_project(
            project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            print_generation_request: bool = False,
    ):
        """
        All app-level TTS generation goes through this function.

        Applies the standard project/model text-preparation pipeline to each
        prompt exactly once, then delegates to the active concrete model's own
        `generate_using_project()` implementation.

        This keeps audiobook generation, realtime playback, server/API usage,
        and LLM chat consistent wrt prompt normalization and model-specific
        transforms such as VibeVoice speaker tagging.
        """
        instance = Tts.get_instance()
        L.i(
            f"Tts.generate_using_project dispatch: type={Tts._type.value.id} "
            f"instance={type(instance).__name__} prompts={len(prompts)} "
            f"has_on_stream_chunk={on_stream_chunk is not None} has_on_stream_end={on_stream_end is not None}"
        )
        prepared_prompts = [instance.prepare_text_for_inference(project, prompt) for prompt in prompts]
        kwargs = {
            "on_stream_chunk": on_stream_chunk,
            "on_stream_end": on_stream_end if on_stream_end is not None else project.on_stream_end,
        }
        if Tts._type.value.is_sgl_omni: # xxx verify
            kwargs["print_generation_request"] = print_generation_request

        return instance.generate_using_project(
            project,
            prepared_prompts,
            force_random_seed,
            **kwargs,
        )

    @staticmethod
    def clear_continuation() -> None:
        instance = Tts.get_instance_if_exists()
        if instance is not None:
            instance.clear_continuation()

    @staticmethod
    def clear_continuation_if_reason(reason: Reason) -> None:
        if reason in { Reason.PARAGRAPH, Reason.SPACE_BREAK, Reason.SECTION_BREAK }:
            Tts.clear_continuation()

    @staticmethod
    def get_instance_if_exists() -> TtsBaseModel | None:
        # Returns instance only if it already exists, else none
        MAP = {
            TtsModelType.OUTE: Tts._oute,
            TtsModelType.CHATTERBOX: Tts._chatterbox,
            TtsModelType.FISH_S1: Tts._fish_s1,
            TtsModelType.FISH_S2: Tts._fish_s2,
            TtsModelType.FISH_S2_SERVER: Tts._fish_s2,
            TtsModelType.HIGGS_V2: Tts._higgs_v2,
            TtsModelType.HIGGS_V3_SERVER: Tts._higgs_v3,
            TtsModelType.VIBEVOICE: Tts._vibevoice,
            TtsModelType.INDEXTTS2: Tts._indextts2,
            TtsModelType.GLM: Tts._glm,
            TtsModelType.MIRA: Tts._mira,
            TtsModelType.MOSS: Tts._moss,
            TtsModelType.MOSS_SERVER: Tts._moss_server,
            TtsModelType.QWEN3TTS: Tts._qwen3,
            TtsModelType.QWEN3TTS_SERVER: Tts._qwen3tts_server,
            TtsModelType.POCKET: Tts._pocket,
            TtsModelType.OMNIVOICE: Tts._omnivoice,
        }
        return MAP.get(Tts._type, None)

    @staticmethod
    def get_chatterbox() -> ChatterboxBaseModel:
        if not Tts._chatterbox:
            model_type = Tts._model_params.get("chatterbox_type")
            assert isinstance(model_type, ChatterboxType), "chatterbox_type not set"
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            Tts._set_instance_info(InstanceDisplayInfo(model_type.label, device))
            
            from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(model_type, device)
            printt()
        return Tts._chatterbox

    @staticmethod
    def get_fish_s1() -> FishS1BaseModel:
        if not Tts._fish_s1:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()

            if device == "cuda":
                compile_enabled = Tts._model_params.get("fish_compile_enabled", True) # TODO: needs to be hooked up
            else:
                compile_enabled = False
            
            if device == "cuda":
                extra = f"compile: {compile_enabled}"
            else:
                extra = ""
            
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"], device, extra))
            
            from tts_audiobook_tool.tts_models.fish_s1_model import FishS1Model
            Tts._fish_s1 = FishS1Model(device, compile_enabled)
            printt()

        return Tts._fish_s1

    @staticmethod
    def get_fish_s2() -> FishS2BaseModel:
        if not Tts._fish_s2:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()

            if device == "cuda":
                compile_enabled = Tts._model_params.get("fish_s2_compile_enabled", True)
            else:
                compile_enabled = False
            
            if device == "cuda":
                extra = f"compile: {compile_enabled}"
            else:
                extra = ""
            
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"], device, extra))
            
            from tts_audiobook_tool.tts_models.fish_s2_model import FishS2Model
            Tts._fish_s2 = FishS2Model(device, compile_enabled)
            printt()

        return Tts._fish_s2

    @staticmethod
    def get_fish_s2_server() -> FishS2ServerBaseModel:
        if not Tts._fish_s2_server:
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"]), and_print=False)
            from tts_audiobook_tool.tts_models.fish_s2_server_model import FishS2ServerModel
            Tts._fish_s2_server = FishS2ServerModel()
            printt()
        return Tts._fish_s2_server

    @staticmethod
    def get_glm() -> GlmBaseModel:
        if not Tts._glm:
            device = "cuda" # cpu not currently supported
            sr = Tts._model_params["glm_sr"]
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"], device, f"{sr}hz"))

            from tts_audiobook_tool.tts_models.glm_model import GlmModel
            Tts._glm = GlmModel(device, sr)
            printt()
        return Tts._glm

    @staticmethod
    def get_higgs() -> HiggsV2BaseModel:
        if not Tts._higgs_v2:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"], device))
            
            from tts_audiobook_tool.tts_models.higgs_v2_model import HiggsV2Model
            Tts._higgs_v2 = HiggsV2Model(device)
            printt()

        return Tts._higgs_v2

    @staticmethod
    def get_higgs_v3() -> HiggsV3ServerBaseModel:
        if not Tts._higgs_v3:
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"]), and_print=False)
            from tts_audiobook_tool.tts_models.higgs_v3_server_model import HiggsV3ServerModel
            Tts._higgs_v3 = HiggsV3ServerModel()
            printt()
        return Tts._higgs_v3

    @staticmethod
    def get_indextts2() -> IndexTts2BaseModel:
        if not Tts._indextts2:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            use_fp16 = Tts._model_params.get("indextts2_use_fp16", False)
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"], device, f"fp16: {use_fp16}"))

            from tts_audiobook_tool.tts_models.indextts2_model import IndexTts2Model
            Tts._indextts2 = IndexTts2Model(use_fp16=use_fp16) # model will use cuda if available
            printt()
        return Tts._indextts2

    @staticmethod
    def get_mira() -> MiraBaseModel:
        if not Tts._mira:
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"], "cuda"))
            
            from tts_audiobook_tool.tts_models.mira_model import MiraModel
            Tts._mira = MiraModel()
            printt()
        return Tts._mira

    @staticmethod
    def get_moss() -> MossBaseModel:
        if not Tts._moss:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            target = Tts._model_params.get("moss_target", "") or MossConfigs.get_default_repo_id()
            target_string = target.removeprefix("OpenMOSS-Team/")
            target_string = ellipsize_path_for_menu(target_string)

            looks_like_path = os.path.isabs(target) or target.startswith(("./", "../")) or "\\" in target
            if looks_like_path and not os.path.exists(target):
                raise ValueError(f"MOSS model path not found: '{target}'")

            Tts._set_instance_info(InstanceDisplayInfo(target_string, device))

            from tts_audiobook_tool.tts_models.moss_model import MossModel
            Tts._moss = MossModel(device=device, model_target=target)
            printt()
        return Tts._moss

    @staticmethod
    def get_moss_server() -> MossServerBaseModel:
        if not Tts._moss_server:
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"]), and_print=False)
            from tts_audiobook_tool.tts_models.moss_server_model import MossServerModel
            Tts._moss_server = MossServerModel()
            printt()
        return Tts._moss_server

    @staticmethod
    def get_omnivoice() -> OmniVoiceBaseModel:
        if not Tts._omnivoice:
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            model_target = Tts._model_params.get("omnivoice_target", "") \
                        or OmniVoiceBaseModel.DEFAULT_REPO_ID
            short_name = Tts.get_type().value.ui["short_name"]

            target_string = model_target
            target_string = target_string.removeprefix("k2-fsa/")
            target_string = ellipsize_path_for_menu(target_string)

            if model_target == OmniVoiceBaseModel.DEFAULT_REPO_ID:
                model_description = short_name
            else:
                model_description = f"{short_name} {COL_DIM}({target_string}){COL_DEFAULT}"

            Tts._set_instance_info(InstanceDisplayInfo(model_description, device))

            from tts_audiobook_tool.tts_models.omnivoice_model import OmniVoiceModel
            Tts._omnivoice = OmniVoiceModel(
                device=device,
                model_target=model_target,
            )
            printt()
        return Tts._omnivoice

    @staticmethod
    def get_oute() -> OuteBaseModel:
        if not Tts._oute:
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"]))
            from tts_audiobook_tool.tts_models.oute_model import OuteModel
            Tts._oute = OuteModel()
            printt()
        return Tts._oute

    @staticmethod
    def get_pocket() -> PocketBaseModel:
        if not Tts._pocket:
            language = Tts._model_params.get("pocket_model_code", "")
            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()
            Tts._set_instance_info(InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"], device))
            
            from tts_audiobook_tool.tts_models.pocket_model import PocketModel
            Tts._pocket = PocketModel(device=device, language=language)
            printt()
        return Tts._pocket
    
    @staticmethod
    def get_qwen3() -> Qwen3BaseModel:
        
        if not Tts._qwen3:

            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()

            target = Tts._model_params["qwen3_target"] or Qwen3BaseModel.DEFAULT_REPO_ID            

            target_string = target
            target_string = target_string.removeprefix("Qwen/")
            target_string = ellipsize_path_for_menu(target_string)

            looks_like_path = os.path.isabs(target) or target.startswith(("./", "../")) or "\\" in target
            if looks_like_path and not os.path.exists(target):
                raise ValueError(f"Qwen3 model path not found: '{target}'")

            Tts._set_instance_info(InstanceDisplayInfo(target_string, device))

            from tts_audiobook_tool.tts_models.qwen3_model import Qwen3Model
            try:
                Tts._qwen3 = Qwen3Model(target, device)
            except Exception as e:
                Tts._qwen3 = None
                raise RuntimeError(f"Failed to load Qwen3 model from '{target}': {e}") from e
            printt()

        return Tts._qwen3

    @staticmethod
    def get_qwen3tts_server() -> Qwen3ServerBaseModel:
        if not Tts._qwen3tts_server:
            info = InstanceDisplayInfo(Tts.get_type().value.ui["proper_name"])
            Tts._set_instance_info(info, and_print=False)
            from tts_audiobook_tool.tts_models.qwen3_server_model import Qwen3ServerModel
            Tts._qwen3tts_server = Qwen3ServerModel()
            printt()
        return Tts._qwen3tts_server

    @staticmethod
    def get_vibevoice() -> VibeVoiceBaseModel:

        if not Tts._vibevoice:

            device = "cpu" if Tts._force_cpu else Tts.get_resolved_torch_device()

            target = Tts._model_params.get("vibevoice_target", "") or VibeVoiceBaseModel.DEFAULT_REPO_ID
            
            target_string = target
            target_string = target_string.removeprefix("microsoft/")
            target_string = target_string.removeprefix("vibevoice/")
            target_string = ellipsize_path_for_menu(target_string)
            
            lora_path = Tts._model_params.get("vibevoice_lora_path", "")
            extra = "with lora" if lora_path else ""

            Tts._set_instance_info(InstanceDisplayInfo(target_string, device, extra))

            from tts_audiobook_tool.tts_models.vibe_voice_model import VibeVoiceModel
            Tts._vibevoice = VibeVoiceModel(
                device_map=device,
                model_target=target,
                lora_path=lora_path,
                max_new_tokens=VibeVoiceBaseModel.MAX_TOKENS
            )
            printt()

        return Tts._vibevoice

    @staticmethod
    def clear_tts_model() -> None:
        model = Tts.get_instance_if_exists()
        if model:
            model.kill()
            Tts._oute = None
            Tts._chatterbox = None
            Tts._fish_s1 = None
            Tts._fish_s2 = None
            Tts._higgs_v2 = None
            Tts._higgs_v3 = None
            Tts._vibevoice = None
            Tts._indextts2 = None
            Tts._glm = None
            Tts._mira = None
            Tts._moss = None
            Tts._qwen3 = None
            Tts._qwen3tts_server = None
            Tts._pocket = None
            Tts._omnivoice = None
            Tts._instance_display_info = None
        app_memory.gc_ram_vram()

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
        supported_devices = Tts.get_type().value.local_torch_devices
        intersection = [item for item in available_devices if item in supported_devices]
        return intersection[0] if intersection else ""

    @staticmethod
    def _set_instance_info(info: InstanceDisplayInfo, and_print: bool=True) -> None:
        # TODO: refactor yek
        
        Tts._instance_display_info = info

        if not info.device and not info.extra:
            extra = ""
        elif info.device and info.extra:
            extra = f"{info.device}, {info.extra}"
        elif info.device:
            extra = info.device
        else: # extra
            extra = info.extra        
        if and_print:
            print_model_init(model_description=info.model_description, extra=extra)

    @staticmethod
    def update_tts_type() -> None:
        """ 
        Applies only when tts type is NONE or is_sgl_omni

        Dynamically updates tts type based on SGL Omni model name,
        and also updates _sgl_model_id.
        """

        if Tts.is_local_model():
            return
        
        original_type = Tts.get_type()

        if not SglOmniUtil.get_base_url() and Tts.get_type() != TtsModelType.NONE:
            Tts.set_type(TtsModelType.NONE)
            # print(f"xxx changed tts type from {original_type} to {Tts.get_type()}")
            return
        
        if Tts._sgl_omni_type is None:
            # Auto-detect
            SglOmniUtil.update_model_id()

            new_type = TtsModelType.find_tts_type_using_sgl_omni_model_id( SglOmniUtil.get_model_id() )
            if new_type is None:
                new_type = TtsModelType.NONE
        else:
            new_type = Tts._sgl_omni_type
        if new_type == original_type:
            return        
        Tts.set_type(new_type)
        # print(f"xxx changed tts type from {original_type} to {new_type}")


# ---

@dataclass
class InstanceDisplayInfo:
    """ Info about the instantiated TTS model used for UI """
    
    # Short descriptor of model instance; required
    # Could be name of model or something more specific, like the model's hf repo id
    model_description: str
    
    # Should usually be populated, depending on model
    device: str = ""

    # Extra info (eg, "fp16: True", etc)
    extra: str = ""
