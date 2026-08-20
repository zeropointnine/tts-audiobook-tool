from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from importlib import util
import os
import threading
from typing import Callable

from tts_audiobook_tool.app_types import DeviceType, StreamChunkCallback, StreamEndCallback
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
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelSpec, TtsModelType
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.tts_models.zonos2_server_base_model import Zonos2ServerBaseModel
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *

class Tts:
    """
    Static class for accessing the TTS model.

    The process-level backend mode (LOCAL or SGL_OMNI) is determined once
    at startup from the presence of the SGL-Omni sentinel package and is
    immutable for the life of the process.

    In local mode, the model type is derived from the state of the virtual
    environment and remains unchanged during the app's runtime. In
    SGL-Omni mode, the selected type can change at runtime (an explicit
    selection, or auto-detection from the server).
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
    _zonos2_server: Zonos2ServerBaseModel | None = None

    _sgl_omni_type: TtsModelType | None = None

    # Process-level backend mode (LOCAL or SGL_OMNI), probed once at
    # startup from the SGL-Omni sentinel package and immutable for the
    # life of the process
    _backend_mode: TtsBackendKind | None = None

    _model_params: dict = {}
    _force_cpu: bool = False
    _voice_auto_advance_counter: int = 0
    _voice_auto_advance_lock = threading.Lock()

    @staticmethod
    def get_next_voice_selection_index() -> int:
        with Tts._voice_auto_advance_lock:
            index = Tts._voice_auto_advance_counter
            Tts._voice_auto_advance_counter += 1
        return index

    @staticmethod
    def reset_voice_selection_index() -> None:
        with Tts._voice_auto_advance_lock:
            Tts._voice_auto_advance_counter = 0

    @staticmethod
    def get_voice_tag_for_selection_index(project, voice_selection_index: int) -> str:
        info = Tts.get_info()
        if not info.voice_target_attr:
            return Tts.get_class().get_voice_tag(project)

        from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
        voice_value = ProjectVoiceUtil.current_voice_value(project, Tts.get_type(), voice_selection_index)
        if not voice_value:
            return Tts.get_class().get_voice_tag(project)
        return Tts.get_class().get_voice_tag_for_value(voice_value)

    @staticmethod
    def get_best_supported_device_type(model_type: TtsModelType) -> DeviceType:
        supported_devices = model_type.value.local_torch_devices
        if Tts._force_cpu:
            if DeviceType.CPU in supported_devices:
                return DeviceType.CPU
            raise RuntimeError(f"{model_type.value.ui.get('proper_name') or model_type.value.id} does not support CPU inference")

        available_devices = Tts.get_available_device_types()
        intersection = [item for item in available_devices if item in supported_devices]
        if not intersection:
            supported = ", ".join(item.value for item in supported_devices) or "none"
            raise RuntimeError(f"No supported torch device is available for {model_type.value.ui.get('proper_name') or model_type.value.id} ({supported})")
        return intersection[0]

    @staticmethod
    def init_local_model_type() -> tuple[TtsModelType, int]:
        """
        Sets the tts model type by checking the state of the current virtual environment.
        Does not instantiate the TtsModel as such.
        Must be run on startup.

        First probes the SGL-Omni sentinel package to fix the process-level
        backend mode (immutable for the life of the process):

        - SGL-Omni mode: the local model probe is skipped entirely, even in
          a dual-capable venv that also holds a local model library
          (SGL-Omni wins; such a venv is user error). The type starts as
          NONE and is resolved by prefs / update_tts_type().
        - Local mode: local model libraries are probed as before.

        Returns the model type that was set, and num matches (always 0 in
        SGL-Omni mode, 0 or 1 in local mode).
        """
        Tts._backend_mode = Tts._probe_backend_mode()

        if Tts._backend_mode == TtsBackendKind.SGL_OMNI:
            Tts._type = TtsModelType.NONE
            return Tts._type, 0

        def get_matches() -> list[TtsModelType]:
            model_infos = []
            for model_info in TtsModelType:
                exists = False
                try:
                    module_test = model_info.value.local_module_test
                    if not module_test:
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
        if not hasattr(Tts, "_type") or Tts._type is None:
            raise Exception("TTS model type has not been set. Must first call init_local_model_type().`")
        return Tts._type

    @staticmethod
    def get_backend_mode() -> TtsBackendKind:
        """
        Returns the process-level backend mode: LOCAL (TTS is run by
        model libraries inside the current venv) or SGL_OMNI (TTS is
        served by an external SGL-Omni server).

        The mode is determined once, from the presence of the SGL-Omni
        sentinel package, and is immutable for the life of the process.
        `init_local_model_type()` probes it eagerly at startup; this
        getter probes lazily if that has not happened yet.
        """
        if Tts._backend_mode is None:
            Tts._backend_mode = Tts._probe_backend_mode()
        return Tts._backend_mode

    @staticmethod
    def _probe_backend_mode() -> TtsBackendKind:
        """
        Probes the SGL-Omni sentinel package
        (`tts_audiobook_tool_sgl_omni_marker`, installed only by
        requirements-sgl-omni.txt) to determine the backend mode.
        A missing (or unreadable) sentinel means local mode.
        """
        try:
            present = util.find_spec("tts_audiobook_tool_sgl_omni_marker") is not None
        except Exception:
            present = False
        return TtsBackendKind.SGL_OMNI if present else TtsBackendKind.LOCAL

    @staticmethod
    def set_type(value: TtsModelType) -> None:
        if Tts._type != value:
            Tts.clear_tts_model()
        Tts._type = value

    @staticmethod
    def set_sgl_omni_type(value: TtsModelType | None) -> None:
        """
        Sets the explicitly selected SGL-Omni TTS type (None = auto-detect).

        In local backend mode the stored value is inert: SGL-Omni is not
        active there, so no type resolution happens. The value still
        persists in prefs, so a later switch to a SGL-Omni venv picks it
        up.
        """
        if value is not None and not TtsModelType.is_valid_sgl_omni_type(value):
            value = None
        Tts._sgl_omni_type = value
        if Tts.get_backend_mode() == TtsBackendKind.LOCAL:
            return
        Tts.update_tts_type()

    @staticmethod
    def is_local_model() -> bool:
        return Tts._type != TtsModelType.NONE and Tts._type.value.backend_kind == TtsBackendKind.LOCAL

    @staticmethod
    def is_sgl_mode() -> bool:
        """
        Whether the process is running in SGL-Omni backend mode, i.e.
        whether the SGL-Omni sentinel package is present in the current
        venv.

        The mode is fixed at startup (see init_local_model_type()) and is
        immutable for the life of the process; it does not depend on
        which catalog member is currently selected.
        """
        return Tts.get_backend_mode() == TtsBackendKind.SGL_OMNI

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
        entry = Tts._model_registry_entry(Tts._type)
        if entry is None or entry[0] is None:
            raise Exception(f"Not implemented: {Tts._type}")
        return entry[0]
    
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
            Tts._zonos2_server,
        ]
        for item in items:
            if item is not None:
                return True
        return False

    @staticmethod
    def get_instance() -> TtsBaseModel:
        # Returns existing or newly instantiated instance
        entry = Tts._model_registry_entry(Tts._type)
        if entry is None or entry[1] is None:
            raise Exception(f"Lookup failed for {Tts._type}")
        return entry[1]()

    @staticmethod
    def generate_using_project(
            project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            print_generation_request: bool = False,
            voice_selection_index: int | None = None,
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
        if voice_selection_index is None:
            voice_selection_index = Tts.get_next_voice_selection_index()
        L.i(
            f"Tts.generate_using_project dispatch: type={Tts._type.value.id} "
            f"instance={type(instance).__name__} prompts={len(prompts)} "
            f"voice_selection_index={voice_selection_index} "
            f"has_on_stream_chunk={on_stream_chunk is not None} has_on_stream_end={on_stream_end is not None}"
        )
        prepared_prompts = [instance.prepare_text_for_inference(project, prompt) for prompt in prompts]
        kwargs = {
            "on_stream_chunk": on_stream_chunk,
            "on_stream_end": on_stream_end if on_stream_end is not None else project.on_stream_end,
            "voice_selection_index": voice_selection_index,
        }
        if Tts._type.value.backend_kind == TtsBackendKind.SGL_OMNI:
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
    def _model_registry_entry(tts_type: TtsModelType) -> tuple[type[TtsBaseModel], Callable[[], TtsBaseModel] | None, str] | None:
        """
        Shared lookup backing get_class() / get_instance() /
        get_instance_if_exists(). The NONE placeholder maps to
        (NoneBaseModel, no factory, no instance attribute); an unknown
        value (impossible for catalog members) maps to nothing at all.
        """
        if tts_type == TtsModelType.NONE:
            return NoneBaseModel, None, ""
        return Tts._MODEL_REGISTRY.get(tts_type)

    @staticmethod
    def get_instance_if_exists() -> TtsBaseModel | None:
        # Returns instance only if it already exists, else none
        entry = Tts._model_registry_entry(Tts._type)
        if entry is None or not entry[2]:
            return None
        return getattr(Tts, entry[2])

    @staticmethod
    def get_chatterbox() -> ChatterboxBaseModel:
        if not Tts._chatterbox:
            model_type = Tts._model_params.get("chatterbox_type")
            assert isinstance(model_type, ChatterboxType), "chatterbox_type not set"
            device_type = Tts.get_best_supported_device_type(TtsModelType.CHATTERBOX)
            
            from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(model_type, device_type)
            printt()
        return Tts._chatterbox

    @staticmethod
    def get_fish_s1() -> FishS1BaseModel:
        if not Tts._fish_s1:
            device_type = Tts.get_best_supported_device_type(TtsModelType.FISH_S1)

            if device_type == DeviceType.CUDA:
                compile_enabled = Tts._model_params.get("fish_compile_enabled", True) # TODO: needs to be hooked up
            else:
                compile_enabled = False
            
            if device_type == DeviceType.CUDA:
                extra = f"compile: {compile_enabled}"
            else:
                extra = ""
            
            from tts_audiobook_tool.tts_models.fish_s1_model import FishS1Model
            Tts._fish_s1 = FishS1Model(device_type, compile_enabled)
            printt()

        return Tts._fish_s1

    @staticmethod
    def get_fish_s2() -> FishS2BaseModel:
        if not Tts._fish_s2:
            device_type = Tts.get_best_supported_device_type(TtsModelType.FISH_S2)

            if device_type == DeviceType.CUDA:
                compile_enabled = Tts._model_params.get("fish_s2_compile_enabled", True)
            else:
                compile_enabled = False
            
            if device_type == DeviceType.CUDA:
                extra = f"compile: {compile_enabled}"
            else:
                extra = ""
            
            from tts_audiobook_tool.tts_models.fish_s2_model import FishS2Model
            Tts._fish_s2 = FishS2Model(device_type, compile_enabled)
            printt()

        return Tts._fish_s2

    @staticmethod
    def get_fish_s2_server() -> FishS2ServerBaseModel:
        if not Tts._fish_s2_server:
            from tts_audiobook_tool.tts_models.fish_s2_server_model import FishS2ServerModel
            Tts._fish_s2_server = FishS2ServerModel()
            printt()
        return Tts._fish_s2_server

    @staticmethod
    def get_glm() -> GlmBaseModel:
        if not Tts._glm:
            device_type = Tts.get_best_supported_device_type(TtsModelType.GLM)
            sr = Tts._model_params["glm_sr"]

            from tts_audiobook_tool.tts_models.glm_model import GlmModel
            Tts._glm = GlmModel(device_type, sr)
            printt()
        return Tts._glm

    @staticmethod
    def get_higgs() -> HiggsV2BaseModel:
        if not Tts._higgs_v2:
            device_type = Tts.get_best_supported_device_type(TtsModelType.HIGGS_V2)
            from tts_audiobook_tool.tts_models.higgs_v2_model import HiggsV2Model
            Tts._higgs_v2 = HiggsV2Model(device_type)
            printt()

        return Tts._higgs_v2

    @staticmethod
    def get_higgs_v3() -> HiggsV3ServerBaseModel:
        if not Tts._higgs_v3:
            from tts_audiobook_tool.tts_models.higgs_v3_server_model import HiggsV3ServerModel
            Tts._higgs_v3 = HiggsV3ServerModel()
            printt()
        return Tts._higgs_v3

    @staticmethod
    def get_indextts2() -> IndexTts2BaseModel:
        if not Tts._indextts2:
            use_fp16 = Tts._model_params.get("indextts2_use_fp16", False)
            from tts_audiobook_tool.tts_models.indextts2_model import IndexTts2Model
            Tts._indextts2 = IndexTts2Model(use_fp16=use_fp16) # model will use cuda if available
            printt()
        return Tts._indextts2

    @staticmethod
    def get_mira() -> MiraBaseModel:
        if not Tts._mira:
            from tts_audiobook_tool.tts_models.mira_model import MiraModel
            Tts._mira = MiraModel()
            printt()
        return Tts._mira

    @staticmethod
    def get_moss() -> MossBaseModel:
        if not Tts._moss:
            device_type = Tts.get_best_supported_device_type(TtsModelType.MOSS)
            target = Tts._model_params.get("moss_target", "") or MossConfigs.get_default_repo_id()

            looks_like_path = os.path.isabs(target) or target.startswith(("./", "../")) or "\\" in target
            if looks_like_path and not os.path.exists(target):
                raise ValueError(f"MOSS model path not found: '{target}'")

            from tts_audiobook_tool.tts_models.moss_model import MossModel
            Tts._moss = MossModel(device=device_type, model_target=target)
            printt()
        return Tts._moss

    @staticmethod
    def get_moss_server() -> MossServerBaseModel:
        if not Tts._moss_server:
            from tts_audiobook_tool.tts_models.moss_server_model import MossServerModel
            Tts._moss_server = MossServerModel()
            printt()
        return Tts._moss_server

    @staticmethod
    def get_omnivoice() -> OmniVoiceBaseModel:
        if not Tts._omnivoice:
            device_type = Tts.get_best_supported_device_type(TtsModelType.OMNIVOICE)
            model_target = Tts._model_params.get("omnivoice_target", "") \
                        or OmniVoiceBaseModel.DEFAULT_REPO_ID
            from tts_audiobook_tool.tts_models.omnivoice_model import OmniVoiceModel
            Tts._omnivoice = OmniVoiceModel(device=device_type, model_target=model_target)
            printt()
        return Tts._omnivoice

    @staticmethod
    def get_oute() -> OuteBaseModel:
        if not Tts._oute:
            from tts_audiobook_tool.tts_models.oute_model import OuteModel
            Tts._oute = OuteModel()
            printt()
        return Tts._oute

    @staticmethod
    def get_pocket() -> PocketBaseModel:
        if not Tts._pocket:
            language = Tts._model_params.get("pocket_model_code", "")
            device_type = Tts.get_best_supported_device_type(TtsModelType.POCKET)
            from tts_audiobook_tool.tts_models.pocket_model import PocketModel
            Tts._pocket = PocketModel(device=device_type, language=language)
            printt()
        return Tts._pocket
    
    @staticmethod
    def get_qwen3() -> Qwen3BaseModel:
        
        if not Tts._qwen3:

            device_type = Tts.get_best_supported_device_type(TtsModelType.QWEN3TTS)
            target = Tts._model_params["qwen3_target"] or Qwen3BaseModel.DEFAULT_REPO_ID            

            looks_like_path = os.path.isabs(target) or target.startswith(("./", "../")) or "\\" in target
            if looks_like_path and not os.path.exists(target):
                raise ValueError(f"Qwen3 model path not found: '{target}'")

            from tts_audiobook_tool.tts_models.qwen3_model import Qwen3Model
            try:
                Tts._qwen3 = Qwen3Model(target, device_type)
            except Exception as e:
                Tts._qwen3 = None
                raise RuntimeError(f"Failed to load Qwen3 model from '{target}': {e}") from e
            printt()

        return Tts._qwen3

    @staticmethod
    def get_qwen3tts_server() -> Qwen3ServerBaseModel:
        if not Tts._qwen3tts_server:
            from tts_audiobook_tool.tts_models.qwen3_server_model import Qwen3ServerModel
            Tts._qwen3tts_server = Qwen3ServerModel()
            printt()
        return Tts._qwen3tts_server

    @staticmethod
    def get_zonos2_server() -> Zonos2ServerBaseModel:
        if not Tts._zonos2_server:
            from tts_audiobook_tool.tts_models.zonos2_server_model import Zonos2ServerModel
            Tts._zonos2_server = Zonos2ServerModel()
            printt()
        return Tts._zonos2_server

    @staticmethod
    def get_vibevoice() -> VibeVoiceBaseModel:

        if not Tts._vibevoice:

            device_type = Tts.get_best_supported_device_type(TtsModelType.VIBEVOICE)
            target = Tts._model_params.get("vibevoice_target", "") or VibeVoiceBaseModel.DEFAULT_REPO_ID
            lora_path = Tts._model_params.get("vibevoice_lora_path", "") 
            
            from tts_audiobook_tool.tts_models.vibe_voice_model import VibeVoiceModel
            Tts._vibevoice = VibeVoiceModel(
                device=device_type,
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
            # Null out all instance attributes, not just the current
            # type's (there must be at most one live instance)
            for entry in Tts._MODEL_REGISTRY.values():
                setattr(Tts, entry[2], None)
        app_memory.gc_ram_vram()

    @staticmethod
    def get_available_device_types() -> list[DeviceType]:
        """Gets available torch device types in preferred inference order."""
        import torch
        available_devices: list[DeviceType] = []
        if torch.cuda.is_available():
            available_devices.append(DeviceType.CUDA)
        if torch.backends.mps.is_available():
            available_devices.append(DeviceType.MPS)
        available_devices.append(DeviceType.CPU)
        return available_devices

    @staticmethod
    def update_tts_type() -> None:
        """
        Applies only in SGL-Omni backend mode, and within that only when
        the current selection is the NONE placeholder or an SGL-Omni
        variant (i.e. not a local model).

        Dynamically updates tts type based on SGL Omni model name,
        and also updates the SGL-Omni model id state.
        """

        if Tts.get_backend_mode() == TtsBackendKind.LOCAL:
            return

        if Tts.is_local_model():
            return

        original_type = Tts.get_type()

        if not SglOmniUtil.get_base_url() and Tts.get_type() != TtsModelType.NONE:
            Tts.set_type(TtsModelType.NONE)
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

    @staticmethod
    def get_requirements_file_name() -> str:
        """
        The requirements file a user should install for the current TTS
        selection.

        Mode-aware for the NONE placeholder: in SGL-Omni mode it means
        "server not configured", so the placeholder's own (sgl-omni)
        requirements file applies; in local mode it means "no TTS model
        library in the venv", which corresponds to the base venv.
        """
        if Tts._type == TtsModelType.NONE:
            if Tts.get_backend_mode() == TtsBackendKind.SGL_OMNI:
                return TtsModelType.NONE.value.requirements_file_name
            return "requirements-base.txt"
        return Tts._type.value.requirements_file_name

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

# ---

# One entry per TTS variant, shared by Tts.get_class() / Tts.get_instance() /
# Tts.get_instance_if_exists() and used to clear instances:
#   (model class, factory returning the existing-or-new instance,
#    name of the Tts class attribute holding the live instance)
# Built after the class body so the factory static methods are available.
Tts._MODEL_REGISTRY: dict[TtsModelType, tuple[type[TtsBaseModel], Callable[[], TtsBaseModel], str]] = {
    TtsModelType.CHATTERBOX: (ChatterboxBaseModel, Tts.get_chatterbox, "_chatterbox"),
    TtsModelType.FISH_S1: (FishS1BaseModel, Tts.get_fish_s1, "_fish_s1"),
    TtsModelType.FISH_S2: (FishS2BaseModel, Tts.get_fish_s2, "_fish_s2"),
    TtsModelType.FISH_S2_SERVER: (FishS2ServerBaseModel, Tts.get_fish_s2_server, "_fish_s2_server"),
    TtsModelType.GLM: (GlmBaseModel, Tts.get_glm, "_glm"),
    TtsModelType.HIGGS_V2: (HiggsV2BaseModel, Tts.get_higgs, "_higgs_v2"),
    TtsModelType.HIGGS_V3_SERVER: (HiggsV3ServerBaseModel, Tts.get_higgs_v3, "_higgs_v3"),
    TtsModelType.INDEXTTS2: (IndexTts2BaseModel, Tts.get_indextts2, "_indextts2"),
    TtsModelType.MIRA: (MiraBaseModel, Tts.get_mira, "_mira"),
    TtsModelType.MOSS: (MossBaseModel, Tts.get_moss, "_moss"),
    TtsModelType.MOSS_SERVER: (MossServerModel, Tts.get_moss_server, "_moss_server"),
    TtsModelType.OMNIVOICE: (OmniVoiceBaseModel, Tts.get_omnivoice, "_omnivoice"),
    TtsModelType.OUTE: (OuteBaseModel, Tts.get_oute, "_oute"),
    TtsModelType.POCKET: (PocketBaseModel, Tts.get_pocket, "_pocket"),
    TtsModelType.QWEN3TTS: (Qwen3BaseModel, Tts.get_qwen3, "_qwen3"),
    TtsModelType.QWEN3TTS_SERVER: (Qwen3ServerBaseModel, Tts.get_qwen3tts_server, "_qwen3tts_server"),
    TtsModelType.VIBEVOICE: (VibeVoiceBaseModel, Tts.get_vibevoice, "_vibevoice"),
    TtsModelType.ZONOS2_SERVER: (Zonos2ServerBaseModel, Tts.get_zonos2_server, "_zonos2_server"),
}
