from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from importlib import util
import os
import threading
from typing import Callable, cast

from tts_audiobook_tool.app_types import DeviceType, StreamChunkCallback, StreamEndCallback, VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Reason

from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
from tts_audiobook_tool.tts_models.dots_base_model import DotsBaseModel
from tts_audiobook_tool.tts_models.fish_s1_base_model import FishS1BaseModel
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.higgs_v2_base_model import HiggsV2BaseModel
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_models.mira_base_model import MiraBaseModel
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.none_base_model import NoneBaseModel
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelSpec, TtsModelType
from tts_audiobook_tool.tts_models.sgl_omni_configured import SglOmniBackendAdapter, ConfiguredModelSupport
from tts_audiobook_tool.tts_models.sgl_omni_definition import SglOmniModelDefinition
from tts_audiobook_tool.tts_models.model_support import ModelSupport
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.model_runtime import (
    ModelRuntimeRole,
    current_role,
    require_model_owner,
)
from tts_audiobook_tool.project_support.model_settings import REGISTRY
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
    _dots: DotsBaseModel | None = None
    _fish_s1: FishS1BaseModel | None = None
    _fish_s2: FishS2BaseModel | None = None
    _glm: GlmBaseModel | None = None
    _higgs_v2: HiggsV2BaseModel | None = None
    _configured_runtime: SglOmniBackendAdapter | None = None
    _configured_definitions: dict[str, SglOmniModelDefinition] = {}
    _config_fingerprint: str = ""
    _catalog_initialized: bool = False
    _indextts2: IndexTts2BaseModel | None = None
    _mira: MiraBaseModel | None = None
    _moss: MossBaseModel | None = None
    _omnivoice: OmniVoiceBaseModel | None = None
    _pocket: PocketBaseModel | None = None
    _qwen3: Qwen3BaseModel | None = None
    _vibevoice: VibeVoiceBaseModel | None = None

    _sgl_omni_type: TtsModelType | None = None

    # Process-level backend mode (LOCAL or SGL_OMNI), probed once at
    # startup from the SGL-Omni sentinel package and immutable for the
    # life of the process
    _backend_mode: TtsBackendKind | None = None
    _MODEL_REGISTRY: dict[
        TtsModelType,
        tuple[type[TtsBaseModel], Callable[[], TtsBaseModel], str],
    ]

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
    def get_voice_value_count(project) -> int:
        """Number of configured voice samples for the active TTS model type."""
        if REGISTRY.voice_binding(Tts.get_type().id) is None:
            return 0

        from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
        return len(ProjectVoiceUtil.get_voice_values(project, Tts.get_type()))

    @staticmethod
    def get_voice_tag_for_selection_index(project, voice_selection_index: int) -> str:
        if REGISTRY.voice_binding(Tts.get_type().id) is None:
            return Tts.get_model_support().get_voice_tag(project)

        from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
        voice_value = ProjectVoiceUtil.current_voice_value(project, Tts.get_type(), voice_selection_index)
        if not voice_value:
            return Tts.get_model_support().get_voice_tag(project)
        return Tts.get_model_support().get_voice_tag_for_value(voice_value)

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
        # Startup/reinitialization must not retain an adapter or a stale
        # definition if the next configuration fails validation.
        Tts._configured_runtime = None
        Tts._configured_definitions = {}
        Tts._config_fingerprint = ""
        TtsModelType.reset_catalog()
        # Configured storage declarations are rebuilt from scratch on every
        # (re)initialization, so local runs never keep them.
        REGISTRY.reset_to_builtins()
        # The configured JSON definitions are the sole SGL-Omni server
        # implementation; local mode never opens the JSON file.
        if Tts._backend_mode == TtsBackendKind.SGL_OMNI:
            from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
            definitions = load_definitions()
            for definition in definitions.models.values():
                if definition.spec.id in TtsModelType._builtin_specs:
                    TtsModelType.overlay_spec(definition.spec)
                else:
                    TtsModelType.register_spec(definition.spec)
                REGISTRY.register_configured_model(definition)
            Tts._configured_definitions = definitions.models
            Tts._config_fingerprint = definitions.fingerprint
        Tts._catalog_initialized = True

        if Tts._backend_mode == TtsBackendKind.SGL_OMNI:
            Tts._type = TtsModelType.NONE
            return Tts._type, 0

        def get_matches() -> list[TtsModelType]:
            model_infos = []
            for model_info in TtsModelType.all():
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
        if (
            Tts._type != value
            and current_role() is not ModelRuntimeRole.INTERACTIVE_MAIN
        ):
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
    def get_model_params_using_project(project) -> dict[str, object]:
        """Return the model configuration represented by a live Project.

        Inspection commands use this snapshot because their project changes may
        intentionally be unsaved while a custom model or adapter is validated.
        """
        from tts_audiobook_tool.project import Project
        assert isinstance(project, Project)

        return {
            "chatterbox_type": project.get_model_setting('chatterbox', 'type'),
            "dots_target": project.get_model_setting('dots', 'target'),
            "dots_compile": project.get_model_setting('dots', 'compile'),
            "vibevoice_target": project.get_model_setting('vibevoice', 'target'),
            "vibevoice_lora_path": project.get_model_setting('vibevoice', 'lora_target'),
            "indextts2_use_fp16": project.get_model_setting('indextts2', 'use_fp16'),
            "glm_sr": project.get_model_setting('glm', 'sr'),
            "moss_target": project.get_model_setting('moss', 'target'),
            "qwen3_target": project.get_model_setting('qwen3tts', 'target'),
            "fish_s1_compile_enabled": project.get_model_setting('fish_s1', 'compile_enabled'),
            "fish_s2_compile_enabled": project.get_model_setting('fish_s2', 'compile_enabled'),
            "pocket_model_code": project.get_model_setting('pocket', 'model_code'),
            "omnivoice_target": project.get_model_setting('omnivoice', 'target'),
        }

    @staticmethod
    def set_model_params_using_project(project) -> None:
        Tts.set_model_params(Tts.get_model_params_using_project(project))

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
        dirty |= new_params.get("dots_target", "") != old_params.get("dots_target", "")
        dirty |= new_params.get("dots_compile", False) != old_params.get("dots_compile", False)
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
        if dirty and current_role() is not ModelRuntimeRole.INTERACTIVE_MAIN:
            Tts.clear_tts_model()

    @staticmethod
    def set_force_cpu(value: bool) -> None:
        if Tts._force_cpu != value:
            Tts._force_cpu = value
            # Interactive main owns configuration only; worker reconciles it.
            if current_role() is not ModelRuntimeRole.INTERACTIVE_MAIN:
                Tts.clear_tts_model()

    @staticmethod
    def get_class() -> type[TtsBaseModel]:
        """
        Gets the current tts model's class, used for accessing static methods.
        """
        return Tts.get_class_for_type(Tts._type)

    @staticmethod
    def get_class_for_type(tts_type: TtsModelType) -> type[TtsBaseModel]:
        """
        Gets the model class registered for the given catalog member, used
        for accessing static/class methods without a live instance.
        """
        entry = Tts._model_registry_entry(tts_type)
        if entry is None or entry[0] is None:
            raise Exception(f"Not implemented: {tts_type}")
        return entry[0]

    @staticmethod
    def get_configured_definition(tts_type: TtsModelType | None = None) -> SglOmniModelDefinition | None:
        return Tts._configured_definitions.get((tts_type or Tts.get_type()).value.id)

    @staticmethod
    def get_model_support() -> ModelSupport:
        return Tts.get_model_support_for_type(Tts.get_type())

    @staticmethod
    def get_model_support_for_type(tts_type: TtsModelType) -> ModelSupport:
        definition = Tts.get_configured_definition(tts_type)
        if definition is not None:
            return ConfiguredModelSupport(definition)
        # Legacy class methods expose the same support surface without a runtime.
        return cast(ModelSupport, Tts.get_class_for_type(tts_type))

    @staticmethod
    def get_info() -> TtsModelSpec:
        return Tts.get_type().value

    @staticmethod
    def instance_exists() -> bool:
        require_model_owner("TTS")
        items = [
            Tts._chatterbox,
            Tts._dots,
            Tts._fish_s1,
            Tts._fish_s2,
            Tts._glm,
            Tts._higgs_v2,
            Tts._configured_runtime,
            Tts._indextts2,
            Tts._mira,
            Tts._moss,
            Tts._omnivoice,
            Tts._pocket,
            Tts._qwen3,
            Tts._vibevoice,
        ]
        for item in items:
            if item is not None:
                return True
        return False

    @staticmethod
    def get_instance() -> TtsBaseModel | SglOmniBackendAdapter:
        require_model_owner("TTS")
        definition = Tts.get_configured_definition()
        if definition is not None:
            if Tts._configured_runtime is None:
                from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredModelSupport
                Tts._configured_runtime = SglOmniBackendAdapter(definition, ConfiguredModelSupport(definition))
            return Tts._configured_runtime
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
            print_params: bool = False,
            voice_selection_index: int | None = None,
            apply_word_substitutions: bool = True,
    ):
        """
        All app-level TTS generation goes through this function.

        Applies the standard project/model text-preparation pipeline to each
        prompt exactly once, then delegates to the active concrete model's own
        `generate_using_project()` implementation.

        This keeps audiobook generation, realtime playback, server/API usage,
        and LLM chat consistent wrt prompt normalization and model-specific
        transforms such as VibeVoice speaker tagging. Diagnostic callers may
        disable only project word substitutions while retaining the rest of
        the preparation pipeline.
        """
        instance = Tts.get_instance()
        if voice_selection_index is None:
            # Rotate voices only in auto-advance mode with more than one
            # voice sample configured; every other condition (including
            # single-voice auto-advance, e.g. LLM chat synthesis) always
            # uses the first voice when the caller does not pass an
            # explicit index.
            if (
                project.voice_select_mode == VoiceSelectMode.AUTO_ADVANCE
                and Tts.get_voice_value_count(project) > 1
            ):
                voice_selection_index = Tts.get_next_voice_selection_index()
            else:
                voice_selection_index = 0
        L.i(
            f"Tts.generate_using_project dispatch: type={Tts._type.value.id} "
            f"instance={type(instance).__name__} prompts={len(prompts)} "
            f"voice_selection_index={voice_selection_index} "
            f"has_on_stream_chunk={on_stream_chunk is not None} has_on_stream_end={on_stream_end is not None}"
        )

        prepared_prompts = [
            instance.prepare_text_for_inference(
                project,
                prompt,
                apply_word_substitutions=apply_word_substitutions,
            )
            for prompt in prompts
        ]
        kwargs = {
            "on_stream_chunk": on_stream_chunk,
            "on_stream_end": on_stream_end if on_stream_end is not None else project.on_stream_end,
            "voice_selection_index": voice_selection_index,
            "print_params": print_params,
        }
        if Tts._type.value.backend_kind == TtsBackendKind.SGL_OMNI:
            kwargs["print_generation_request"] = print_generation_request

        return instance.generate_using_project(
            project=project,
            prompts=prepared_prompts,
            force_random_seed=force_random_seed,
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
    def get_instance_if_exists() -> TtsBaseModel | SglOmniBackendAdapter | None:
        require_model_owner("TTS")
        if Tts.get_configured_definition() is not None:
            return Tts._configured_runtime
        # Returns instance only if it already exists, else none
        entry = Tts._model_registry_entry(Tts._type)
        if entry is None or not entry[2]:
            return None
        return getattr(Tts, entry[2])

    @staticmethod
    def get_chatterbox() -> ChatterboxBaseModel:
        require_model_owner("TTS")
        if not Tts._chatterbox:
            model_type = Tts._model_params.get("chatterbox_type")
            assert isinstance(model_type, ChatterboxType), "chatterbox_type not set"
            device_type = Tts.get_best_supported_device_type(TtsModelType.CHATTERBOX)

            from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(model_type, device_type)
            printt()
        return Tts._chatterbox

    @staticmethod
    def get_dots() -> DotsBaseModel:
        require_model_owner("TTS")
        if not Tts._dots:
            device_type = Tts.get_best_supported_device_type(TtsModelType.DOTS)
            target = Tts._model_params.get("dots_target", "") or DotsBaseModel.DEFAULT_REPO_ID
            from tts_audiobook_tool.tts_models.dots_base_model import DotsCompileMode
            compile_enabled = Tts._model_params.get(
                "dots_compile", DotsCompileMode.default().enabled
            )
            from tts_audiobook_tool.tts_models.dots_model import DotsModel
            Tts._dots = DotsModel(
                target,
                device_type,
                compile_enabled=compile_enabled,
            )
            printt()
        return Tts._dots

    @staticmethod
    def get_fish_s1() -> FishS1BaseModel:
        require_model_owner("TTS")
        if not Tts._fish_s1:
            device_type = Tts.get_best_supported_device_type(TtsModelType.FISH_S1)

            if device_type == DeviceType.CUDA:
                compile_enabled = Tts._model_params.get("fish_s1_compile_enabled", False)
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
        require_model_owner("TTS")
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
    def get_glm() -> GlmBaseModel:
        require_model_owner("TTS")
        if not Tts._glm:
            device_type = Tts.get_best_supported_device_type(TtsModelType.GLM)
            sr = Tts._model_params["glm_sr"]

            from tts_audiobook_tool.tts_models.glm_model import GlmModel
            Tts._glm = GlmModel(device_type, sr)
            printt()
        return Tts._glm

    @staticmethod
    def get_higgs() -> HiggsV2BaseModel:
        require_model_owner("TTS")
        if not Tts._higgs_v2:
            device_type = Tts.get_best_supported_device_type(TtsModelType.HIGGS_V2)
            from tts_audiobook_tool.tts_models.higgs_v2_model import HiggsV2Model
            Tts._higgs_v2 = HiggsV2Model(device_type)
            printt()

        return Tts._higgs_v2

    @staticmethod
    def get_indextts2() -> IndexTts2BaseModel:
        require_model_owner("TTS")
        if not Tts._indextts2:
            use_fp16 = Tts._model_params.get("indextts2_use_fp16", False)
            from tts_audiobook_tool.tts_models.indextts2_model import IndexTts2Model
            Tts._indextts2 = IndexTts2Model(use_fp16=use_fp16) # model will use cuda if available
            printt()
        return Tts._indextts2

    @staticmethod
    def get_mira() -> MiraBaseModel:
        require_model_owner("TTS")
        if not Tts._mira:
            from tts_audiobook_tool.tts_models.mira_model import MiraModel
            Tts._mira = MiraModel()
            printt()
        return Tts._mira

    @staticmethod
    def get_moss() -> MossBaseModel:
        require_model_owner("TTS")
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
    def get_omnivoice() -> OmniVoiceBaseModel:
        require_model_owner("TTS")
        if not Tts._omnivoice:
            device_type = Tts.get_best_supported_device_type(TtsModelType.OMNIVOICE)
            model_target = Tts._model_params.get("omnivoice_target", "") \
                        or OmniVoiceBaseModel.DEFAULT_REPO_ID
            from tts_audiobook_tool.tts_models.omnivoice_model import OmniVoiceModel
            Tts._omnivoice = OmniVoiceModel(device=device_type, model_target=model_target)
            printt()
        return Tts._omnivoice

    @staticmethod
    def get_pocket() -> PocketBaseModel:
        require_model_owner("TTS")
        if not Tts._pocket:
            language = Tts._model_params.get("pocket_model_code", "")
            device_type = Tts.get_best_supported_device_type(TtsModelType.POCKET)
            from tts_audiobook_tool.tts_models.pocket_model import PocketModel
            Tts._pocket = PocketModel(device=device_type, language=language)
            printt()
        return Tts._pocket

    @staticmethod
    def get_qwen3() -> Qwen3BaseModel:
        require_model_owner("TTS")

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
    def get_vibevoice() -> VibeVoiceBaseModel:
        require_model_owner("TTS")

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
        require_model_owner("TTS")
        model = Tts.get_instance_if_exists()
        if model:
            model.clear_voice_clone_cache()
            model.kill()
            # Null out all instance attributes, not just the current
            # type's (there must be at most one live instance)
            for entry in Tts._MODEL_REGISTRY.values():
                setattr(Tts, entry[2], None)
            Tts._configured_runtime = None
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
Tts._MODEL_REGISTRY = {
    TtsModelType.CHATTERBOX: (ChatterboxBaseModel, Tts.get_chatterbox, "_chatterbox"),
    TtsModelType.DOTS: (DotsBaseModel, Tts.get_dots, "_dots"),
    TtsModelType.FISH_S1: (FishS1BaseModel, Tts.get_fish_s1, "_fish_s1"),
    TtsModelType.FISH_S2: (FishS2BaseModel, Tts.get_fish_s2, "_fish_s2"),
    TtsModelType.GLM: (GlmBaseModel, Tts.get_glm, "_glm"),
    TtsModelType.HIGGS_V2: (HiggsV2BaseModel, Tts.get_higgs, "_higgs_v2"),
    TtsModelType.INDEXTTS2: (IndexTts2BaseModel, Tts.get_indextts2, "_indextts2"),
    TtsModelType.MIRA: (MiraBaseModel, Tts.get_mira, "_mira"),
    TtsModelType.MOSS: (MossBaseModel, Tts.get_moss, "_moss"),
    TtsModelType.OMNIVOICE: (OmniVoiceBaseModel, Tts.get_omnivoice, "_omnivoice"),
    TtsModelType.POCKET: (PocketBaseModel, Tts.get_pocket, "_pocket"),
    TtsModelType.QWEN3TTS: (Qwen3BaseModel, Tts.get_qwen3, "_qwen3"),
    TtsModelType.VIBEVOICE: (VibeVoiceBaseModel, Tts.get_vibevoice, "_vibevoice"),
}
