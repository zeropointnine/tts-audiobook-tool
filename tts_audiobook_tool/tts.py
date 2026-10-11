from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from importlib import util
import os
import threading
from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsSnapshot
    from tts_audiobook_tool.project import Project

from tts_audiobook_tool.app_types import DeviceType, ReadinessIssue, StreamChunkCallback, StreamEndCallback, VoiceSelectMode
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
from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppModelDefinition
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter
from tts_audiobook_tool.tts_models.model_support import ModelSupport
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.launcher_marker import REMOTE_CLIENT_MARKER_MODULES
from tts_audiobook_tool.model_runtime import (
    ModelRuntimeRole,
    current_role,
    require_model_owner,
)
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.seed_util import resolve_max_random_seed
from tts_audiobook_tool.util import *


class TtsRuntimeMode(Enum):
    LOCAL = "local"
    REMOTE_CLIENT = "remote_client"


class Tts:
    """
    Static class for accessing the TTS model.

    The process-level mode (local or remote TTS client) is fixed at startup
    from the launcher marker. The catalog backend (SGL-Omni or audio.cpp)
    is instead determined from the remote server at connection/refresh.
    """

    _type: TtsModelType

    _chatterbox: ChatterboxBaseModel | None = None
    _dots: DotsBaseModel | None = None
    _fish_s1: FishS1BaseModel | None = None
    _fish_s2: FishS2BaseModel | None = None
    _glm: GlmBaseModel | None = None
    _higgs_v2: HiggsV2BaseModel | None = None
    _configured_runtime: SglOmniBackendAdapter | AudioCppBackendAdapter | None = None
    _configured_definitions: dict[str, SglOmniModelDefinition] = {}
    _audio_cpp_definitions: dict[str, AudioCppModelDefinition] = {}
    _config_fingerprint: str = ""
    _catalog_initialized: bool = False
    _indextts2: IndexTts2BaseModel | None = None
    _mira: MiraBaseModel | None = None
    _moss: MossBaseModel | None = None
    _omnivoice: OmniVoiceBaseModel | None = None
    _pocket: PocketBaseModel | None = None
    _qwen3: Qwen3BaseModel | None = None
    _vibevoice: VibeVoiceBaseModel | None = None

    # Capability discovery is independent of the project-bound runtime type.
    _available_local_models: tuple[TtsModelType, ...] = ()
    _selected_server_model_id: str = ""
    _remote_issue: str = ""
    _binding_issue: ReadinessIssue | None = None
    _bound_project_type_id: str = ""

    # Process-level client-vs-local mode, independent of catalog backend.
    _backend_mode: TtsRuntimeMode | None = None
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
    def get_voice_value_count(project: Project) -> int:
        """Number of configured voice samples for this project's selection."""
        model_type = project.get_tts_model_type()
        if REGISTRY.voice_binding(model_type.id) is None or (
            model_type.id == "pocket_local" and project.get_model_setting(model_type.id, "predefined_voice")
        ):
            return 0

        from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
        return len(ProjectVoiceUtil.get_voice_values(project, model_type))

    @staticmethod
    def get_voice_tag_for_selection_index(project: Project, voice_selection_index: int) -> str:
        model_type = project.get_tts_model_type()
        support = Tts.get_model_support(project)
        if REGISTRY.voice_binding(model_type.id) is None or (
            model_type.id == "pocket_local" and project.get_model_setting(model_type.id, "predefined_voice")
        ):
            return support.get_voice_tag(project)

        from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
        voice_value = ProjectVoiceUtil.current_voice_value(project, model_type, voice_selection_index)
        if not voice_value:
            return support.get_voice_tag(project)
        return support.get_voice_tag_for_value(voice_value)

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
        Initialize metadata and detect the current virtual environment's capability.
        Leaves the active runtime unselected until a Project is bound.
        Does not instantiate an inference model. Must be run on startup.

        First probes the remote-client markers (including the historical
        SGL-Omni name) to fix the process-level backend mode (immutable for
        the life of the process):

        - Remote-client mode: the local model probe is skipped entirely, even
          in a dual-capable venv that also holds a local model library
          (remote-client wins; such a venv is user error). The type starts as
          NONE and is resolved from a Project by bind_project().
        - Local mode: local model libraries are probed as before.

        Returns the model type that was set, and num matches (always 0 in
        remote-client mode, 0 or 1 in local mode).
        """
        Tts._backend_mode = Tts._probe_backend_mode()
        # Startup/reinitialization must not retain an adapter or a stale
        # definition if the next configuration fails validation.
        Tts._configured_runtime = None
        Tts._configured_definitions = {}
        Tts._audio_cpp_definitions = {}
        Tts._selected_server_model_id = ""
        Tts._remote_issue = ""
        Tts._config_fingerprint = ""
        TtsModelType.reset_catalog()
        # Project metadata/settings are available even in the wrong runtime
        # or while its server is offline. These loaders never load inference.
        REGISTRY.reset_to_catalog()
        from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
        from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
        definitions = load_definitions()
        audio_definitions = load_audio_cpp_definitions()
        if definitions.fingerprint != audio_definitions.fingerprint:
            raise RuntimeError("Remote backend definitions have different catalog fingerprints")
        if definitions.setting_groups is not None:
            REGISTRY.members = dict(definitions.setting_groups)
        # Build each audio.cpp behavior once so a subclass whose declared
        # parameters disagree with the catalog fails here, at startup, rather
        # than when that model is first selected.
        from tts_audiobook_tool.tts_models.audio_cpp_behavior import get_audio_cpp_behavior
        for audio_definition in audio_definitions.models.values():
            get_audio_cpp_behavior(audio_definition)
        for definition in (*definitions.models.values(), *audio_definitions.models.values()):
            TtsModelType.install_spec(definition.spec)
            REGISTRY.register_model_settings(definition.spec.id, definition.settings)
        Tts._configured_definitions = definitions.models
        Tts._audio_cpp_definitions = audio_definitions.models
        Tts._config_fingerprint = definitions.fingerprint
        Tts._catalog_initialized = True
        Tts._available_local_models = ()
        Tts._binding_issue = None
        Tts._bound_project_type_id = ""
        Tts._type = TtsModelType.require_by_id("none")

        if Tts._backend_mode is TtsRuntimeMode.REMOTE_CLIENT:
            Tts._type = TtsModelType.require_by_id("none")
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
        Tts._available_local_models = tuple(matches)
        # The detected capability is not an active project/runtime selection.
        return (matches[0] if len(matches) == 1 else TtsModelType.require_by_id("none")), len(matches)

    @staticmethod
    def get_active_type() -> TtsModelType:
        if not hasattr(Tts, "_type") or Tts._type is None:
            raise Exception("TTS model type has not been set. Must first call init_local_model_type().`")
        return Tts._type

    @staticmethod
    def get_backend_mode() -> TtsRuntimeMode:
        """Process mode is local or remote-client, not the server's protocol."""
        if Tts._backend_mode is None:
            Tts._backend_mode = Tts._probe_backend_mode()
        return Tts._backend_mode

    @staticmethod
    def _probe_backend_mode() -> TtsRuntimeMode:
        for module in REMOTE_CLIENT_MARKER_MODULES:
            try:
                if util.find_spec(module) is not None:
                    return TtsRuntimeMode.REMOTE_CLIENT
            except Exception:
                # An unreadable marker must not prevent probing the other name.
                continue
        return TtsRuntimeMode.LOCAL

    @staticmethod
    def set_type(value: TtsModelType) -> None:
        if (
            Tts._type != value
            and current_role() is not ModelRuntimeRole.INTERACTIVE_MAIN
        ):
            Tts.clear_tts_model()
        Tts._type = value

    @staticmethod
    def get_available_tts_models(*, refresh: bool = False) -> list[TtsModelType]:
        """Distinct discoverable types, independent of project/runtime selection.

        Local capabilities are probed once at initialization. Remote observations
        reuse discovery TTL/backoff unless an explicit action requests refresh.
        Exact server-entry pairs remain in the snapshot for runtime resolution.
        """
        if not Tts.is_remote_mode():
            if len(Tts._available_local_models) > 1:
                raise RuntimeError("More than one supported local TTS model library is installed")
            return list(Tts._available_local_models)
        from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
        snapshot = RemoteTtsDiscovery.refresh(force=refresh)
        if snapshot.issue is not None:
            return []
        return list(dict.fromkeys(model for model, _ in snapshot.candidates if model.id != "none"))


    @staticmethod
    def reconcile_project_model(project: Project) -> TtsModelType | None:
        """Reconcile an interactive selection; return the changed type, if any.

        Local mode forces the installed model, or NONE when none is available.
        In remote mode, failed/empty discovery must not discard a saved selection.
        Runtime binding remains separate and never mutates the Project, so
        workers and metadata-only callers do not auto-select.
        """
        available = Tts.get_available_tts_models()
        if Tts.is_remote_mode():
            if not available or any(model.id == project.tts_model_type for model in available):
                return None
        model = available[0] if len(available) == 1 else TtsModelType.require_by_id("none")
        if project.tts_model_type == model.id:
            return None
        project.tts_model_type = model.id
        return model

    @staticmethod
    def get_local_model_type() -> TtsModelType:
        """Startup capability, not the active or saved project selection."""
        return Tts._available_local_models[0] if len(Tts._available_local_models) == 1 else TtsModelType.require_by_id("none")

    @staticmethod
    def is_local_model() -> bool:
        return Tts._type.id != "none" and Tts._type.value.backend_kind == TtsBackendKind.LOCAL

    @staticmethod
    def is_remote_mode() -> bool:
        return Tts.get_backend_mode() is TtsRuntimeMode.REMOTE_CLIENT

    @staticmethod
    def is_sgl_mode() -> bool:
        """Legacy name: true for either remote server in the client venv."""
        return Tts.is_remote_mode()

    @staticmethod
    def get_model_params_using_project(project) -> dict[str, object]:
        """Return the model configuration represented by a live Project.

        Inspection commands use this snapshot because their project changes may
        intentionally be unsaved while a custom model or adapter is validated.
        """
        from tts_audiobook_tool.project import Project
        assert isinstance(project, Project)

        return {
            "chatterbox_type": project.get_model_setting('chatterbox_local', 'type'),
            "dots_target": project.get_model_setting('dots_local', 'target'),
            "dots_compile": project.get_model_setting('dots_local', 'compile'),
            "vibevoice_target": project.get_model_setting('vibevoice_local', 'target'),
            "vibevoice_lora_path": project.get_model_setting('vibevoice_local', 'lora_target'),
            "indextts2_use_fp16": project.get_model_setting('indextts2_local', 'use_fp16'),
            "glm_sr": project.get_model_setting('glm_local', 'sr'),
            "moss_target": project.get_model_setting('moss_local', 'target'),
            "qwen3_target": project.get_model_setting('qwen3tts_local', 'target'),
            "fish_s1_compile_enabled": project.get_model_setting('fish_s1_local', 'compile_enabled'),
            "fish_s2_compile_enabled": project.get_model_setting('fish_s2_local', 'compile_enabled'),
            "pocket_model_code": project.get_model_setting('pocket_local', 'model_code'),
            "omnivoice_target": project.get_model_setting('omnivoice_local', 'target'),
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
    def get_configured_definition(tts_type: TtsModelType) -> SglOmniModelDefinition | None:
        return Tts._configured_definitions.get(tts_type.id)

    @staticmethod
    def get_audio_cpp_definition(tts_type: TtsModelType) -> AudioCppModelDefinition | None:
        return Tts._audio_cpp_definitions.get(tts_type.id)

    @staticmethod
    def can_batch(tts_type: TtsModelType) -> bool:
        """Whether the catalog lets this model offer/use a concurrency value.

        Registry storage presence is a separate question (see
        `TtsModelType.can_batch()`): a declared model may keep its storage
        binding for project compatibility while this capability is false.
        Models without a loaded definition answer True, which preserves local
        and legacy behavior.
        """
        definition = Tts._configured_definitions.get(tts_type.id)
        return definition.can_batch if definition is not None else True

    @staticmethod
    def requires_reference_transcript(project: Project) -> bool:
        """Whether generating with this project's current settings requires
        the voice sample transcript.

        The single runtime authority for the voice pre-flight, server startup
        and the audio.cpp adapter. Differs from `REGISTRY.transcript_binding()`,
        which only says whether the model stores (and lets the user edit) a
        transcript: each model's support answers for its current settings
        (eg MOSS local never reads it; Qwen3 only for its Base checkpoint;
        CosyVoice3 on audio.cpp only in Zero-shot mode).
        """
        model_type = project.get_tts_model_type()
        if REGISTRY.transcript_binding(model_type.id) is None:
            return False  # Nothing stored, so nothing can be required.
        return Tts.get_model_support_for_type(model_type).uses_reference_transcript(project)

    @staticmethod
    def get_model_support(project: Project) -> ModelSupport:
        return Tts.get_model_support_for_type(project.get_tts_model_type())

    @staticmethod
    def get_model_support_for_type(tts_type: TtsModelType) -> ModelSupport:
        definition = Tts.get_configured_definition(tts_type)
        if definition is not None:
            return ConfiguredModelSupport(definition)
        audio_definition = Tts.get_audio_cpp_definition(tts_type)
        if audio_definition is not None:
            from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppModelSupport
            return AudioCppModelSupport(audio_definition)
        # Legacy class methods expose the same support surface without a runtime.
        return cast(ModelSupport, Tts.get_class_for_type(tts_type))

    @staticmethod
    def get_info(project: Project) -> TtsModelSpec:
        return project.get_tts_model_type().value

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
    def get_instance() -> TtsBaseModel | SglOmniBackendAdapter | AudioCppBackendAdapter:
        require_model_owner("TTS")
        if Tts._type.id == "none":
            raise RuntimeError(Tts._binding_issue.verbose if Tts._binding_issue else "No TTS model is bound")
        definition = Tts.get_configured_definition(Tts._type)
        if definition is not None:
            if Tts._configured_runtime is None:
                Tts._configured_runtime = SglOmniBackendAdapter(definition, ConfiguredModelSupport(definition))
            return Tts._configured_runtime
        audio_definition = Tts.get_audio_cpp_definition(Tts._type)
        if audio_definition is not None:
            if not Tts._selected_server_model_id:
                raise RuntimeError(Tts._remote_issue or "No compatible audio.cpp server model is selected")
            if Tts._configured_runtime is None:
                from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport
                Tts._configured_runtime = AudioCppBackendAdapter(
                    audio_definition, AudioCppModelSupport(audio_definition), Tts._selected_server_model_id)
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
            max_random_seed: int = -1,
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
        the preparation pipeline. An optional inclusive max_random_seed can
        tighten the catalog's random-seed cap, but cannot widen it. With -1,
        only the catalog policy applies; unconfigured models keep their native
        ranges. Fixed seeds are never limited by this cap.
        """
        if project.get_tts_model_type().id == "none" or Tts.get_active_type() != project.get_tts_model_type():
            return Tts._binding_issue.verbose if Tts._binding_issue else "Project TTS model is not bound"
        try:
            max_random_seed = resolve_max_random_seed(Tts._type.value.max_random_seed, max_random_seed)
        except ValueError as exc:
            return str(exc)
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
        if Tts._type.value.backend_kind in (TtsBackendKind.SGL_OMNI, TtsBackendKind.AUDIO_CPP):
            kwargs["print_generation_request"] = print_generation_request

        return instance.generate_using_project(
            project=project,
            prompts=prepared_prompts,
            force_random_seed=force_random_seed,
            max_random_seed=max_random_seed,
            **kwargs,
        )

    @staticmethod
    def clear_continuation() -> None:
        instance = Tts.get_instance_if_exists()
        if instance is not None:
            instance.clear_continuation()

    @staticmethod
    def clear_continuation_if_reason(reason: Reason) -> None:
        if reason in { Reason.PARAGRAPH, Reason.HEADING, Reason.SPACE_BREAK, Reason.SECTION_BREAK }:
            Tts.clear_continuation()

    @staticmethod
    def _model_registry_entry(tts_type: TtsModelType) -> tuple[type[TtsBaseModel], Callable[[], TtsBaseModel] | None, str] | None:
        """
        Shared lookup backing get_class_for_type() / get_instance() /
        get_instance_if_exists(). The "none" placeholder maps to
        (NoneBaseModel, no factory, no instance attribute); an unknown
        value (impossible for catalog members) maps to nothing at all.
        """
        if tts_type.id == "none":
            return NoneBaseModel, None, ""
        return Tts._MODEL_REGISTRY.get(tts_type)

    @staticmethod
    def get_instance_if_exists() -> TtsBaseModel | SglOmniBackendAdapter | AudioCppBackendAdapter | None:
        require_model_owner("TTS")
        if Tts.get_configured_definition(Tts._type) is not None or Tts.get_audio_cpp_definition(Tts._type) is not None:
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
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("chatterbox_local"))

            from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel
            Tts._chatterbox = ChatterboxModel(model_type, device_type)
            printt()
        return Tts._chatterbox

    @staticmethod
    def get_dots() -> DotsBaseModel:
        require_model_owner("TTS")
        if not Tts._dots:
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("dots_local"))
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
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("fish_s1_local"))

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
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("fish_s2_local"))

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
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("glm_local"))
            sr = Tts._model_params["glm_sr"]

            from tts_audiobook_tool.tts_models.glm_model import GlmModel
            Tts._glm = GlmModel(device_type, sr)
            printt()
        return Tts._glm

    @staticmethod
    def get_higgs() -> HiggsV2BaseModel:
        require_model_owner("TTS")
        if not Tts._higgs_v2:
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("higgs_v2_local"))
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
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("moss_local"))
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
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("omnivoice_local"))
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
            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("pocket_local"))
            from tts_audiobook_tool.tts_models.pocket_model import PocketModel
            Tts._pocket = PocketModel(device=device_type, language=language)
            printt()
        return Tts._pocket

    @staticmethod
    def get_qwen3() -> Qwen3BaseModel:
        require_model_owner("TTS")

        if not Tts._qwen3:

            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("qwen3tts_local"))
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

            device_type = Tts.get_best_supported_device_type(TtsModelType.require_by_id("vibevoice_local"))
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
    def bind_project(project: Project, *, refresh: bool = False) -> ReadinessIssue | None:
        """Bind a validated project selection without changing its saved ID.

        Metadata reads use the Project directly. Only this runtime boundary
        resolves an exact remote entry, which remains ephemeral. audio.cpp uses
        the first matching entry in discovery order; other remote backends require
        a unique match. Repeated bindings reuse discovery and unchanged instances.
        """
        selected = project.get_tts_model_type()
        issue: ReadinessIssue | None = None
        new_id = ""
        if selected.id == "none":
            message = (f"Unknown TTS model: {project.tts_model_type}" if project.tts_model_type != "none"
                       else "Select a TTS model for this project")
            issue = ReadinessIssue("TTS model", message)
        elif not Tts.is_remote_mode():
            if selected not in Tts.get_available_tts_models():
                issue = ReadinessIssue("TTS model", "Selected TTS model is not available in this environment")
        elif selected.value.backend_kind is TtsBackendKind.LOCAL:
            issue = ReadinessIssue("TTS model", "Selected TTS model is not available in this environment")
        else:
            from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
            Tts.get_available_tts_models(refresh=refresh)
            snapshot = RemoteTtsDiscovery.get_snapshot()
            if snapshot.issue is not None:
                issue = ReadinessIssue("Remote TTS server", snapshot.issue.message)
            else:
                candidates = [pair for pair in snapshot.candidates if pair[0] == selected]
                if candidates and (
                    selected.value.backend_kind is TtsBackendKind.AUDIO_CPP
                    or len(candidates) == 1
                ):
                    # Discovery preserves /v1/models order; never sort or prefer loaded entries.
                    new_id = candidates[0][1]
                elif not candidates:
                    issue = ReadinessIssue("TTS model", _missing_remote_model_issue(snapshot, selected))
                else:
                    issue = ReadinessIssue("TTS model", (
                        "Multiple server entries match the selected TTS model"
                        f" ({', '.join(server_id for _, server_id in candidates)})"))
        new_type = selected if issue is None else TtsModelType.require_by_id("none")
        if new_type != Tts.get_active_type() or new_id != Tts._selected_server_model_id:
            if current_role() is not ModelRuntimeRole.INTERACTIVE_MAIN:
                Tts.clear_tts_model()
            Tts._type = new_type
            Tts._selected_server_model_id = new_id
        Tts._binding_issue = issue
        Tts._bound_project_type_id = project.tts_model_type
        Tts._remote_issue = issue.verbose if issue is not None and Tts.is_remote_mode() else ""
        SglOmniUtil._model_id = new_id if selected.value.backend_kind is TtsBackendKind.SGL_OMNI else ""
        Tts.set_model_params_using_project(project)
        return issue

    @staticmethod
    def get_requirements_file_name() -> str:
        """Choose install requirements from environment capability, not Project selection."""
        if Tts.is_remote_mode():
            return "requirements-remote.txt"
        local_type = Tts.get_local_model_type()
        return local_type.value.requirements_file_name if local_type.id != "none" else "requirements-base.txt"

# ---

def _available_server_ids_text(snapshot: RemoteTtsSnapshot, limit: int = 3) -> str:
    """Comma-separated server IDs from a snapshot, elided past `limit`."""
    server_ids = [server_id for _, server_id in snapshot.candidates]
    text = ", ".join(server_ids[:limit])
    if len(server_ids) > limit:
        text += f", +{len(server_ids) - limit} more"
    return text


def _missing_remote_model_issue(
    snapshot: RemoteTtsSnapshot,
    preferred_type: TtsModelType,
) -> str:
    """Explain why the project's selected type has no server entry."""
    proper_name = preferred_type.value.ui["proper_name"]
    issue = f"No server model matches the selected {proper_name} variant"
    available = _available_server_ids_text(snapshot)
    return f"{issue} (available: {available})" if available else f"{issue}."

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

# One entry per TTS variant, shared by Tts.get_class_for_type() / Tts.get_instance() /
# Tts.get_instance_if_exists() and used to clear instances:
#   (model class, factory returning the existing-or-new instance,
#    name of the Tts class attribute holding the live instance)
# Built after the class body so the factory static methods are available.
Tts._MODEL_REGISTRY = {
    TtsModelType.require_by_id("chatterbox_local"): (ChatterboxBaseModel, Tts.get_chatterbox, "_chatterbox"),
    TtsModelType.require_by_id("dots_local"): (DotsBaseModel, Tts.get_dots, "_dots"),
    TtsModelType.require_by_id("fish_s1_local"): (FishS1BaseModel, Tts.get_fish_s1, "_fish_s1"),
    TtsModelType.require_by_id("fish_s2_local"): (FishS2BaseModel, Tts.get_fish_s2, "_fish_s2"),
    TtsModelType.require_by_id("glm_local"): (GlmBaseModel, Tts.get_glm, "_glm"),
    TtsModelType.require_by_id("higgs_v2_local"): (HiggsV2BaseModel, Tts.get_higgs, "_higgs_v2"),
    TtsModelType.require_by_id("indextts2_local"): (IndexTts2BaseModel, Tts.get_indextts2, "_indextts2"),
    TtsModelType.require_by_id("mira_local"): (MiraBaseModel, Tts.get_mira, "_mira"),
    TtsModelType.require_by_id("moss_local"): (MossBaseModel, Tts.get_moss, "_moss"),
    TtsModelType.require_by_id("omnivoice_local"): (OmniVoiceBaseModel, Tts.get_omnivoice, "_omnivoice"),
    TtsModelType.require_by_id("pocket_local"): (PocketBaseModel, Tts.get_pocket, "_pocket"),
    TtsModelType.require_by_id("qwen3tts_local"): (Qwen3BaseModel, Tts.get_qwen3, "_qwen3"),
    TtsModelType.require_by_id("vibevoice_local"): (VibeVoiceBaseModel, Tts.get_vibevoice, "_vibevoice"),
}
