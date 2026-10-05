"""Backend-neutral ownership and reconciliation of model-scoped project settings.

The registry describes *permissions*, never project-authored sharing rules. Unknown
whole objects are retained so projects can travel between model environments.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from typing import TYPE_CHECKING, NamedTuple, Any, Mapping, cast

if TYPE_CHECKING:
    from tts_audiobook_tool.tts_models.sgl_omni_definition import SglOmniModelDefinition

from tts_audiobook_tool.app_support import path_norm


class SettingRef(NamedTuple):
    """Explicit runtime reference to a declared model-scoped setting.

    Replaces the old flat attribute-name strings in menus and save paths:
    a setting's identity is `(model ID, setting name)`, never a Python
    project field name. Interpretation (bounds, defaults, sentinels)
    belongs to the caller, not to the reference.
    """

    model_id: str
    name: str


@dataclass(frozen=True)
class Binding:
    model_id: str
    name: str
    section: str
    group: str = ""
    default: Any = None
    sentinel: Any = None
    has_sentinel: bool = False
    value_type: type = object
    list_item_type: type = object
    # Seeds are never "just the default": -1 means random and any other value
    # is a deliberate choice, so a legacy value equal to the declared default
    # still migrates as an explicit override.
    preserve_default: bool = False


@dataclass
class ModelSettings:
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    shared: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"models": deepcopy(self.models), "shared": deepcopy(self.shared)}


# Names retired from a section. The ownership check below cannot tell "another
# model owns this name" from "this model used to own it" -- names are global --
# so a retired value is dropped instead of failing the project.
# audio.cpp retired `orchestration.concurrent_requests`: its server serializes
# requests per model, so the setting never did anything.
RETIRED_SETTINGS: dict[str, frozenset[str]] = {
    "orchestration": frozenset({"concurrent_requests"}),
}
# Names used in the old file which differ from the actual Python field names.
LEGACY_ALIASES = {
    "fish_s1_voice_transcript": ("fish_s1_voice_text",),
    "glm_voice_transcript": ("glm_voice_text",),
    "higgs_voice_transcript": ("higgs_voice_text",),
    "vibevoice_lora_target": ("vibevoice_lora_path",),
    "omnivoice_num_step": ("omnivoice_steps",),
}


class ModelSettingsRegistry:
    def __init__(self) -> None:
        self.bindings: dict[tuple[str, str], Binding] = {}
        self.legacy: dict[str, Binding] = {}
        self.members: dict[str, tuple[str, ...]] = {}
        # Backend bounds supplement storage types only during load reconciliation.
        self.parameter_bounds: dict[tuple[str, str], tuple[int | float, int | float]] = {}

    def add(self, binding: Binding, legacy_attr: str = "") -> None:
        key = (binding.model_id, binding.name)
        if key in self.bindings:
            raise ValueError(f"Duplicate model setting {key}")
        if binding.group and binding.model_id not in self.members.get(binding.group, ()):
            raise ValueError(f"Unauthorized shared group {binding.group} for {binding.model_id}")
        self.bindings[key] = binding
        if legacy_attr:
            self.legacy[legacy_attr] = binding

    def for_model(self, model_id: str) -> list[Binding]:
        return [b for (id, _), b in self.bindings.items() if id == model_id]

    @staticmethod
    def _binding(model_id: str, setting: dict[str, Any]) -> Binding:
        kind = setting["type"]
        types = {"int": (int, object), "float": (float, object), "str": (str, object),
                 "bool": (bool, object), "list[str]": (list, str), "list[float]": (list, float)}
        if kind == "enum:chatterbox":
            from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
            value_type, item_type = ChatterboxType, object
            if ChatterboxType.get_by_id(setting["default"]) is None:
                raise ValueError(f"Invalid {model_id}.{setting['name']} enum default")
        else:
            value_type, item_type = types[kind]
        return Binding(model_id, setting["name"], setting["section"], setting.get("group", ""),
                       setting["default"], setting.get("sentinel", -1), "sentinel" in setting,
                       value_type, item_type, setting.get("preserve_default", False))

    def register_model_settings(self, model_id: str, settings: tuple[dict[str, Any], ...]) -> None:
        """Install a model's catalog storage identically for every backend/ID."""
        bindings = [self._binding(model_id, setting) for setting in settings]
        for binding in bindings:
            if binding.group and model_id not in self.members.get(binding.group, ()):
                raise ValueError(f"Unauthorized shared group {binding.group} for {model_id}")
        for key in list(self.bindings):
            if key[0] == model_id:
                del self.bindings[key]
        for binding in bindings:
            self.add(binding)

    def load_catalog_settings(self) -> None:
        from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH, _load_catalog
        from tts_audiobook_tool.tts_models.catalog_settings import parse_model_settings
        from tts_audiobook_tool.project_support.model_settings_compat import legacy_bindings
        document, _, _, _ = _load_catalog(CATALOG_PATH)
        pending = ModelSettingsRegistry()
        pending.members = {name: tuple(ids) for name, ids in document.get("setting_groups", {}).items()}
        for entry in document["models"]:
            model_id = entry["id"]
            pending.register_model_settings(model_id, parse_model_settings(entry))
            for name, parameter in entry.get(entry.get("backend_kind", ""), {}).get("parameters", {}).items():
                if parameter["type"] in ("int", "float"):
                    pending.parameter_bounds[(model_id, name)] = (parameter["min"], parameter["max"])
        # The local MOSS implementation uses the same architecture-specific UI
        # bounds as the catalog's server controls. Keep validating local saved
        # values after retiring their shared storage, without coupling overrides.
        for (model_id, name), bounds in tuple(pending.parameter_bounds.items()):
            if model_id in ("moss_delay_sglomni", "moss_local_sglomni") and ("moss_local", name) in pending.bindings:
                pending.parameter_bounds[("moss_local", name)] = bounds
        # The v1.5 Local Transformer preset has no server counterpart to copy
        # bounds from, so its local-only settings carry the preset's own.
        from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
        v15 = MossConfigs.LOCAL_V15.value
        pending.parameter_bounds.update({
            ("moss_local", "local_v15_temperature"): (v15.temperature_min, v15.temperature_max),
            ("moss_local", "local_v15_top_p"): (v15.audio_top_p_min, v15.audio_top_p_max),
            ("moss_local", "local_v15_top_k"): (v15.audio_top_k_min, v15.audio_top_k_max),
        })
        # Migration defaults/owners are frozen separately from current defaults.
        legacy = legacy_bindings()
        self.bindings, self.members, self.legacy = pending.bindings, pending.members, legacy
        self.parameter_bounds = pending.parameter_bounds

    def register_configured_model(self, definition: SglOmniModelDefinition) -> None:
        self.register_model_settings(definition.spec.id, definition.settings)
        self._register_parameter_bounds(definition)

    def register_audio_cpp_model(self, definition: Any) -> None:
        self.register_model_settings(definition.spec.id, definition.settings)
        self._register_parameter_bounds(definition)

    def _register_parameter_bounds(self, definition: Any) -> None:
        model_id = definition.spec.id
        self.parameter_bounds = {key: bounds for key, bounds in self.parameter_bounds.items() if key[0] != model_id}
        for name, parameter in definition.parameters.items():
            if parameter.type in ("int", "float"):
                self.parameter_bounds[(model_id, name)] = (parameter.min, parameter.max)

    def reset_to_catalog(self) -> None:
        """Restore all shipped catalog settings, not a historical model subset."""
        self.load_catalog_settings()

    def reset_to_builtins(self) -> None:
        """Compatibility spelling; there is no built-in-only registration path."""
        self.reset_to_catalog()

    def get(self, model_id: str, name: str) -> Binding:
        try:
            return self.bindings[(model_id, name)]
        except KeyError as exc:
            raise ValueError(f"Unknown setting {model_id}.{name}") from exc

    def _section_binding(self, model_id: str, name: str, section: str) -> Binding | None:
        binding = self.bindings.get((model_id, name))
        return binding if binding is not None and binding.section == section else None

    def voice_binding(self, model_id: str) -> Binding | None:
        """The model's voice-reference list storage, when it stores voice samples."""
        return self._section_binding(model_id, "file_name", "voice_references")

    def transcript_binding(self, model_id: str) -> Binding | None:
        """The model's voice transcript storage, when transcripts are used."""
        return self._section_binding(model_id, "transcript", "voice_references")

    def orchestration_binding(self, model_id: str) -> Binding | None:
        """The model's batch size / concurrent requests storage, when it has one."""
        for name in ("concurrent_requests", "batch_size"):
            binding = self._section_binding(model_id, name, "orchestration")
            if binding is not None:
                return binding
        return None

    def serialize(self, store: ModelSettings) -> dict[str, Any]:
        """Document all declared parameters without pinning unset defaults.

        Internal storage stays sparse. JSON null means use the current default;
        old numeric unset sentinels get the same portable spelling. Unknown
        model/group objects and non-parameter sections remain untouched.
        """
        result = store.to_dict()
        for binding in self.bindings.values():
            if binding.section != "parameters":
                continue
            if binding.group:
                obj = result["shared"].setdefault(binding.group, {"model_ids": list(self.members[binding.group])})
            else:
                obj = result["models"].setdefault(binding.model_id, {})
            parameters = obj.setdefault("parameters", {})
            value = parameters.get(binding.name)
            if binding.has_sentinel and value == binding.sentinel:
                value = None
            parameters[binding.name] = deepcopy(self._json_value(value))
        return result

    def resolve(self, store: ModelSettings, binding: Binding) -> Any:
        obj = store.shared.get(binding.group, {}) if binding.group else store.models.get(binding.model_id, {})
        if binding.section == "voice_references":
            return [ref.get(binding.name, "") for ref in obj.get("voice_references", [])]
        value = obj.get(binding.section, {}).get(binding.name, binding.default)
        if isinstance(binding.value_type, type) and issubclass(binding.value_type, Enum) and isinstance(value, str):
            resolver = getattr(binding.value_type, "get_by_id", None)
            resolved = resolver(value) if callable(resolver) else None
            if resolved is None:
                raise ValueError(f"{binding.model_id}.{binding.name}: unrecognized value {value!r}")
            return resolved
        return deepcopy(value)

    def assign(self, store: ModelSettings, binding: Binding, value: Any, *, reset: bool = False) -> None:
        obj = store.shared.setdefault(binding.group, {"model_ids": list(self.members[binding.group])}) if binding.group else store.models.setdefault(binding.model_id, {})
        if binding.section == "voice_references":
            if reset:
                value = []
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{binding.model_id}.{binding.name}: expected a list of strings")
            refs = obj.setdefault("voice_references", [])
            if binding.name == "transcript":
                # Transcripts only annotate existing references. A longer list
                # must not fabricate voice entries with an empty file name.
                for index, ref in enumerate(refs):
                    ref["transcript"] = value[index] if index < len(value) else ""
                return
            for index, item in enumerate(value):
                if index < len(refs):
                    refs[index]["file_name"] = item
                else:
                    refs.append({"file_name": item, "transcript": ""})
            del refs[len(value):]
            return
        section = obj.setdefault(binding.section, {})
        if reset:
            section.pop(binding.name, None)
        else:
            self.validate_value(binding, value)
            section[binding.name] = deepcopy(self._json_value(value))

    @staticmethod
    def validate_value(binding: Binding, value: Any) -> None:
        typ = binding.value_type
        if isinstance(typ, type) and issubclass(typ, Enum):
            resolver = getattr(typ, "get_by_id", None)
            valid = isinstance(value, typ) or isinstance(value, str) and callable(resolver) and resolver(value) is not None
        elif typ is float:
            valid = isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        elif typ is int:
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = typ is object or isinstance(value, typ)
        if valid and typ is list:
            item_type = binding.list_item_type
            valid = all(isinstance(item, item_type) and not isinstance(item, bool) for item in cast(list[Any], value))
        if not valid:
            raise ValueError(f"{binding.model_id}.{binding.name}: expected {typ.__name__}, got {value!r}")

    def reconcile(self, source: Any, legacy: Mapping[str, Any] | None = None) -> ModelSettings:
        if source is None:
            source = {}
        if isinstance(source, ModelSettings):
            source = source.to_dict()
        if not isinstance(source, dict):
            raise ValueError("model_settings must be an object")
        raw_models = source.get("models", {})
        raw_shared = source.get("shared", {})
        if not isinstance(raw_models, dict) or not isinstance(raw_shared, dict):
            raise ValueError("model_settings.models and model_settings.shared must be objects")
        # Fork retired ownership before current cleaning removes raw nulls and
        # empty lists. Remember the originally supplied private objects for the
        # existing whole-object precedence of flat private fields (e.g. target).
        provided_models = raw_models
        from tts_audiobook_tool.project_support.model_settings_compat import fork_moss_settings, split_moss_local_seed
        source, legacy = fork_moss_settings(source, legacy, self.legacy, self.bindings)
        source = split_moss_local_seed(source, self.bindings)
        raw_models, raw_shared = source.get("models", {}), source.get("shared", {})
        result = ModelSettings()
        known_models = {id for id, _ in self.bindings}
        for model_id, raw in raw_models.items():
            if not isinstance(raw, dict):
                raise ValueError(f"model_settings.models.{model_id} must be an object")
            if model_id not in known_models:
                result.models[model_id] = deepcopy(raw)
                continue
            cleaned = self._clean_object(raw, [b for b in self.for_model(model_id) if not b.group], f"models.{model_id}")
            result.models[model_id] = cleaned
        for group, raw in raw_shared.items():
            if not isinstance(raw, dict):
                raise ValueError(f"model_settings.shared.{group} must be an object")
            if group not in self.members:
                result.shared[group] = deepcopy(raw)
                continue
            members = self.members[group]
            recorded = raw.get("model_ids")
            if (not isinstance(recorded, list) or len(recorded) != len(members)
                    or not all(isinstance(id, str) for id in recorded)
                    or set(recorded) != set(members)):
                raise ValueError(f"model_settings.shared.{group}.model_ids: expected members {list(members)!r}")
            bindings = [b for b in self.bindings.values() if b.group == group]
            cleaned = self._clean_object(raw, bindings, f"shared.{group}")
            cleaned["model_ids"] = list(members)
            result.shared[group] = cleaned
        if legacy:
            for attr, value in legacy.items():
                binding = self.legacy.get(attr)
                if binding is None:
                    continue
                if binding.group and binding.group in raw_shared or not binding.group and binding.model_id in provided_models:
                    continue
                if binding.section == "voice_references":
                    continue
                # An old flat file records every field, including ones the
                # user never touched, so a value equal to the declared default
                # migrates to "absent" rather than an explicit override. Real
                # seed values (including -1 for random) are preserved because
                # they are never treated as defaults to drop.
                if not binding.preserve_default and self._json_value(value) == binding.default:
                    continue
                self.assign(result, binding, self._json_value(value))
            # Pair voice lists by position, including optional transcripts.
            for attr, binding in self.legacy.items():
                if binding.section != "voice_references" or binding.name != "file_name" or attr not in legacy:
                    continue
                if binding.group and binding.group in raw_shared or not binding.group and binding.model_id in provided_models:
                    continue
                voices = legacy[attr]
                voices = [voices] if isinstance(voices, str) and voices else voices
                if not isinstance(voices, list):
                    continue
                transcript_attr = attr.removesuffix("_file_name") + "_transcript"
                transcripts = legacy.get(transcript_attr)
                if transcripts is None:
                    transcripts = next((legacy[alias] for alias in LEGACY_ALIASES.get(transcript_attr, ()) if alias in legacy), [])
                transcripts = [transcripts] if isinstance(transcripts, str) and transcripts else transcripts
                if not isinstance(transcripts, list):
                    transcripts = []
                refs = [{"file_name": voice, "transcript": transcripts[i] if i < len(transcripts) else ""}
                        for i, voice in enumerate(voices) if isinstance(voice, str) and voice]
                if refs:
                    obj = result.shared.setdefault(binding.group, {"model_ids": list(self.members[binding.group])}) if binding.group else result.models.setdefault(binding.model_id, {})
                    obj["voice_references"] = refs
        return result

    @staticmethod
    def _is_retired_setting(section: str, name: str) -> bool:
        """Whether a section/name pair was retired and should be dropped on load."""
        return name in RETIRED_SETTINGS.get(section, ())

    def _validate_loaded_parameter(self, binding: Binding, value: Any) -> None:
        self.validate_value(binding, value)
        if binding.has_sentinel and value == binding.sentinel:
            return
        # Shared overrides must be valid for every declared backend consumer,
        # including when the local member has no backend parameter declaration.
        model_ids = self.members[binding.group] if binding.group else (binding.model_id,)
        for model_id in model_ids:
            bounds = self.parameter_bounds.get((model_id, binding.name))
            if bounds is not None and not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{model_id}.{binding.name}: expected a value between {bounds[0]} and {bounds[1]}")

    def _clean_object(self, raw: dict, bindings: list[Binding], where: str) -> dict[str, Any]:
        allowed: dict[str, dict[str, Binding]] = {}
        for binding in bindings:
            if binding.section != "voice_references":
                allowed.setdefault(binding.section, {})[binding.name] = binding
        known: dict[str, set[str]] = {}
        for binding in self.bindings.values():
            known.setdefault(binding.section, set()).add(binding.name)
        result: dict[str, Any] = {}
        if "voice_references" in raw and not any(b.section == "voice_references" for b in bindings):
            raise ValueError(f"model_settings.{where}.voice_references belongs to a different storage owner")
        for section, values in raw.items():
            if section in known and section not in ("voice_references", "parameters") and section not in allowed and isinstance(values, dict):
                unauthorized = {name for name in known[section].intersection(values)
                                if not self._is_retired_setting(section, name)}
                if unauthorized:
                    name = sorted(unauthorized)[0]
                    raise ValueError(f"model_settings.{where}.{section}.{name} belongs to a different storage owner")
        for section, fields in allowed.items():
            values = raw.get(section, {})
            if not isinstance(values, dict):
                if section == "parameters":
                    continue
                raise ValueError(f"model_settings.{where}.{section} must be an object")
            clean = {}
            for name, value in values.items():
                if name not in fields:
                    # Model definitions can retire parameters, and a whole
                    # declared setting can be retired for a family (see
                    # RETIRED_SETTINGS). A removed name might now belong to
                    # another model (e.g. a former custom 'speed' setting), so
                    # global name collisions are not evidence of invalid
                    # ownership in saved projects.
                    if (section != "parameters" and name in known.get(section, ())
                            and not self._is_retired_setting(section, name)):
                        raise ValueError(f"model_settings.{where}.{section}.{name} belongs to a different storage owner")
                    continue
                # JSON null intentionally follows the current default, just
                # like an omitted key in older sparse project files.
                if section == "parameters" and value is None:
                    continue
                try:
                    if section == "parameters":
                        self._validate_loaded_parameter(fields[name], value)
                    else:
                        self.validate_value(fields[name], value)
                except ValueError as exc:
                    if section == "parameters":
                        logging.getLogger("tts-audiobook-tool").warning(
                            "Invalid saved model setting model_settings.%s.%s.%s=%r; "
                            "resetting to default %r: %s",
                            where, section, name, value, fields[name].default, exc,
                        )
                        continue
                    raise
                clean[name] = deepcopy(value)
            if clean:
                result[section] = clean
        if any(b.section == "voice_references" for b in bindings):
            refs = raw.get("voice_references", [])
            if not isinstance(refs, list):
                raise ValueError(f"model_settings.{where}.voice_references must be an array")
            clean_refs = []
            for index, ref in enumerate(refs):
                if not isinstance(ref, dict) or not isinstance(ref.get("file_name"), str) or not isinstance(ref.get("transcript", ""), str):
                    raise ValueError(f"model_settings.{where}.voice_references[{index}]: expected file_name and optional transcript strings")
                clean_refs.append({"file_name": ref["file_name"], "transcript": ref.get("transcript", "")})
            if clean_refs:
                result["voice_references"] = clean_refs
        return result

    @staticmethod
    def _json_value(value: Any) -> Any:
        return getattr(value, "id") if isinstance(value, Enum) and hasattr(value, "id") else value


REGISTRY = ModelSettingsRegistry()


def register_builtin_fields() -> None:
    """Compatibility name for callers: initialize all shipped catalog settings."""
    if not REGISTRY.bindings:
        REGISTRY.load_catalog_settings()


register_builtin_fields()


def normalize_paths(store: ModelSettings, registry: ModelSettingsRegistry = REGISTRY) -> bool:
    """Canonicalize only declared file paths; never touch opaque repository IDs."""
    changed = False
    for collection, known in ((store.models, {id for id, _ in registry.bindings}),
                              (store.shared, set(registry.members))):
        for key, obj in collection.items():
            if key not in known:
                continue
            for ref in obj.get("voice_references", []):
                ref["file_name"], did_change = path_norm.normalize_stored_relative_path(ref["file_name"])
                changed |= did_change
            for name, value in obj.get("files", {}).items():
                if isinstance(value, str):
                    obj["files"][name], did_change = path_norm.normalize_stored_relative_path(value)
                    changed |= did_change
    return changed
