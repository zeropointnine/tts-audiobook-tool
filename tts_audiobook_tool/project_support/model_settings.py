"""Backend-neutral ownership and reconciliation of model-scoped project settings.

The registry describes *permissions*, never project-authored sharing rules. Unknown
whole objects are retained so projects can travel between model environments.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import TYPE_CHECKING, NamedTuple, Any, Mapping, cast

from tts_audiobook_tool.project_support.model_settings_declarations import BUILTIN_LEGACY_FIELDS

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


# An explicit list of existing shared ownership; equal names alone do not share.
SHARED_MEMBERS: dict[str, tuple[str, ...]] = {
    "auk": ("server_auk", "server_auk_flash"),
    "fish_s2": ("fish_s2", "server_fish_s2"),
    "moss": ("moss", "server_moss_delay", "server_moss_local"),
    "qwen3": ("qwen3tts", "server_qwen3tts"),
}
PREFIX_MODELS: dict[str, str] = {
    "auk": "server_auk", "chatterbox": "chatterbox", "dots": "dots",
    "fish_s1": "fish_s1", "fish_s2": "fish_s2", "higgs": "higgs_v2",
    "higgs_v3": "server_higgs_v3", "vibevoice": "vibevoice",
    "indextts2": "indextts2", "glm": "glm", "mira": "mira",
    "moss": "moss", "qwen3": "qwen3tts", "zonos2": "server_zonos2",
    "pocket": "pocket", "omnivoice": "omnivoice",
}
PRIVATE_FIELDS: dict[str, set[str]] = {
    "fish_s2": {"rolling_cont", "compile_enabled", "server_concurrent_requests"},
    "moss": {"target", "rolling_cont"},
    "qwen3": {"target", "model_type", "rolling_cont", "speaker_id", "instructions", "batch_size", "server_concurrent_requests"},
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
        self.members: dict[str, tuple[str, ...]] = dict(SHARED_MEMBERS)

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

    def register_configured_model(self, definition: SglOmniModelDefinition) -> None:
        """Accept configured IDs without manufacturing Python project fields.

        Built-in overlays retain their existing storage ownership and defaults;
        variant-specific interpretation belongs to the consumer definition.
        A new ID gets fresh private bindings named after its parameters.
        """
        model_id = definition.spec.id
        from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
        is_overlay = model_id in TtsModelType._builtin_specs
        for parameter in definition.parameters.values():
            key = (model_id, parameter.name)
            value_type = int if parameter.type == "int" else float
            if key in self.bindings:
                old = self.bindings[key]
                if old.section != "parameters" or old.value_type is not value_type:
                    raise ValueError(f"Configured {model_id}.{parameter.name} conflicts with existing storage ownership")
                # A consumer definition cannot rewrite a shared storage default.
                continue
            if is_overlay:
                raise ValueError(f"Configured {model_id}.{parameter.name} has no built-in storage binding")
            self.add(Binding(model_id, parameter.name, "parameters", "", parameter.default,
                             parameter.default_sentinel, parameter.default_sentinel is not None, value_type))
        if is_overlay:
            # Built-in voice, transcript and orchestration declarations remain
            # authoritative, regardless of which server definitions are loaded.
            return
        self.add(Binding(model_id, "file_name", "voice_references", "", [], value_type=list, list_item_type=str))
        if definition.transcript_policy != "omitted":
            self.add(Binding(model_id, "transcript", "voice_references", "", [], value_type=list, list_item_type=str))
        if definition.orchestration_name:
            self.add(Binding(model_id, definition.orchestration_name, "orchestration", "", default=1, value_type=int))

    def reset_to_builtins(self) -> None:
        """Forget optional definitions on backend/flag reinitialization."""
        self.bindings.clear()
        self.legacy.clear()
        self.members = dict(SHARED_MEMBERS)
        if self is REGISTRY:
            register_builtin_fields()

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
                if binding.group and binding.group in raw_shared or not binding.group and binding.model_id in raw_models:
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
                if binding.group and binding.group in raw_shared or not binding.group and binding.model_id in raw_models:
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
                unauthorized = known[section].intersection(values)
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
                    # Model definitions can retire parameters. A removed name
                    # might now belong to another model (e.g. a former custom
                    # 'speed' setting), so global name collisions are not
                    # evidence of invalid ownership in saved projects.
                    if section != "parameters" and name in known.get(section, ()):
                        raise ValueError(f"model_settings.{where}.{section}.{name} belongs to a different storage owner")
                    continue
                try:
                    self.validate_value(fields[name], value)
                except ValueError:
                    if section == "parameters":
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
    """Register all variants once, regardless of which backend is installed."""
    if REGISTRY.legacy:
        return
    value_types = {"int": (int, object), "float": (float, object),
                   "str": (str, object), "bool": (bool, object),
                   "list[str]": (list, str), "list[float]": (list, float)}
    for prefix, model_id in PREFIX_MODELS.items():
        for attr, (kind, default) in BUILTIN_LEGACY_FIELDS.items():
            if not attr.startswith(prefix + "_") or any(attr.startswith(other + "_") for other in PREFIX_MODELS if other.startswith(prefix + "_")):
                continue
            suffix = attr[len(prefix) + 1:]
            if suffix.startswith("server_"):
                target = {"fish_s2": "server_fish_s2", "qwen3": "server_qwen3tts"}.get(prefix, model_id)
                name = "concurrent_requests" if suffix == "server_concurrent_requests" else suffix
            else:
                target = model_id
                name = suffix
            group = prefix if prefix in SHARED_MEMBERS and suffix not in PRIVATE_FIELDS.get(prefix, set()) else ""
            if suffix in ("voice_file_name", "server_voice_file_name"):
                section, name = "voice_references", "file_name"
            elif suffix == "voice_transcript":
                section, name = "voice_references", "transcript"
            elif suffix in ("server_concurrent_requests", "batch_size"):
                section, name = "orchestration", "concurrent_requests" if suffix == "server_concurrent_requests" else "batch_size"
            elif suffix.endswith("_file_name"):
                section, name = "files", suffix.removesuffix("_file_name")
            else:
                section = "parameters"
            if kind == "enum:chatterbox":
                from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
                value_type, item_type = ChatterboxType, object
            else:
                value_type, item_type = value_types[kind]
            has_sentinel = default == -1 and suffix != "seed"
            binding = Binding(target, name, section, group, default, -1, has_sentinel,
                              value_type, item_type, preserve_default=(suffix == "seed"))
            REGISTRY.add(binding, attr)
            if group:
                for member in SHARED_MEMBERS[group]:
                    if member != target:
                        REGISTRY.add(Binding(member, name, section, group, default, -1, has_sentinel,
                                             value_type, item_type, preserve_default=(suffix == "seed")))
    if set(REGISTRY.legacy) != set(BUILTIN_LEGACY_FIELDS):
        raise ValueError(f"Unbound built-in fields: {set(BUILTIN_LEGACY_FIELDS) - set(REGISTRY.legacy)}")
    # Explicit aliases are input-only; the canonical name still owns the value.


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
