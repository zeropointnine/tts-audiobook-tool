"""Versioned SGL-Omni definitions. No project persistence is defined here."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Never

from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelSpec, TtsModelType
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH, load_catalog, parse_spec

DEFINITION_PATH = CATALOG_PATH
_RESERVED = {"input", "stream", "references", "seed", "language", "max_new_tokens", "stage_params"}
_VALUE_TYPES = {"int": int, "float": float}
_SEED_POLICIES = {"unsupported", "resolved"}
_TRANSCRIPT_POLICIES = {"required_when_voice_present", "optional", "omitted"}
_PROMPT_POLICIES = {"zonos_tokens", "auk_seconds"}
def _fail(where: str, message: str) -> Never:
    raise ValueError(f"SGL-Omni definition {where}: {message}")


def _object(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "expected an object")
    unknown = set(value) - keys
    if unknown:
        _fail(where, f"unsupported field(s): {', '.join(sorted(unknown))}")
    return value


def _required(obj: dict, key: str, where: str, typ: type) -> Any:
    value = obj.get(key)
    if not isinstance(value, typ) or (typ in (int, float) and isinstance(value, bool)):
        _fail(f"{where}.{key}", f"expected {typ.__name__}")
    return value


@dataclass(frozen=True)
class NumericParameter:
    model_id: str
    name: str
    type: str
    request_key: str
    default: int | float
    default_sentinel: int | float | None
    min: int | float
    max: int | float
    omit_when_unset: bool = False
    max_request_value: int | float | None = None

    def validate(self, value: object) -> int | float:
        if isinstance(value, bool) or not isinstance(value, int if self.type == "int" else (int, float)):
            raise ValueError(f"{self.model_id}.{self.name} must be a {self.type}")
        if not math.isfinite(value) or not self.min <= value <= self.max:
            raise ValueError(f"{self.model_id}.{self.name} must be between {self.min} and {self.max}")
        return value


@dataclass(frozen=True)
class MenuControl:
    kind: str
    parameter: str = ""
    label: str = ""
    group: str = ""


@dataclass(frozen=True)
class SglOmniModelDefinition:
    spec: TtsModelSpec
    parameters: dict[str, NumericParameter]
    menu: tuple[MenuControl, ...]
    request_defaults: dict[str, int | float | str]
    request_stream_defaults: dict[str, int | float | str]
    transcript_policy: str
    seed_policy: str
    prompt_policy: str = ""
    language_policy: str = ""
    music_and_trim: bool = False
    orchestration_name: str = ""


@dataclass(frozen=True)
class SglOmniDefinitions:
    models: dict[str, SglOmniModelDefinition]
    fingerprint: str


def load_definitions(path: Path = DEFINITION_PATH) -> SglOmniDefinitions:
    """
    Validate the whole file before any caller installs an overlay.

    A definition either overlays an existing SGL-Omni built-in (keeping its
    stable handle and existing registry ownership) or introduces a new model
    ID with private model-scoped storage. No Python project field is added.
    """
    builtins, entries, fingerprint = load_catalog(path)
    installed = TtsModelType._builtin_specs
    missing = installed.keys() - {spec.id for _, spec in builtins}
    if missing:
        _fail("models", f"missing built-in model definition(s): {', '.join(sorted(missing))}")
    for symbol, spec in builtins:
        handle = getattr(TtsModelType, symbol, None)
        baseline = installed.get(spec.id)
        if handle is None or handle.id != spec.id or baseline is None or spec.backend_kind is not baseline.backend_kind:
            _fail(f"model {spec.id}", "built-in ID, symbol and backend must match the installed catalog")
        if spec.backend_kind is not TtsBackendKind.SGL_OMNI and spec != baseline:
            _fail(f"model {spec.id}", "local built-in metadata must match the installed catalog")

    from tts_audiobook_tool.project_support.model_settings import REGISTRY

    models: dict[str, SglOmniModelDefinition] = {}
    used_file_tags = {spec.file_tag for spec in TtsModelType._builtin_specs.values()}

    for index, entry in enumerate(entries):
        obj = entry  # Already validated as part of the complete v3 catalog.
        id = _required(obj, "id", f"models[{index}]", str)
        where = f"model {id}"
        if not id or id == "none":
            _fail(where, "invalid model ID")
        if id in models:
            _fail(where, "duplicate model ID")
        builtin = TtsModelType._builtin_specs.get(id)
        if builtin is not None and builtin.backend_kind is not TtsBackendKind.SGL_OMNI:
            _fail(where, f"cannot overlay the non-SGL-Omni built-in model {id}")
        is_overlay = builtin is not None

        behavior = _object(obj.get("behavior"), f"{where}.behavior", {"streaming", "seed", "voice_required", "transcript", "prompt_policy", "language", "music_and_trim", "orchestration"})
        streaming = _required(behavior, "streaming", f"{where}.behavior", bool)
        voice_required = _required(behavior, "voice_required", f"{where}.behavior", bool)
        seed = _required(behavior, "seed", f"{where}.behavior", str)
        transcript = _required(behavior, "transcript", f"{where}.behavior", str)
        if seed not in _SEED_POLICIES:
            _fail(f"{where}.behavior.seed", f"unsupported seed policy {seed!r}")
        if transcript not in _TRANSCRIPT_POLICIES:
            _fail(f"{where}.behavior.transcript", f"unsupported transcript policy {transcript!r}")
        if is_overlay:
            if "orchestration" in behavior:
                _fail(f"{where}.behavior.orchestration", "built-in storage is declared by the registry")
            if REGISTRY.voice_binding(id) is None:
                _fail(f"{where}.behavior.voice_required", "built-in model has no voice storage")
            has_transcript = REGISTRY.transcript_binding(id) is not None
            if (transcript != "omitted") != has_transcript:
                _fail(f"{where}.behavior.transcript", "must match the built-in transcript binding")
            orchestration_name = ""
        else:
            orchestration_name = behavior.get("orchestration", "")
            if orchestration_name not in ("", "batch_size", "concurrent_requests"):
                _fail(f"{where}.behavior.orchestration", "expected batch_size or concurrent_requests")
        prompt_policy = behavior.get("prompt_policy", "")
        if prompt_policy not in ("", *_PROMPT_POLICIES):
            _fail(f"{where}.behavior.prompt_policy", f"unsupported prompt policy {prompt_policy!r}")
        if prompt_policy == "auk_seconds" and (not voice_required or transcript != "required_when_voice_present" or streaming):
            _fail(f"{where}.behavior.prompt_policy", "AuK duration requires voice, transcript and nonstreaming")
        language = behavior.get("language", "")
        if language not in ("", "moss"):
            _fail(f"{where}.behavior.language", f"unsupported language policy {language!r}")
        music_and_trim = behavior.get("music_and_trim", False)
        if not isinstance(music_and_trim, bool):
            _fail(f"{where}.behavior.music_and_trim", "expected boolean")

        spec = parse_spec(obj)
        file_tag = spec.file_tag
        if is_overlay:
            if file_tag != builtin.file_tag:
                _fail(f"{where}.spec.file_tag", f"expected existing {id} file tag {builtin.file_tag!r}, got {file_tag!r}")
        elif file_tag in used_file_tags:
            _fail(f"{where}.spec.file_tag", f"file tag {file_tag!r} is already used")
        used_file_tags.add(file_tag)

        params_obj = obj.get("parameters", {})
        if not isinstance(params_obj, dict):
            _fail(f"{where}.parameters", "expected an object")
        parameters: dict[str, NumericParameter] = {}
        request_keys: set[str] = {"input", "stream", "references", "seed", "language", "stage_params"}
        for name, data in params_obj.items():
            item_where = f"{where}.parameters.{name}"
            if not name or not isinstance(name, str):
                _fail(item_where, "parameter name must be a nonempty string")
            if name in {"file_name", "transcript", "batch_size", "concurrent_requests", "seed"}:
                _fail(item_where, "reserved model setting name")
            param = _object(data, item_where, {"type", "request_key", "default", "default_sentinel", "min", "max", "omit_when_unset", "max_request_value"})
            typ = _required(param, "type", item_where, str)
            if typ not in _VALUE_TYPES:
                _fail(f"{item_where}.type", f"unsupported numeric type {typ!r}")
            if is_overlay:
                binding = REGISTRY.bindings.get((id, name))
                if (binding is None or binding.section != "parameters"
                        or binding.model_id != id
                        or binding.group and id not in REGISTRY.members.get(binding.group, ())):
                    _fail(item_where, f"no authorized {id}.{name} parameter binding")
                if binding.value_type is not _VALUE_TYPES[typ]:
                    _fail(f"{item_where}.type", "must match the existing storage type")
            request_key = _required(param, "request_key", item_where, str)
            if request_key in _RESERVED or request_key in request_keys:
                _fail(f"{item_where}.request_key", "reserved or duplicate request key")
            request_keys.add(request_key)
            number_type = _VALUE_TYPES[typ]
            values = {key: _required(param, key, item_where, number_type) for key in ("default", "min", "max")}
            sentinel = param.get("default_sentinel")
            if sentinel is not None and (isinstance(sentinel, bool) or not isinstance(sentinel, (int, float)) or not math.isfinite(sentinel)):
                _fail(f"{item_where}.default_sentinel", "expected a finite number")
            if is_overlay:
                binding = REGISTRY.get(id, name)
                if (binding.has_sentinel and sentinel != binding.sentinel
                        or not binding.has_sentinel and sentinel is not None):
                    _fail(f"{item_where}.default_sentinel", "must preserve the existing storage sentinel")
            omit = param.get("omit_when_unset", False)
            if not isinstance(omit, bool) or omit and sentinel is None:
                _fail(f"{item_where}.omit_when_unset", "requires a boolean and a sentinel")
            cap = param.get("max_request_value")
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, number_type) or not math.isfinite(cap)
                                    or not values["min"] <= cap <= values["max"]):
                _fail(f"{item_where}.max_request_value", "expected a value within the parameter bounds")
            p = NumericParameter(id, name, typ, request_key, values["default"], sentinel,
                                 values["min"], values["max"], omit, cap)
            if p.min > p.max or sentinel is not None and p.min <= sentinel <= p.max:
                _fail(item_where, "bounds must be ordered and exclude the sentinel")
            try:
                p.validate(p.default)
            except ValueError as exc:
                _fail(f"{item_where}.default", str(exc))
            parameters[name] = p

        def parse_defaults(field: str, occupied: set[str]) -> dict[str, int | float | str]:
            source = obj.get(field, {})
            if not isinstance(source, dict):
                _fail(f"{where}.{field}", "expected an object")
            result: dict[str, int | float | str] = {}
            for key, value in source.items():
                if not isinstance(key, str) or not key:
                    _fail(f"{where}.{field}", "keys must be non-empty strings")
                if key in occupied or key in _RESERVED and not (key == "max_new_tokens" and field == "request_defaults" and id in ("server_moss_delay", "server_moss_local")):
                    _fail(f"{where}.{field}.{key}", "reserved or duplicate request key")
                if isinstance(value, bool) or not isinstance(value, (int, float, str)) or isinstance(value, (int, float)) and not math.isfinite(value):
                    _fail(f"{where}.{field}.{key}", "expected a finite number or string")
                occupied.add(key)
                result[key] = value
            return result

        defaults = parse_defaults("request_defaults", request_keys)
        stream_defaults = parse_defaults("request_stream_defaults", request_keys)
        if prompt_policy == "auk_seconds" and "speed" not in parameters:
            _fail(f"{where}.behavior.prompt_policy", "requires a speed parameter")
        if prompt_policy == "zonos_tokens" and "max_new_tokens" in defaults:
            _fail(f"{where}.request_defaults.max_new_tokens", "conflicts with prompt policy")
        if seed == "resolved" and (id, "seed") not in REGISTRY.bindings:
            _fail(f"{where}.behavior.seed", "requires a declared seed binding")

        menu_data = obj.get("menu")
        if not isinstance(menu_data, list):
            _fail(f"{where}.menu", "expected an array")
        controls = []
        for i, data in enumerate(menu_data):
            menu_where = f"{where}.menu[{i}]"
            control = _object(data, menu_where, {"kind", "parameter", "label", "group"})
            if "kind" in control:
                kind = control["kind"]
                if control != {"kind": kind} or kind not in ("voice_samples", "seed"):
                    _fail(menu_where, "unsupported menu control")
                if kind == "seed" and seed != "resolved":
                    _fail(menu_where, "seed control requires resolved seed")
                controls.append(MenuControl(kind))
            else:
                name = _required(control, "parameter", menu_where, str)
                if name not in parameters:
                    _fail(f"{menu_where}.parameter", f"undeclared parameter {name}")
                label = _required(control, "label", menu_where, str)
                if not label:
                    _fail(f"{menu_where}.label", "must not be empty")
                group = control.get("group", "")
                if group not in ("", "advanced"):
                    _fail(f"{menu_where}.group", "expected advanced or empty")
                controls.append(MenuControl("parameter", name, label, group))
        declared = [c.parameter for c in controls if c.kind == "parameter"]
        if (sum(c.kind == "voice_samples" for c in controls) != 1
                or sorted(declared) != sorted(parameters)
                or len(declared) != len(set(declared))
                or sum(c.kind == "seed" for c in controls) != (seed == "resolved")):
            _fail(f"{where}.menu", "expected one voice control and each parameter exactly once")
        models[id] = SglOmniModelDefinition(spec, parameters, tuple(controls), dict(defaults), dict(stream_defaults), transcript, seed, prompt_policy, language, music_and_trim, orchestration_name)
    required = {id for id, spec in TtsModelType._builtin_specs.items() if spec.backend_kind is TtsBackendKind.SGL_OMNI}
    missing = required - models.keys()
    if missing:
        _fail("models", f"missing built-in SGL-Omni definition(s): {', '.join(sorted(missing))}")
    return SglOmniDefinitions(models, fingerprint)
