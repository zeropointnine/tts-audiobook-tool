"""Validated SGL-Omni request policies and catalog-owned settings declarations."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Never

from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelSpec, TtsModelType
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH, _load_catalog, parse_spec
from tts_audiobook_tool.tts_models.catalog_settings import parse_model_settings

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
    input_prompt_suffix: str = ""

    def validate(self, value: object) -> int | float:
        if isinstance(value, bool) or not isinstance(value, int if self.type == "int" else (int, float)):
            raise ValueError(f"{self.model_id}.{self.name} must be a {self.type}")
        if not math.isfinite(value) or not self.min <= value <= self.max:
            raise ValueError(f"{self.model_id}.{self.name} must be between {self.min} and {self.max}")
        return value


@dataclass(frozen=True)
class MenuControl:
    kind: str
    # Voice samples are implicitly routed; settings always declare a target.
    target_menu: str | None
    parameter: str = ""
    label: str = ""


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
    # Declares whether this model offers and uses a batch-size / concurrency
    # value. Storage ownership stays with the model-settings registry: a false
    # capability hides the setting and pins the effective value to one without
    # retiring the stored `orchestration` value.
    can_batch: bool = True
    settings: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SglOmniDefinitions:
    models: dict[str, SglOmniModelDefinition]
    fingerprint: str
    setting_groups: dict[str, tuple[str, ...]] | None = None


def load_definitions(path: Path = DEFINITION_PATH) -> SglOmniDefinitions:
    """Validate request policies and catalog storage uniformly for every ID."""
    document, specs, entries, fingerprint = _load_catalog(path)
    installed = TtsModelType._initial_specs
    missing = installed.keys() - {spec.id for spec in specs}
    if missing:
        _fail("models", f"missing built-in model definition(s): {', '.join(sorted(missing))}")
    for spec in specs:
        baseline = installed.get(spec.id)
        if baseline is None and spec.backend_kind is TtsBackendKind.SGL_OMNI:
            continue  # Additional server models need no Python declaration.
        if baseline is None or spec.backend_kind is not baseline.backend_kind:
            _fail(f"model {spec.id}", "built-in ID and backend must match the installed catalog")
        if spec.backend_kind is not TtsBackendKind.SGL_OMNI and spec != baseline:
            _fail(f"model {spec.id}", "local built-in metadata must match the installed catalog")

    models: dict[str, SglOmniModelDefinition] = {}
    used_file_tags = {spec.file_tag for spec in installed.values()}

    for index, entry in enumerate(entries):
        if entry["backend_kind"] != TtsBackendKind.SGL_OMNI.value:
            continue  # The shared catalog also contains audio.cpp server entries.
        id = _required(entry, "id", f"models[{index}]", str)
        obj = entry["sgl_omni"]  # Already checked as part of the complete catalog.
        where = f"model {id}"
        if not id or id == "none":
            _fail(where, "invalid model ID")
        if id in models:
            _fail(where, "duplicate model ID")
        builtin = installed.get(id)
        if builtin is not None and builtin.backend_kind is not TtsBackendKind.SGL_OMNI:
            _fail(where, f"cannot overlay the non-SGL-Omni built-in model {id}")
        is_overlay = builtin is not None

        behavior = _object(obj.get("behavior"), f"{where}.behavior", {"streaming", "seed", "voice_required", "transcript", "prompt_policy", "language", "music_and_trim", "orchestration", "can_batch"})
        streaming = _required(behavior, "streaming", f"{where}.behavior", bool)
        voice_required = _required(behavior, "voice_required", f"{where}.behavior", bool)
        seed = _required(behavior, "seed", f"{where}.behavior", str)
        transcript = _required(behavior, "transcript", f"{where}.behavior", str)
        if seed not in _SEED_POLICIES:
            _fail(f"{where}.behavior.seed", f"unsupported seed policy {seed!r}")
        if transcript not in _TRANSCRIPT_POLICIES:
            _fail(f"{where}.behavior.transcript", f"unsupported transcript policy {transcript!r}")
        settings = parse_model_settings(entry)
        storage = {setting["name"]: setting for setting in settings}
        if "file_name" not in storage:
            _fail(f"{where}.behavior.voice_required", "model has no voice storage")
        has_transcript = "transcript" in storage
        if (transcript != "omitted") != has_transcript:
            _fail(f"{where}.behavior.transcript", "must match catalog transcript storage")
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
        can_batch = behavior.get("can_batch", True)
        if not isinstance(can_batch, bool):
            _fail(f"{where}.behavior.can_batch", "expected boolean")

        spec = parse_spec(entry)
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
            param = _object(data, item_where, {"type", "request_key", "default", "default_sentinel", "min", "max", "omit_when_unset", "max_request_value", "input_prompt_suffix"})
            suffix = param.get("input_prompt_suffix", "")
            if not isinstance(suffix, str):
                _fail(f"{item_where}.input_prompt_suffix", "expected a string")
            typ = _required(param, "type", item_where, str)
            if typ not in _VALUE_TYPES:
                _fail(f"{item_where}.type", f"unsupported numeric type {typ!r}")
            binding = storage.get(name)
            if binding is None or binding["section"] != "parameters":
                _fail(item_where, f"no catalog {id}.{name} parameter binding")
            if binding["type"] != typ:
                _fail(f"{item_where}.type", "must match the catalog storage type")
            request_key = _required(param, "request_key", item_where, str)
            if request_key in _RESERVED or request_key in request_keys:
                _fail(f"{item_where}.request_key", "reserved or duplicate request key")
            request_keys.add(request_key)
            number_type = _VALUE_TYPES[typ]
            values = {key: _required(param, key, item_where, number_type) for key in ("default", "min", "max")}
            sentinel = param.get("default_sentinel")
            if sentinel is not None and (isinstance(sentinel, bool) or not isinstance(sentinel, (int, float)) or not math.isfinite(sentinel)):
                _fail(f"{item_where}.default_sentinel", "expected a finite number")
            if binding.get("sentinel") != sentinel:
                _fail(f"{item_where}.default_sentinel", "must match the catalog storage sentinel")
            omit = param.get("omit_when_unset", False)
            if not isinstance(omit, bool) or omit and sentinel is None:
                _fail(f"{item_where}.omit_when_unset", "requires a boolean and a sentinel")
            cap = param.get("max_request_value")
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, number_type) or not math.isfinite(cap)
                                    or not values["min"] <= cap <= values["max"]):
                _fail(f"{item_where}.max_request_value", "expected a value within the parameter bounds")
            p = NumericParameter(id, name, typ, request_key, values["default"], sentinel,
                                 values["min"], values["max"], omit, cap, suffix)
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
                if key in occupied or key in _RESERVED and not (key == "max_new_tokens" and field == "request_defaults" and id in ("moss_delay_sglomni", "moss_local_sglomni")):
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
        if seed == "resolved" and "seed" not in storage:
            _fail(f"{where}.behavior.seed", "requires a declared seed binding")

        menu_data = obj.get("menu")
        if not isinstance(menu_data, list):
            _fail(f"{where}.menu", "expected an array")
        controls = []
        for i, data in enumerate(menu_data):
            menu_where = f"{where}.menu[{i}]"
            control = _object(data, menu_where, {"kind", "parameter", "label", "target_menu"})
            if "kind" in control:
                kind = control["kind"]
                if kind == "voice_samples":
                    _object(control, menu_where, {"kind"})
                    controls.append(MenuControl(kind, None))
                    continue
                if kind != "seed":
                    _fail(menu_where, "unsupported menu control")
                _object(control, menu_where, {"kind", "target_menu"})
                if seed != "resolved":
                    _fail(menu_where, "seed control requires resolved seed")
            target_menu = _required(control, "target_menu", menu_where, str)
            if target_menu not in ("model", "voice"):
                _fail(f"{menu_where}.target_menu", "expected model or voice")
            if "kind" in control:
                controls.append(MenuControl("seed", target_menu))
            else:
                name = _required(control, "parameter", menu_where, str)
                if name not in parameters:
                    _fail(f"{menu_where}.parameter", f"undeclared parameter {name}")
                label = _required(control, "label", menu_where, str)
                if not label:
                    _fail(f"{menu_where}.label", "must not be empty")
                controls.append(MenuControl("parameter", target_menu, name, label))
        declared = [c.parameter for c in controls if c.kind == "parameter"]
        if (sum(c.kind == "voice_samples" for c in controls) != 1
                or sorted(declared) != sorted(parameters)
                or len(declared) != len(set(declared))
                or sum(c.kind == "seed" for c in controls) != (seed == "resolved")):
            _fail(f"{where}.menu", "expected one voice control and each parameter exactly once")
        models[id] = SglOmniModelDefinition(spec, parameters, tuple(controls), dict(defaults), dict(stream_defaults), transcript, seed, prompt_policy, language, music_and_trim, orchestration_name, can_batch, settings)
    required = {id for id, spec in installed.items() if spec.backend_kind is TtsBackendKind.SGL_OMNI}
    missing = required - models.keys()
    if missing:
        _fail("models", f"missing built-in SGL-Omni definition(s): {', '.join(sorted(missing))}")
    groups = {name: tuple(ids) for name, ids in document.get("setting_groups", {}).items()}
    return SglOmniDefinitions(models, fingerprint, groups)
