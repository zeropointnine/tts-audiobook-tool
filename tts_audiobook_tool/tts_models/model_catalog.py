"""Load the versioned, application-shipped model catalog without runtime dependencies.

This module parses model metadata at import time, before local model classes
capture their INFO specs. SGL-Omni request/storage validation happens later in
sgl_omni_definition.py, after the project settings registry is available.
"""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
import re
from typing import Any

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind, TtsModelSpec

CATALOG_PATH = Path(__file__).with_name("model_catalog.toml")
CATALOG_SCHEMA_VERSION = 3
_LOCAL_SPEC_KEYS = {
    "local_module_test", "local_torch_devices", "file_tag", "default_output_sample_rate",
    "requires_voice", "can_stream", "requires_ffmpeg_libs", "un_all_caps",
    "requirements_file_name", "ui", "output_filters", "substitutions",
}
_SERVER_SPEC_KEYS = {"file_tag", "default_output_sample_rate", "un_all_caps", "requirements_file_name", "ui", "substitutions"}
_UI_KEYS = {"proper_name", "short_name", "voice_path_console", "voice_path_requestor", "project_links"}


def _fail(where: str, reason: str) -> None:
    raise ValueError(f"Model catalog {where}: {reason}")


def _object(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "expected an object")
    extra = set(value) - keys
    if extra:
        _fail(where, f"unsupported field(s): {', '.join(sorted(extra))}")
    return value


def _field(obj: dict, key: str, where: str, typ: type) -> Any:
    value = obj.get(key)
    if type(value) is not typ:
        _fail(f"{where}.{key}", f"expected {typ.__name__}")
    return value


def parse_spec(entry: dict[str, Any]) -> TtsModelSpec:
    """Derive one complete spec from its sole declaration in the catalog."""
    id = entry["id"]
    where = f"model {id}"
    # TOML has no null: only the none placeholder omits its backend kind.
    if "backend_kind" not in entry and id != "none":
        _fail(f"{where}.backend_kind", "required for real models")
    backend = entry.get("backend_kind")
    if backend not in (None, TtsBackendKind.LOCAL.value, TtsBackendKind.SGL_OMNI.value):
        _fail(f"{where}.backend_kind", "expected local or sgl_omni (or omitted for none)")
    if id == "none" and backend is not None:
        _fail(f"{where}.backend_kind", "the none placeholder cannot have a backend")
    is_server = backend == TtsBackendKind.SGL_OMNI.value
    meta = _object(entry.get("spec"), f"{where}.spec", _SERVER_SPEC_KEYS if is_server else _LOCAL_SPEC_KEYS)
    file_tag = _field(meta, "file_tag", f"{where}.spec", str)
    rate = _field(meta, "default_output_sample_rate", f"{where}.spec", int)
    if id == "none":
        if file_tag or rate != 0:
            _fail(f"{where}.spec", "the none placeholder requires an empty file tag and zero sample rate")
    elif not file_tag or rate <= 0:
        _fail(f"{where}.spec", "real models require a file tag and positive sample rate")
    caps = _field(meta, "un_all_caps", f"{where}.spec", bool)
    requirements = _field(meta, "requirements_file_name", f"{where}.spec", str)
    if not requirements:
        _fail(f"{where}.spec.requirements_file_name", "must not be empty")
    ui = _object(meta.get("ui"), f"{where}.spec.ui", _UI_KEYS | ({"opt_in_url"} if not is_server else set()))
    for key in ("proper_name", "short_name", "voice_path_console", "voice_path_requestor"):
        _field(ui, key, f"{where}.spec.ui", str)
    links = _field(ui, "project_links", f"{where}.spec.ui", list)
    if not all(isinstance(link, str) for link in links):
        _fail(f"{where}.spec.ui.project_links", "expected strings")
    if "opt_in_url" in ui:
        _field(ui, "opt_in_url", f"{where}.spec.ui", str)
    replacements = _field(meta, "substitutions", f"{where}.spec", list)
    if not all(isinstance(pair, list) and len(pair) == 2 and all(isinstance(s, str) for s in pair) for pair in replacements):
        _fail(f"{where}.spec.substitutions", "expected [before, after] string pairs")
    if is_server:
        match = _object(entry.get("match"), f"{where}.match", {"model_id_substring"})
        substring = _field(match, "model_id_substring", f"{where}.match", str)
        if not substring.strip():
            _fail(f"{where}.match.model_id_substring", "must not be empty")
        behavior = _object(entry.get("behavior"), f"{where}.behavior",
                           {"streaming", "seed", "voice_required", "transcript", "prompt_policy", "language", "music_and_trim", "orchestration"})
        streaming = _field(behavior, "streaming", f"{where}.behavior", bool)
        voice_required = _field(behavior, "voice_required", f"{where}.behavior", bool)
        devices: list[DeviceType] = []
        module = ""
        requires_ffmpeg = False
        filters: list[str] = []
    else:
        if id != "none" and backend != TtsBackendKind.LOCAL.value:
            _fail(f"{where}.backend_kind", "only the none placeholder can lack a backend")
        if "match" in entry or "behavior" in entry:
            _fail(where, "local and placeholder models cannot have server behavior")
        substring = ""
        module = _field(meta, "local_module_test", f"{where}.spec", str)
        raw_devices = _field(meta, "local_torch_devices", f"{where}.spec", list)
        devices = []
        try:
            devices = [DeviceType(device) for device in raw_devices if isinstance(device, str)]
        except ValueError:
            _fail(f"{where}.spec.local_torch_devices", "unknown device")
        if len(devices) != len(raw_devices):
            _fail(f"{where}.spec.local_torch_devices", "expected device strings")
        voice_required = _field(meta, "requires_voice", f"{where}.spec", bool)
        streaming = _field(meta, "can_stream", f"{where}.spec", bool)
        requires_ffmpeg = _field(meta, "requires_ffmpeg_libs", f"{where}.spec", bool)
        filters = _field(meta, "output_filters", f"{where}.spec", list)
        if not all(isinstance(s, str) for s in filters):
            _fail(f"{where}.spec.output_filters", "expected strings")
        if id == "none":
            if module or devices or voice_required or streaming or requires_ffmpeg:
                _fail(f"{where}.spec", "the none placeholder cannot have local inference settings")
        elif not module.strip():
            _fail(f"{where}.spec.local_module_test", "local models require a module probe")
    return TtsModelSpec(id, TtsBackendKind(backend) if backend else None, substring,
                        module, devices, file_tag, rate, voice_required, streaming,
                        requires_ffmpeg, caps, requirements, ui, filters,
                        [tuple(pair) for pair in replacements])


def load_catalog(path: Path = CATALOG_PATH) -> tuple[list[tuple[str, TtsModelSpec]], list[dict[str, Any]], str]:
    """Return ordered built-in handles, server entries, and a full-file fingerprint."""
    try:
        raw = path.read_bytes()
        document = tomllib.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Model catalog {path}: {exc}") from exc
    root = _object(document, "root", {"schema_version", "models"})
    if type(root.get("schema_version")) is not int or root["schema_version"] != CATALOG_SCHEMA_VERSION:
        _fail("schema_version", f"expected supported version {CATALOG_SCHEMA_VERSION}")
    entries = _field(root, "models", "root", list)
    builtins: list[tuple[str, TtsModelSpec]] = []
    servers: list[dict[str, Any]] = []
    ids: set[str] = set()
    symbols: set[str] = set()
    for index, entry in enumerate(entries):
        item = _object(entry, f"models[{index}]", {"id", "symbol", "backend_kind", "spec", "match", "behavior", "parameters", "request_defaults", "request_stream_defaults", "menu"})
        id = _field(item, "id", f"models[{index}]", str)
        if not id or not re.fullmatch(r"[a-z][a-z0-9_]*", id) or id in ids:
            _fail(f"models[{index}].id", "expected a unique, lowercase model ID")
        ids.add(id)
        spec = parse_spec(item)
        symbol = item.get("symbol")
        if symbol is not None:
            if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", symbol) or symbol in symbols:
                _fail(f"model {id}.symbol", "expected a unique uppercase handle name")
            symbols.add(symbol)
            builtins.append((symbol, spec))
        elif spec.backend_kind is not TtsBackendKind.SGL_OMNI:
            _fail(f"model {id}.symbol", "local models and the placeholder must have a built-in handle")
        if spec.backend_kind is TtsBackendKind.SGL_OMNI:
            servers.append(item)
        elif set(item) & {"parameters", "request_defaults", "request_stream_defaults", "menu"}:
            _fail(f"model {id}", "local models cannot have server request definitions")
    if not any(symbol == "NONE" and spec.id == "none" for symbol, spec in builtins):
        _fail("models", "missing built-in none placeholder")
    return builtins, servers, hashlib.sha256(raw).hexdigest()
