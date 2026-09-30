"""Catalog-owned persisted settings, independent of inference and legacy fields."""
from __future__ import annotations

import math
from typing import Any, Never

_TYPES = {"int", "float", "str", "bool", "list[str]", "list[float]", "enum:chatterbox"}
_SECTIONS = {"parameters", "voice_references", "orchestration", "files"}
_KEYS = {"name", "section", "type", "default", "group", "sentinel", "preserve_default"}


def _fail(where: str, message: str) -> Never:
    raise ValueError(f"Model catalog {where}: {message}")


def _valid(value: Any, typ: str) -> bool:
    if typ == "float":
        return type(value) in (int, float) and math.isfinite(value)
    if typ == "int":
        return type(value) is int
    if typ == "bool":
        return type(value) is bool
    if typ in ("str", "enum:chatterbox"):
        return isinstance(value, str)
    if typ.startswith("list["):
        item_type = typ[5:-1]
        return isinstance(value, list) and all(_valid(item, item_type) for item in value)
    return False


def parse_model_settings(entry: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Derive backend-standard storage, then apply explicit catalog declarations.

    Explicit storage can preserve shared defaults/sentinels and settings not sent
    to a server. The same rule applies to every ID, with or without a Python alias.
    """
    settings: dict[str, dict[str, Any]] = {}
    backend = entry.get("backend_kind")
    if backend in ("sgl_omni", "audio_cpp"):
        group = entry[backend]
        parameters = group.get("parameters", {})
        if not isinstance(parameters, dict):
            _fail(f"model {entry['id']}.{backend}.parameters", "expected an object")
        for name, param in parameters.items():
            where = f"model {entry['id']}.{backend}.parameters.{name}"
            if not isinstance(param, dict) or param.get("type") not in ("int", "float", "str"):
                _fail(where, "expected a numeric or string parameter declaration")
            if "default" not in param or not _valid(param["default"], param["type"]):
                _fail(f"{where}.default", f"expected {param['type']}")
            settings[name] = {"name": name, "section": "parameters", "type": param["type"],
                              "default": param["default"]}
            if "default_sentinel" in param:
                settings[name]["sentinel"] = param["default_sentinel"]
        settings["file_name"] = {"name": "file_name", "section": "voice_references", "type": "list[str]", "default": []}
        behavior = group.get("behavior", {})
        transcript = (behavior.get("transcript") != "omitted" if backend == "sgl_omni"
                      else group.get("reference_transcript", False))
        if transcript:
            settings["transcript"] = {"name": "transcript", "section": "voice_references", "type": "list[str]", "default": []}
        if backend == "audio_cpp" or behavior.get("seed") == "resolved":
            settings["seed"] = {"name": "seed", "section": "parameters", "type": "int", "default": -1, "preserve_default": True}
        orchestration = behavior.get("orchestration", "")
        if orchestration:
            if orchestration not in ("batch_size", "concurrent_requests"):
                _fail(f"model {entry['id']}.behavior.orchestration", "expected batch_size or concurrent_requests")
            settings[orchestration] = {"name": orchestration, "section": "orchestration", "type": "int", "default": 1}
    explicit = entry.get("settings", [])
    if not isinstance(explicit, list):
        _fail(f"model {entry['id']}.settings", "expected an array")
    names: set[str] = set()
    for raw in explicit:
        where = f"model {entry['id']}.settings"
        if not isinstance(raw, dict) or set(raw) - _KEYS:
            _fail(where, "unsupported setting declaration")
        name = raw.get("name")
        typ = raw.get("type")
        section = raw.get("section")
        if not isinstance(name, str) or not name or name in names:
            _fail(where, "expected unique nonempty setting names")
        if not isinstance(typ, str) or typ not in _TYPES or not isinstance(section, str) or section not in _SECTIONS:
            _fail(f"{where}.{name}", "unsupported type or section")
        if "default" not in raw or not _valid(raw["default"], typ):
            _fail(f"{where}.{name}.default", f"expected {typ}")
        if "sentinel" in raw and not _valid(raw["sentinel"], typ):
            _fail(f"{where}.{name}.sentinel", f"expected {typ}")
        if not isinstance(raw.get("group", ""), str) or type(raw.get("preserve_default", False)) is not bool:
            _fail(f"{where}.{name}", "invalid group or preserve_default")
        if section == "voice_references" and (name not in ("file_name", "transcript") or typ != "list[str]"):
            _fail(f"{where}.{name}", "voice references require file_name/transcript string lists")
        if name in settings and (settings[name]["type"] != typ or settings[name]["section"] != section):
            _fail(f"{where}.{name}", "must match the backend setting type and section")
        names.add(name)
        settings[name] = dict(raw)
    if "transcript" in settings:
        voice = settings.get("file_name")
        if voice is None or voice.get("group", "") != settings["transcript"].get("group", ""):
            _fail(f"model {entry['id']}.settings.transcript", "must share the file_name storage owner")
    if backend in ("sgl_omni", "audio_cpp"):
        for name, param in entry[backend].get("parameters", {}).items():
            storage = settings[name]
            where = f"model {entry['id']}.settings.{name}"
            sentinel = param.get("default_sentinel")
            if storage.get("sentinel") != sentinel:
                _fail(f"{where}.sentinel", "must match the backend default_sentinel")
            if param["type"] == "str":
                continue  # A string control is validated by its declared default alone.
            default = storage["default"]
            if default == sentinel and sentinel is not None:
                continue
            low, high = param.get("min"), param.get("max")
            if not _valid(low, param["type"]):
                _fail(f"model {entry['id']}.{backend}.parameters.{name}.min", "expected a finite number")
            if not _valid(high, param["type"]):
                _fail(f"model {entry['id']}.{backend}.parameters.{name}.max", "expected a finite number")
            if not low <= default <= high:
                _fail(f"{where}.default", "must be within backend bounds or equal its sentinel")
    return tuple(settings.values())


def validate_settings(document: dict[str, Any]) -> None:
    """Validate ownership and shared contracts before installing any bindings."""
    groups = document.get("setting_groups", {})
    if not isinstance(groups, dict):
        _fail("setting_groups", "expected an object")
    ids = {entry["id"] for entry in document["models"]}
    for group, members in groups.items():
        if (not group or not isinstance(members, list) or not members
                or any(not isinstance(id, str) or id not in ids for id in members)
                or len(set(members)) != len(members)):
            _fail(f"setting_groups.{group}", "expected unique catalog model IDs")
    shared: dict[tuple[str, str], dict[str, Any]] = {}
    owners: dict[tuple[str, str], set[str]] = {}
    for entry in document["models"]:
        for setting in parse_model_settings(entry):
            group = setting.get("group", "")
            if not group:
                continue
            if entry["id"] not in groups.get(group, []):
                _fail(f"model {entry['id']}.settings.{setting['name']}", f"unauthorized shared group {group}")
            key = (group, setting["name"])
            if key in shared and setting != shared[key]:
                _fail(f"setting_groups.{group}.{setting['name']}", "shared storage declarations must agree")
            shared[key] = setting
            owners.setdefault(key, set()).add(entry["id"])
    for (group, name), members in owners.items():
        if members != set(groups[group]):
            _fail(f"setting_groups.{group}.{name}", "shared setting must be declared for every member")
