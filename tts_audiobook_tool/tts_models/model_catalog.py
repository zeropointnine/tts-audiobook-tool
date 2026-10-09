"""Load the versioned, application-shipped model catalog without runtime dependencies.

This module parses model metadata at import time, before local model classes
capture their INFO specs. SGL-Omni request/storage validation happens later in
sgl_omni_definition.py, after the project settings registry is available.
"""
from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path
import re
from typing import Any, NoReturn

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind, TtsModelSpec

CATALOG_PATH = Path(__file__).with_name("model_catalog.toml")
CATALOG_SCHEMA_VERSION = 1
_LOCAL_SPEC_KEYS = {
    "local_module_test", "local_torch_devices", "file_tag", "default_output_sample_rate",
    "requires_voice", "can_stream", "requires_ffmpeg_libs", "un_all_caps",
    "requirements_file_name", "ui", "output_filters", "substitutions", "max_random_seed",
}
_SERVER_SPEC_KEYS = {"file_tag", "default_output_sample_rate", "un_all_caps", "requirements_file_name", "ui", "substitutions", "max_random_seed"}
_UI_KEYS = {"proper_name", "short_name", "voice_path_console", "voice_path_requestor", "project_links", "settings_note",
            "voice_sample_max_duration_s"}


def _fail(where: str, reason: str) -> NoReturn:
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


# audio.cpp entries name their own request controls; the backend detector and
# adapter interpret whatever a family declares here.
_AUDIO_CPP_GROUP_KEYS = {"match", "parameters", "menu", "reference_transcript", "max_words_range_reco",
                         "request_options", "voice_required", "language_policy", "language_target", "music_and_trim"}
_AUDIO_CPP_MATCH_KEYS = {"task", "tasks", "family", "mode", "session_options"}
_AUDIO_CPP_PARAMETER_KEYS = {"type", "default", "min", "max", "default_sentinel", "request_key", "target",
                             "input_prompt_suffix"}
_AUDIO_CPP_TARGETS = ("top_level", "options")


def _audio_cpp_number(param: dict[str, Any], key: str, where: str, typ: str, value_type: Any) -> int | float:
    value = param.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, value_type) or not math.isfinite(value):
        _fail(f"{where}.{key}", f"expected a finite {typ}")
    return value


def _parse_audio_cpp_parameters(value: Any, where: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        _fail(where, "expected an object")
    parameters: dict[str, dict[str, Any]] = {}
    for name, raw in value.items():
        item_where = f"{where}.{name}"
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            _fail(where, f"invalid parameter name {name!r}")
        param = _object(raw, item_where, _AUDIO_CPP_PARAMETER_KEYS)
        typ = _field(param, "type", item_where, str)
        if typ not in ("float", "int", "str"):
            _fail(f"{item_where}.type", "expected float, int or str")
        request_key = param.get("request_key", name)
        if type(request_key) is not str or not request_key.strip():
            _fail(f"{item_where}.request_key", "expected a nonempty string")
        target = param.get("target", "top_level")
        if target not in _AUDIO_CPP_TARGETS:
            _fail(f"{item_where}.target", "expected top_level or options")
        suffix = param.get("input_prompt_suffix", "")
        if not isinstance(suffix, str):
            _fail(f"{item_where}.input_prompt_suffix", "expected a string")
        if typ == "str":
            if suffix:
                _fail(f"{item_where}.input_prompt_suffix", "only supported for numeric input prompts")
            # A text control has no bounds; audio.cpp reads the only such
            # control this backend declares (`instruction`) from `options`.
            for key in ("min", "max", "default_sentinel"):
                if param.get(key) is not None:
                    _fail(f"{item_where}.{key}", "numeric bounds are not valid for a string parameter")
            default = _field(param, "default", item_where, str)
            if target != "options":
                _fail(f"{item_where}.target", "a string parameter must travel in options")
            parameters[name] = {"type": typ, "default": default, "min": None, "max": None,
                                "default_sentinel": None, "request_key": request_key, "target": target}
            continue
        value_type = int if typ == "int" else (int, float)
        minimum = _audio_cpp_number(param, "min", item_where, typ, value_type)
        default = _audio_cpp_number(param, "default", item_where, typ, value_type)
        maximum = _audio_cpp_number(param, "max", item_where, typ, value_type)
        if not minimum <= default <= maximum:
            _fail(item_where, "expected min <= default <= max")
        sentinel: int | float | None = None
        if param.get("default_sentinel") is not None:
            sentinel = _audio_cpp_number(param, "default_sentinel", item_where, typ, value_type)
            if minimum <= sentinel <= maximum:
                _fail(f"{item_where}.default_sentinel", "expected a sentinel outside min..max")
        parameters[name] = {"type": typ, "default": default, "min": minimum, "max": maximum,
                            "default_sentinel": sentinel, "request_key": request_key, "target": target,
                            "input_prompt_suffix": suffix}
    return parameters


def _parse_audio_cpp_menu(value: Any, parameters: dict[str, dict[str, Any]], where: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        _fail(where, "expected a list")
    menu: list[dict[str, str]] = []
    for index, item in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(item, dict):
            _fail(item_where, "expected controls")
        kind = item.get("kind", "parameter")
        if kind == "voice_samples":
            _object(item, item_where, {"kind"})
            menu.append({"kind": kind})
            continue
        if kind == "seed":
            control = _object(item, item_where, {"kind", "target_menu"})
        elif kind == "voice_instructions":
            control = _object(item, item_where, {"kind", "parameter", "target_menu"})
        elif "kind" not in item:
            control = _object(item, item_where, {"parameter", "label", "target_menu"})
        else:
            _fail(item_where, "invalid control")
        target_menu = _field(control, "target_menu", item_where, str)
        if target_menu not in ("model", "voice"):
            _fail(f"{item_where}.target_menu", "expected model or voice")
        if kind == "seed":
            menu.append({"kind": kind, "target_menu": target_menu})
            continue
        name = _field(control, "parameter", item_where, str)
        if name not in parameters:
            _fail(f"{item_where}.{name}", "unknown parameter")
        if kind == "voice_instructions":
            # A text parameter is edited by its own prompt, not by the numeric
            # control, so name it explicitly rather than by position.
            if parameters[name]["type"] != "str":
                _fail(f"{item_where}.{name}", "voice_instructions requires a string parameter")
            menu.append({"kind": kind, "parameter": name, "target_menu": target_menu})
            continue
        if parameters[name]["type"] == "str":
            _fail(f"{item_where}.{name}", "a string parameter requires the voice_instructions control")
        label = _field(control, "label", item_where, str)
        if not label:
            _fail(f"{item_where}.{name}.label", "must not be empty")
        menu.append({"kind": "parameter", "parameter": name, "label": label, "target_menu": target_menu})
    if (sorted(item["parameter"] for item in menu if item["kind"] in ("parameter", "voice_instructions"))
            != sorted(parameters)):
        _fail(where, "expected each parameter exactly once")
    if (sum(item["kind"] == "seed" for item in menu) != 1
            or sum(item["kind"] == "voice_samples" for item in menu) != 1):
        _fail(where, "expected voice samples and seed exactly once")
    return menu


def _parse_audio_cpp_request_options(value: Any, where: str) -> dict[str, int | float]:
    """Validate values the adapter pins into every request's ``options`` object.

    These are constants, not project settings: no menu control, no persisted
    storage, and no way to change them from the app. A family uses one for a
    fixed knob it must always send, the way a local model hardcodes its own
    module constant. Declared user controls live in ``parameters`` instead.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail(where, "expected an object")
    options: dict[str, int | float] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            _fail(where, f"invalid request option name {name!r}")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            _fail(f"{where}.{name}", "expected a finite number")
        options[name] = raw
    return options


def _parse_audio_cpp_word_range(value: Any, where: str) -> tuple[int, int, str] | None:
    """Validate optional ``[min, max]`` / ``[min, max, note]`` per family.

    A family whose per-request cost or quality depends on how much text one
    request carries can declare a tighter segment recommendation than the
    shared default. Shape matches the local-model recommendation tuples,
    including the optional trailing label used in warning text.
    """
    if value is None:
        return None
    if not isinstance(value, list) or len(value) not in (2, 3):
        _fail(where, "expected [min, max] or [min, max, note]")
    minimum, maximum = value[0], value[1]
    for item in (minimum, maximum):
        if isinstance(item, bool) or not isinstance(item, int):
            _fail(where, "expected integer word counts")
    if minimum < 1 or maximum < minimum:
        _fail(where, "expected 1 <= min <= max")
    note = value[2] if len(value) == 3 else ""
    if type(note) is not str:
        _fail(where, "expected a string note")
    return (minimum, maximum, note)


def _parse_audio_cpp_tasks(match: dict[str, Any], where: str) -> tuple[str, ...]:
    """Validate a family's accepted task token(s).

    An entry usually names one task. A family whose two advertised routes are
    the same inference path (audio.cpp's GLM-TTS serves identical
    reference-conditioned synthesis as both `tts` and `clon`) can accept
    several, because the server advertises whichever token its operator
    configured rather than a canonical one.
    """
    single = match.get("task")
    multiple = match.get("tasks")
    if single is not None and multiple is not None:
        _fail(f"{where}.tasks", "expected either task or tasks, not both")
    if single is not None:
        if type(single) is not str or not single.strip():
            _fail(f"{where}.task", "must not be empty")
        return (single,)
    if not isinstance(multiple, list) or not multiple:
        _fail(f"{where}.tasks", "expected a nonempty array of task names")
    tasks: list[str] = []
    for item in multiple:
        if type(item) is not str or not item.strip():
            _fail(f"{where}.tasks", "expected nonempty strings")
        if item in tasks:
            _fail(f"{where}.tasks", f"duplicate task {item!r}")
        tasks.append(item)
    return tuple(tasks)


def parse_audio_cpp_declaration(entry: dict[str, Any], where: str) -> dict[str, Any]:
    """Validate one audio.cpp group and return its normalized declaration.

    This is the single source of truth shared by ``parse_spec`` (import-time
    gate) and the audio.cpp definition loader (runtime dataclasses).
    """
    group = _object(entry.get("audio_cpp"), f"{where}.audio_cpp", _AUDIO_CPP_GROUP_KEYS)
    match = _object(group.get("match"), f"{where}.audio_cpp.match", _AUDIO_CPP_MATCH_KEYS)
    for key in ("family", "mode"):
        if not _field(match, key, f"{where}.audio_cpp.match", str).strip():
            _fail(f"{where}.audio_cpp.match.{key}", "must not be empty")
    tasks = _parse_audio_cpp_tasks(match, f"{where}.audio_cpp.match")
    session_options = match.get("session_options", {})
    if not isinstance(session_options, dict) or any(
        type(key) is not str or not key.strip() or type(value) is not str or not value.strip()
        for key, value in session_options.items()
    ):
        _fail(f"{where}.audio_cpp.match.session_options", "expected nonempty string keys and values")
    reference_transcript = group.get("reference_transcript", False)
    if type(reference_transcript) is not bool:
        _fail(f"{where}.audio_cpp.reference_transcript", "expected a boolean")
    voice_required = group.get("voice_required", True)
    if type(voice_required) is not bool:
        _fail(f"{where}.audio_cpp.voice_required", "expected a boolean")
    language_policy = group.get("language_policy", "project_code")
    if language_policy not in ("project_code", "moss", "omit", "normalized"):
        _fail(f"{where}.audio_cpp.language_policy", "expected project_code, moss, omit or normalized")
    language_target = group.get("language_target", "top_level")
    if language_target not in _AUDIO_CPP_TARGETS:
        _fail(f"{where}.audio_cpp.language_target", "expected top_level or options")
    music_and_trim = group.get("music_and_trim", False)
    if type(music_and_trim) is not bool:
        _fail(f"{where}.audio_cpp.music_and_trim", "expected a boolean")
    parameters = _parse_audio_cpp_parameters(group.get("parameters", {}), f"{where}.audio_cpp.parameters")
    menu = _parse_audio_cpp_menu(group.get("menu"), parameters, f"{where}.audio_cpp.menu")
    word_range = _parse_audio_cpp_word_range(
        group.get("max_words_range_reco"), f"{where}.audio_cpp.max_words_range_reco")
    request_options = _parse_audio_cpp_request_options(
        group.get("request_options"), f"{where}.audio_cpp.request_options")
    return {"family": match["family"], "tasks": tasks, "mode": match["mode"],
            "session_options": dict(session_options), "reference_transcript": reference_transcript,
            "voice_required": voice_required,
            "language_policy": language_policy, "language_target": language_target,
            "music_and_trim": music_and_trim,
            "parameters": parameters, "menu": menu, "max_words_range_reco": word_range,
            "request_options": request_options}


def parse_spec(entry: dict[str, Any]) -> TtsModelSpec:
    """Derive one complete spec from its sole declaration in the catalog."""
    id = entry["id"]
    where = f"model {id}"
    # TOML has no null: only the none placeholder omits its backend kind.
    if "backend_kind" not in entry and id != "none":
        _fail(f"{where}.backend_kind", "required for real models")
    backend = entry.get("backend_kind")
    if backend not in (None, *(kind.value for kind in TtsBackendKind)):
        _fail(f"{where}.backend_kind", "expected local, sgl_omni or audio_cpp (or omitted for none)")
    if id == "none" and backend is not None:
        _fail(f"{where}.backend_kind", "the none placeholder cannot have a backend")
    is_server = backend in (TtsBackendKind.SGL_OMNI.value, TtsBackendKind.AUDIO_CPP.value)
    meta = _object(entry.get("spec"), f"{where}.spec", _SERVER_SPEC_KEYS if is_server else _LOCAL_SPEC_KEYS)
    file_tag = _field(meta, "file_tag", f"{where}.spec", str)
    rate = _field(meta, "default_output_sample_rate", f"{where}.spec", int)
    if id == "none":
        if file_tag or rate != 0:
            _fail(f"{where}.spec", "the none placeholder requires an empty file tag and zero sample rate")
    elif not file_tag or rate <= 0:
        _fail(f"{where}.spec", "real models require a file tag and positive sample rate")
    max_random_seed = meta.get("max_random_seed", -1)
    if type(max_random_seed) is not int:
        _fail(f"{where}.spec.max_random_seed", "expected int")
    if max_random_seed < -1:
        _fail(f"{where}.spec.max_random_seed", "expected >= -1")
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
    if "settings_note" in ui:
        _field(ui, "settings_note", f"{where}.spec.ui", str)
    if "voice_sample_max_duration_s" in ui:
        duration = ui["voice_sample_max_duration_s"]
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            _fail(f"{where}.spec.ui.voice_sample_max_duration_s", "expected a positive finite number")
    replacements = _field(meta, "substitutions", f"{where}.spec", list)
    if not all(isinstance(pair, list) and len(pair) == 2 and all(isinstance(s, str) for s in pair) for pair in replacements):
        _fail(f"{where}.spec.substitutions", "expected [before, after] string pairs")
    module = ""
    devices: list[DeviceType] = []
    requires_ffmpeg = False
    filters: list[str] = []
    if backend == TtsBackendKind.SGL_OMNI.value:
        if "audio_cpp" in entry:
            _fail(where, "sgl_omni models cannot contain audio_cpp definitions")
        group = _object(entry.get("sgl_omni"), f"{where}.sgl_omni",
                        {"match", "behavior", "parameters", "request_defaults", "request_stream_defaults", "menu"})
        match = _object(group.get("match"), f"{where}.sgl_omni.match", {"model_id_substring"})
        substring = _field(match, "model_id_substring", f"{where}.sgl_omni.match", str)
        if not substring.strip():
            _fail(f"{where}.sgl_omni.match.model_id_substring", "must not be empty")
        behavior = _object(group.get("behavior"), f"{where}.sgl_omni.behavior",
                           {"streaming", "seed", "voice_required", "transcript", "prompt_policy", "language", "music_and_trim", "orchestration", "can_batch"})
        streaming = _field(behavior, "streaming", f"{where}.sgl_omni.behavior", bool)
        voice_required = _field(behavior, "voice_required", f"{where}.sgl_omni.behavior", bool)
        if "can_batch" in behavior:
            # Declares whether the model offers/uses a batch-size or concurrency value.
            _field(behavior, "can_batch", f"{where}.sgl_omni.behavior", bool)
    elif backend == TtsBackendKind.AUDIO_CPP.value:
        if "sgl_omni" in entry:
            _fail(where, "audio_cpp models cannot contain sgl_omni definitions")
        declaration = parse_audio_cpp_declaration(entry, where)
        # audio.cpp families are served offline; streaming families need a
        # separate reader before they can opt in. Most are cloned from a
        # reference WAV, but a family whose route also serves reference-less
        # requests (OmniVoice design/auto voice) declares `voice_required`.
        streaming = False
        voice_required = declaration["voice_required"]
    else:
        if id != "none" and backend != TtsBackendKind.LOCAL.value:
            _fail(f"{where}.backend_kind", "only the none placeholder can lack a backend")
        if "sgl_omni" in entry or "audio_cpp" in entry:
            _fail(where, "local and placeholder models cannot have server behavior")
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
    return TtsModelSpec(id, TtsBackendKind(backend) if backend else None,
                        module, devices, file_tag, rate, voice_required, streaming,
                        requires_ffmpeg, caps, requirements, ui, filters,
                        [tuple(pair) for pair in replacements], max_random_seed)


def _load_catalog(path: Path) -> tuple[dict[str, Any], list[TtsModelSpec], list[dict[str, Any]], str]:
    """Read and validate one immutable catalog snapshot."""
    try:
        raw = path.read_bytes()
        document = tomllib.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Model catalog {path}: {exc}") from exc
    root = _object(document, "root", {"schema_version", "models", "setting_groups"})
    if type(root.get("schema_version")) is not int or root["schema_version"] != CATALOG_SCHEMA_VERSION:
        _fail("schema_version", f"expected supported version {CATALOG_SCHEMA_VERSION}")
    entries = _field(root, "models", "root", list)
    specs: list[TtsModelSpec] = []
    servers: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, entry in enumerate(entries):
        item = _object(entry, f"models[{index}]", {"id", "backend_kind", "spec", "sgl_omni", "audio_cpp", "settings"})
        id = _field(item, "id", f"models[{index}]", str)
        if not id or not re.fullmatch(r"[a-z][a-z0-9_]*", id) or id in ids:
            _fail(f"models[{index}].id", "expected a unique, lowercase model ID")
        ids.add(id)
        spec = parse_spec(item)
        specs.append(spec)
        if spec.backend_kind in (TtsBackendKind.SGL_OMNI, TtsBackendKind.AUDIO_CPP):
            servers.append(item)
    if "none" not in ids:
        _fail("models", "missing built-in none placeholder")
    from tts_audiobook_tool.tts_models.catalog_settings import validate_settings
    validate_settings(document)
    return document, specs, servers, hashlib.sha256(raw).hexdigest()


def load_catalog(path: Path = CATALOG_PATH) -> tuple[list[TtsModelSpec], list[dict[str, Any]], str]:
    """Return all specs in catalog order, server entries, and a full-file fingerprint."""
    _, specs, servers, fingerprint = _load_catalog(path)
    return specs, servers, fingerprint


def load_settings_catalog(path: Path = CATALOG_PATH) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[dict[str, Any], ...]]]:
    """Load current settings for all backends; never consult legacy declarations."""
    document, _, _, _ = _load_catalog(path)
    from tts_audiobook_tool.tts_models.catalog_settings import parse_model_settings
    groups = {name: tuple(ids) for name, ids in document.get("setting_groups", {}).items()}
    settings = {entry["id"]: parse_model_settings(entry) for entry in document["models"]}
    return groups, settings
