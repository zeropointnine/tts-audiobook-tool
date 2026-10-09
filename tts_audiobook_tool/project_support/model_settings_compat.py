"""Frozen project-settings history and explicit retired-ownership migrations."""
from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping
from tts_audiobook_tool.project_support.model_settings_declarations import BUILTIN_LEGACY_FIELDS
if TYPE_CHECKING:
    from tts_audiobook_tool.project_support.model_settings import Binding

# Historical ownership, including the now-retired MOSS and Fish S2 groups. Do
# not edit these mappings to match current catalog ownership: old projects
# still need them.
SHARED_MEMBERS: dict[str, tuple[str, ...]] = {
    "auk": ("auk_sglomni", "auk_flash_sglomni"),
    "fish_s2": ("fish_s2_local", "fish_s2_sglomni"),
    "moss": ("moss_local", "moss_delay_sglomni", "moss_local_sglomni"),
    "qwen3": ("qwen3tts_local", "qwen3tts_sglomni"),
}
PREFIX_MODELS: dict[str, str] = {
    "auk": "auk_sglomni", "chatterbox": "chatterbox_local", "dots": "dots_local",
    "fish_s1": "fish_s1_local", "fish_s2": "fish_s2_local", "higgs": "higgs_v2_local",
    "higgs_v3": "higgs_v3_sglomni", "vibevoice": "vibevoice_local",
    "indextts2": "indextts2_local", "glm": "glm_local", "mira": "mira_local",
    "moss": "moss_local", "qwen3": "qwen3tts_local", "zonos2": "zonos2_sglomni",
    "pocket": "pocket_local", "omnivoice": "omnivoice_local",
}
PRIVATE_FIELDS: dict[str, set[str]] = {
    "fish_s2": {"rolling_cont", "compile_enabled", "server_concurrent_requests"},
    "moss": {"target", "rolling_cont"},
    "qwen3": {"target", "model_type", "rolling_cont", "speaker_id", "instructions", "batch_size", "server_concurrent_requests"},
}

# Shared groups that no longer exist in the catalog. On load, each one is
# consumed by copying its values into its historical members' private objects
# (see `fork_retired_group`). To retire another group: remove it from the
# catalog's `setting_groups` and its settings' `group` keys, then append it here.
# A local member without catalog parameter bounds should also borrow its server
# counterpart's (`BORROWED_PARAMETER_BOUNDS` in model_settings.py).
RETIRED_GROUPS: tuple[str, ...] = ("moss", "fish_s2")

# moss_local retired its shared `seed` parameter in favor of one seed per
# preset. Historical values migrate to all three, following the same
# copy-to-each-member rule the original MOSS split used.
MOSS_LOCAL_SEED_ALIASES: tuple[str, ...] = ("delay_seed", "local_seed", "local_v15_seed")


def legacy_bindings() -> dict[str, Binding]:
    """Frozen pre-v3 field conversion; never initializes current storage."""
    from tts_audiobook_tool.project_support.model_settings import Binding
    legacy: dict[str, Binding] = {}
    value_types = {"int": (int, object), "float": (float, object),
                   "str": (str, object), "bool": (bool, object),
                   "list[str]": (list, str), "list[float]": (list, float)}
    for prefix, model_id in PREFIX_MODELS.items():
        for attr, (kind, default) in BUILTIN_LEGACY_FIELDS.items():
            if not attr.startswith(prefix + "_") or any(attr.startswith(other + "_") for other in PREFIX_MODELS if other.startswith(prefix + "_")):
                continue
            suffix = attr[len(prefix) + 1:]
            if suffix.startswith("server_"):
                target = {"fish_s2": "fish_s2_sglomni", "qwen3": "qwen3tts_sglomni"}.get(prefix, model_id)
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
            legacy[attr] = binding
    if set(legacy) != set(BUILTIN_LEGACY_FIELDS):
        raise ValueError(f"Unbound built-in fields: {set(BUILTIN_LEGACY_FIELDS) - set(legacy)}")
    return legacy


def _current_targets(
    model_id: str, section: str, name: str, current: Mapping[tuple[str, str], Any]
) -> tuple[str, ...]:
    """Current setting names a historical value migrates to for this member.

    The historical shared `seed` forks to moss_local's per-preset seeds
    instead of its retired `seed` name; every other name maps to itself.
    """
    targets = (
        MOSS_LOCAL_SEED_ALIASES
        if model_id == "moss_local" and section == "parameters" and name == "seed"
        else (name,)
    )
    return tuple(
        target for target in targets
        if (model_id, target) in current and current[(model_id, target)].section == section
    )


def fork_retired_groups(
    source: dict[str, Any],
    legacy: Mapping[str, Any] | None,
    historical: Mapping[str, Binding],
    current: Mapping[tuple[str, str], Binding],
    members: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """Consume every retired shared group, in retirement order."""
    for group in RETIRED_GROUPS:
        if group in members:
            continue  # Defensive: a group reinstated in the catalog is current again.
        source, legacy = fork_retired_group(group, source, legacy, historical, current)
    return source, legacy


def fork_retired_group(
    group: str,
    source: dict[str, Any],
    legacy: Mapping[str, Any] | None,
    historical: Mapping[str, Binding],
    current: Mapping[tuple[str, str], Binding],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """Consume historical sharing, preserving raw private-key precedence.

    Each historical value is copied to every original member that still owns a
    same-named setting in the same section; members that no longer own it
    simply do not receive it.

    This runs before reconciliation prunes nulls/empty voice lists, so those
    explicit resets cannot be replaced with an old override. Other families and
    genuinely unknown whole objects keep their normal reconciliation behavior.
    """
    members = SHARED_MEMBERS[group]
    # A standalone registry may not know this family at all; leave the group
    # for normal unknown-object retention.
    known_models = {model_id for model_id, _ in current}
    if not all(model_id in known_models for model_id in members):
        return source, legacy
    group_fields = {attr: binding for attr, binding in historical.items() if binding.group == group}
    supplied = {attr: value for attr, value in (legacy or {}).items() if attr in group_fields}
    shared = source.get("shared", {})
    if not isinstance(shared, dict) or group not in shared and not supplied:
        return source, legacy

    # Work on a copy: a later validation failure must leave the complete source
    # available to the project's existing non-destructive warning/fallback path.
    source = deepcopy(source)
    models = source.setdefault("models", {})
    shared = source.setdefault("shared", {})
    remaining = {attr: value for attr, value in (legacy or {}).items() if attr not in group_fields}
    where = f"model_settings.shared.{group}"
    if group in shared:
        raw = shared[group]
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be an object")
        recorded = raw.get("model_ids")
        if (not isinstance(recorded, list) or len(recorded) != len(members)
                or not all(isinstance(model_id, str) for model_id in recorded)
                or set(recorded) != set(members)):
            raise ValueError(f"{where}.model_ids: expected members {list(members)!r}")
    else:
        # Pre-v3 flat fields used the same historical sharing. Materialize only
        # supplied overrides, following the frozen default/seed migration rules.
        raw = {}
        voice_attr, transcript_attr = f"{group}_voice_file_name", f"{group}_voice_transcript"
        for attr, value in supplied.items():
            binding = group_fields[attr]
            if binding.section == "voice_references":
                continue
            if not binding.preserve_default and value == binding.default:
                continue
            raw.setdefault(binding.section, {})[binding.name] = deepcopy(value)
        if voice_attr in supplied:
            voices = supplied[voice_attr]
            voices = [voices] if isinstance(voices, str) and voices else voices
            transcripts = supplied.get(transcript_attr, [])
            transcripts = [transcripts] if isinstance(transcripts, str) and transcripts else transcripts
            if not isinstance(transcripts, list):
                transcripts = []
            if isinstance(voices, list):
                raw["voice_references"] = [
                    {"file_name": voice, "transcript": transcripts[index] if index < len(transcripts) else ""}
                    for index, voice in enumerate(voices) if isinstance(voice, str) and voice
                ]

    # Keep the same structural checks as historical registry reconciliation,
    # even when new private values would mask every field in the old object.
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}  # Malformed parameter sections follow the usual pruning policy.
    orchestration = raw.get("orchestration", {})
    if not isinstance(orchestration, dict):
        raise ValueError(f"{where}.orchestration must be an object")
    if "voice_references" in raw:
        references = raw["voice_references"]
        if not isinstance(references, list):
            raise ValueError(f"{where}.voice_references must be an array")
        for index, ref in enumerate(references):
            if (not isinstance(ref, dict) or not isinstance(ref.get("file_name"), str)
                    or not isinstance(ref.get("transcript", ""), str)):
                raise ValueError(
                    f"{where}.voice_references[{index}]: "
                    "expected file_name and optional transcript strings")

    historical_names = {(binding.section, binding.name) for binding in group_fields.values()}
    for model_id in members:
        obj = models.setdefault(model_id, {})
        if not isinstance(obj, dict):
            raise ValueError(f"model_settings.models.{model_id} must be an object")
        for section, values in (("parameters", parameters), ("orchestration", orchestration)):
            missing = {name: value for name, value in values.items()
                       if (section, name) in historical_names
                       and _current_targets(model_id, section, name, current)}
            if not missing:
                continue
            destination = obj.setdefault(section, {})
            if isinstance(destination, dict):
                for name, value in missing.items():
                    for target in _current_targets(model_id, section, name, current):
                        destination.setdefault(target, deepcopy(value))
        if "voice_references" in raw and "voice_references" not in obj:
            obj["voice_references"] = deepcopy(raw["voice_references"])
    shared.pop(group, None)
    return source, remaining or None


def split_moss_local_seed(source: dict[str, Any], current: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    """Fork a stored pre-split private `seed` to moss_local's per-preset seeds.

    Runs before reconciliation cleaning drops the retired name. Explicit
    per-preset seeds always win; the old value only fills absent ones.
    """
    models = source.get("models")
    raw = models.get("moss_local") if isinstance(models, dict) else None
    params = raw.get("parameters") if isinstance(raw, dict) else None
    if not isinstance(params, dict) or "seed" not in params:
        return source
    source = deepcopy(source)
    params = source["models"]["moss_local"]["parameters"]
    value = params.pop("seed")
    for name in MOSS_LOCAL_SEED_ALIASES:
        binding = current.get(("moss_local", name))
        if binding is not None and binding.section == "parameters" and name not in params:
            params[name] = deepcopy(value)
    return source

