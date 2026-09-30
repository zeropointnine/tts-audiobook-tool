"""Frozen pre-v3 storage mappings, used only to convert legacy project fields."""
from __future__ import annotations
from typing import TYPE_CHECKING
from tts_audiobook_tool.project_support.model_settings_declarations import BUILTIN_LEGACY_FIELDS
if TYPE_CHECKING:
    from tts_audiobook_tool.project_support.model_settings import Binding

# An explicit list of existing shared ownership; equal names alone do not share.
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

