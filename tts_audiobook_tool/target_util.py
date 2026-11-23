"""
String-based target helpers, kept for callers not yet migrated to `ModelTarget`
(see `app_types/model_target.py`). Each untyped string is read with the legacy
inference rule (`infer_legacy_kind`).
"""
from __future__ import annotations

from tts_audiobook_tool.app_types.model_target import (
    ModelTarget,
    is_hf_repo_id_syntax as is_hf_repo_id_syntax,
    split_hf_subfolder,
)


def _from_string(target: str) -> ModelTarget:
    result = ModelTarget.from_stored(target, None)
    if result is None:
        raise ValueError("Target must be a non-empty string")
    return result


def split_hf_subfolder_target(s: str) -> tuple[str, str] | None:
    return split_hf_subfolder(s)


def resolve_to_local_dir(target: str, file_names: list[str]) -> str:
    return _from_string(target).resolve_local_dir(file_names)


def make_target_display_name(target: str) -> str:
    return _from_string(target).display_name()


def exist_test(
        target: str,
        local_file_markers=["config.json", "model.safetensors", "adapter_config.json"]
) -> tuple[str, str]:
    if not target.strip():
        return "", "Target must be a non-empty string"
    return _from_string(target).exist_test(local_file_markers)


def is_same_target(a: str, b: str) -> bool:
    if not a.strip() or not b.strip():
        return a == b
    return _from_string(a).is_same(_from_string(b))
