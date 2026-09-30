"""SGL-Omni model endpoint matching, owned by the SGL backend.

Only SGL entries are considered. A more specific model-ID fragment wins;
equal-length fragments retain catalog order. The endpoint model's original ID
is returned so the request adapter can address that exact served model.
"""
from __future__ import annotations

from functools import lru_cache

from tts_audiobook_tool.tts_models.model_catalog import load_catalog
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


@lru_cache(maxsize=1)
def _matchers() -> tuple[tuple[str, str], ...]:
    _, entries, _ = load_catalog()
    return tuple((entry["id"], entry["sgl_omni"]["match"]["model_id_substring"].lower())
                 for entry in entries if entry["backend_kind"] == TtsBackendKind.SGL_OMNI.value)


def detect_sgl_omni_models(models: list[dict]) -> list[tuple[TtsModelType, str]]:
    """Return matched (catalog handle, served ID) pairs for SGL endpoint models."""
    result: list[tuple[TtsModelType, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        server_id = model.get("id")
        if not isinstance(server_id, str) or not server_id.strip():
            continue
        lowered = server_id.strip().lower()
        best_id = ""
        best_length = 0
        for catalog_id, substring in _matchers():
            if substring in lowered and len(substring) > best_length:
                best_id, best_length = catalog_id, len(substring)
        if best_id:
            handle = TtsModelType.require_by_id(best_id)
            if handle.id != "none" and handle.value.backend_kind is TtsBackendKind.SGL_OMNI:
                result.append((handle, server_id))
    return result
