"""Match audio.cpp model metadata to supported, catalog-declared variants.

Server IDs are opaque: a user-defined ID must never be used to infer the
model family or which family checkpoint/variant the server is actually
serving. Each catalog entry declares the identifying `family`/`task`/`mode`
and any session-option hints it requires; an entry without hints (for
example a family with no variant options) matches on family/task/mode alone.

An entry may also accept several task tokens (`tasks`). audio.cpp advertises
the token its operator configured, which for a family whose routes are the
same inference path can be either spelling: GLM-TTS serves identical
reference-conditioned synthesis as `tts` and as `clon`.
"""

from __future__ import annotations

import tomllib
from typing import Any

from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def _declared_tasks(match: dict[str, Any]) -> tuple[str, ...]:
    """Accepted task tokens from either the single `task` or the `tasks` field."""
    single = match.get("task")
    if isinstance(single, str) and single:
        return (single,)
    multiple = match.get("tasks")
    if isinstance(multiple, list) and all(isinstance(item, str) and item for item in multiple):
        return tuple(multiple)
    return ()


def detect_audio_cpp_models(models: list[dict[str, Any]]) -> list[tuple[TtsModelType, str]]:
    """Return every supported (catalog handle, exact configured server ID) pair.

    A declared session-option hint is an exact requirement. Entries without
    hints accept the server-selected variant on the declared family/task/mode;
    an ID containing a version string changes nothing.
    """
    entries = tomllib.loads(CATALOG_PATH.read_text(encoding="utf-8"))["models"]
    declared: list[tuple[TtsModelType, dict[str, Any]]] = []
    for entry in entries:
        if entry.get("backend_kind") != TtsBackendKind.AUDIO_CPP.value:
            continue
        handle = TtsModelType.require_by_id(entry["id"])
        if handle.id == "none" or handle.value.backend_kind is not TtsBackendKind.AUDIO_CPP:
            continue
        match = entry.get("audio_cpp", {}).get("match", {})
        if isinstance(match, dict) and _declared_tasks(match) and all(
            isinstance(match.get(key), str) and match[key]
            for key in ("family", "mode")
        ):
            declared.append((handle, match))

    candidates: list[tuple[TtsModelType, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if not all(isinstance(model.get(key), str) and model[key] for key in ("family", "task", "mode")):
            continue
        options = model.get("session_options", {})
        if not isinstance(options, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in options.items()
        ):
            continue
        for handle, match in declared:
            if model["family"] != match["family"] or model["mode"] != match["mode"]:
                continue
            if model["task"] not in _declared_tasks(match):
                continue
            hints = match.get("session_options", {})
            if not isinstance(hints, dict) or any(options.get(key) != value for key, value in hints.items()):
                continue
            candidates.append((handle, model_id))
    return candidates
