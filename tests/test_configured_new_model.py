"""A user-defined SGL-Omni model: declared in TOML, stored without new Python fields."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from catalog_toml_support import read_catalog, write_catalog
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredModelSupport
from tts_audiobook_tool.tts_models.sgl_omni_definition import DEFINITION_PATH, load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

CUSTOM_ID = "server_custom"


def make_custom_entry(source_entry: dict[str, Any]) -> dict[str, Any]:
    """Derive a brand-new model whose protocol matches the generic adapter."""
    entry = json.loads(json.dumps(source_entry))
    entry["id"] = CUSTOM_ID
    entry.pop("symbol", None)  # A custom server model has no built-in handle.
    entry["match"] = {"model_id_substring": "custom-model"}
    entry["spec"]["file_tag"] = "custom_model"
    entry["spec"]["ui"]["proper_name"] = "Custom Model"
    entry["spec"]["ui"]["short_name"] = "custom"
    entry["behavior"]["orchestration"] = "concurrent_requests"
    return entry


def write_definitions(tmp_path, entries: list[dict[str, Any]]):
    catalog = read_catalog(DEFINITION_PATH)
    catalog["models"] = entries  # Callers supply the entire catalog, including local/none.
    return write_catalog(tmp_path / "model_catalog.toml", catalog)


def shipped_entries() -> list[dict[str, Any]]:
    return read_catalog(DEFINITION_PATH)["models"]


def higgs_entry() -> dict[str, Any]:
    return next(entry for entry in shipped_entries() if entry["id"] == "server_higgs_v3")


@pytest.fixture
def custom_definitions(tmp_path):
    """The shipped Higgs overlay plus one brand-new model."""
    entries = shipped_entries()
    entries.append(make_custom_entry(higgs_entry()))
    return load_definitions(write_definitions(tmp_path, entries))


@pytest.fixture
def configured_custom(monkeypatch, custom_definitions):
    """Install the custom definitions the way startup does, then restore."""
    old = (Tts._backend_mode, getattr(Tts, "_type", None),
           Tts._config_fingerprint, Tts._configured_definitions, Tts._configured_runtime,
           Tts._catalog_initialized)
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsBackendKind.SGL_OMNI))
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.sgl_omni_definition.load_definitions",
        lambda: custom_definitions,
    )
    try:
        Tts.init_local_model_type()
        yield custom_definitions
    finally:
        TtsModelType.reset_catalog()
        REGISTRY.reset_to_builtins()
        (Tts._backend_mode, previous_type,
         Tts._config_fingerprint, Tts._configured_definitions, Tts._configured_runtime,
         Tts._catalog_initialized) = old
        if previous_type is None:
            if hasattr(Tts, "_type"):
                delattr(Tts, "_type")
        else:
            Tts._type = previous_type


def test_loader_accepts_a_new_model_next_to_the_builtin_overlay(tmp_path) -> None:
    entries = shipped_entries()
    entries.append(make_custom_entry(higgs_entry()))

    definitions = load_definitions(write_definitions(tmp_path, entries))

    assert set(definitions.models) == {
        entry["id"] for entry in shipped_entries() if entry.get("backend_kind") == "sgl_omni"
    } | {CUSTOM_ID}
    custom = definitions.models[CUSTOM_ID]
    assert custom.spec.file_tag == "custom_model"
    assert custom.spec.can_stream
    assert set(custom.parameters) == {"temperature", "top_p", "top_k"}
    assert definitions.fingerprint != load_definitions().fingerprint


def test_loader_rejects_flat_storage_names_in_schema_v3(tmp_path) -> None:
    entries = shipped_entries()
    custom = make_custom_entry(higgs_entry())
    custom["spec"]["voice_target_attr"] = "higgs_v3_voice_file_name"
    entries.append(custom)

    with pytest.raises(ValueError, match=r"spec: unsupported field.*voice_target_attr"):
        load_definitions(write_definitions(tmp_path, entries))

    custom["spec"].pop("voice_target_attr")
    custom["parameters"]["temperature"]["project_attr"] = "higgs_v3_temperature"
    with pytest.raises(ValueError, match=r"parameters.temperature: unsupported field.*project_attr"):
        load_definitions(write_definitions(tmp_path, entries))


def test_loader_rejects_replacing_a_local_builtin_with_server(tmp_path) -> None:
    entries = [entry for entry in shipped_entries() if entry["id"] != "chatterbox"]
    entry = make_custom_entry(higgs_entry())
    entry["id"] = "chatterbox"
    entries.append(entry)

    with pytest.raises(ValueError, match="missing built-in.*chatterbox"):
        load_definitions(write_definitions(tmp_path, entries))


def test_startup_registers_new_model_storage_without_python_fields(configured_custom) -> None:
    definition = configured_custom.models[CUSTOM_ID]

    handle = TtsModelType.get_by_id(CUSTOM_ID)
    assert handle is not TtsModelType.NONE
    assert handle.value is definition.spec
    assert Tts.get_configured_definition(handle) is definition
    assert TtsModelType.get_by_id("server_higgs_v3") is TtsModelType.HIGGS_V3_SERVER
    assert TtsModelType.all().count(handle) == 1

    # No Python project fields were invented for the new model.
    assert not [name for name in Project.model_fields if name.startswith("server_custom_")]
    assert REGISTRY.get(CUSTOM_ID, "temperature").default == 1.0
    assert REGISTRY.voice_binding(CUSTOM_ID) is not None
    assert REGISTRY.transcript_binding(CUSTOM_ID) is not None
    assert REGISTRY.orchestration_binding(CUSTOM_ID).name == "concurrent_requests"
    assert not any(name.startswith("server_custom_") for name in REGISTRY.legacy)

    project = Project()
    assert project.get_model_setting(CUSTOM_ID, "temperature") == 1.0
    assert project.get_model_setting(CUSTOM_ID, "top_k") == 100
    project.set_model_setting(CUSTOM_ID, "temperature", 0.42)
    project.set_model_setting(CUSTOM_ID, "top_k", 12)

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    stored = payload["model_settings"]["models"][CUSTOM_ID]["parameters"]
    assert stored == {"temperature": 0.42, "top_k": 12}

    reloaded = Project.model_validate(payload)
    assert reloaded.get_model_setting(CUSTOM_ID, "temperature") == 0.42
    assert reloaded.get_model_setting(CUSTOM_ID, "top_k") == 12


def test_new_model_support_and_generation_payload(configured_custom, monkeypatch) -> None:
    from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil

    Tts._type = TtsModelType.get_by_id(CUSTOM_ID)
    assert isinstance(Tts.get_model_support(), ConfiguredModelSupport)

    project = Project()
    project.set_model_setting(CUSTOM_ID, "temperature", 0.7)
    project.set_model_setting(CUSTOM_ID, "top_p", 0.8)

    captured: list[dict[str, Any]] = []

    def generate_concurrent(base_url, payloads, print_request=False, **kwargs):
        captured.extend(payloads)
        return [Sound(np.array([0.1], dtype=np.float32), 24_000) for _ in payloads]

    monkeypatch.setattr(SglOmniUtil, "get_base_url", staticmethod(lambda: "http://example.test"))
    monkeypatch.setattr(SglOmniUtil, "generate_concurrent", staticmethod(generate_concurrent))

    instance = Tts.get_instance()
    result = instance.generate_using_project(project, ["ready A", "ready B"], force_random_seed=True)

    assert isinstance(result, list) and len(result) == 2
    assert captured == [
        {"max_tokens": 1536, "temperature": 0.7, "top_p": 0.8, "top_k": 100,
         "input": prompt, "stream": False}
        for prompt in ("ready A", "ready B")
    ]
    assert not any("seed" in payload for payload in captured)


def test_local_mode_forgets_the_configured_model_and_its_storage(configured_custom, monkeypatch) -> None:
    assert REGISTRY.get(CUSTOM_ID, "temperature")
    assert TtsModelType.get_by_id(CUSTOM_ID) is not TtsModelType.NONE

    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsBackendKind.LOCAL))
    Tts.init_local_model_type()

    assert TtsModelType.get_by_id(CUSTOM_ID) is TtsModelType.NONE
    with pytest.raises(ValueError, match="Unknown setting"):
        REGISTRY.get(CUSTOM_ID, "temperature")
    assert Tts.get_configured_definition(TtsModelType.HIGGS_V3_SERVER) is None
