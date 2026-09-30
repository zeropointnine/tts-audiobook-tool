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
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredModelSupport
from tts_audiobook_tool.tts_models.sgl_omni_definition import DEFINITION_PATH, load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

CUSTOM_ID = "server_custom"


def make_custom_entry(source_entry: dict[str, Any]) -> dict[str, Any]:
    """Derive a brand-new model whose protocol matches the generic adapter."""
    entry = json.loads(json.dumps(source_entry))
    entry["id"] = CUSTOM_ID
    entry.pop("settings", None)  # Use private backend-derived storage, not Higgs' explicit contracts.
    entry["sgl_omni"]["match"] = {"model_id_substring": "custom-model"}
    entry["spec"]["file_tag"] = "custom_model"
    entry["spec"]["ui"]["proper_name"] = "Custom Model"
    entry["spec"]["ui"]["short_name"] = "custom"
    entry["sgl_omni"]["behavior"]["orchestration"] = "concurrent_requests"
    return entry


def write_definitions(tmp_path, entries: list[dict[str, Any]]):
    catalog = read_catalog(DEFINITION_PATH)
    catalog["models"] = entries  # Callers supply the entire catalog, including local/none.
    return write_catalog(tmp_path / "model_catalog.toml", catalog)


def shipped_entries() -> list[dict[str, Any]]:
    return read_catalog(DEFINITION_PATH)["models"]


def higgs_entry() -> dict[str, Any]:
    return next(entry for entry in shipped_entries() if entry["id"] == "higgs_v3_sglomni")


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
           Tts._catalog_initialized, Tts._audio_cpp_definitions,
           Tts._selected_server_model_id, Tts._remote_issue)
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.REMOTE_CLIENT))
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.sgl_omni_definition.load_definitions",
        lambda: custom_definitions,
    )
    from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppDefinitions, load_audio_cpp_definitions
    audio_models = load_audio_cpp_definitions().models
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.audio_cpp_definition.load_audio_cpp_definitions",
        lambda: AudioCppDefinitions(audio_models, custom_definitions.fingerprint),
    )
    try:
        Tts.init_local_model_type()
        yield custom_definitions
    finally:
        TtsModelType.reset_catalog()
        REGISTRY.reset_to_builtins()
        (Tts._backend_mode, previous_type,
         Tts._config_fingerprint, Tts._configured_definitions, Tts._configured_runtime,
         Tts._catalog_initialized, Tts._audio_cpp_definitions,
          Tts._selected_server_model_id, Tts._remote_issue) = old
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
    assert custom.spec.id == CUSTOM_ID
    assert CUSTOM_ID not in {entry["id"] for entry in shipped_entries()}
    assert all("symbol" not in entry for entry in entries)
    assert custom.spec.file_tag == "custom_model"
    assert custom.spec.can_stream
    assert set(custom.parameters) == {"temperature", "top_p", "top_k"}
    assert definitions.fingerprint != load_definitions().fingerprint


@pytest.mark.parametrize("orchestration", ["batch_size", "concurrent_requests"])
def test_orchestration_storage_is_identical_with_and_without_alias(tmp_path, orchestration) -> None:
    entries = shipped_entries()
    custom = make_custom_entry(higgs_entry())
    custom["sgl_omni"]["behavior"]["orchestration"] = orchestration
    builtin = next(entry for entry in entries if entry["id"] == "higgs_v3_sglomni")
    builtin.pop("settings", None)
    builtin["sgl_omni"]["behavior"]["orchestration"] = orchestration
    entries.append(custom)
    definitions = load_definitions(write_definitions(tmp_path, entries))
    assert definitions.models[CUSTOM_ID].settings == definitions.models["higgs_v3_sglomni"].settings
    registry = type(REGISTRY)()
    for definition in (definitions.models[CUSTOM_ID], definitions.models["higgs_v3_sglomni"]):
        registry.register_configured_model(definition)
        binding = registry.orchestration_binding(definition.spec.id)
        assert binding.name == orchestration
        assert binding.default == 1
        assert binding.group == ""
        top_k = registry.get(definition.spec.id, "top_k")
        assert top_k.default == 100
        assert top_k.has_sentinel and top_k.sentinel == -1


def test_loader_rejects_flat_storage_names_in_schema_v1(tmp_path) -> None:
    entries = shipped_entries()
    custom = make_custom_entry(higgs_entry())
    custom["spec"]["voice_target_attr"] = "higgs_v3_voice_file_name"
    entries.append(custom)

    with pytest.raises(ValueError, match=r"spec: unsupported field.*voice_target_attr"):
        load_definitions(write_definitions(tmp_path, entries))

    custom["spec"].pop("voice_target_attr")
    custom["sgl_omni"]["parameters"]["temperature"]["project_attr"] = "higgs_v3_temperature"
    with pytest.raises(ValueError, match=r"parameters.temperature: unsupported field.*project_attr"):
        load_definitions(write_definitions(tmp_path, entries))


def test_loader_rejects_replacing_a_local_builtin_with_server(tmp_path) -> None:
    entries = [entry for entry in shipped_entries() if entry["id"] != "chatterbox_local"]
    entry = make_custom_entry(higgs_entry())
    entry["id"] = "chatterbox_local"
    entries.append(entry)

    with pytest.raises(ValueError, match="model chatterbox_local: built-in ID and backend must match"):
        load_definitions(write_definitions(tmp_path, entries))


def test_startup_registers_new_model_storage_without_python_fields(configured_custom) -> None:
    definition = configured_custom.models[CUSTOM_ID]

    handle = TtsModelType.require_by_id(CUSTOM_ID)
    assert handle is not TtsModelType.require_by_id("none")
    assert handle.value is definition.spec
    assert Tts.get_configured_definition(handle) is definition
    assert TtsModelType.require_by_id("higgs_v3_sglomni") is TtsModelType.require_by_id("higgs_v3_sglomni")
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
    assert stored == {"temperature": 0.42, "top_p": None, "top_k": 12}

    reloaded = Project.model_validate(payload)
    assert reloaded.get_model_setting(CUSTOM_ID, "temperature") == 0.42
    assert reloaded.get_model_setting(CUSTOM_ID, "top_k") == 12


def test_new_model_support_and_generation_payload(configured_custom, monkeypatch) -> None:
    from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil

    Tts._type = TtsModelType.require_by_id(CUSTOM_ID)
    project = Project(tts_model_type=CUSTOM_ID)
    assert isinstance(Tts.get_model_support(project), ConfiguredModelSupport)
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


def test_local_mode_preserves_configured_model_metadata_and_storage(configured_custom, monkeypatch) -> None:
    assert REGISTRY.get(CUSTOM_ID, "temperature")
    assert TtsModelType.require_by_id(CUSTOM_ID) is not TtsModelType.require_by_id("none")

    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.LOCAL))
    Tts.init_local_model_type()

    assert TtsModelType.require_by_id(CUSTOM_ID) is not TtsModelType.require_by_id("none")
    assert REGISTRY.get(CUSTOM_ID, "temperature")
    assert Tts.get_configured_definition(TtsModelType.require_by_id("higgs_v3_sglomni")) is not None
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
