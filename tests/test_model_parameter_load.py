"""Saved catalog parameters fall back to defaults before menus/inference read them."""
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import ModelSettingsRegistry, REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppSettings
from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppParameter, load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions


@pytest.mark.parametrize("invalid", [0.0, 2.01, True, "0.8", [], {}, float("nan"), float("inf")])
def test_invalid_audio_cpp_saved_parameter_resets_without_menu_error(invalid, caplog):
    model_id = "breeze_tts_2_audiocpp"
    parameter = cast(AudioCppParameter, load_audio_cpp_definitions().models[model_id].parameters["temperature"])
    source = {"models": {model_id: {"parameters": {"temperature": invalid, "top_p": 0.7, "seed": 123}}}}
    project = Project.model_validate({"version": 3, "tts_model_type": model_id, "model_settings": source})

    assert project.get_model_setting(model_id, "temperature") == parameter.default
    assert AudioCppSettings.get(project, parameter) == parameter.default
    assert project.model_settings.models[model_id]["parameters"] == {"top_p": 0.7, "seed": 123}
    assert source["models"][model_id]["parameters"]["temperature"] is invalid
    assert f"models.{model_id}.parameters.temperature" in caplog.text
    assert "resetting to default" in caplog.text

    state = cast(State, SimpleNamespace(project=project))
    item = VoiceAudioCppMenu.make_parameter_item(state, parameter, "Temperature")
    assert callable(item.label)
    assert "invalid" not in item.label(state)
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["model_settings"]["models"][model_id]["parameters"]["temperature"] is None


@pytest.mark.parametrize("value", [0.01, 2.0, 1, 0.8])
def test_valid_audio_cpp_saved_parameter_is_preserved(value, caplog):
    model_id = "breeze_tts_2_audiocpp"
    settings = REGISTRY.reconcile({"models": {model_id: {"parameters": {"temperature": value}}}})
    assert settings.models[model_id]["parameters"]["temperature"] == value
    assert not caplog.records


@pytest.mark.parametrize("value", [0, 101, 30.0, False, "30"])
def test_invalid_shared_sgl_parameter_resets_for_both_members(value, caplog):
    parameter = load_definitions().models["fish_s2_sglomni"].parameters["top_k"]
    project = Project.model_validate({"version": 3, "model_settings": {"shared": {"fish_s2": {
        "model_ids": list(REGISTRY.members["fish_s2"]),
        "parameters": {"top_k": value, "temperature": 0.7},
        "voice_references": [{"file_name": "voice.flac", "transcript": "Hello."}],
    }}}})
    assert project.get_model_setting("fish_s2_local", "top_k") == parameter.default_sentinel
    assert ConfiguredSettings.get(project, parameter) == parameter.default
    assert project.model_settings.shared["fish_s2"]["parameters"] == {"temperature": 0.7}
    assert project.model_settings.shared["fish_s2"]["voice_references"][0]["file_name"] == "voice.flac"
    assert "shared.fish_s2.parameters.top_k" in caplog.text


def test_saved_sentinels_and_missing_parameters_do_not_warn(caplog):
    parameter = load_definitions().models["zonos2_sglomni"].parameters["temperature"]
    for parameters in ({}, {"temperature": parameter.default_sentinel}):
        project = Project.model_validate({"version": 3, "model_settings": {"models": {
            parameter.model_id: {"parameters": parameters},
        }}})
        assert ConfiguredSettings.get(project, parameter) == parameter.default
    assert not caplog.records


@pytest.mark.parametrize("backend", ["audio_cpp", "sgl_omni"])
def test_dynamic_registration_uses_updated_bounds_and_keeps_unknown_objects(backend, caplog):
    registry = ModelSettingsRegistry()
    if backend == "audio_cpp":
        definition = load_audio_cpp_definitions().models["breeze_tts_2_audiocpp"]
        register = registry.register_audio_cpp_model
    else:
        definition = load_definitions().models["zonos2_sglomni"]
        register = registry.register_configured_model
    custom = replace(definition, spec=definition.spec._replace(id="custom"))
    source = {"models": {"custom": {"parameters": {"temperature": 1.5}},
                         "unknown": {"parameters": {"temperature": None}}}}
    register(custom)
    assert registry.reconcile(source).models == source["models"]
    parameters = dict(custom.parameters)
    parameters["temperature"] = replace(parameters["temperature"], max=1.0)
    register(replace(custom, parameters=parameters))
    cleaned = registry.reconcile(source)
    assert cleaned.models["custom"] == {}
    assert cleaned.models["unknown"] == source["models"]["unknown"]
    assert "models.custom.parameters.temperature" in caplog.text
