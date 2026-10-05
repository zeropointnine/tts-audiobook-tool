"""Phase-2 storage declaration and one-way migration building blocks."""
from __future__ import annotations

import pytest
from dataclasses import replace

from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import (
    Binding, ModelSettings, ModelSettingsRegistry, REGISTRY, normalize_paths,
    register_builtin_fields,
)
from tts_audiobook_tool.project_support.model_settings_declarations import BUILTIN_LEGACY_FIELDS


@pytest.fixture(autouse=True)
def builtins() -> None:
    register_builtin_fields()


def test_declarations_cover_all_legacy_model_fields() -> None:
    # Version 3 removed the flat model fields from Project entirely; the
    # declaration table is now the only record of their storage bindings.
    assert not set(BUILTIN_LEGACY_FIELDS) & set(Project.model_fields)
    assert set(REGISTRY.legacy) == set(BUILTIN_LEGACY_FIELDS)


def test_catalog_voice_and_batch_presence_comes_from_storage_bindings() -> None:
    for model in TtsModelType.all():
        voice = REGISTRY.voice_binding(model.id)
        transcript = REGISTRY.transcript_binding(model.id)
        orchestration = REGISTRY.orchestration_binding(model.id)
        if transcript is not None:
            assert voice is not None
            assert transcript.section == "voice_references"
        if voice is not None:
            assert voice.section == "voice_references"
        assert model.can_batch() == (orchestration is not None)
        if orchestration is not None:
            assert orchestration.section == "orchestration"
    assert REGISTRY.voice_binding("fish_s2_local").group == REGISTRY.voice_binding("fish_s2_sglomni").group == "fish_s2"
    assert REGISTRY.transcript_binding("zonos2_sglomni") is None


def test_audio_cpp_models_have_no_orchestration_setting() -> None:
    # audio.cpp's server serializes requests per model, so no family stores a
    # batch/concurrency value and `can_batch()` must stay False for all of them.
    models = TtsModelType.get_items_by_backend(TtsBackendKind.AUDIO_CPP)
    assert models
    for model in models:
        assert REGISTRY.orchestration_binding(model.id) is None
        assert not model.can_batch()


def test_retired_audio_cpp_concurrent_requests_is_dropped_on_load() -> None:
    from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
    from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions

    model_id = "chatterbox_audiocpp"
    if not REGISTRY.for_model(model_id):
        REGISTRY.register_audio_cpp_model(load_audio_cpp_definitions().models[model_id])
    source = {"models": {model_id: {"orchestration": {"concurrent_requests": 3},
                                    "parameters": {"top_p": 0.9}}}, "shared": {}}
    # A project saved when audio.cpp still had the setting must still load, with
    # the now-unknown value dropped rather than treated as foreign ownership.
    settings = REGISTRY.reconcile(source)
    assert settings.models[model_id] == {"parameters": {"top_p": 0.9}}
    project = Project.model_validate(
        {"version": 3, "tts_model_type": model_id, "model_settings": source})
    assert project.model_settings.models[model_id] == {"parameters": {"top_p": 0.9}}
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "orchestration" not in saved["model_settings"]["models"][model_id]


def test_shared_group_owns_only_declared_fields() -> None:
    assert REGISTRY.get("fish_s2_local", "temperature").group == "fish_s2"
    assert REGISTRY.get("fish_s2_sglomni", "temperature").group == "fish_s2"
    assert REGISTRY.get("fish_s2_local", "compile_enabled").group == ""
    assert REGISTRY.get("fish_s2_sglomni", "concurrent_requests").group == ""
    assert REGISTRY.get("auk_flash_sglomni", "speed").group == "auk"
    assert REGISTRY.get("moss_delay_sglomni", "delay_top_k").group == ""
    with pytest.raises(ValueError, match="Unknown setting"):
        REGISTRY.get("moss_delay_sglomni", "local_top_k")
    assert REGISTRY.get("qwen3tts_sglomni", "temperature").group == "qwen3"
    with pytest.raises(ValueError, match="Unknown setting"):
        REGISTRY.get("qwen3tts_sglomni", "rolling_cont")


def test_legacy_migration_pairs_voices_and_preserves_shared_ownership() -> None:
    settings = REGISTRY.reconcile({}, {
        "fish_s2_voice_file_name": ["a.flac", "b.flac"],
        "fish_s2_voice_transcript": ["first"],
        "fish_s2_temperature": -1,
        "fish_s2_top_k": 50,
        "fish_s2_seed": -1,
        "fish_s2_compile_enabled": False,
        "fish_s2_server_concurrent_requests": 3,
    })
    assert settings.shared["fish_s2"]["model_ids"] == ["fish_s2_local", "fish_s2_sglomni"]
    assert settings.shared["fish_s2"]["voice_references"] == [
        {"file_name": "a.flac", "transcript": "first"},
        {"file_name": "b.flac", "transcript": ""},
    ]
    assert settings.shared["fish_s2"]["parameters"] == {"top_k": 50, "seed": -1}
    assert settings.models["fish_s2_local"]["parameters"] == {"compile_enabled": False}
    assert settings.models["fish_s2_sglomni"]["orchestration"] == {"concurrent_requests": 3}
    assert REGISTRY.resolve(settings, REGISTRY.get("fish_s2_local", "temperature")) == -1
    REGISTRY.assign(settings, REGISTRY.get("fish_s2_sglomni", "temperature"), 0.8)
    assert REGISTRY.resolve(settings, REGISTRY.get("fish_s2_local", "temperature")) == 0.8
    REGISTRY.assign(settings, REGISTRY.get("fish_s2_local", "temperature"), None, reset=True)
    assert "temperature" not in settings.shared["fish_s2"].get("parameters", {})


def test_new_objects_take_precedence_and_unknown_objects_round_trip() -> None:
    source = {
        "models": {
            "fish_s2_local": {"parameters": {"compile_enabled": False, "old_unused": 12}},
            "absent_custom": {"experimental": {"anything": [1, 2]}},
        },
        "shared": {
            "fish_s2": {"model_ids": ["fish_s2_local", "fish_s2_sglomni"], "parameters": {"top_k": 60, "old_unused": 12}},
            "unavailable": {"model_ids": ["absent_custom"], "future": "untouched"},
        },
    }
    settings = REGISTRY.reconcile(source, {
        "fish_s2_compile_enabled": True, "fish_s2_temperature": 0.7,
        "fish_s2_top_k": 99, "fish_s2_voice_file_name": ["old.flac"],
    })
    assert settings.models["fish_s2_local"] == {"parameters": {"compile_enabled": False}}
    assert settings.shared["fish_s2"] == {
        "model_ids": ["fish_s2_local", "fish_s2_sglomni"], "parameters": {"top_k": 60},
    }
    assert settings.models["absent_custom"] == source["models"]["absent_custom"]
    assert settings.shared["unavailable"] == source["shared"]["unavailable"]
    assert REGISTRY.reconcile(settings.to_dict()).to_dict() == settings.to_dict()


def test_unauthorized_membership_rejected_without_mutating_source() -> None:
    source = {"shared": {"fish_s2": {"model_ids": ["fish_s2_local", "unauthorized"]}}}
    with pytest.raises(ValueError, match="model_ids"):
        REGISTRY.reconcile(source)
    assert source["shared"]["fish_s2"]["model_ids"] == ["fish_s2_local", "unauthorized"]


def test_invalid_parameters_are_pruned_without_losing_valid_overrides() -> None:
    source = {"models": {"fish_s2_local": {"parameters": {
        "temperature": 0.8,  # shared owner, not this private model object
        "future_unknown": 3,
        "rolling_cont": 2,
        "compile_enabled": "not a boolean",
    }}}, "shared": {"fish_s2": {
        "model_ids": ["fish_s2_local", "fish_s2_sglomni"],
        "parameters": {"compile_enabled": True, "temperature": 0.7},
    }}}
    settings = REGISTRY.reconcile(source)
    assert settings.models["fish_s2_local"] == {"parameters": {"rolling_cont": 2}}
    assert settings.shared["fish_s2"]["parameters"] == {"temperature": 0.7}
    assert source["models"]["fish_s2_local"]["parameters"]["temperature"] == 0.8
    assert REGISTRY.reconcile({"models": {"fish_s2_local": {"parameters": []}}}).models["fish_s2_local"] == {}

    # File references and shared-group membership retain their stricter
    # ownership/structure checks; model parameters and explicitly retired
    # settings (RETIRED_SETTINGS) are the disposable ones.
    with pytest.raises(ValueError, match="different storage owner"):
        REGISTRY.reconcile({"models": {"fish_s2_local": {"files": {"emo_voice": "sample.flac"}}}})
    # A model that really owns concurrent_requests keeps its value.
    assert REGISTRY.reconcile({"models": {
        "fish_s2_sglomni": {"orchestration": {"concurrent_requests": 3}}}}).models["fish_s2_sglomni"] == {
            "orchestration": {"concurrent_requests": 3}}


def test_removed_parameter_colliding_with_another_models_name_is_pruned() -> None:
    registry = ModelSettingsRegistry()
    registry.add(Binding("other", "speed", "parameters", default=1.0, value_type=float))
    registry.add(Binding("custom", "temperature", "parameters", default=0.7, value_type=float))
    source = {"models": {"custom": {"parameters": {"speed": 1.3, "temperature": 0.6}}}}
    assert registry.reconcile(source).models["custom"] == {"parameters": {"temperature": 0.6}}


def test_scalar_paths_normalize_without_rewriting_opaque_strings_or_unknown_models() -> None:
    settings = REGISTRY.reconcile({"models": {
        "indextts2_local": {"files": {"emo_voice": "C:\\elsewhere\\voice\\emo.flac"},
                      "voice_references": [{"file_name": "C:\\elsewhere\\voice\\main.flac"}]},
        "unknown": {"files": {"path": "C:\\untouched\\sound.flac"}},
    }})
    assert normalize_paths(settings)
    assert settings.models["indextts2_local"]["files"]["emo_voice"] == "emo.flac"
    assert settings.models["indextts2_local"]["voice_references"][0]["file_name"] == "main.flac"
    assert settings.models["unknown"]["files"]["path"] == "C:\\untouched\\sound.flac"


def test_new_configured_id_has_no_python_project_field() -> None:
    definition = load_definitions().models["higgs_v3_sglomni"]
    custom = replace(definition, spec=definition.spec._replace(id="server_custom"))
    registry = ModelSettingsRegistry()
    registry.register_configured_model(custom)
    assert "server_custom_temperature" not in Project.model_fields
    settings = registry.reconcile({"models": {"server_custom": {
        "parameters": {"temperature": 0.8},
        "voice_references": [{"file_name": "example.flac", "transcript": "Example."}],
    }}})
    assert registry.resolve(settings, registry.get("server_custom", "temperature")) == 0.8
    assert custom.settings is definition.settings
    assert registry.resolve(settings, registry.get("server_custom", "top_k")) == -1
    assert settings.to_dict()["models"]["server_custom"]["voice_references"][0]["file_name"] == "example.flac"
    registry.reset_to_builtins()
    assert registry.reconcile(settings.to_dict()).models["server_custom"] == settings.models["server_custom"]


def test_explicit_value_equal_to_default_is_not_removed() -> None:
    registry = ModelSettingsRegistry()
    registry.add(Binding("custom", "temperature", "parameters", default=0.9, value_type=float))
    store = ModelSettings()
    registry.assign(store, registry.get("custom", "temperature"), 0.9)
    assert store.models["custom"]["parameters"] == {"temperature": 0.9}
    registry.assign(store, registry.get("custom", "temperature"), None, reset=True)
    assert registry.resolve(store, registry.get("custom", "temperature")) == 0.9


def test_every_current_binding_is_declared_by_the_catalog() -> None:
    from tts_audiobook_tool.tts_models.model_catalog import load_settings_catalog

    groups, models = load_settings_catalog()
    assert REGISTRY.members == groups
    expected = {(model_id, setting["name"]): REGISTRY._binding(model_id, setting)
                for model_id, settings in models.items() for setting in settings}
    assert REGISTRY.bindings == expected


def test_legacy_defaults_do_not_initialize_current_settings(monkeypatch) -> None:
    before = REGISTRY.get("fish_s2_local", "temperature")
    with monkeypatch.context() as patch:
        patch.setitem(BUILTIN_LEGACY_FIELDS, "fish_s2_temperature", ("float", 0.123))
        registry = ModelSettingsRegistry()
        registry.reset_to_builtins()
        assert registry.get("fish_s2_local", "temperature") == before
        assert registry.legacy["fish_s2_temperature"].default == 0.123


def test_current_bindings_retain_frozen_legacy_storage_contracts() -> None:
    from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType

    types = {"int": (int, object), "float": (float, object),
             "str": (str, object), "bool": (bool, object),
             "list[str]": (list, str), "list[float]": (list, float),
             "enum:chatterbox": (ChatterboxType, object)}
    for attr, legacy in REGISTRY.legacy.items():
        kind, default = BUILTIN_LEGACY_FIELDS[attr]
        current = REGISTRY.get(legacy.model_id, legacy.name)
        if legacy.group == "moss":
            # Historical sharing remains frozen; current MOSS ownership forked.
            assert "moss" not in REGISTRY.members
            assert current.group == ""
            assert replace(current, group="moss") == legacy, attr
        else:
            assert current == legacy, attr
        assert (current.value_type, current.list_item_type) == types[kind], attr
        assert current.default == default, attr
        assert current.has_sentinel == (default == -1 and not attr.endswith("_seed")), attr
        assert current.preserve_default == attr.endswith("_seed"), attr
        if current.group:
            for member in REGISTRY.members[current.group]:
                assert replace(REGISTRY.get(member, current.name), model_id=current.model_id) == current, attr
