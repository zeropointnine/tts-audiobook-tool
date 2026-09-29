"""Phase-2 storage declaration and one-way migration building blocks."""
from __future__ import annotations

import pytest
from dataclasses import replace

from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

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
    assert REGISTRY.voice_binding("fish_s2").group == REGISTRY.voice_binding("server_fish_s2").group == "fish_s2"
    assert REGISTRY.transcript_binding("server_zonos2") is None


def test_shared_group_owns_only_declared_fields() -> None:
    assert REGISTRY.get("fish_s2", "temperature").group == "fish_s2"
    assert REGISTRY.get("server_fish_s2", "temperature").group == "fish_s2"
    assert REGISTRY.get("fish_s2", "compile_enabled").group == ""
    assert REGISTRY.get("server_fish_s2", "concurrent_requests").group == ""
    assert REGISTRY.get("server_auk_flash", "speed").group == "auk"
    assert REGISTRY.get("server_moss_delay", "local_top_k").group == "moss"
    assert REGISTRY.get("server_qwen3tts", "temperature").group == "qwen3"
    with pytest.raises(ValueError, match="Unknown setting"):
        REGISTRY.get("server_qwen3tts", "rolling_cont")


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
    assert settings.shared["fish_s2"]["model_ids"] == ["fish_s2", "server_fish_s2"]
    assert settings.shared["fish_s2"]["voice_references"] == [
        {"file_name": "a.flac", "transcript": "first"},
        {"file_name": "b.flac", "transcript": ""},
    ]
    assert settings.shared["fish_s2"]["parameters"] == {"top_k": 50, "seed": -1}
    assert settings.models["fish_s2"]["parameters"] == {"compile_enabled": False}
    assert settings.models["server_fish_s2"]["orchestration"] == {"concurrent_requests": 3}
    assert REGISTRY.resolve(settings, REGISTRY.get("fish_s2", "temperature")) == -1
    REGISTRY.assign(settings, REGISTRY.get("server_fish_s2", "temperature"), 0.8)
    assert REGISTRY.resolve(settings, REGISTRY.get("fish_s2", "temperature")) == 0.8
    REGISTRY.assign(settings, REGISTRY.get("fish_s2", "temperature"), None, reset=True)
    assert "temperature" not in settings.shared["fish_s2"].get("parameters", {})


def test_new_objects_take_precedence_and_unknown_objects_round_trip() -> None:
    source = {
        "models": {
            "fish_s2": {"parameters": {"compile_enabled": False, "old_unused": 12}},
            "absent_custom": {"experimental": {"anything": [1, 2]}},
        },
        "shared": {
            "fish_s2": {"model_ids": ["fish_s2", "server_fish_s2"], "parameters": {"top_k": 60, "old_unused": 12}},
            "unavailable": {"model_ids": ["absent_custom"], "future": "untouched"},
        },
    }
    settings = REGISTRY.reconcile(source, {
        "fish_s2_compile_enabled": True, "fish_s2_temperature": 0.7,
        "fish_s2_top_k": 99, "fish_s2_voice_file_name": ["old.flac"],
    })
    assert settings.models["fish_s2"] == {"parameters": {"compile_enabled": False}}
    assert settings.shared["fish_s2"] == {
        "model_ids": ["fish_s2", "server_fish_s2"], "parameters": {"top_k": 60},
    }
    assert settings.models["absent_custom"] == source["models"]["absent_custom"]
    assert settings.shared["unavailable"] == source["shared"]["unavailable"]
    assert REGISTRY.reconcile(settings.to_dict()).to_dict() == settings.to_dict()


def test_unauthorized_membership_rejected_without_mutating_source() -> None:
    source = {"shared": {"fish_s2": {"model_ids": ["fish_s2", "unauthorized"]}}}
    with pytest.raises(ValueError, match="model_ids"):
        REGISTRY.reconcile(source)
    assert source["shared"]["fish_s2"]["model_ids"] == ["fish_s2", "unauthorized"]


def test_invalid_parameters_are_pruned_without_losing_valid_overrides() -> None:
    source = {"models": {"fish_s2": {"parameters": {
        "temperature": 0.8,  # shared owner, not this private model object
        "future_unknown": 3,
        "rolling_cont": 2,
        "compile_enabled": "not a boolean",
    }}}, "shared": {"fish_s2": {
        "model_ids": ["fish_s2", "server_fish_s2"],
        "parameters": {"compile_enabled": True, "temperature": 0.7},
    }}}
    settings = REGISTRY.reconcile(source)
    assert settings.models["fish_s2"] == {"parameters": {"rolling_cont": 2}}
    assert settings.shared["fish_s2"]["parameters"] == {"temperature": 0.7}
    assert source["models"]["fish_s2"]["parameters"]["temperature"] == 0.8
    assert REGISTRY.reconcile({"models": {"fish_s2": {"parameters": []}}}).models["fish_s2"] == {}

    # File references and shared-group membership retain their stricter
    # ownership/structure checks; only model parameters are disposable.
    with pytest.raises(ValueError, match="different storage owner"):
        REGISTRY.reconcile({"models": {"fish_s2": {"files": {"emo_voice": "sample.flac"}}}})


def test_removed_parameter_colliding_with_another_models_name_is_pruned() -> None:
    registry = ModelSettingsRegistry()
    registry.add(Binding("other", "speed", "parameters", default=1.0, value_type=float))
    registry.add(Binding("custom", "temperature", "parameters", default=0.7, value_type=float))
    source = {"models": {"custom": {"parameters": {"speed": 1.3, "temperature": 0.6}}}}
    assert registry.reconcile(source).models["custom"] == {"parameters": {"temperature": 0.6}}


def test_scalar_paths_normalize_without_rewriting_opaque_strings_or_unknown_models() -> None:
    settings = REGISTRY.reconcile({"models": {
        "indextts2": {"files": {"emo_voice": "C:\\elsewhere\\voice\\emo.flac"},
                      "voice_references": [{"file_name": "C:\\elsewhere\\voice\\main.flac"}]},
        "unknown": {"files": {"path": "C:\\untouched\\sound.flac"}},
    }})
    assert normalize_paths(settings)
    assert settings.models["indextts2"]["files"]["emo_voice"] == "emo.flac"
    assert settings.models["indextts2"]["voice_references"][0]["file_name"] == "main.flac"
    assert settings.models["unknown"]["files"]["path"] == "C:\\untouched\\sound.flac"


def test_new_configured_id_has_no_python_project_field() -> None:
    definition = load_definitions().models["server_higgs_v3"]
    custom = replace(definition, spec=definition.spec._replace(id="server_custom"))
    registry = ModelSettingsRegistry()
    registry.register_configured_model(custom)
    assert "server_custom_temperature" not in Project.model_fields
    settings = registry.reconcile({"models": {"server_custom": {
        "parameters": {"temperature": 0.8},
        "voice_references": [{"file_name": "example.flac", "transcript": "Example."}],
    }}})
    assert registry.resolve(settings, registry.get("server_custom", "temperature")) == 0.8
    assert registry.resolve(settings, registry.get("server_custom", "top_k")) == 100
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
