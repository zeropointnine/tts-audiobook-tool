"""Document declared parameters while retaining unset/default intent in JSON."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import Binding, ModelSettings, ModelSettingsRegistry, REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions


def test_save_documents_every_declared_parameter_without_mutating_store(caplog):
    project = Project()
    before = project.model_settings.to_dict()
    saved = json.loads(json.dumps(ProjectSerializationUtil.to_project_json_dict(project)))
    for binding in REGISTRY.bindings.values():
        if binding.section != "parameters":
            continue
        obj = saved["model_settings"]["models"][binding.model_id]
        assert obj["parameters"][binding.name] is None
    # Every parameter is private, so nothing is documented under a shared group.
    assert saved["model_settings"]["shared"] == {}
    assert project.model_settings.to_dict() == before

    restored = Project.model_validate(deepcopy(saved))
    for binding in REGISTRY.bindings.values():
        if binding.section == "parameters":
            assert restored.get_model_setting(binding.model_id, binding.name) == project.get_model_setting(binding.model_id, binding.name)
    assert ProjectSerializationUtil.to_project_json_dict(restored) == saved
    assert not caplog.records


def test_null_and_missing_keys_follow_changed_defaults_but_explicit_values_stay_pinned(caplog):
    registry = ModelSettingsRegistry()
    binding = Binding("custom", "temperature", "parameters", default=0.8, value_type=float)
    registry.add(binding)
    unset = registry.serialize(ModelSettings())
    pinned_store = ModelSettings()
    registry.assign(pinned_store, binding, 0.8)
    pinned = registry.serialize(pinned_store)
    assert unset["models"]["custom"]["parameters"] == {"temperature": None}
    assert pinned["models"]["custom"]["parameters"] == {"temperature": 0.8}

    changed = replace(binding, default=0.9)
    registry.bindings[("custom", "temperature")] = changed
    for source in (unset, {"models": {"custom": {"parameters": {}}}}, {}):
        assert registry.resolve(registry.reconcile(source), changed) == 0.9
    assert registry.resolve(registry.reconcile(pinned), changed) == 0.8
    assert not caplog.records


def test_null_is_unset_for_every_parameter_type_without_warning(caplog):
    registry = ModelSettingsRegistry()
    bindings = [
        Binding("custom", "number", "parameters", default=3, value_type=int),
        Binding("custom", "decimal", "parameters", default=0.8, value_type=float),
        Binding("custom", "text", "parameters", default="hello", value_type=str),
        Binding("custom", "enabled", "parameters", default=True, value_type=bool),
        Binding("custom", "items", "parameters", default=[1.0], value_type=list, list_item_type=float),
    ]
    for binding in bindings:
        registry.add(binding)
    source = {"models": {"custom": {"parameters": {b.name: None for b in bindings}}}}
    store = registry.reconcile(source)
    assert store.models["custom"] == {}
    for binding in bindings:
        assert registry.resolve(store, binding) == binding.default
    assert not caplog.records


def test_old_unset_sentinels_serialize_as_null_but_random_seed_stays_minus_one(caplog):
    project = Project.model_validate({"version": 3, "model_settings": {"models": {"zonos2_sglomni": {
        "parameters": {"temperature": -1.0},
    }, "qwen3tts_local": {
        "parameters": {"top_k": -1, "seed": -1, "temperature": 0.7},
    }}}})
    saved = ProjectSerializationUtil.to_project_json_dict(project)["model_settings"]
    assert saved["models"]["zonos2_sglomni"]["parameters"]["temperature"] is None
    assert saved["models"]["qwen3tts_local"]["parameters"]["top_k"] is None
    assert saved["models"]["qwen3tts_local"]["parameters"]["seed"] == -1
    assert saved["models"]["qwen3tts_local"]["parameters"]["temperature"] == 0.7
    restored = Project.model_validate({"version": 3, "model_settings": saved})
    assert restored.get_model_setting("zonos2_sglomni", "temperature") == -1.0
    assert restored.get_model_setting("qwen3tts_local", "top_k") == -1
    assert restored.get_model_setting("qwen3tts_local", "seed") == -1
    assert not caplog.records


def test_serialization_preserves_false_zero_empty_text_and_nonparameter_sections():
    project = Project()
    project.set_model_setting("fish_s2_local", "compile_enabled", False)
    project.set_model_setting("fish_s2_local", "seed", 0)
    project.set_model_setting("omnivoice_local", "target", "")
    project.set_model_setting("fish_s2_sglomni", "concurrent_requests", 3)
    saved = ProjectSerializationUtil.to_project_json_dict(project)["model_settings"]
    assert saved["models"]["fish_s2_local"]["parameters"]["compile_enabled"] is False
    assert saved["models"]["fish_s2_local"]["parameters"]["seed"] == 0
    assert saved["models"]["omnivoice_local"]["parameters"]["target"] == ""
    assert saved["models"]["fish_s2_sglomni"]["orchestration"] == {"concurrent_requests": 3}
    assert "orchestration" not in saved["models"]["breeze_tts_2_audiocpp"]


@pytest.mark.parametrize(("model_id", "old_name"), [
    ("omnivoice_local", "instruct"),
    ("qwen3tts_local", "instructions"),
    ("breeze_tts_2_audiocpp", "instruct"),
    ("omnivoice_audiocpp", "instruct"),
    ("cosyvoice3_audiocpp", "instruction"),
])
def test_instructions_are_only_voice_data_not_model_settings(model_id, old_name):
    voices = [Voice(instructions={model_id: "Speak softly."}),
              Voice(instructions={model_id: "Speak brightly.", "other_model": "Unchanged."})]
    source = ProjectSerializationUtil.to_project_json_dict(Project(voice_references=voices))
    # Obsolete global values are ordinary undeclared parameters: dropped, not
    # migrated, even when a voice already has its own instruction for this model.
    source["model_settings"]["models"][model_id]["parameters"][old_name] = "Discard global value."
    restored = Project.model_validate(deepcopy(source))
    assert restored.voice_references == voices
    with pytest.raises(ValueError, match="Unknown setting"):
        restored.get_model_setting(model_id, old_name)
    with pytest.raises(ValueError, match="Unknown setting"):
        restored.set_model_setting(model_id, old_name, "No global editor")
    saved = ProjectSerializationUtil.to_project_json_dict(restored)
    assert old_name not in saved["model_settings"]["models"][model_id]["parameters"]
    assert saved["voice_references"] == [voice.to_dict() for voice in voices]
    assert Project.model_validate(deepcopy(saved)).voice_references == voices


def test_flat_qwen_instruction_field_has_no_legacy_binding_or_migration():
    assert "qwen3_instructions" not in REGISTRY.legacy
    restored = Project.model_validate({"qwen3_instructions": "Discard global value."})
    assert all(not voice.instructions for voice in restored.voice_references)
    assert "qwen3_instructions" not in ProjectSerializationUtil.to_project_json_dict(restored)


def test_dynamic_models_and_unknown_objects_keep_their_owners():
    registry = ModelSettingsRegistry()
    definition = load_audio_cpp_definitions().models["breeze_tts_2_audiocpp"]
    custom = replace(definition, spec=definition.spec._replace(id="custom"))
    registry.register_audio_cpp_model(custom)
    source = {"models": {"unknown": {"parameters": {"temperature": None, "future": -1}}},
              "shared": {"future_group": {"model_ids": ["unknown"], "future": [1, 2]}}}
    before = deepcopy(source)
    store = registry.reconcile(source)
    saved = registry.serialize(store)
    assert saved["models"]["custom"]["parameters"] == {name: None for name in (*custom.parameters, "seed")}
    assert saved["models"]["unknown"] == source["models"]["unknown"]
    assert saved["shared"]["future_group"] == source["shared"]["future_group"]
    assert source == before
    assert store.to_dict() == source
