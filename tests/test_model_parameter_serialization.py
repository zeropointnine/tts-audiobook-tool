"""Document declared parameters while retaining unset/default intent in JSON."""
from copy import deepcopy
from dataclasses import replace
import json

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
        collection, owner = ("shared", binding.group) if binding.group else ("models", binding.model_id)
        obj = saved["model_settings"][collection][owner]
        assert obj["parameters"][binding.name] is None
        if binding.group:
            assert obj["model_ids"] == list(REGISTRY.members[binding.group])
            private = saved["model_settings"]["models"].get(binding.model_id, {}).get("parameters", {})
            assert binding.name not in private
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
    }}, "shared": {"qwen3": {
        "model_ids": list(REGISTRY.members["qwen3"]),
        "parameters": {"top_k": -1, "seed": -1, "temperature": 0.7},
    }}}})
    saved = ProjectSerializationUtil.to_project_json_dict(project)["model_settings"]
    assert saved["models"]["zonos2_sglomni"]["parameters"]["temperature"] is None
    assert saved["shared"]["qwen3"]["parameters"]["top_k"] is None
    assert saved["shared"]["qwen3"]["parameters"]["seed"] == -1
    assert saved["shared"]["qwen3"]["parameters"]["temperature"] == 0.7
    restored = Project.model_validate({"version": 3, "model_settings": saved})
    assert restored.get_model_setting("zonos2_sglomni", "temperature") == -1.0
    assert restored.get_model_setting("qwen3tts_local", "top_k") == -1
    assert restored.get_model_setting("qwen3tts_local", "seed") == -1
    assert not caplog.records


def test_serialization_preserves_false_zero_empty_text_and_nonparameter_sections():
    project = Project()
    project.set_model_setting("fish_s2_local", "compile_enabled", False)
    project.set_model_setting("fish_s2_local", "seed", 0)
    project.set_model_setting("omnivoice_local", "instruct", "")
    project.set_model_setting("fish_s2_sglomni", "concurrent_requests", 3)
    saved = ProjectSerializationUtil.to_project_json_dict(project)["model_settings"]
    assert saved["models"]["fish_s2_local"]["parameters"]["compile_enabled"] is False
    assert saved["models"]["fish_s2_local"]["parameters"]["seed"] == 0
    assert saved["models"]["omnivoice_local"]["parameters"]["instruct"] == ""
    assert saved["models"]["fish_s2_sglomni"]["orchestration"] == {"concurrent_requests": 3}
    assert "orchestration" not in saved["models"]["breeze_tts_2_audiocpp"]


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
