"""Catalog-driven setting destinations reuse Model editors in both menus."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
from tts_audiobook_tool.menus.model.model_configured_sgl_omni_menu import ModelConfiguredSglOmniMenu
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu
from tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu import VoiceConfiguredSglOmniMenu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppMenuControl, load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.sgl_omni_definition import MenuControl, load_definitions


BACKENDS = [
    (load_audio_cpp_definitions, "breeze_tts_2_audiocpp", AudioCppMenuControl,
     ModelAudioCppMenu, VoiceAudioCppMenu),
    (load_definitions, "qwen3tts_sglomni", MenuControl,
     ModelConfiguredSglOmniMenu, VoiceConfiguredSglOmniMenu),
]


@pytest.fixture(params=BACKENDS, ids=["audio_cpp", "sgl_omni"])
def backend(request):
    loader, model_id, control_type, model_menu, voice_menu = request.param
    return loader().models[model_id], control_type, model_menu, voice_menu


def test_mixed_destinations_preserve_order_and_exclusively_partition_controls(backend, monkeypatch):
    definition, control_type, model_menu, voice_menu = backend
    controls = [
        control_type("parameter", "voice", "temperature", "Voice temperature"),
        control_type("seed", "model"),
        control_type("voice_samples", None),
        control_type("parameter", "model", "top_p", "Model Top-P"),
        control_type("parameter", "voice", "top_k", "Voice Top-K"),
    ]
    if model_menu is ModelAudioCppMenu:
        controls.insert(4, control_type("voice_instructions", "voice", "instruct"))
    definition = replace(definition, menu=tuple(controls))
    state = SimpleNamespace(project=Project(tts_model_type=definition.spec.id))
    sample_calls = []

    def make_samples(current, model_type):
        assert current is state and model_type.id == definition.spec.id
        sample_calls.append(model_type.id)
        return [MenuItem("Add sample", lambda *_: None), MenuItem("Select samples", lambda *_: None)]

    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", make_samples)
    # Routing must use the existing Model editors, not introduce Voice editors.
    monkeypatch.setattr(model_menu, "make_parameter_item",
                        lambda _state, _parameter, label, *_, **__: MenuItem(label, lambda *_: None))
    monkeypatch.setattr(ModelMenuShared, "make_seed_item",
                        lambda *_, **__: MenuItem("Seed", lambda *_: None))
    monkeypatch.setattr(ModelMenuShared, "make_voice_instructions_item",
                        lambda *_, **__: [MenuItem("Edit instructions", lambda *_: None),
                                          MenuItem("Clear instructions", lambda *_: None)])
    voice_items = voice_menu.make_items(state, definition)
    model_items = model_menu.make_items(state, definition)
    expected_voice = ["Voice temperature", "Add sample", "Select samples", "Voice Top-K"]
    if model_menu is ModelAudioCppMenu:
        expected_voice[3:3] = ["Edit instructions", "Clear instructions"]
    assert [item.label for item in voice_items] == expected_voice
    assert [item.label for item in model_items] == ["Seed", "Model Top-P"]
    assert {item.label for item in voice_items}.isdisjoint(item.label for item in model_items)
    assert all(not item.superlabel for item in voice_items + model_items)
    assert [item.blank_line_before for item in voice_items] == [True] + [False] * (len(voice_items) - 1)
    assert all(not item.blank_line_before for item in model_items)
    assert sample_calls == [definition.spec.id]
    assert not hasattr(voice_menu, "make_parameter_item")


def test_each_destination_delegates_only_its_settings_to_model_control_helper(backend, monkeypatch):
    definition, control_type, model_menu, voice_menu = backend
    controls = (
        control_type("seed", "voice"),
        control_type("voice_samples", None),
        control_type("parameter", "model", "temperature", "Temperature"),
        control_type("parameter", "voice", "top_k", "Top-K"),
    )
    definition = replace(definition, menu=controls)
    state = SimpleNamespace(project=Project(tts_model_type=definition.spec.id))
    calls = []

    def expand(current, current_definition, control):
        assert current is state and current_definition is definition
        calls.append(control)
        return [MenuItem(control.kind, lambda *_: None)]

    monkeypatch.setattr(model_menu, "make_control_items", expand)
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", lambda *_: [])
    model_menu.make_items(state, definition)
    assert calls == [controls[2]]
    calls.clear()
    voice_menu.make_items(state, definition)
    assert calls == [controls[0], controls[3]]


@pytest.mark.parametrize("expansion_sizes", [(), (1, 1), (0, 2, 1), (0, 0)])
def test_only_first_emitted_voice_setting_gets_separator(backend, monkeypatch, expansion_sizes):
    definition, control_type, model_menu, voice_menu = backend
    definition = replace(definition, menu=(
        control_type("voice_samples", None),
        *(control_type("seed", "voice") for _ in expansion_sizes),
    ))
    state = SimpleNamespace(project=Project(tts_model_type=definition.spec.id))
    sizes = iter(expansion_sizes)
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items",
                        lambda *_: [MenuItem("Samples", lambda *_: None)])
    monkeypatch.setattr(model_menu, "make_control_items",
                        lambda *_: [MenuItem("Setting", lambda *_: None) for _ in range(next(sizes))])

    items = voice_menu.make_items(state, definition)
    expected = [False]
    if sum(expansion_sizes):
        expected += [True] + [False] * (sum(expansion_sizes) - 1)
    assert [item.blank_line_before for item in items] == expected


def test_voice_numeric_and_seed_edit_definition_owner_not_selected_model(backend, monkeypatch):
    definition, control_type, model_menu, voice_menu = backend
    model_id = definition.spec.id
    unrelated_id = "higgs_v3_sglomni"
    project = Project(tts_model_type=unrelated_id)
    project.set_model_setting(model_id, "temperature", 0.6)
    project.set_model_setting(unrelated_id, "temperature", 0.8)
    project.set_model_setting(model_id, "seed", 11)
    project.set_model_setting(unrelated_id, "seed", 22)
    definition = replace(definition, menu=(
        control_type("parameter", "voice", "temperature", "Temperature"),
        control_type("seed", "voice"),
    ))
    state = SimpleNamespace(project=project)
    saves, errors, prefills = [], [], []
    responses = iter(["0.7", "123"])

    def ask_input(**kwargs):
        prefills.append(kwargs["prefill"])
        return next(responses)

    monkeypatch.setattr(ask, "ask_input", ask_input)
    monkeypatch.setattr(ask, "ask_error", errors.append)
    monkeypatch.setattr(ask, "print_feedback", lambda *_, **__: None)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self) or "")
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_configured_sgl_omni_menu.print_feedback",
                        lambda *_, **__: None)
    items = voice_menu.make_items(state, definition)
    assert len(items) == 2
    assert get_string_from(state, items[0].label).startswith("Temperature ")
    assert model_menu.make_items(state, definition) == []
    for item in items:
        item.handler(state, item)
    assert errors == []
    assert prefills == ["0.6", "11"]
    assert saves == [project, project]
    assert project.get_model_setting(model_id, "temperature") == 0.7
    assert project.get_model_setting(model_id, "seed") == 123
    assert project.get_model_setting(unrelated_id, "temperature") == 0.8
    assert project.get_model_setting(unrelated_id, "seed") == 22
    if model_id == "qwen3tts_sglomni":
        # Shared Qwen3 settings still belong to the registry's shared group.
        assert project.get_model_setting("qwen3tts_local", "temperature") == 0.7
        assert project.get_model_setting("qwen3tts_local", "seed") == 123
        shared = project.model_dump(exclude={"reason_pauses"})["model_settings"]["shared"]["qwen3"]["parameters"]
        assert shared["temperature"] == 0.7
        assert shared["seed"] == 123


@pytest.mark.parametrize("model_id,label,setting_count", [
    ("breeze_tts_2_audiocpp", "Instructions", 5),
    ("omnivoice_audiocpp", "Voice design instructions", 4),
])
@pytest.mark.parametrize("instructions", ["", "Speak warmly and naturally."])
def test_shipped_audio_cpp_instructions_and_clear_belong_to_voice_menu(
        monkeypatch, model_id, label, setting_count, instructions):
    definition = load_audio_cpp_definitions().models[model_id]
    project = Project(tts_model_type=definition.spec.id)
    project.set_model_setting(definition.spec.id, "instruct", instructions)
    state = SimpleNamespace(project=project)
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items",
                        lambda *_: [MenuItem("Samples", lambda *_: None)])

    voice_items = VoiceAudioCppMenu.make_items(state, definition)
    assert [item.blank_line_before for item in voice_items] == [False, True] + ([False] if instructions else [])
    voice_labels = [get_string_from(state, item.label) for item in voice_items]
    model_labels = [get_string_from(state, item.label)
                    for item in ModelAudioCppMenu.make_items(state, definition)]
    assert voice_labels[0] == "Samples"
    assert voice_labels[1].startswith(label + " ")
    assert voice_labels[2:] == (["Clear instructions"] if instructions else [])
    assert len(model_labels) == setting_count  # Numeric settings and seed.
    assert not any(value.startswith(label + " ") or value == "Clear instructions"
                   for value in model_labels)
