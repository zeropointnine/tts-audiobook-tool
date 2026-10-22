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
    definition = replace(definition, menu=tuple(controls))
    state = SimpleNamespace(project=Project(tts_model_type=definition.spec.id))
    sample_calls = []

    def make_samples(current, model_type, *, preset_choices=None, allow_custom_preset=False):
        # Non-Qwen models pass the shared sample builder's default preset options.
        assert preset_choices is None and not allow_custom_preset
        assert current is state and model_type.id == definition.spec.id
        sample_calls.append(model_type.id)
        return [MenuItem("Add sample", lambda *_: None), MenuItem("Select samples", lambda *_: None)]

    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", make_samples)
    # Routing must use the existing Model editors, not introduce Voice editors.
    monkeypatch.setattr(model_menu, "make_parameter_item",
                        lambda _state, _parameter, label, *_, **__: MenuItem(label, lambda *_: None))
    monkeypatch.setattr(ModelMenuShared, "make_seed_item",
                        lambda *_, **__: MenuItem("Seed", lambda *_: None))
    voice_items = voice_menu.make_items(state, definition)
    model_items = model_menu.make_items(state, definition)
    expected_voice = ["Voice temperature", "Add sample", "Select samples", "Voice Top-K"]
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

    def expand(current, current_definition, control, *_):
        assert current is state and current_definition is definition
        calls.append(control)
        return [MenuItem(control.kind, lambda *_: None)]

    monkeypatch.setattr(model_menu, "make_control_items", expand)
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", lambda *_, **__: [])
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
                        lambda *_, **__: [MenuItem("Samples", lambda *_: None)])
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
        # Server Qwen3 edits are private; local Qwen3 keeps its own values.
        assert project.get_model_setting("qwen3tts_local", "temperature") == -1
        assert project.get_model_setting("qwen3tts_local", "seed") == -1
        saved = project.model_dump(exclude={"reason_pauses"})["model_settings"]
        assert "qwen3" not in saved["shared"]
        assert saved["models"][model_id]["parameters"]["temperature"] == 0.7
        assert saved["models"][model_id]["parameters"]["seed"] == 123


@pytest.mark.parametrize("model_id,setting_count", [
    ("breeze_tts_2_audiocpp", 5), ("omnivoice_audiocpp", 4), ("cosyvoice3_audiocpp", 4),
])
@pytest.mark.parametrize("instructions", ["", "Speak warmly and naturally."])
def test_audio_cpp_instructions_have_no_global_menu_controls(monkeypatch, model_id, setting_count, instructions):
    definition = load_audio_cpp_definitions().models[model_id]
    project = Project(tts_model_type=model_id)
    project.voice_references[0].instructions[model_id] = instructions
    state = SimpleNamespace(project=project)
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items",
                        lambda *_, **__: [MenuItem("Samples", lambda *_: None)])
    voice_items = VoiceAudioCppMenu.make_items(state, definition)
    assert [item.label for item in voice_items] == ["Samples"]
    assert not voice_items[0].blank_line_before
    items = ModelAudioCppMenu.make_items(state, definition)
    assert len(items) == setting_count
    assert not any("instructions" in get_string_from(state, item.label).lower() for item in items)


def test_cosyvoice3_mode_item_shows_choice_label_and_saves_via_option_submenu(monkeypatch):
    # "Mode" is a fixed-choice control: its label shows the choice's display
    # name, and selecting from the option submenu persists the template value.
    from tts_audiobook_tool.menus.menu_util import MenuUtil
    model_id = "cosyvoice3_audiocpp"
    definition = load_audio_cpp_definitions().models[model_id]
    project = Project(tts_model_type=model_id)
    monkeypatch.setattr(Project, "save", lambda self: "")
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_menu_shared.print_feedback",
                        lambda *_, **__: None)
    state = SimpleNamespace(project=project)
    items = ModelAudioCppMenu.make_items(state, definition)
    labels = [get_string_from(state, item.label) for item in items]
    assert labels[0].startswith("Mode ") and "Zero-shot" in labels[0]
    # Per-voice instructions never add a global settings row.
    assert labels[1].startswith("Top-K ")

    offered = {}

    def fake_options_menu(**kwargs):
        offered.update(kwargs)
        kwargs["on_select"]("instruct")

    monkeypatch.setattr(MenuUtil, "options_menu", fake_options_menu)
    items[0].handler(state, items[0])
    assert offered["values"] == ["zero_shot", "cross_lingual", "instruct"]
    assert offered["labels"] == ["Zero-shot", "Cross-lingual", "Instruct"]
    assert offered["current_value"] == offered["default_value"] == "zero_shot"
    assert project.get_model_setting(model_id, "template_name") == "instruct"
    assert "Instruct" in get_string_from(state, items[0].label)


def _cosyvoice3_model_labels(project: Project) -> list[str]:
    model_id = "cosyvoice3_audiocpp"
    definition = load_audio_cpp_definitions().models[model_id]
    state = SimpleNamespace(project=project)
    return [get_string_from(state, item.label) for item in ModelAudioCppMenu.make_items(state, definition)]


@pytest.mark.parametrize("mode", ["zero_shot", "cross_lingual", "instruct"])
def test_cosyvoice3_mode_does_not_add_global_instruction_controls(mode):
    project = Project(tts_model_type="cosyvoice3_audiocpp")
    project.set_model_setting("cosyvoice3_audiocpp", "template_name", mode)
    labels = _cosyvoice3_model_labels(project)
    assert len(labels) == 4
    assert not any("instructions" in label.lower() for label in labels)


def test_invalid_stored_values_remain_editable():
    project = Project(tts_model_type="cosyvoice3_audiocpp")
    project.set_model_setting("cosyvoice3_audiocpp", "top_k", 0)
    labels = _cosyvoice3_model_labels(project)
    assert len(labels) == 4
    assert any(label.startswith("Top-K (invalid:") for label in labels)


def test_cosyvoice3_invalid_mode_is_shown_and_any_choice_repairs_it(monkeypatch):
    # An invalid stored Mode must not masquerade as the default: the label shows
    # the broken value and the submenu marks nothing selected, so picking the
    # default (Zero-shot) still saves instead of being skipped as "unchanged".
    from tts_audiobook_tool.menus.menu_util import MenuUtil
    model_id = "cosyvoice3_audiocpp"
    definition = load_audio_cpp_definitions().models[model_id]
    project = Project(tts_model_type=model_id)
    project.set_model_setting(model_id, "template_name", "sft")
    monkeypatch.setattr(Project, "save", lambda self: "")
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_menu_shared.print_feedback",
                        lambda *_, **__: None)
    state = SimpleNamespace(project=project)
    item = ModelAudioCppMenu.make_items(state, definition)[0]
    label = get_string_from(state, item.label)
    assert label.startswith("Mode ") and "invalid: 'sft'" in label and "Zero-shot" not in label

    offered = {}

    def fake_options_menu(**kwargs):
        offered.update(kwargs)
        # Mirrors MenuUtil.options_menu: a pick equal to current_value is skipped.
        if kwargs["current_value"] != "zero_shot":
            kwargs["on_select"]("zero_shot")

    monkeypatch.setattr(MenuUtil, "options_menu", fake_options_menu)
    item.handler(state, item)
    assert offered["current_value"] is None
    # Selecting the default resets storage to "unset", which reads back as valid.
    assert project.get_model_setting(model_id, "template_name") == "zero_shot"
    assert "Zero-shot" in get_string_from(state, item.label)
