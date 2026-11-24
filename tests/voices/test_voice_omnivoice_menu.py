"""OmniVoice delegates per-voice LoRA controls to the shared voice_lora_items machinery."""
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.voice import voice_lora_items as lora_items
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice import voice_omnivoice_menu as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes

MODEL_ID = "omnivoice_local"


def make_state(dir_path: Path) -> State:
    # Isolate storage even if a menu handler unexpectedly saves.
    project = Project(dir_path=str(dir_path), tts_model_type=MODEL_ID, voice_references=[
        Voice("voice-1.flac", "First."),
        Voice(loras={MODEL_ID: VoiceLora(ModelTarget.hf("owner/second-speaker"), 0.5)}),
    ])
    return cast(State, SimpleNamespace(project=project))


def labels_of(state, items) -> list[str]:
    return [strip_ansi_codes(get_string_from(state, item.label)) for item in items]


def test_voices_menu_dispatches_to_the_omnivoice_menu(monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr(menu.VoiceOmniVoiceMenu, "menu", staticmethod(shown.append))
    state = make_state(tmp_path)
    voice_menu_shared.VoiceMenuShared.menu(state)
    assert shown == [state]


def test_omnivoice_menu_wires_per_voice_lora_controls(monkeypatch, tmp_path):
    # The voice list has no global LoRA action. Voice 2's submenu uses its own
    # LoRA picker/strength controls, with Omni's adapter markers.
    state = make_state(tmp_path)
    captured, opened, asked, validated = {}, {}, {}, []
    monkeypatch.setattr(menu.VoiceMenuShared, "menu_wrapper", lambda state, items: captured.update(items=items))
    monkeypatch.setattr(voice_menu_shared.VoiceMenuShared, "voice_sample_submenu",
                        lambda s, tts_type, index, *_, **kwargs: opened.update(index=index, **kwargs))
    monkeypatch.setattr(ModelMenuShared, "ask_target", lambda **kwargs: asked.update(kwargs))
    monkeypatch.setattr(lora_items, "validate_and_set_voice_lora", lambda *args: validated.append(args))
    monkeypatch.setattr("tts_audiobook_tool.model_worker.ModelWorker.inspect_tts_blocking",
                        lambda *_, **__: pytest.fail("Opening voice controls must not load the model"))

    menu.VoiceOmniVoiceMenu.menu(state)
    items = captured["items"](state)
    labels = labels_of(state, items)
    assert not any(label.startswith(("LoRA", "Select LoRA", "Clear LoRA")) for label in labels)
    voice_2 = next(item for item, label in zip(items, labels) if label.startswith("Voice 2"))
    voice_2.handler(state, voice_2)
    assert opened["index"] == 1
    extra = opened["voice_extra_items"](state, opened["index"])
    labels = labels_of(state, extra)
    assert len(extra) == 2
    assert labels[0].startswith("LoRA") and "second-speaker" in labels[0]
    assert labels[1].startswith("LoRA strength") and "0.50" in labels[1]
    assert "Clear LoRA" not in labels

    extra[0].handler(state, extra[0])
    assert asked["current_target"] == ModelTarget.hf("owner/second-speaker")
    assert asked["spec"].local_file_markers == ["adapter_config.json", "adapter_model.safetensors"]
    assert "OmniVoice LoRA" in asked["spec"].subject
    asked["callback"](state.project, ModelTarget.hf("new-lora"))
    [(received_state, spec, index, target)] = validated
    assert received_state is state
    assert spec.model_id == MODEL_ID
    assert (index, target) == (1, ModelTarget.hf("new-lora"))
    assert not (tmp_path / "project.json").exists()
    monkeypatch.setattr(lora_items, "print_feedback", lambda *_, **__: None)
    asked["on_clear"]()
    assert state.project.voice_references[1].lora_for(MODEL_ID) is None


def test_replacing_an_omnivoice_lora_keeps_the_voices_strength(monkeypatch, tmp_path):
    # The new adapter is checked without the model, against the project's base (the default
    # here), and saved at the strength the voice already had.
    from tts_audiobook_tool.tts_models import omnivoice_lora_check
    from tts_audiobook_tool.tts_models.lora_files import LoraCheck
    from tts_audiobook_tool.model_worker import ModelWorker
    state = make_state(tmp_path)
    checks = []
    monkeypatch.setattr(omnivoice_lora_check, "check_adapter",
                        lambda target, base, strength: checks.append((target, base, strength)) or LoraCheck())
    monkeypatch.setattr(ModelWorker, "inspect_tts_blocking", lambda *_, **__: pytest.fail("model probed"))
    lora_items.validate_and_set_voice_lora(state, menu.make_lora_spec(), 1, ModelTarget.hf("owner/new-speaker"))
    assert checks == [(ModelTarget.hf("owner/new-speaker"), ModelTarget.hf("k2-fsa/OmniVoice"), 0.5)]
    assert state.project.voice_references[1].lora_for(MODEL_ID) == VoiceLora(ModelTarget.hf("owner/new-speaker"), 0.5)
