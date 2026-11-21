"""Breeze TTS 2 (local) LoRA controls follow the OmniVoice pattern, plus a strength item."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice import voice_breeze_menu as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts_models.breeze_lora import LOCAL_FILE_MARKERS

MODEL_ID = "breeze_tts_2_local"


def make_state(lora_target=""):
    project = Project(tts_model_type=MODEL_ID, voice_references=[
        {"file_name": "voice-1.flac", "transcript": "First."},
    ])
    project.set_model_setting(MODEL_ID, "lora_target", lora_target)
    return cast(State, SimpleNamespace(project=project))


def test_voices_menu_dispatches_to_the_breeze_menu(monkeypatch):
    shown = []
    monkeypatch.setattr(menu.VoiceBreezeMenu, "menu", staticmethod(shown.append))
    state = make_state()
    voice_menu_shared.VoiceMenuShared.menu(state)
    assert shown == [state]


@pytest.mark.parametrize("lora_target", ["", "/runs/fw-leda/lora/checkpoint-step-000500"])
def test_lora_select_strength_and_clear(monkeypatch, lora_target):
    # Strength and Clear appear only once a LoRA is set; selection accepts raw checkpoint
    # directories (Breeze markers, not the generic hf ones)
    state = make_state(lora_target)
    captured, asked, applied, saved, cleared = {}, {}, [], [], []
    monkeypatch.setattr(menu.VoiceMenuShared, "menu_wrapper",
                        lambda state, items: captured.update(items=items))
    monkeypatch.setattr(ModelMenuShared, "ask_target", lambda **kwargs: asked.update(kwargs))
    monkeypatch.setattr(menu, "apply_lora_and_validate", lambda *args: applied.append(args))
    monkeypatch.setattr(Project, "save", lambda self: saved.append(self))
    monkeypatch.setattr(menu.ModelWorker, "clear_models_if_running_blocking", lambda: cleared.append(True))
    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("Opening voice controls must not load the model"))
    menu.VoiceBreezeMenu.menu(state)
    items = captured["items"](state)
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in items]
    lora_items = [item for item, label in zip(items, labels) if "LoRA" in label]

    select_label = strip_ansi_codes(get_string_from(state, lora_items[0].label))
    assert select_label.startswith("LoRA " if lora_target else "Select LoRA ")
    assert len(lora_items) == (3 if lora_target else 1)
    lora_items[0].handler(state, lora_items[0])
    assert asked["current_target"] == lora_target
    assert asked["local_file_markers"] == LOCAL_FILE_MARKERS
    assert "Instavar" in asked["prompt"]
    asked["callback"](state.project, "new-lora")
    assert applied == [(state, "new-lora")]
    if lora_target:
        strength_label = strip_ansi_codes(get_string_from(state, lora_items[1].label))
        assert strength_label.startswith("LoRA strength")
        clear = lora_items[2]
        clear.handler(state, clear)
        assert state.project.get_model_setting(MODEL_ID, "lora_target") == ""
        assert saved == [state.project]
        assert cleared == [True]


@pytest.mark.parametrize("inspection,error,expected_error", [
    (SimpleNamespace(metadata={"has_lora": True}), "", None),
    (SimpleNamespace(metadata={"has_lora": False, "lora_error": "Adapter was trained on a different base"}),
     "", "Couldn't load LoRA: Adapter was trained on a different base"),
    (None, "Failed to initialize", "Failed to initialize"),
])
def test_lora_validation_saves_or_rolls_back(monkeypatch, inspection, error, expected_error):
    state = make_state("old-lora")
    saved, errors = [], []
    monkeypatch.setattr(Project, "save", lambda self: saved.append(self))
    monkeypatch.setattr(menu.ModelWorker, "clear_models_if_running_blocking", lambda: None)
    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking", lambda *_: (inspection, error))
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(menu.ask, "ask_enter_to_continue", lambda: None)
    menu.apply_lora_and_validate(state, "new-lora")
    assert state.project.get_model_setting(MODEL_ID, "lora_target") == (
        "old-lora" if expected_error else "new-lora")
    assert errors == ([expected_error] if expected_error else [])
    assert saved == ([] if expected_error else [state.project])
