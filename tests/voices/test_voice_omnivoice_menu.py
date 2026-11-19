"""OmniVoice LoRA controls follow the VibeVoice pattern (tests/voices/test_voice_vibevoice_menu.py)."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice import voice_omnivoice_menu as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes


def make_state(lora_target=""):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "voice-1.flac", "transcript": "First."},
    ])
    project.set_model_setting("omnivoice_local", "lora_target", lora_target)
    return cast(State, SimpleNamespace(project=project))


def test_voices_menu_dispatches_to_the_omnivoice_menu(monkeypatch):
    shown = []
    monkeypatch.setattr(menu.VoiceOmniVoiceMenu, "menu", staticmethod(shown.append))
    state = make_state()
    voice_menu_shared.VoiceMenuShared.menu(state)
    assert shown == [state]


@pytest.mark.parametrize("lora_target", ["", "wjcwqc/PaigeVoice/lora"])
def test_lora_select_and_clear_follow_sample_controls(monkeypatch, lora_target):
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
    menu.VoiceOmniVoiceMenu.menu(state)
    items = captured["items"](state)
    lora_items = [item for item in items if "LoRA" in strip_ansi_codes(get_string_from(state, item.label))]

    select = lora_items[0]
    label = strip_ansi_codes(get_string_from(state, select.label))
    # Unset invites selection; once set, the item just names the current LoRA
    assert label.startswith("LoRA " if lora_target else "Select LoRA ")
    assert (lora_target or "(optional)") in label
    assert len(lora_items) == (2 if lora_target else 1)
    select.handler(state, select)
    assert asked["current_target"] == lora_target
    assert "OmniVoice LoRA" in asked["prompt"]
    asked["callback"](state.project, "new-lora")
    assert applied == [(state, "new-lora")]
    if lora_target:
        clear = lora_items[1]
        clear.handler(state, clear)
        assert state.project.get_model_setting("omnivoice_local", "lora_target") == ""
        assert saved == [state.project]
        assert cleared == [True]


@pytest.mark.parametrize("inspection,error,expected_error", [
    (SimpleNamespace(metadata={"has_lora": True}), "", None),
    # The loader's reason is shown when the worker reports one
    (SimpleNamespace(metadata={"has_lora": False, "lora_error": "Unsupported adapter feature: use_dora = True"}),
     "", "Couldn't load LoRA: Unsupported adapter feature: use_dora = True"),
    (SimpleNamespace(metadata={"has_lora": False}), "", "Couldn't load LoRA"),
    (None, "Failed to initialize", "Failed to initialize"),
])
def test_lora_validation_saves_or_rolls_back(monkeypatch, inspection, error, expected_error):
    state = make_state("old-lora")
    saved, errors, inspected_targets = [], [], []
    monkeypatch.setattr(Project, "save", lambda self: saved.append(self))
    monkeypatch.setattr(menu.ModelWorker, "clear_models_if_running_blocking", lambda: None)

    def inspect(current):
        inspected_targets.append(current.project.get_model_setting("omnivoice_local", "lora_target"))
        return inspection, error

    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking", inspect)
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(menu.ask, "ask_enter_to_continue", lambda: None)
    menu.apply_lora_and_validate(state, "new-lora")
    assert inspected_targets == ["new-lora"]
    assert state.project.get_model_setting("omnivoice_local", "lora_target") == (
        "old-lora" if expected_error else "new-lora")
    assert errors == ([expected_error] if expected_error else [])
    assert saved == ([] if expected_error else [state.project])
