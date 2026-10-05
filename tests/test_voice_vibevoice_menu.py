"""VibeVoice LoRA controls live with voice selection without changing validation."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice import voice_vibevoice_menu as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes


def make_state(lora_target=""):
    project = Project(tts_model_type="vibevoice_local", voice_references=[
        {"file_name": "voice-1.flac", "transcript": "First."},
        {"file_name": "voice-2.flac", "transcript": "Second."},
    ])
    project.set_model_setting("vibevoice_local", "lora_target", lora_target)
    return cast(State, SimpleNamespace(project=project))


@pytest.mark.parametrize("lora_target", ["", "vibevoice-community/klett"])
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
    menu.VoiceVibeVoiceMenu.menu(state)
    items = captured["items"](state)
    select = items[7]
    assert get_string_from(state, items[6].label).startswith(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    label = strip_ansi_codes(get_string_from(state, select.label))
    assert label.startswith("Select LoRA")
    assert (lora_target or "(optional)") in label
    assert select.blank_line_before
    assert len(items) == (9 if lora_target else 8)
    assert all(not item.superlabel for item in items)
    select.handler(state, select)
    assert asked["project"] is state.project
    assert asked["current_target"] == lora_target
    assert "VibeVoice LoRA" in asked["prompt"]
    asked["callback"](state.project, "new-lora")
    assert applied == [(state, "new-lora")]
    if lora_target:
        clear = items[-1]
        assert clear.label == "Clear LoRA"
        assert not clear.blank_line_before
        clear.handler(state, clear)
        assert state.project.get_model_setting("vibevoice_local", "lora_target") == ""
        assert len(captured["items"](state)) == 8
        assert saved == [state.project]
        assert cleared == [True]
        assert len(state.project.voice_references) == 2


@pytest.mark.parametrize("inspection,error,expected_error", [
    (SimpleNamespace(metadata={"has_lora": True}), "", None),
    (SimpleNamespace(metadata={"has_lora": False}), "", "Couldn't load LoRA"),
    (None, "Failed to initialize", "Failed to initialize"),
])
def test_lora_validation_preserves_save_reload_and_rollback(monkeypatch, inspection, error, expected_error):
    state = make_state("old-lora")
    saved, cleared, errors, continued, inspected_targets = [], [], [], [], []
    monkeypatch.setattr(Project, "save", lambda self: saved.append(self))
    monkeypatch.setattr(menu.ModelWorker, "clear_models_if_running_blocking", lambda: cleared.append(True))

    def inspect(current):
        inspected_targets.append(current.project.get_model_setting("vibevoice_local", "lora_target"))
        return inspection, error

    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking", inspect)
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(menu.ask, "ask_enter_to_continue", lambda: continued.append(True))
    menu.apply_lora_and_validate(state, "new-lora")
    assert inspected_targets == ["new-lora"]
    assert state.project.get_model_setting("vibevoice_local", "lora_target") == (
        "old-lora" if expected_error else "new-lora")
    assert errors == ([expected_error] if expected_error else [])
    assert saved == ([] if expected_error else [state.project])
    assert len(cleared) == (2 if expected_error else 1)
    assert continued == ([] if expected_error else [True])
    assert len(state.project.voice_references) == 2
