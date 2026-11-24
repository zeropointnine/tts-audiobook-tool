"""Breeze TTS 2 (local) per-voice LoRA controls: LoRA picker (including clear) and strength
items (`voice_lora_items`); selecting a LoRA checks it without the model (`breeze_lora_check`) before saving."""
import json
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
from tts_audiobook_tool.menus.voice import voice_breeze_menu as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts_models.breeze_lora_source import LOCAL_FILE_MARKERS
from tts_audiobook_tool.tts_models.lora_files import LoraCheck

MODEL_ID = "breeze_tts_2_local"
LEDA = ModelTarget.local("/runs/fw-leda/lora/checkpoint-step-000500")


def make_state(dir_path: Path, lora: VoiceLora | None = None) -> State:
    # Every menu state has isolated storage, even if a handler unexpectedly saves.
    project = Project(dir_path=str(dir_path), tts_model_type=MODEL_ID, voice_references=[
        Voice("voice-1.flac", "First.", loras={MODEL_ID: lora} if lora else {}),
        Voice(),
    ])
    return cast(State, SimpleNamespace(project=project))


def labels_of(state, items) -> list[str]:
    return [strip_ansi_codes(get_string_from(state, item.label)) for item in items]


def test_voices_menu_dispatches_to_the_breeze_menu(monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr(menu.VoiceBreezeMenu, "menu", staticmethod(shown.append))
    state = make_state(tmp_path)
    voice_menu_shared.VoiceMenuShared.menu(state)
    assert shown == [state]


def test_breeze_menu_puts_lora_items_in_each_voice_submenu(monkeypatch, tmp_path):
    # The voice list itself has no LoRA items; the per-voice hook reaches the submenu
    state = make_state(tmp_path)
    captured, opened = {}, {}
    monkeypatch.setattr(menu.VoiceMenuShared, "menu_wrapper", lambda state, items: captured.update(items=items))
    monkeypatch.setattr(voice_menu_shared.VoiceMenuShared, "voice_sample_submenu",
                        lambda s, tts_type, index, *_, **kwargs: opened.update(index=index, **kwargs))
    menu.VoiceBreezeMenu.menu(state)
    items = captured["items"](state)
    labels = labels_of(state, items)
    assert not any(label.startswith(("LoRA", "Select LoRA", "Clear LoRA")) for label in labels)
    voice_2 = next(item for item, label in zip(items, labels_of(state, items)) if label.startswith("Voice 2"))
    voice_2.handler(state, voice_2)
    extra = opened["voice_extra_items"](state, opened["index"])
    assert labels_of(state, extra)[0].startswith("LoRA")


@pytest.mark.parametrize("lora", [None, VoiceLora(LEDA, 0.5)])
def test_lora_select_strength_and_clear(monkeypatch, tmp_path, lora):
    # Strength appears only with a LoRA; clearing lives in the picker. Selection accepts raw
    # checkpoint directories (Breeze markers, not the generic hf ones); nothing loads the model
    state = make_state(tmp_path, lora)
    asked, validated = {}, []
    monkeypatch.setattr(ModelMenuShared, "ask_target", lambda **kwargs: asked.update(kwargs))
    monkeypatch.setattr(lora_items, "validate_and_set_voice_lora", lambda *args: validated.append(args))
    monkeypatch.setattr(lora_items, "print_feedback", lambda *_, **__: None)
    monkeypatch.setattr("tts_audiobook_tool.model_worker.ModelWorker.inspect_tts_blocking",
                        lambda *_, **__: pytest.fail("Opening voice controls must not load the model"))
    spec = menu.make_lora_spec()
    items = lora_items.make_voice_lora_items(state, spec, 0)
    labels = labels_of(state, items)
    assert len(items) == (2 if lora else 1)
    assert "Clear LoRA" not in labels
    assert labels[0].startswith("LoRA")
    items[0].handler(state, items[0])
    assert asked["current_target"] == (lora.target if lora else None)
    assert asked["spec"].local_file_markers == LOCAL_FILE_MARKERS
    assert "Instavar" in asked["spec"].subject
    asked["callback"](state.project, ModelTarget.hf("new-lora"))
    assert [args[3] for args in validated] == [ModelTarget.hf("new-lora")]
    if lora:
        assert labels[1].startswith("LoRA strength") and "0.50" in labels[1]
        asked["on_clear"]()
        assert state.project.voice_references[0].loras == {}
        assert "loras" not in json.loads((tmp_path / "project.json").read_text())["voice_references"][0]
    else:
        assert not (tmp_path / "project.json").exists()


@pytest.mark.parametrize("result,expected_error,expected_warning", [
    (LoraCheck(), None, None),
    (LoraCheck(error="Adapter was trained on a different base"), "Couldn't load LoRA: Adapter was trained on a different base", None),
    (LoraCheck(warnings=("All of the adapter's LoRA B factors are zero",)), None, "All of the adapter's LoRA B factors are zero"),
])
def test_lora_is_saved_only_when_the_check_accepts_it(monkeypatch, tmp_path, result, expected_error, expected_warning):
    # The unsaved LoRA is checked without the model (never probing the worker), keeping the
    # voice's strength; a warning doesn't block saving but is shown with a pause.
    from tts_audiobook_tool.tts_models import breeze_lora_check
    state = make_state(tmp_path, VoiceLora(ModelTarget.hf("old-lora"), 0.5))
    checks, errors, printed, pauses = [], [], [], []
    monkeypatch.setattr(breeze_lora_check, "check_adapter",
                        lambda target, strength: checks.append(VoiceLora(target, strength)) or result)
    monkeypatch.setattr("tts_audiobook_tool.model_worker.ModelWorker.inspect_tts_blocking",
                        lambda *_, **__: pytest.fail("model probed"))
    monkeypatch.setattr(lora_items.ask, "ask_error", errors.append)
    monkeypatch.setattr(lora_items.ask, "ask_enter_to_continue", lambda *_: pauses.append(True))
    monkeypatch.setattr(lora_items, "printt", lambda value="", *_, **__: printed.append(strip_ansi_codes(value)))
    monkeypatch.setattr(lora_items, "print_feedback", lambda *_, **__: None)
    lora_items.validate_and_set_voice_lora(state, menu.make_lora_spec(), 0, ModelTarget.hf("new-lora"))
    assert checks == [VoiceLora(ModelTarget.hf("new-lora"), 0.5)]
    assert state.project.voice_references[0].lora_for(MODEL_ID) == VoiceLora(
        ModelTarget.hf("old-lora" if expected_error else "new-lora"), 0.5)
    assert errors == ([expected_error] if expected_error else [])
    assert (f"Warning: {expected_warning}" in printed) == bool(expected_warning)
    assert pauses == ([True] if expected_warning else [])
    stored_path = tmp_path / "project.json"
    if expected_error:
        assert not stored_path.exists()
    else:
        assert json.loads(stored_path.read_text())["voice_references"][0]["loras"] == {
            MODEL_ID: {"target": "new-lora", "target_type": "hf", "strength": 0.5},
        }


@pytest.mark.parametrize("entered,expected,error", [
    ("1.5", 1.5, None), ("9", 0.5, "Out of range"), ("x", 0.5, "Bad value"), ("", 0.5, None),
])
def test_strength_prompt_validates_and_saves(monkeypatch, tmp_path, entered, expected, error):
    # Only a valid, changed strength should create a saved project.
    state = make_state(tmp_path, VoiceLora(LEDA, 0.5))
    errors = []
    monkeypatch.setattr(lora_items.ask, "ask_input", lambda **_: entered)
    monkeypatch.setattr(lora_items.ask, "ask_error", errors.append)
    monkeypatch.setattr(lora_items, "print_feedback", lambda *_, **__: None)
    lora_items.ask_voice_lora_strength(state, menu.make_lora_spec(), 0)
    assert state.project.voice_references[0].lora_for(MODEL_ID) == VoiceLora(LEDA, expected)
    assert errors == ([error] if error else [])
    stored_path = tmp_path / "project.json"
    if expected != 0.5:
        assert json.loads(stored_path.read_text())["voice_references"][0]["loras"] == {
            MODEL_ID: {"target": LEDA.value, "target_type": "local", "strength": expected},
        }
    else:
        assert not stored_path.exists()
