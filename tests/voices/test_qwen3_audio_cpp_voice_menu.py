"""Qwen audio.cpp voice menus use manual type settings without inspection."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared as shared
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts


MODEL_ID = "qwen3tts_audiocpp"


@pytest.fixture(autouse=True)
def isolate_menu(monkeypatch):
    monkeypatch.setattr(shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    monkeypatch.setattr(shared, "print_feedback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Project, "save", lambda _: pytest.fail("Unexpected save"))
    monkeypatch.setattr(ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("audio.cpp has no checkpoint speaker discovery"))


def make_state():
    project = Project(tts_model_type=MODEL_ID, voice_references=[
        Voice("reference.flac", "Keep this transcript.",
              presets={MODEL_ID: "vivian", "qwen3tts_local": "Ryan", "pocket_local": "alba"},
              instructions={MODEL_ID: "Speak warmly.", "qwen3tts_local": "Local instructions."}),
        Voice("second.flac", "Second reference."),
    ])
    return cast(State, SimpleNamespace(project=project))


def label(state, item):
    return strip_ansi_codes(get_string_from(state, item.label))


def items(state, menu):
    value = menu["items"]
    return value(state) if callable(value) else value


def labels(state, menu):
    return [label(state, item) for item in items(state, menu)]


def select(state, menu, prefix):
    item = next(item for item in items(state, menu) if label(state, item).startswith(prefix))
    item.handler(state, item)


def open_voice_menu(monkeypatch, state):
    captured = {}
    monkeypatch.setattr(shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    # Enter through real backend dispatch, the audio.cpp wrapper and Voice 1.
    shared.VoiceMenuShared.menu(state)
    outer = dict(captured)
    select(state, outer, "Voice 1:")
    return outer, dict(captured), captured


def test_switching_manual_types_redraws_controls_without_losing_voice_data(monkeypatch):
    # A single open submenu follows settings changes; inactive values never get cleared.
    state = make_state()
    project = state.project
    original = list(project.voice_references)
    outer, submenu, _ = open_voice_menu(monkeypatch, state)
    cases = [
        ("base", True, True, False, False),
        ("custom_voice", False, False, True, True),
        ("voice_design", False, False, False, True),
        ("base", True, True, False, False),
    ]
    for model_type, sample, transcript, speaker, instructions in cases:
        project.set_model_setting(MODEL_ID, "model_type", model_type)
        current = labels(state, submenu)
        assert not any("clone mode" in value.lower() for value in current)
        for prefix, visible in [
            ("Voice sample", sample), ("Clear voice sample", sample), ("Trim or play", sample),
            ("Transcription", transcript), ("Speaker", speaker), ("Clear speaker", speaker),
            ("Instructions/", instructions), ("Clear instructions", instructions),
        ]:
            assert any(value.startswith(prefix) for value in current) is visible, (model_type, prefix)
        assert Tts.requires_reference_transcript(project) is transcript
        row = labels(state, outer)[0]
        assert ("reference" in row) is sample
        assert ("vivian" in row) is speaker
        assert ("Speak warmly." in row) is instructions
        assert project.voice_references == original


@pytest.mark.parametrize("selection", ["aiden", "ono_anna", "uncle_fu", "vivian"])
def test_official_speaker_choices_save_only_exact_model_and_voice(monkeypatch, selection):
    # Official ids are offered without discovery; the local Qwen preset is unrelated.
    state = make_state()
    state.project.set_model_setting(MODEL_ID, "model_type", "custom_voice")
    saves = []
    monkeypatch.setattr(Project, "save", lambda project: saves.append(project) or "")
    _, submenu, captured = open_voice_menu(monkeypatch, state)
    select(state, submenu, "Speaker")
    assert labels(state, captured) == [
        "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu",
        "vivian (selected)", "Custom speaker...",
    ]
    select(state, captured, selection)
    voice = state.project.voice_references[0]
    assert voice.presets == {MODEL_ID: selection, "qwen3tts_local": "Ryan", "pocket_local": "alba"}
    assert state.project.voice_references[1].presets == {}
    assert voice.file_name == "reference.flac" and voice.transcript == "Keep this transcript."
    assert voice.instructions[MODEL_ID] == "Speak warmly."
    assert len(saves) == (0 if selection == "vivian" else 1)


@pytest.mark.parametrize("value,error", [
    ("  My Custom Speaker  ", ""), ("", ""), ("   ", ""), ("New Speaker", "Disk full"),
])
def test_custom_speaker_save_cancel_and_failure_preserve_existing_data(monkeypatch, value, error):
    # Custom input uses the shared transactional helper, including rollback on save failure.
    state = make_state()
    state.project.set_model_setting(MODEL_ID, "model_type", "custom_voice")
    original = state.project.voice_references
    saves, prompts, errors = [], [], []
    monkeypatch.setattr(Project, "save", lambda project: saves.append(project) or error)
    monkeypatch.setattr(shared.ask, "ask_input", lambda **kwargs: prompts.append(kwargs) or value)
    monkeypatch.setattr(shared.ask, "ask_error", errors.append)
    _, submenu, captured = open_voice_menu(monkeypatch, state)
    select(state, submenu, "Speaker")
    select(state, captured, "Custom speaker...")
    assert prompts == [{"lower": False, "prefill": "vivian"}]
    if value.strip() and not error:
        assert state.project.voice_references[0].presets[MODEL_ID] == "My Custom Speaker"
        assert any("My Custom Speaker" in label for label in labels(state, submenu))
        # Clearing a custom speaker still uses the existing shared clear action.
        select(state, submenu, "Clear speaker")
        assert MODEL_ID not in state.project.voice_references[0].presets
        assert "required" in next(value for value in labels(state, submenu) if value.startswith("Speaker"))
    else:
        assert state.project.voice_references is original
    assert len(saves) == (2 if value.strip() and not error else int(bool(value.strip())))
    assert errors == ([error] if error else [])
    voice = state.project.voice_references[0]
    assert voice.presets["qwen3tts_local"] == "Ryan" and voice.presets["pocket_local"] == "alba"
    assert voice.file_name == original[0].file_name and voice.transcript == original[0].transcript
    assert voice.instructions == original[0].instructions
    assert state.project.voice_references[1] == original[1]


def test_instruction_edit_remains_model_specific_when_sample_is_inactive(monkeypatch):
    # VoiceDesign edits its own instructions without disturbing saved cloning/preset state.
    state = make_state()
    state.project.set_model_setting(MODEL_ID, "model_type", "voice_design")
    original = state.project.voice_references[0]
    monkeypatch.setattr(Project, "save", lambda _: "")
    monkeypatch.setattr(shared.ask, "ask_input", lambda **_: "A quiet storyteller.")
    _, submenu, _ = open_voice_menu(monkeypatch, state)
    select(state, submenu, "Instructions/")
    voice = state.project.voice_references[0]
    assert voice.instructions == {MODEL_ID: "A quiet storyteller.", "qwen3tts_local": "Local instructions."}
    assert voice.presets == original.presets
    assert (voice.file_name, voice.transcript) == (original.file_name, original.transcript)
