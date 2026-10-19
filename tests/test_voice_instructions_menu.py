"""Shared per-voice instruction controls, persistence, and generation routing."""
import re
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.app_types import RoomReverb
from tts_audiobook_tool.app_types.voice import Voice, VoiceCrop
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions


BREEZE = "breeze_tts_2_audiocpp"
COSY = "cosyvoice3_audiocpp"


@pytest.fixture(autouse=True)
def isolate_instruction_ui(monkeypatch):
    monkeypatch.setattr(menu.AudioMetaUtil, "get_audio_duration", lambda _: None)
    monkeypatch.setattr(menu, "print_feedback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Project, "save", lambda _: pytest.fail("unexpected project save"))


def open_submenu(monkeypatch, project, index=0):
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    menu.VoiceMenuShared.voice_sample_submenu(state, project.get_tts_model_type(), index)
    return state, captured


def label(state, item):
    return re.sub(r"\x1b\[[0-9;]*m", "", get_string_from(state, item.label))


def labels(state, submenu):
    return [label(state, item) for item in submenu["items"](state)]


def item_named(state, submenu, prefix):
    return next(item for item in submenu["items"](state) if label(state, item).startswith(prefix))


def select(state, submenu, prefix):
    item = item_named(state, submenu, prefix)
    return item.handler(state, item)


@pytest.mark.parametrize("model_id,setting,supported", [
    (BREEZE, None, True),
    ("omnivoice_local", None, True),
    ("dots_local", None, False),
    ("qwen3tts_local", "base", False),
    ("qwen3tts_local", "custom_voice", True),
])
def test_controls_use_real_model_capability(monkeypatch, model_id, setting, supported):
    project = Project(tts_model_type=model_id, voice_references=[
        Voice(instructions={model_id: "Keep my instructions"}),
    ])
    if setting is not None:
        project.set_model_setting(model_id, "model_type", setting)
    state, submenu = open_submenu(monkeypatch, project)

    assert Tts.get_model_support(project).can_instructions(project) is supported
    current_labels = labels(state, submenu)
    ui = project.get_tts_model_type().value.ui
    expected_label = f"{ui.get('instructions_label', 'Instructions')}/{ui['short_name']} (currently: "
    assert any(value.startswith(expected_label) for value in current_labels) is supported
    assert ("Clear instructions" in current_labels) is supported
    assert project.voice_references[0].instructions == {model_id: "Keep my instructions"}


SAMPLE_ITEM_PREFIXES = (
    menu.LABEL_SET_VOICE_SAMPLE, menu.LABEL_CLEAR_VOICE_SAMPLE,
    menu.LABEL_CROP_VOICE_SAMPLE, menu.LABEL_EDIT_VOICE_TRANSCRIPTION,
)


@pytest.mark.parametrize("model_type,can_clone", [
    ("base", True), ("", True), ("custom_voice", False), ("voice_design", False),
])
def test_qwen_submenu_hides_sample_items_when_checkpoint_cannot_clone(monkeypatch, model_type, can_clone):
    original = Voice("sample.flac", "Sample words", instructions={"qwen3tts_local": "A calm voice"})
    project = Project(tts_model_type="qwen3tts_local", voice_references=[original])
    project.set_model_setting("qwen3tts_local", "model_type", model_type)
    state, submenu = open_submenu(monkeypatch, project)

    current_labels = labels(state, submenu)
    shown = [prefix for prefix in SAMPLE_ITEM_PREFIXES if any(value.startswith(prefix) for value in current_labels)]
    assert shown == (list(SAMPLE_ITEM_PREFIXES) if can_clone else [])
    # Instructions follow their own capability; the hidden sample is kept
    assert any(value.startswith("Instructions/Qwen3-TTS") for value in current_labels) is (not can_clone)
    assert project.voice_references == [original]


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("instructions", ["", "Speak Softly"])
def test_instruction_controls_have_current_label_and_expected_position(monkeypatch, sample, instructions):
    project = Project(tts_model_type=BREEZE, voice_references=[
        Voice("sample.flac" if sample else "", "Sample transcript", instructions={
            BREEZE: instructions, "another_model": "Not this model's instructions",
        }),
        Voice(),
    ])
    state, submenu = open_submenu(monkeypatch, project)
    current_labels = labels(state, submenu)
    expected_value = f'"{instructions}"' if instructions else "none"
    set_index = current_labels.index(f"Instructions/Breeze TTS 2 (currently: {expected_value})")
    previous = "Transcription" if sample else "Voice sample"
    assert current_labels[set_index - 1].startswith(previous)
    following = ["Clear instructions"] if instructions else []
    following += [menu.LABEL_MOVE_VOICE_SAMPLE, menu.LABEL_REMOVE_VOICE_SAMPLE]
    assert current_labels[set_index + 1:set_index + 1 + len(following)] == following
    assert ("Clear instructions" in current_labels) is bool(instructions)


@pytest.mark.parametrize("length", [40, 41])
def test_instruction_preview_flattens_whitespace_and_ellipsizes_at_40(monkeypatch, length):
    flattened = "Speak Softly " + "x" * (length - len("Speak Softly "))
    full_value = "  Speak\n\tSoftly  " + flattened[len("Speak Softly "):] + "\n"
    project = Project(tts_model_type=BREEZE, voice_references=[Voice(instructions={BREEZE: full_value})])
    state, submenu = open_submenu(monkeypatch, project)
    expected = flattened if length == 40 else flattened[:37] + "..."

    assert label(state, item_named(state, submenu, "Instructions")) == (
        f'Instructions/Breeze TTS 2 (currently: "{expected}")'
    )
    assert project.voice_references[0].instructions[BREEZE] == full_value


@pytest.mark.parametrize("model_id,expected_label,has_override", [
    (BREEZE, "Instructions/Breeze TTS 2", True),
    ("omnivoice_local", "Voice design/OmniVoice", True),
    ("omnivoice_audiocpp", "Voice design/OmniVoice", True),
])
def test_instruction_label_and_prompt_use_catalog_override_or_default(monkeypatch, model_id, expected_label, has_override):
    project = Project(tts_model_type=model_id)
    state, submenu = open_submenu(monkeypatch, project)
    printed = []
    monkeypatch.setattr(menu, "printt", printed.append)
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: "")

    assert label(state, item_named(state, submenu, expected_label)) == f"{expected_label} (currently: none)"
    select(state, submenu, expected_label)

    ui = project.get_tts_model_type().value.ui
    assert ("instructions_message" in ui) is has_override
    assert printed == [menu.format_instructions_message(ui.get("instructions_message", menu.DEFAULT_INSTRUCTIONS_MESSAGE))]


def test_instructions_message_dims_only_parenthesized_lines():
    assert menu.format_instructions_message(menu.DEFAULT_INSTRUCTIONS_MESSAGE) == menu.DEFAULT_INSTRUCTIONS_MESSAGE
    assert menu.format_instructions_message("Enter:\n(hint one)\n(hint two)") == (
        f"Enter:\n{menu.COL_DIM}(hint one){menu.COL_DEFAULT}\n{menu.COL_DIM}(hint two){menu.COL_DEFAULT}"
    )


def install_fake_omnivoice_resolver(monkeypatch):
    """Stand-in for OmniVoice's `_resolve_instruct` (not importable in every venv)."""
    calls = []

    def resolve(instruct):
        calls.append(instruct)
        if "bogus" in instruct:
            raise ValueError(f"Unsupported instruct items found in {instruct}")
        items = [item.strip().lower() for item in instruct.split(",") if item.strip()]
        return ", ".join(items) or None

    module = ModuleType("omnivoice.models.omnivoice")
    module._resolve_instruct = resolve  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omnivoice", ModuleType("omnivoice"))
    monkeypatch.setitem(sys.modules, "omnivoice.models", ModuleType("omnivoice.models"))
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice", module)
    return calls


def edit_omnivoice_local_instructions(monkeypatch, answer):
    original = Voice(instructions={"omnivoice_local": "male", "other": "Retained"})
    project = Project(tts_model_type="omnivoice_local", voice_references=[original])
    state, submenu = open_submenu(monkeypatch, project)
    saved, errors = [], []
    monkeypatch.setattr(menu, "printt", lambda *_: None)
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: answer)
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(Project, "save", lambda self: saved.append(list(self.voice_references)) or "")
    select(state, submenu, "Voice design")
    return project, original, saved, errors


def test_omnivoice_local_saves_valid_instructions_as_entered(monkeypatch):
    calls = install_fake_omnivoice_resolver(monkeypatch)
    project, _, saved, errors = edit_omnivoice_local_instructions(monkeypatch, "  Female, High Pitch \n")

    assert calls == ["Female, High Pitch"]
    assert errors == []
    # Validation only: not replaced by the resolver's normalized form
    assert project.voice_references[0].instructions == {"omnivoice_local": "Female, High Pitch", "other": "Retained"}
    assert len(saved) == 1


@pytest.mark.parametrize("answer,expected_error", [
    ("female, bogus", "Unsupported instruct items found in female, bogus"),
    (", ,", "No instruction items found"),
])
def test_omnivoice_local_rejects_invalid_instructions_without_saving(monkeypatch, answer, expected_error):
    install_fake_omnivoice_resolver(monkeypatch)
    project, original, saved, errors = edit_omnivoice_local_instructions(monkeypatch, answer)

    assert errors == [expected_error]
    assert saved == []
    assert project.voice_references == [original]


def test_omnivoice_local_saves_when_validator_unavailable(monkeypatch):
    # A None entry makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice", None)
    project, _, saved, errors = edit_omnivoice_local_instructions(monkeypatch, "anything, goes")

    assert errors == []
    assert project.voice_references[0].instructions["omnivoice_local"] == "anything, goes"
    assert len(saved) == 1


@pytest.mark.parametrize("model_id", [BREEZE, "omnivoice_audiocpp", "qwen3tts_local"])
def test_other_models_do_not_use_omnivoice_validation(monkeypatch, model_id):
    calls = install_fake_omnivoice_resolver(monkeypatch)
    project = Project(tts_model_type=model_id)

    assert Tts.get_model_support(project).validate_instructions("bogus free text") == ""
    assert calls == []


def test_set_prefills_full_value_preserves_case_and_only_strips_outer_whitespace(monkeypatch):
    old_value = "Existing Mixed CASE\n" + "long instructions " * 5
    project = Project(tts_model_type=BREEZE, voice_references=[Voice(instructions={BREEZE: old_value})])
    state, submenu = open_submenu(monkeypatch, project)
    asks, printed, saved = [], [], []

    def ask_input(*args, **kwargs):
        asks.append((args, kwargs))
        return " \tWhisper Like a Moonlit Robot\n  with My Own Vocabulary. \n"

    monkeypatch.setattr(menu.ask, "ask_input", ask_input)
    monkeypatch.setattr(menu, "printt", printed.append)
    monkeypatch.setattr(Project, "save", lambda self: saved.append(
        ProjectSerializationUtil.to_project_json_dict(self)) or "")
    select(state, submenu, "Instructions")

    expected = "Whisper Like a Moonlit Robot\n  with My Own Vocabulary."
    assert asks == [((), {"lower": False, "prefill": old_value})]
    assert any("instructions" in str(message).lower() for message in printed)
    assert all("breeze" not in str(message).lower() for message in printed)
    assert project.voice_references[0].instructions[BREEZE] == expected
    assert len(saved) == 1
    assert Project.model_validate(saved[0]).voice_references[0].instructions[BREEZE] == expected


@pytest.mark.parametrize("answer", ["", " \t\n ", "  Keep Mixed CASE\n"])
def test_empty_cancelled_or_unchanged_input_does_not_save(monkeypatch, answer):
    original = Voice(instructions={BREEZE: "Keep Mixed CASE", "other": "Retained"})
    project = Project(tts_model_type=BREEZE, voice_references=[original])
    state, submenu = open_submenu(monkeypatch, project)
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: answer)
    monkeypatch.setattr(menu.ask, "ask_error", lambda _: pytest.fail("unexpected validation error"))

    select(state, submenu, "Instructions")

    assert project.voice_references == [original]


@pytest.mark.parametrize("length", [4000, 4001])
def test_instruction_length_limit_rejects_without_truncating(monkeypatch, length):
    original = Voice(instructions={BREEZE: "Original"})
    project = Project(tts_model_type=BREEZE, voice_references=[original])
    state, submenu = open_submenu(monkeypatch, project)
    saved, errors = [], []
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: " \n" + "A" * length + " \t")
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(Project, "save", lambda self: saved.append(list(self.voice_references)) or "")

    select(state, submenu, "Instructions")

    if length == 4000:
        assert not errors
        assert len(saved) == 1
        assert project.voice_references[0].instructions[BREEZE] == "A" * 4000
    else:
        assert len(errors) == 1 and "4000" in errors[0]
        assert not saved
        assert project.voice_references == [original]


def test_storage_set_and_clear_preserve_other_namespaces_fields_and_voices(monkeypatch):
    original = Voice(
        "original.flac", "Original transcript", VoiceCrop("trim.flac", 1, 5, "Trim transcript"),
        room_reverb=RoomReverb.get_by_id("small_drr22"), pan=-0.3, presets={"pocket_local": "preset-name"},
        instructions={BREEZE: "Original", BREEZE + "_extra": "Exact key only", "other": "Keep"},
    )
    other = Voice("other.flac", "Other transcript", instructions={BREEZE: "Other voice"})
    project = Project(tts_model_type=BREEZE, voice_references=[original, other])
    snapshots = []
    monkeypatch.setattr(Project, "save", lambda self: snapshots.append(
        ProjectSerializationUtil.to_project_json_dict(self)) or "")

    assert ProjectVoiceUtil.set_voice_instructions_at_index_and_save(project, 0, BREEZE, "  Updated  ") == ""
    expected_map = {**original.instructions, BREEZE: "Updated"}
    assert project.voice_references == [replace(original, instructions=expected_map), other]
    assert ProjectVoiceUtil.set_voice_instructions_at_index_and_save(project, 0, BREEZE, "") == ""
    del expected_map[BREEZE]
    assert project.voice_references == [replace(original, instructions=expected_map), other]
    assert project.voice_references[1] is other
    assert original.instructions[BREEZE] == "Original"
    assert len(snapshots) == 2
    assert Project.model_validate(snapshots[0]).voice_references[0].instructions[BREEZE] == "Updated"
    assert Project.model_validate(snapshots[1]).voice_references == project.voice_references
    # Clearing an already absent namespace is a storage no-op too.
    assert ProjectVoiceUtil.set_voice_instructions_at_index_and_save(project, 0, BREEZE, "") == ""
    assert len(snapshots) == 2


@pytest.mark.parametrize("action", ["Instructions", "Clear instructions"])
def test_save_error_is_reported_and_rolls_back_the_voice_list(monkeypatch, action):
    project = Project(tts_model_type=BREEZE, voice_references=[
        Voice(instructions={BREEZE: "Original", "other": "Retain"}), Voice(),
    ])
    original_list = project.voice_references
    state, submenu = open_submenu(monkeypatch, project)
    errors, attempted = [], []
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: "Replacement")
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(Project, "save", lambda self: attempted.append(
        dict(self.voice_references[0].instructions)) or "disk full")

    select(state, submenu, action)

    expected = {"other": "Retain"}
    if action == "Instructions":
        expected[BREEZE] = "Replacement"
    assert attempted == [expected]
    assert errors == ["disk full"]
    assert project.voice_references is original_list
    assert project.voice_references[0].instructions == {BREEZE: "Original", "other": "Retain"}


def test_set_and_clear_handlers_follow_voice_after_submenu_move(monkeypatch):
    moved = Voice(instructions={BREEZE: "Move me", "other": "Retain"})
    untouched = Voice(instructions={BREEZE: "Leave me alone"})
    project = Project(tts_model_type=BREEZE, voice_references=[moved, untouched])
    state, submenu = open_submenu(monkeypatch, project)
    set_item = item_named(state, submenu, "Instructions")
    clear_item = item_named(state, submenu, "Clear instructions")
    monkeypatch.setattr(Project, "save", lambda _: "")
    monkeypatch.setattr(menu.VoiceMenuShared, "ask_voice_sample_position", lambda *_: 1)
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: "Moved and updated")

    select(state, submenu, menu.LABEL_MOVE_VOICE_SAMPLE)
    assert get_string_from(state, submenu["heading"]) == "Voice 2"
    set_item.handler(state, set_item)
    assert project.voice_references == [untouched, replace(moved, instructions={
        BREEZE: "Moved and updated", "other": "Retain",
    })]
    clear_item.handler(state, clear_item)
    assert project.voice_references == [untouched, replace(moved, instructions={"other": "Retain"})]
    assert "Clear instructions" not in labels(state, submenu)


def test_cosy_runtime_mode_hides_and_restores_both_controls_without_clearing_data(monkeypatch):
    original = Voice(instructions={COSY: "Preserved while disabled"})
    project = Project(tts_model_type=COSY, voice_references=[original])
    state, submenu = open_submenu(monkeypatch, project)

    for mode, supported in [(None, False), ("instruct", True), ("cross_lingual", False), ("instruct", True)]:
        if mode is not None:
            project.set_model_setting(COSY, "template_name", mode)
        assert Tts.get_model_support(project).can_instructions(project) is supported
        current_labels = labels(state, submenu)
        ui = project.get_tts_model_type().value.ui
        expected_label = f"{ui.get('instructions_label', 'Instructions')}/{ui['short_name']} (currently: "
        assert any(value.startswith(expected_label) for value in current_labels) is supported
        assert ("Clear instructions" in current_labels) is supported
        assert project.voice_references == [original]


def test_breeze_ui_set_and_clear_reach_real_adapter_and_persist(monkeypatch):
    project = Project(tts_model_type=BREEZE, voice_references=[
        Voice(instructions={BREEZE: "Unselected voice"}),
        Voice(instructions={"other": "Not a Breeze instruction"}),
    ])
    state, submenu = open_submenu(monkeypatch, project, index=1)
    definition = load_audio_cpp_definitions().models[BREEZE]
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "breeze-server")
    sent, snapshots = [], []
    monkeypatch.setattr(AudioCppUtil, "generate", lambda _url, payload, **_: sent.append(payload) or SimpleNamespace())
    monkeypatch.setattr(AudioCppUtil, "make_voice_ref", lambda _: pytest.fail("sample-less voice accessed"))
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: "  Speak With Warmth  ")
    monkeypatch.setattr(Project, "save", lambda self: snapshots.append(
        ProjectSerializationUtil.to_project_json_dict(self)) or "")

    select(state, submenu, "Instructions")
    assert "Clear instructions" in labels(state, submenu)
    assert not isinstance(adapter.generate_using_project(project, ["Hello"], voice_selection_index=1), str)
    assert sent[-1]["options"]["instruction"] == "Speak With Warmth"
    assert Project.model_validate(snapshots[-1]).voice_references[1].instructions == {
        BREEZE: "Speak With Warmth", "other": "Not a Breeze instruction",
    }

    select(state, submenu, "Clear instructions")
    assert "Clear instructions" not in labels(state, submenu)
    assert label(state, item_named(state, submenu, "Instructions")) == "Instructions/Breeze TTS 2 (currently: none)"
    assert not isinstance(adapter.generate_using_project(project, ["Hello"], voice_selection_index=1), str)
    assert "instruction" not in sent[-1]["options"]
    reloaded = Project.model_validate(snapshots[-1])
    assert reloaded.voice_references == project.voice_references
    assert reloaded.voice_references[1].instructions == {"other": "Not a Breeze instruction"}
    assert reloaded.voice_references[0].instructions == {BREEZE: "Unselected voice"}
    assert len(snapshots) == 2
