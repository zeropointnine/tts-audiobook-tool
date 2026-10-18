from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.model import model_qwen3_menu
from tts_audiobook_tool.menus.model.model_qwen3_menu import ModelQwen3Menu
from tts_audiobook_tool.menus.voice import voice_qwen3_menu
from tts_audiobook_tool.menus.voice.voice_qwen3_menu import VoiceQwen3Menu
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.model_worker_protocol import TtsInspected
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from project_settings_test_support import set_setting


def test_model_selection_refreshes_qwen_menu_model_type(monkeypatch) -> None:
    project = Project.model_validate(
        {
            "qwen3_target": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "qwen3_model_type": "custom_voice",
        }
    )
    state = cast(State, SimpleNamespace(project=project))
    initial_inspection = TtsInspected(
        operation_id="initial",
        tts_type_id="qwen3tts_local",
        metadata={
            "model_type": "custom_voice",
            "supported_speakers": ["Vivian"],
        },
    )
    rendered_labels: list[list[str]] = []

    def fake_model_target_submenu(state: State, apply_target) -> None:
        apply_target("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    def fake_apply_model_and_validate(
        state: State, target: str, on_applied=None
    ) -> None:
        set_setting(state.project, "qwen3_target", target)
        set_setting(state.project, "qwen3_model_type", "base")
        updated_inspection = TtsInspected(
            operation_id="updated",
            tts_type_id="qwen3tts_local",
            metadata={"model_type": "base"},
        )
        if on_applied is not None:
            on_applied(updated_inspection)

    def fake_menu_wrapper(state: State, make_items) -> None:
        custom_items: list[MenuItem] = make_items(state)
        rendered_labels.append(
            [strip_ansi_codes(get_string_from(state, item.label)) for item in custom_items]
        )
        target_item = next(
            item
            for item in custom_items
            if strip_ansi_codes(get_string_from(state, item.label)).startswith(
                "Select Qwen3-TTS model"
            )
        )
        target_item.handler(state, target_item)

        base_items: list[MenuItem] = make_items(state)
        rendered_labels.append(
            [strip_ansi_codes(get_string_from(state, item.label)) for item in base_items]
        )

    monkeypatch.setattr(
        model_qwen3_menu, "model_target_submenu", fake_model_target_submenu
    )
    monkeypatch.setattr(
        model_qwen3_menu, "apply_model_and_validate", fake_apply_model_and_validate
    )
    monkeypatch.setattr(
        model_qwen3_menu.ModelMenuShared, "menu_wrapper", fake_menu_wrapper
    )

    ModelQwen3Menu.menu(state, initial_inspection)

    assert any("model type: custom_voice" in label for label in rendered_labels[0])
    assert not any(label.startswith(("Set speaker", "Instructions")) for label in rendered_labels[0])
    assert not any(
        label.startswith("Select voice clone sample") for label in rendered_labels[0]
    )
    assert not any(
        label.startswith("Select voice clone sample") for label in rendered_labels[1]
    )
    assert not any(label.startswith("Set speaker") for label in rendered_labels[1])
    assert any("model type: base" in label for label in rendered_labels[1])


@pytest.mark.parametrize("model_type", ["", "unknown", "base", "custom_voice", "voice_design"])
@pytest.mark.parametrize("sample_count", [0, 1, 2, 9])
def test_qwen_voice_samples_are_available_without_inspection(monkeypatch, model_type, sample_count) -> None:
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "model_type", model_type)
    # No samples: the project's one default (sample-less) voice
    project.voice_references = [
        Voice(f"voice_{i}.flac", "Reference words.")
        for i in range(sample_count)
    ] or [Voice()]
    state = cast(State, SimpleNamespace(project=project))
    rendered_labels: list[str] = []
    raw_labels: list[str] = []

    def capture_menu(current: State, make_items) -> None:
        # Keep the wrapper's own composition (selection controls appended last)
        make_items = VoiceMenuShared.with_voice_selection_items(make_items)
        raw_labels.extend(get_string_from(current, item.label) for item in make_items(current))
        rendered_labels.extend(strip_ansi_codes(label) for label in raw_labels)

    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", capture_menu)
    monkeypatch.setattr(
        model_qwen3_menu.ModelWorker,
        "inspect_tts_blocking",
        lambda *_: pytest.fail("Voice management must not inspect a checkpoint"),
    )
    VoiceQwen3Menu.menu(state)

    assert rendered_labels
    first = f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} 1: "
    if model_type in ("custom_voice", "voice_design"):
        # These checkpoints can't clone: sample details are hidden even when
        # stored, and no voice has instructions here. Only VoiceDesign treats
        # that as a (error-colored) hint; CustomVoice uses the speaker's delivery.
        assert rendered_labels[0] == f"{first}no instructions"
        assert (voice_qwen3_menu.COL_ERROR in raw_labels[0]) == (model_type == "voice_design")
    elif sample_count == 0:
        # The Base checkpoint uses (and requires) a voice clone sample
        assert rendered_labels[0] == f"{first}no sample (required)"
        assert voice_qwen3_menu.COL_ERROR in raw_labels[0]
    else:
        assert rendered_labels[0] == f"{first}voice_0.flac"
    assert (voice_menu_shared.LABEL_ADD_VOICE_SAMPLE in rendered_labels) == (sample_count < 9)
    assert not any(label.startswith(("Select Qwen3-TTS model", "Temperature", "Seed",
                                     "Add/remove voice samples")) for label in rendered_labels)
    expected = [f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} {i + 1}: " for i in range(max(sample_count, 1))]
    if sample_count < 9:
        expected.append(voice_menu_shared.LABEL_ADD_VOICE_SAMPLE)
    # Model-specific items precede the shared selection controls
    if model_type == "custom_voice":
        expected.append("Set speaker")
    if sample_count > 1:
        expected.append(voice_menu_shared.LABEL_VOICE_SELECTION_MODE)
    expected.append(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    assert len(rendered_labels) == len(expected)
    assert all(label.startswith(prefix) for label, prefix in zip(rendered_labels, expected))


@pytest.mark.parametrize("model_type", ["base", "custom_voice", "voice_design"])
def test_qwen_model_settings_exclude_voice_controls(monkeypatch, model_type):
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "speaker_id", "Vivian")
    project.voice_references = [Voice(instructions={"qwen3tts_local": "Speak warmly"})]
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(model_qwen3_menu.ModelMenuShared, "menu_wrapper",
                        lambda state, items: captured.update(items=items(state)))
    inspection = TtsInspected(operation_id="test", tts_type_id="qwen3tts_local",
                              metadata={"model_type": model_type, "supported_speakers": ["Vivian"]})
    ModelQwen3Menu.menu(state, inspection)
    labels = [get_string_from(state, item.label) for item in captured["items"]]
    assert labels[0].startswith("Select Qwen3-TTS model")
    assert not any(label.startswith(("Set speaker", "Clear speaker", "Instructions", "Clear instructions"))
                   for label in labels)


@pytest.mark.parametrize("speakers", [["Vivian"], ["Vivian", "Ryan"]])
def test_qwen_speaker_selection_inspects_on_demand_and_keeps_validation(monkeypatch, speakers):
    project = Project(tts_model_type="qwen3tts_local", voice_references=[
        {"file_name": "voice.flac", "transcript": "Reference."},
    ])
    project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    project.set_model_setting("qwen3tts_local", "speaker_id", "Invalid")
    state = cast(State, SimpleNamespace(project=project))
    captured, edits, inspections, saves, feedback = {}, [], [], [], []
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items: captured.update(items=items))
    monkeypatch.setattr(voice_qwen3_menu.ask, "ask_string_and_save",
                        lambda *args, **kwargs: edits.append((args, kwargs)))
    monkeypatch.setattr(voice_qwen3_menu, "print_feedback", feedback.append)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))
    inspection = TtsInspected(operation_id="test", tts_type_id="qwen3tts_local",
                              metadata={"model_type": "custom_voice", "supported_speakers": speakers})
    monkeypatch.setattr(voice_qwen3_menu.ModelWorker, "inspect_tts_blocking",
                        lambda current: (inspections.append(current) or inspection, ""))
    VoiceQwen3Menu.menu(state)
    items = captured["items"](state)
    assert inspections == []
    speaker = next(item for item in items if get_string_from(state, item.label).startswith("Set speaker"))
    assert speaker.blank_line_before
    speaker.handler(state, speaker)
    assert inspections == [state]
    label = strip_ansi_codes(get_string_from(state, speaker.label))
    if len(speakers) == 1:
        assert "Vivian" in label and "invalid" not in label
        assert edits == []
        assert feedback == ["Model has only one speaker id (Vivian)"]
        # Saved, so instance-less readiness sees a speaker id
        assert project.get_model_setting("qwen3tts_local", "speaker_id") == "Vivian"
        assert saves == [project]
        saves.clear()
    else:
        assert "current id is invalid" in label
        args, kwargs = edits[0]
        assert args[0] is project
        assert args[2] == SettingRef("qwen3tts_local", "speaker_id")
        assert kwargs["validator"]("Vivian") == ""
        assert kwargs["validator"]("Invalid") == "Invalid speaker id"
        project.set_model_setting("qwen3tts_local", "speaker_id", "Vivian")
        assert "invalid" not in get_string_from(state, speaker.label)
    clear = next(item for item in captured["items"](state) if item.label == "Clear speaker")
    clear.handler(state, clear)
    assert project.get_model_setting("qwen3tts_local", "speaker_id") == ""
    assert saves == [project]
    assert len(project.voice_references) == 1
    assert not any(item.label == "Clear speaker" for item in captured["items"](state))


@pytest.mark.parametrize("inspected_type,error", [(None, "Model worker is busy"), (None, ""), ("base", "")])
def test_qwen_speaker_edit_handles_inspection_failure_or_changed_checkpoint(monkeypatch, inspected_type, error):
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    state = cast(State, SimpleNamespace(project=project))
    captured, errors = {}, []
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items: captured.update(items=items))
    inspection = (TtsInspected(operation_id="test", tts_type_id="qwen3tts_local",
                               metadata={"model_type": inspected_type}) if inspected_type else None)
    monkeypatch.setattr(voice_qwen3_menu.ModelWorker, "inspect_tts_blocking", lambda _: (inspection, error))
    monkeypatch.setattr(voice_qwen3_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(voice_qwen3_menu, "ask_speaker_id", lambda *_: pytest.fail("No valid speaker inventory"))
    VoiceQwen3Menu.menu(state)
    speaker = next(item for item in captured["items"](state)
                   if get_string_from(state, item.label).startswith("Set speaker"))
    speaker.handler(state, speaker)
    assert errors == [error or ("Speaker selection requires a CustomVoice checkpoint" if inspected_type
                               else "Couldn't inspect Qwen3-TTS model")]
    if inspected_type == "base":
        assert not any(get_string_from(state, item.label).startswith(("Set speaker", "Instructions"))
                       for item in captured["items"](state))


@pytest.mark.parametrize("model_type", ["base", "custom_voice", "voice_design"])
@pytest.mark.parametrize("instructions", ["", "Speak warmly"])
def test_qwen_voice_menu_has_no_model_global_instruction_controls(monkeypatch, model_type, instructions):
    project = Project(tts_model_type="qwen3tts_local", voice_references=[
        Voice(instructions={"qwen3tts_local": instructions}),
    ])
    project.set_model_setting("qwen3tts_local", "model_type", model_type)
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items: captured.update(
                            items=VoiceMenuShared.with_voice_selection_items(items)(state)))
    monkeypatch.setattr(voice_qwen3_menu.ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("Opening voice management must not load a model"))
    original_get = Project.get_model_setting

    def get_setting(self, model_id, name):
        assert name != "instructions", "Global instructions must never be read"
        return original_get(self, model_id, name)

    monkeypatch.setattr(Project, "get_model_setting", get_setting)
    VoiceQwen3Menu.menu(state)
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in captured["items"]]
    assert not any(label.startswith(("Instructions", "Clear instructions")) for label in labels)
    # Shared voice rows may still preview this voice's instructions.
    # Default voice, Add, CustomVoice's speaker, then Edit voice/line selections.
    assert len(labels) == 3 + (model_type == "custom_voice")
    if model_type == "custom_voice":
        assert labels[2].startswith("Set speaker")
    assert labels[-1].startswith(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
