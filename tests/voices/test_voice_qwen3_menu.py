from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.model import model_qwen3_menu
from tts_audiobook_tool.menus.model.model_qwen3_menu import ModelQwen3Menu
from tts_audiobook_tool.menus.voice import voice_qwen3_menu
from tts_audiobook_tool.menus.voice.voice_qwen3_menu import VoiceQwen3Menu
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
    assert not any(label.startswith(("Speaker", "Instructions")) for label in rendered_labels[0])
    assert not any(
        label.startswith("Select voice clone sample") for label in rendered_labels[0]
    )
    assert not any(
        label.startswith("Select voice clone sample") for label in rendered_labels[1]
    )
    assert not any(label.startswith("Speaker") for label in rendered_labels[1])
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
        # stored, and no voice has instructions here. VoiceDesign treats that
        # as a non-blocking hint; CustomVoice needs (error-colored) a speaker per voice.
        if model_type == "voice_design":
            assert rendered_labels[0] == f"{first}no instructions set"
            assert voice_menu_shared.COL_ERROR not in raw_labels[0]
        else:
            assert rendered_labels[0] == f"{first}no speaker (required)"
            assert voice_menu_shared.COL_ERROR in raw_labels[0]
    elif sample_count == 0:
        # The Base checkpoint uses (and requires) a voice clone sample
        assert rendered_labels[0] == f"{first}required"
        assert voice_menu_shared.COL_ERROR in raw_labels[0]
    else:
        assert rendered_labels[0] == f"{first}voice_0"
    assert not any(label.startswith(("Select Qwen3-TTS model", "Temperature", "Seed",
                                     "Add/remove voice samples")) for label in rendered_labels)
    expected = [f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} {i + 1}: " for i in range(max(sample_count, 1))]
    expected.append(voice_menu_shared.LABEL_ADD_VOICE_SAMPLE)
    if sample_count:  # Not shown for the one default voice
        expected.append(voice_menu_shared.LABEL_CLEAR_VOICES)
    # Speakers are per voice (voice submenu), not Voices menu items
    expected.append(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    expected.append(voice_menu_shared.LABEL_AUTO_ASSIGN_CHARACTER_VOICES)
    assert len(rendered_labels) == len(expected)
    assert all(label.startswith(prefix) for label, prefix in zip(rendered_labels, expected))


@pytest.mark.parametrize("model_type", ["base", "custom_voice", "voice_design"])
def test_qwen_model_settings_exclude_voice_controls(monkeypatch, model_type):
    project = Project(tts_model_type="qwen3tts_local")
    project.voice_references = [Voice(presets={"qwen3tts_local": "Vivian"},
                                      instructions={"qwen3tts_local": "Speak warmly"})]
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(model_qwen3_menu.ModelMenuShared, "menu_wrapper",
                        lambda state, items: captured.update(items=items(state)))
    inspection = TtsInspected(operation_id="test", tts_type_id="qwen3tts_local",
                              metadata={"model_type": model_type, "supported_speakers": ["Vivian"]})
    ModelQwen3Menu.menu(state, inspection)
    labels = [get_string_from(state, item.label) for item in captured["items"]]
    assert labels[0].startswith("Select Qwen3-TTS model")
    assert not any(label.startswith(("Speaker", "Clear speaker", "Instructions", "Clear instructions"))
                   for label in labels)


def open_voice_submenu(monkeypatch, state, index=0):
    """Open Voice N's submenu from the Voices menu; returns its item maker."""
    captured, submenu = {}, {}
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", lambda state, items: captured.update(items=items))
    VoiceQwen3Menu.menu(state)
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: submenu.update(kwargs))
    voice_item = captured["items"](state)[index]
    voice_item.handler(state, voice_item)
    return lambda: submenu["items"](state)


def find_item(state, items, prefix):
    return next((item for item in items
                 if strip_ansi_codes(get_string_from(state, item.label)).startswith(prefix)), None)


@pytest.mark.parametrize("speakers", [["Vivian"], ["Vivian", "Ryan"]])
def test_qwen_speaker_selection_is_per_voice_and_inspects_on_demand(monkeypatch, speakers):
    project = Project(tts_model_type="qwen3tts_local", voice_references=[
        Voice("voice.flac", "Reference.", presets={"qwen3tts_local": "Invalid"}), Voice(),
    ])
    project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    state = cast(State, SimpleNamespace(project=project))
    options, inspections, saves, feedback = {}, [], [], []
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "options_menu", lambda **kwargs: options.update(kwargs))
    monkeypatch.setattr(voice_menu_shared, "print_feedback", feedback.append)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self) or "")
    inspection = TtsInspected(operation_id="test", tts_type_id="qwen3tts_local",
                              metadata={"model_type": "custom_voice", "supported_speakers": speakers})
    monkeypatch.setattr(voice_qwen3_menu.ModelWorker, "inspect_tts_blocking",
                        lambda current: (inspections.append(current) or inspection, ""))
    make_items = open_voice_submenu(monkeypatch, state)
    speaker = find_item(state, make_items(), "Speaker")
    assert speaker is not None and "Invalid" in strip_ansi_codes(get_string_from(state, speaker.label))
    assert inspections == []
    speaker.handler(state, speaker)
    assert inspections == [state]
    if len(speakers) == 1:
        assert options == {}
        assert feedback == ["Model has only one speaker (Vivian)"]
    else:
        assert options["values"] == speakers and options["current_value"] == "Invalid"
        options["on_select"]("Vivian")
    # Saved to this voice only, so instance-less pre-flight sees a speaker
    assert [voice.presets for voice in project.voice_references] == [{"qwen3tts_local": "Vivian"}, {}]
    assert saves == [project]
    clear = find_item(state, make_items(), "Clear speaker")
    assert clear is not None
    clear.handler(state, clear)
    assert [voice.presets for voice in project.voice_references] == [{}, {}]
    assert project.voice_references[0].file_name == "voice.flac"
    assert find_item(state, make_items(), "Clear speaker") is None
    label = strip_ansi_codes(get_string_from(state, find_item(state, make_items(), "Speaker").label))
    assert "required" in label


@pytest.mark.parametrize("model_type", ["", "base", "voice_design"])
def test_qwen_speaker_item_only_for_custom_voice(monkeypatch, model_type):
    project = Project(tts_model_type="qwen3tts_local",
                      voice_references=[Voice(presets={"qwen3tts_local": "Vivian"})])
    project.set_model_setting("qwen3tts_local", "model_type", model_type)
    state = cast(State, SimpleNamespace(project=project))
    make_items = open_voice_submenu(monkeypatch, state)
    assert find_item(state, make_items(), "Speaker") is None
    row = strip_ansi_codes(VoiceMenuShared.make_voice_sample_item_label(
        project, project.get_tts_model_type(), 0)[0])
    assert "speaker" not in row


@pytest.mark.parametrize("inspected_type,error", [(None, "Model worker is busy"), (None, ""), ("base", "")])
def test_qwen_speaker_edit_handles_inspection_failure_or_changed_checkpoint(monkeypatch, inspected_type, error):
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    state = cast(State, SimpleNamespace(project=project))
    errors = []
    inspection = (TtsInspected(operation_id="test", tts_type_id="qwen3tts_local",
                               metadata={"model_type": inspected_type}) if inspected_type else None)
    monkeypatch.setattr(voice_qwen3_menu.ModelWorker, "inspect_tts_blocking", lambda _: (inspection, error))
    monkeypatch.setattr(voice_qwen3_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "options_menu",
                        lambda **_: pytest.fail("No valid speaker inventory"))
    make_items = open_voice_submenu(monkeypatch, state)
    speaker = find_item(state, make_items(), "Speaker")
    assert speaker is not None
    speaker.handler(state, speaker)
    assert errors == [error or ("Speaker selection requires a CustomVoice checkpoint" if inspected_type
                               else "Couldn't inspect Qwen3-TTS model")]
    assert project.voice_references[0].presets == {}


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
    # The voice, Add, Clear, then Edit voice/line selections and Auto-assign
    # character voices (speakers are per voice).
    assert len(labels) == 5
    assert labels[-2].startswith(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    assert labels[-1].startswith(voice_menu_shared.LABEL_AUTO_ASSIGN_CHARACTER_VOICES)
