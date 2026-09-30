from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.constants import VOICE_ADVANCED_SUPERLABEL
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppStylesheetFailed,
    EditorSaveFailed,
)
from textual_editor_stubs import StubPhraseGroup, StubProject


@pytest.mark.parametrize("model_id", ["chatterbox_audiocpp", "echo_tts_audiocpp", "mira_local"])
@pytest.mark.parametrize("subheading", [None, "Existing guidance", lambda current: current.project.language_code])
def test_menu_wrapper_displays_model_note_and_preserves_subheading(monkeypatch, model_id, subheading):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type=model_id, language_code="fr")))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))

    VoiceMenuShared.menu_wrapper(state, [], subheading=subheading)

    note = state.project.get_tts_model_type().value.ui.get("settings_note", "").strip()
    existing = get_string_from(state, subheading) if subheading else ""
    assert get_string_from(state, captured["subheading"]) == "\n\n".join(part for part in (note, existing) if part)
    assert captured["heading"] == "Voice clone and model settings"
    assert captured["breadcrumb"] == "Voice"


def test_menu_wrapper_note_tracks_selected_model_and_remains_visible(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="chatterbox_audiocpp")))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu_wrapper(state, [])

    subheading = captured["subheading"]
    note = get_string_from(state, subheading)
    assert "English always uses dedicated English-only weights." in note
    assert "Other languages use\nMultilingual V2 by default." in note
    assert "To select Multilingual V3, use audio.cpp setting:" in note
    assert 'chatterbox.multilingual_t3 = "v3"' in note
    assert get_string_from(state, subheading) == note  # informational, not dismiss-once
    state.project.tts_model_type = "echo_tts_audiocpp"
    assert get_string_from(state, subheading) == ""


@pytest.mark.parametrize("initial_id, next_id, parameter, label, value", [
    ("omnivoice_audiocpp", "breeze_tts_2_audiocpp", "temperature", "Temperature", 0.6),
    ("omnivoice_audiocpp", "higgs_v3_sglomni", "temperature", "Temperature", 0.6),
    ("higgs_v3_sglomni", "omnivoice_audiocpp", "num_inference_steps", "Steps", 12),
])
def test_remote_voice_menu_redraw_resolves_reconciled_model_and_setting_owner(
        monkeypatch, initial_id, next_id, parameter, label, value):
    from tts_audiobook_tool.menus.menu_status import MenuStatus
    from tts_audiobook_tool.tts import Tts, TtsRuntimeMode

    state = cast(State, SimpleNamespace(project=Project(tts_model_type=initial_id)))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "bind_project", lambda project: None)
    monkeypatch.setattr(Project, "save", lambda self: "")
    monkeypatch.setattr(voice_menu_shared.ask, "ask_input", lambda **kwargs: str(value))
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_audio_cpp_menu.print_feedback",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.print_feedback",
                        lambda *args, **kwargs: None)
    VoiceMenuShared.menu(state)
    factory = captured["items"]
    assert factory is VoiceMenuShared.make_remote_items
    assert factory(state)

    initial_type = state.project.get_tts_model_type()
    initial_definition = Tts.get_audio_cpp_definition(initial_type) or Tts.get_configured_definition(initial_type)
    assert initial_definition is not None
    original_values = {name: state.project.get_model_setting(initial_id, name)
                       for name in initial_definition.parameters}
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda: [TtsModelType.require_by_id(next_id)])
    MenuStatus.prepare_tts(state)
    assert state.project.tts_model_type == next_id
    items = factory(state)
    control = next(item for item in items if get_string_from(state, item.label).startswith(label + " "))
    control.handler(state, control)
    assert state.project.get_model_setting(next_id, parameter) == value
    assert {name: state.project.get_model_setting(initial_id, name)
            for name in initial_definition.parameters} == original_values
    if next_id != "omnivoice_audiocpp":
        assert not any(get_string_from(state, item.label).startswith("Steps ") for item in items)


@pytest.mark.parametrize("model_id", ["none", "unknown-model"])
def test_remote_voice_menu_removes_controls_when_selection_is_unresolved(monkeypatch, model_id):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="omnivoice_audiocpp")))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu(state)
    assert captured["items"](state)
    state.project.tts_model_type = model_id
    assert captured["items"](state) == []


def test_remote_voice_sample_submenu_tracks_selection_on_redraw_and_edit(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="omnivoice_audiocpp")))
    state.project.set_model_setting("omnivoice_audiocpp", "file_name", ["old.flac"])
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    item = VoiceMenuShared.make_manage_voice_samples_item(state, state.project.get_tts_model_type())
    item.handler(state, item)

    state.project.tts_model_type = "breeze_tts_2_audiocpp"
    state.project.set_model_setting("breeze_tts_2_audiocpp", "file_name", ["new.flac"])
    subheading = get_string_from(state, captured["subheading"])
    assert "new.flac" in subheading and "old.flac" not in subheading
    calls = []
    monkeypatch.setattr(VoiceMenuShared, "ask_and_set_voice_file",
                        lambda current, model, **kwargs: calls.append((model.id, kwargs)))
    monkeypatch.setattr(VoiceMenuShared, "remove_voice_sample_from_menu",
                        lambda current, model: calls.append((model.id, "remove")) or False)
    items = captured["items"](state)
    items[0].handler(state, items[0])
    items[1].handler(state, items[1])
    assert calls == [("breeze_tts_2_audiocpp", {"append": True}), ("breeze_tts_2_audiocpp", "remove")]
    state.project.tts_model_type = "none"
    assert captured["items"](state) == []
    assert get_string_from(state, captured["subheading"]) == ""


@pytest.mark.parametrize(
    ("run_result", "expected_message"),
    [
        (
            ContentAppCompleted(EditorSaveFailed("Save failed: disk full")),
            "Save failed: disk full",
        ),
        (
            ContentAppStylesheetFailed("Couldn't load textual css"),
            "Couldn't load textual css",
        ),
    ],
)
def test_voice_sample_assignment_reports_editor_failures(
    run_result, expected_message: str, monkeypatch
) -> None:
    state = cast(
        State,
        SimpleNamespace(project=StubProject([StubPhraseGroup("Line 1")])),
    )
    feedback_calls: list[str] = []
    monkeypatch.setattr(
        voice_menu_shared,
        "VoiceLineEditorTextualApp",
        lambda _: object(),
    )
    monkeypatch.setattr(
        voice_menu_shared,
        "run_content_textual_app",
        lambda _: run_result,
    )
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", feedback_calls.append)

    voice_menu_shared.VoiceMenuShared.assign_voice_samples_to_text_lines(state)

    assert feedback_calls == [expected_message]

# ---------------------------------------------------------------------------
# Voice sample selection mode menu items (moved from test_voice_selection.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("voice_count", [0, 1, 2])
def test_voice_samples_subheading_has_no_trailing_newline(monkeypatch, voice_count):
    project = Project(tts_model_type="mira_local")
    voices = [f"voice-{index}.flac" for index in range(voice_count)]
    monkeypatch.setattr(voice_menu_shared.ProjectVoiceUtil, "get_voice_values", lambda *_: voices)
    monkeypatch.setattr(voice_menu_shared.ProjectVoiceUtil, "make_voice_sample_display_label", lambda _, name, __: name)

    text = VoiceMenuShared.make_voice_samples_subheading(project, TtsModelType.require_by_id("mira_local"))

    assert not text.endswith("\n")
    assert text.count("\n") == max(0, voice_count - 1)
    for index, voice in enumerate(voices, start=1):
        assert f"Voice sample {index}:" in text and voice in text


@pytest.mark.parametrize("qualifier", ["", "Requires batch size 1."])
def test_rolling_continuation_subheading_preserves_paragraph_spacing(monkeypatch, qualifier):
    state = cast(State, SimpleNamespace(project=Project()))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "print_screen_heading", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(voice_menu_shared.ask, "ask_number_and_save", lambda *args, **kwargs: None)

    VoiceMenuShared.ask_rolling_continuation(state, "rolling_cont", 10, qualifier_line=qualifier)

    text = captured["subheading"]
    assert not text.endswith("\n")
    expected = voice_menu_shared.ROLLING_CONTINUATION_DESC
    if qualifier:
        expected += "\n\n" + qualifier
    assert text == expected


@pytest.mark.parametrize("voice_count, expected_item_count", [(0, 2), (1, 2), (2, 3)])
def test_voice_sample_selection_mode_item_requires_multiple_samples(
        voice_count: int,
        expected_item_count: int,
) -> None:
    project = Project.model_validate({
        "mira_voice_file_name": [f"voice-{index}.flac" for index in range(voice_count)],
    })
    state = cast(State, SimpleNamespace(project=project))

    items = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id("mira_local"))

    assert len(items) == expected_item_count
    assert get_string_from(state, items[0].label).startswith("Add/remove voice samples") or voice_count == 0
    if voice_count > 1:
        assert "Voice selection mode" in get_string_from(state, items[1].label)
        assert VoiceSelectMode.AUTO_ADVANCE.current_label in get_string_from(state, items[1].label)


def test_voice_sample_selection_mode_item_label_tracks_project_value() -> None:
    project = Project.model_validate({
        "mira_voice_file_name": ["voice-a.flac", "voice-b.flac"],
    })
    state = cast(State, SimpleNamespace(project=project))
    item = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id("mira_local"))[1]

    assert VoiceSelectMode.AUTO_ADVANCE.current_label in get_string_from(state, item.label)

    project.voice_select_mode = VoiceSelectMode.USER_DEFINED

    assert VoiceSelectMode.USER_DEFINED.current_label in get_string_from(state, item.label)


def test_voice_sample_selection_mode_submenu_uses_options_menu_and_saves_selection() -> None:
    project = Project.model_validate({
        "mira_voice_file_name": ["voice-a.flac", "voice-b.flac"],
    })
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(Project, "save") as save, \
            patch("tts_audiobook_tool.menus.voice.voice_menu_shared.MenuUtil.options_menu") as options_menu:
        VoiceMenuShared.voice_sample_selection_mode_submenu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](VoiceSelectMode.USER_DEFINED)

    assert kwargs["heading_text"] == "Voice selection mode"
    assert kwargs["labels"] == [mode.label for mode in VoiceSelectMode]
    assert kwargs["values"] == list(VoiceSelectMode)
    assert kwargs["current_value"] == VoiceSelectMode.AUTO_ADVANCE
    assert kwargs["default_value"] == VoiceSelectMode.get_default()
    assert kwargs["sublabels"] == [mode.description for mode in VoiceSelectMode]
    assert project.voice_select_mode == VoiceSelectMode.USER_DEFINED
    save.assert_called_once_with()


def test_target_submenu_only_applies_changed_preset(monkeypatch) -> None:
    state = cast(State, SimpleNamespace())
    captured_items = []
    applied_targets: list[str] = []

    def capture_menu(**kwargs) -> None:
        captured_items.extend(kwargs["items"])

    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", capture_menu)

    VoiceMenuShared.target_submenu(
        state=state,
        heading="Model",
        preset_targets=["repo/current", "repo/other"],
        current_target="repo/current",
        default_target="repo/current",
        ask_custom_target=lambda: None,
        apply_target=applied_targets.append,
    )

    captured_items[0].handler(state, captured_items[0])
    captured_items[1].handler(state, captured_items[1])

    assert applied_targets == ["repo/other"]


# ---------------------------------------------------------------------------
# Group superlabels for definition-driven menus
# ---------------------------------------------------------------------------


def make_items(count: int) -> list[MenuItem]:
    return [MenuItem(f"item-{index}", lambda *_: None) for index in range(count)]


def test_group_superlabels_render_once_at_first_member() -> None:
    items = make_items(5)

    VoiceMenuShared.apply_group_superlabels(items, ["advanced", "", "", "advanced", "advanced"])

    assert [item.superlabel for item in items] == [
        VOICE_ADVANCED_SUPERLABEL, "", "", "", "",
    ]


def test_group_superlabels_skip_ungrouped_items() -> None:
    items = make_items(3)

    VoiceMenuShared.apply_group_superlabels(items, ["", "", ""])

    assert all(not item.superlabel for item in items)


def test_group_superlabels_reject_mismatched_lists_and_unknown_groups() -> None:
    with pytest.raises(ValueError):
        VoiceMenuShared.apply_group_superlabels(make_items(2), ["advanced"])

    with pytest.raises(ValueError):
        VoiceMenuShared.apply_group_superlabels(make_items(1), ["mystery"])
