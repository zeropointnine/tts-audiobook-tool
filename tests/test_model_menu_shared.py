from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model import model_menu_shared
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.mark.parametrize("model_id", ["chatterbox_audiocpp", "echo_tts_audiocpp", "mira_local"])
@pytest.mark.parametrize("subheading", [None, "Existing guidance", lambda current: current.project.language_code])
def test_menu_wrapper_displays_model_note_and_preserves_subheading(monkeypatch, model_id, subheading):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type=model_id, language_code="fr")))
    captured = {}
    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))

    ModelMenuShared.menu_wrapper(state, [], subheading=subheading)

    note = state.project.get_tts_model_type().value.ui.get("settings_note", "").strip()
    existing = get_string_from(state, subheading) if subheading else ""
    assert get_string_from(state, captured["subheading"]) == "\n\n".join(part for part in (note, existing) if part)
    assert captured["heading"] == "Model settings"
    assert captured["breadcrumb"] == "Model"


def test_menu_wrapper_note_tracks_selected_model_and_remains_visible(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="chatterbox_audiocpp")))
    captured = {}
    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    ModelMenuShared.menu_wrapper(state, [])

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
def test_remote_model_menu_redraw_resolves_reconciled_model_and_setting_owner(
        monkeypatch, initial_id, next_id, parameter, label, value):
    from tts_audiobook_tool.menus.menu_status import MenuStatus
    from tts_audiobook_tool.tts import Tts, TtsRuntimeMode

    state = cast(State, SimpleNamespace(project=Project(tts_model_type=initial_id)))
    captured = {}
    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "bind_project", lambda project: None)
    monkeypatch.setattr(Project, "save", lambda self: "")
    monkeypatch.setattr(model_menu_shared.ask, "ask_input", lambda **kwargs: str(value))
    monkeypatch.setattr(model_menu_shared.ask, "print_feedback",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_configured_sgl_omni_menu.print_feedback",
                        lambda *args, **kwargs: None)
    ModelMenuShared.menu(state)
    factory = captured["items"]
    assert factory is ModelMenuShared.make_remote_items
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
def test_remote_model_menu_removes_controls_when_selection_is_unresolved(monkeypatch, model_id):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="omnivoice_audiocpp")))
    captured = {}
    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    ModelMenuShared.menu(state)
    assert captured["items"](state)
    state.project.tts_model_type = model_id
    assert captured["items"](state) == []


@pytest.mark.parametrize("model_id", ["none", "omnivoice_audiocpp", "higgs_v3_sglomni"])
def test_remote_model_menu_leads_with_model_picker(monkeypatch, model_id):
    # In remote mode, the TTS model picker is the first Model settings item and
    # the menu opens even with no model selected; model controls follow it.
    from tts_audiobook_tool.tts import Tts, TtsRuntimeMode

    state = cast(State, SimpleNamespace(project=Project(tts_model_type=model_id)))
    captured = {}
    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [])
    ModelMenuShared.menu(state)
    items = captured["items"](state)
    assert get_string_from(state, items[0].label).startswith("TTS model")
    assert not items[0].blank_line_before
    if model_id == "none":
        assert len(items) == 1
    else:
        assert len(items) > 1 and items[1].blank_line_before


@pytest.mark.parametrize("qualifier", ["", "Requires batch size 1."])
def test_rolling_continuation_subheading_preserves_paragraph_spacing(monkeypatch, qualifier):
    state = cast(State, SimpleNamespace(project=Project()))
    captured = {}
    monkeypatch.setattr(model_menu_shared.MenuUtil, "print_screen_heading", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(model_menu_shared.ask, "ask_number_and_save", lambda *args, **kwargs: None)

    ModelMenuShared.ask_rolling_continuation(state, "rolling_cont", 10, qualifier_line=qualifier)

    text = captured["subheading"]
    assert not text.endswith("\n")
    expected = model_menu_shared.ROLLING_CONTINUATION_DESC
    if qualifier:
        expected += "\n\n" + qualifier
    assert text == expected


def test_target_submenu_only_applies_changed_preset(monkeypatch) -> None:
    state = cast(State, SimpleNamespace())
    captured_items = []
    applied_targets: list[str] = []

    def capture_menu(**kwargs) -> None:
        captured_items.extend(kwargs["items"])

    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", capture_menu)

    ModelMenuShared.target_submenu(
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


