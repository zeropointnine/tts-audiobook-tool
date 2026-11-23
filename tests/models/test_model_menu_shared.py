from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.model_target import ModelTarget
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
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_catalog_menu.print_feedback",
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


@pytest.mark.parametrize(("remote", "model_id", "available_count", "expected"), [
    (True, "none", 2, "1"),
    (True, "none", 1, ""),
    (True, "omnivoice_audiocpp", 2, ""),
    (False, "omnivoice_audiocpp", 2, ""),
])
def test_remote_model_menu_opens_picker_immediately_when_selection_required(
        monkeypatch, remote, model_id, available_count, expected):
    from tts_audiobook_tool.tts import Tts, TtsRuntimeMode

    state = cast(State, SimpleNamespace(project=Project(tts_model_type=model_id)))
    captured = {}
    models = [TtsModelType.require_by_id("chatterbox_audiocpp"), TtsModelType.require_by_id("higgs_v3_audiocpp")]
    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    if remote:
        monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: models[:available_count])

    ModelMenuShared.menu(state)

    assert captured["initial_hotkey"] == expected
    if expected:
        picker = captured["items"](state)[0]
        assert get_string_from(state, picker.label).startswith("Switch TTS model")


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
    applied_targets: list[ModelTarget] = []

    def capture_menu(**kwargs) -> None:
        captured_items.extend(kwargs["items"])

    monkeypatch.setattr(model_menu_shared.MenuUtil, "menu", capture_menu)

    ModelMenuShared.target_submenu(
        state=state,
        heading="Model",
        preset_targets=[ModelTarget.hf("repo/current"), ModelTarget.hf("repo/other")],
        current_target=ModelTarget.hf("repo/current"),
        default_target=ModelTarget.hf("repo/current"),
        ask_custom_target=lambda: None,
        apply_target=applied_targets.append,
    )

    captured_items[0].handler(state, captured_items[0])
    captured_items[1].handler(state, captured_items[1])

    assert applied_targets == [ModelTarget.hf("repo/other")]




# --- ask_target: the L/H kind step, then a directory or a repo id ---

def run_ask_target(monkeypatch, *, hotkey, current=None, dir_path="", entered="", exist_error="",
                   spec=None):
    """Runs `ask_target` with stubbed input; returns (callbacks, errors, feedback, dialog kwargs)."""
    from tts_audiobook_tool.app_types.model_target import ModelTarget
    from tts_audiobook_tool.menus.model.model_menu_shared import TargetPromptSpec

    callbacks, errors, feedback, dialogs = [], [], [], []
    monkeypatch.setattr(model_menu_shared.ask, "ask_hotkey", lambda *_a, **_k: hotkey)
    monkeypatch.setattr(model_menu_shared.ask, "ask_dir_path",
                        lambda **kwargs: dialogs.append(kwargs) or dir_path)
    monkeypatch.setattr(model_menu_shared.ask, "ask_input", lambda **_k: entered)
    monkeypatch.setattr(model_menu_shared.ask, "ask_error", errors.append)
    monkeypatch.setattr(model_menu_shared, "print_feedback", lambda *args, **_k: feedback.append(args))
    monkeypatch.setattr(ModelTarget, "exist_test", lambda self, markers=None: ("", exist_error))
    ModelMenuShared.ask_target(
        project=cast(Project, None),
        spec=spec or TargetPromptSpec(subject="Test model", hf_example="owner/repo"),
        current_target=current,
        callback=lambda _project, target: callbacks.append(target),
    )
    return callbacks, errors, feedback, dialogs


def test_ask_target_local_uses_the_directory_request_and_types_the_result(monkeypatch, tmp_path):
    # L opens the app's directory request, starting beside the current local target
    from tts_audiobook_tool.app_types.model_target import ModelTarget
    current = ModelTarget.local(str(tmp_path / "checkpoint-500"))
    callbacks, errors, _, dialogs = run_ask_target(
        monkeypatch, hotkey="l", current=current, dir_path=str(tmp_path / "checkpoint-900"))
    assert callbacks == [ModelTarget.local(str(tmp_path / "checkpoint-900"))] and not errors
    assert dialogs[0]["initialdir"] == str(tmp_path)


def test_ask_target_hf_accepts_repo_ids_subfolders_and_pasted_urls(monkeypatch):
    from tts_audiobook_tool.app_types.model_target import ModelTarget
    entered = "https://huggingface.co/owner/repo/tree/main/sub/dir"
    callbacks, errors, _, dialogs = run_ask_target(monkeypatch, hotkey="h", entered=entered)
    assert callbacks == [ModelTarget.hf("owner/repo/sub/dir")] and not errors and not dialogs


@pytest.mark.parametrize("hotkey,dir_path,entered", [
    ("\r", "", ""),  # neither L nor H
    ("x", "", ""),
    ("\x03", "", ""),  # Ctrl-C
    ("l", "", ""),  # cancelled dialog
    ("h", "", ""),  # empty entry
])
def test_ask_target_aborts_quietly(monkeypatch, hotkey, dir_path, entered):
    callbacks, errors, feedback, _ = run_ask_target(monkeypatch, hotkey=hotkey, dir_path=dir_path, entered=entered)
    assert callbacks == [] and errors == [] and feedback == []


def test_ask_target_hf_rejects_a_local_path_and_points_to_l(monkeypatch):
    callbacks, errors, _, _ = run_ask_target(monkeypatch, hotkey="h", entered="/home/lee/models/x")
    assert callbacks == []
    assert len(errors) == 1 and "Not a valid Hugging Face repo id" in errors[0] and "local directory" in errors[0]


def test_ask_target_same_kind_and_value_is_already_set(monkeypatch):
    from tts_audiobook_tool.app_types.model_target import ModelTarget
    callbacks, _, feedback, _ = run_ask_target(
        monkeypatch, hotkey="h", current=ModelTarget.hf("owner/repo"), entered="owner/repo")
    assert callbacks == [] and feedback == [("Already set",)]


def test_ask_target_switching_kind_with_the_same_value_is_a_change(monkeypatch, tmp_path):
    # A repo-shaped string chosen as a local directory is a different target
    from tts_audiobook_tool.app_types.model_target import ModelTarget
    callbacks, _, _, _ = run_ask_target(
        monkeypatch, hotkey="l", current=ModelTarget.hf("owner/repo"), dir_path="owner/repo")
    assert callbacks == [ModelTarget.local("owner/repo")]


def test_ask_target_reports_a_failed_existence_check(monkeypatch):
    callbacks, errors, _, _ = run_ask_target(monkeypatch, hotkey="h", entered="owner/repo",
                                             exist_error="Hugging Face repo not found or gated: owner/repo")
    assert callbacks == [] and errors == ["Hugging Face repo not found or gated: owner/repo"]
