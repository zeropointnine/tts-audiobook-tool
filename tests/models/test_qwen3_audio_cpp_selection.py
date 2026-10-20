"""Filename convenience is confined to an explicit, exact audio.cpp selection."""
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.menus.model.model_select_menu as select_menu_module
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsSnapshot
from tts_audiobook_tool.menus.model.model_select_menu import ModelSelectMenu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.audio_cpp_behavior_qwen3 import infer_qwen3_model_type_from_path
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_ACCENT, COL_DEFAULT


MODEL_ID = "qwen3tts_audiocpp"
FILENAMES = [
    ("qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf", "base"),
    ("qwen3-tts-12hz-1.7b-base-bf16.gguf", "base"),
    ("qwen3-tts-12hz-1.7b-base-orig.gguf", "base"),
    ("qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf", "custom_voice"),
    ("qwen3-tts-12hz-1.7b-customvoice-bf16.gguf", "custom_voice"),
    ("qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf", "voice_design"),
    ("qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf", "voice_design"),
    ("qwen3-tts-12hz-0.6b-base-q8_0.gguf", "base"),
    ("qwen3-tts-12hz-0.6b-base-bf16.gguf", "base"),
]
BASE_FILENAME = FILENAMES[0][0]
TYPE_LABELS = {"base": "Base", "custom_voice": "CustomVoice", "voice_design": "VoiceDesign"}


@pytest.mark.parametrize(("filename", "expected"), FILENAMES)
@pytest.mark.parametrize("directory", ["", "/srv/models/", "C:\\models\\", "\\\\server\\models\\"])
def test_infer_exact_published_filename_for_server_path_styles(filename, expected, directory):
    # Server paths must work independently of the client's operating system.
    assert infer_qwen3_model_type_from_path(directory + filename) == expected


@pytest.mark.parametrize("path", [
    None, 123, False, {}, Path(BASE_FILENAME), "", "   ", ".", "..",
    "unknown.gguf", "renamed-" + BASE_FILENAME, BASE_FILENAME + ".bak",
    BASE_FILENAME.upper(), BASE_FILENAME.replace("base", "Base"),
    BASE_FILENAME.replace("q8_0_v2", "q8_0"),
    BASE_FILENAME.replace("q8_0_v2", "q8_0_v3"),
    BASE_FILENAME.replace("1.7b", "0.6b"),
    " " + BASE_FILENAME, BASE_FILENAME + " ",
    "/srv/" + BASE_FILENAME + "/unknown.gguf",
    "C:\\models\\" + BASE_FILENAME + "\\renamed.gguf",
    "/srv/" + BASE_FILENAME + "/",
    "C:\\models\\" + BASE_FILENAME + "\\",
])
def test_infer_does_not_guess_from_renames_directories_or_non_strings(path):
    # Only the exact case-sensitive basename is allowlisted, not substrings.
    assert infer_qwen3_model_type_from_path(path) is None


@pytest.fixture
def picker(monkeypatch):
    def open_picker(snapshot, *, manual="voice_design", available=None, save_error=""):
        model = TtsModelType.require_by_id(MODEL_ID)
        project = Project(tts_model_type=MODEL_ID, audio_cpp_model_id="previous-entry")
        project.set_model_setting(MODEL_ID, "model_type", manual)
        state = cast(State, SimpleNamespace(
            project=project, pending_tts_model_change=object(), pending_project_load_checks=False,
        ))
        captured = {}
        events = []

        def record_project(action, current):
            events.append((action, current.tts_model_type, current.audio_cpp_model_id,
                           current.get_model_setting(MODEL_ID, "model_type")))

        def save(current):
            record_project("save", current)
            return save_error

        monkeypatch.setattr(select_menu_module.RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
        monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models",
                            lambda **kwargs: [model] if available is None else available)
        monkeypatch.setattr(select_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))
        monkeypatch.setattr(Project, "save", save)
        monkeypatch.setattr(select_menu_module.Tts, "bind_project", lambda current: record_project("bind", current))
        monkeypatch.setattr(select_menu_module, "print_feedback", lambda *args: None)
        monkeypatch.setattr(select_menu_module.ask, "ask_error", lambda message: events.append(("error", message)))
        monkeypatch.setattr(select_menu_module.ask, "ask_enter_to_continue",
                            lambda message: events.append(("confirm", message)))

        before = (project.tts_model_type, project.audio_cpp_model_id,
                  project.get_model_setting(MODEL_ID, "model_type"))
        ModelSelectMenu.menu(state)
        # Opening and rendering a recognized entry is not an explicit selection.
        assert (project.tts_model_type, project.audio_cpp_model_id,
                project.get_model_setting(MODEL_ID, "model_type")) == before
        assert events == []
        assert state.pending_tts_model_change is not None
        assert state.pending_project_load_checks is False
        return state, captured, events

    return open_picker


def snapshot_for(path, *, server_id="operator-entry"):
    model = TtsModelType.require_by_id(MODEL_ID)
    return RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        models=({"id": server_id, "path": path},),
        candidates=((model, server_id),),
    )


@pytest.mark.parametrize(("filename", "expected"), FILENAMES)
@pytest.mark.parametrize("same_type", [False, True])
def test_explicit_selection_sets_type_before_single_save_and_bind(picker, filename, expected, same_type):
    # Even an already-correct manual value receives the explicit-selection FYI.
    manual = expected if same_type else ("voice_design" if expected != "voice_design" else "base")
    state, captured, events = picker(snapshot_for("C:\\models\\" + filename), manual=manual)
    choice = (TtsModelType.require_by_id(MODEL_ID), "operator-entry")
    assert choice in captured["values"]

    captured["on_select"](choice)

    expected_state = (MODEL_ID, "operator-entry", expected)
    assert events == [
        ("save", *expected_state),
        ("bind", *expected_state),
        ("confirm", "🔔 Qwen3-TTS model type has been automatically set to "
         f"{COL_ACCENT}{TYPE_LABELS[expected]}{COL_DEFAULT}\n   based on the model filename.\n"),
    ]
    assert state.project.get_model_setting(MODEL_ID, "model_type") == expected
    assert state.pending_tts_model_change is None
    assert state.pending_project_load_checks is True


@pytest.mark.parametrize("case", [
    "unknown_path", "missing_path", "none_path", "numeric_path", "missing_metadata",
    "duplicate_metadata", "duplicate_identical_metadata", "wrong_backend", "missing_backend",
    "wrong_family", "missing_candidate", "different_candidate_id", "different_candidate_family",
    "server_id_is_filename", "misleading_directory",
])
def test_ineligible_explicit_selection_preserves_manual_type(picker, case):
    # Metadata must belong uniquely to this exact (catalog type, server ID) pair.
    model = TtsModelType.require_by_id(MODEL_ID)
    other = TtsModelType.require_by_id("chatterbox_audiocpp")
    selected = (model, "operator-entry")
    backend = TtsBackendKind.AUDIO_CPP
    entries = ({"id": "operator-entry", "path": "/srv/" + BASE_FILENAME},)
    candidates = (selected,)
    if case == "unknown_path":
        entries = ({"id": "operator-entry", "path": "/srv/renamed.gguf"},)
    elif case == "missing_path":
        entries = ({"id": "operator-entry"},)
    elif case in ("none_path", "numeric_path"):
        entries = ({"id": "operator-entry", "path": None if case == "none_path" else 123},)
    elif case == "missing_metadata":
        entries = ({"id": "different-entry", "path": BASE_FILENAME},)
    elif case.startswith("duplicate"):
        entries += (dict(entries[0]) if case == "duplicate_identical_metadata" else
                    {"id": "operator-entry", "path": "/srv/renamed.gguf"},)
    elif case == "wrong_backend":
        backend = TtsBackendKind.SGL_OMNI
    elif case == "missing_backend":
        backend = None
    elif case == "wrong_family":
        selected = (other, "operator-entry")
        candidates = (selected,)
    elif case == "missing_candidate":
        candidates = ()
    elif case == "different_candidate_id":
        candidates = ((model, "other-entry"),)
    elif case == "different_candidate_family":
        candidates = ((other, "operator-entry"),)
    elif case == "server_id_is_filename":
        selected = (model, BASE_FILENAME)
        candidates = (selected,)
        entries = ({"id": BASE_FILENAME, "path": "/srv/renamed.gguf"},)
    elif case == "misleading_directory":
        entries = ({"id": "operator-entry", "path": "/srv/" + BASE_FILENAME + "/renamed.gguf"},)
    snapshot = RemoteTtsSnapshot(backend_kind=backend, models=entries, candidates=candidates)
    state, captured, events = picker(snapshot, available=[model, other])

    captured["on_select"](selected)

    expected_state = (selected[0].id, selected[1], "voice_design")
    assert events == [("save", *expected_state), ("bind", *expected_state)]
    assert state.project.get_model_setting(MODEL_ID, "model_type") == "voice_design"


def test_selection_uses_selected_metadata_not_first_qwen_entry(picker):
    # Multiple legitimate Qwen entries are choices, not duplicate metadata IDs.
    model = TtsModelType.require_by_id(MODEL_ID)
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        models=({"id": "first", "path": BASE_FILENAME},
                {"id": "second", "path": FILENAMES[3][0]}),
        candidates=((model, "first"), (model, "second")),
    )
    state, captured, events = picker(snapshot)

    captured["on_select"]((model, "second"))

    assert state.project.get_model_setting(MODEL_ID, "model_type") == "custom_voice"
    assert events[:2] == [("save", MODEL_ID, "second", "custom_voice"),
                          ("bind", MODEL_ID, "second", "custom_voice")]
    assert [event[0] for event in events] == ["save", "bind", "confirm"]


def test_failed_save_keeps_in_memory_selection_and_reports_error_without_success(picker):
    # Preserve the picker's existing unsaved-memory/bind policy on save failure.
    state, captured, events = picker(snapshot_for(BASE_FILENAME), save_error="cannot save project")

    captured["on_select"]((TtsModelType.require_by_id(MODEL_ID), "operator-entry"))

    expected_state = (MODEL_ID, "operator-entry", "base")
    assert events == [("save", *expected_state), ("error", "cannot save project"),
                      ("bind", *expected_state)]
    assert state.project.get_model_setting(MODEL_ID, "model_type") == "base"
    assert state.project.audio_cpp_model_id == "operator-entry"
    assert state.pending_tts_model_change is None
    assert state.pending_project_load_checks is True
