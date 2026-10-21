"""Project identity, available capability, and selected metadata are independent."""
from __future__ import annotations

import json
import sys

import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsIssue, RemoteTtsSnapshot
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


@pytest.fixture(autouse=True)
def metadata_catalog(monkeypatch):
    fields = ("_backend_mode", "_type", "_configured_runtime", "_config_fingerprint", "_catalog_initialized",
              "_configured_definitions", "_audio_cpp_definitions", "_available_local_models", "_binding_issue",
              "_bound_project_type_id", "_bound_project_audio_cpp_model_id", "_selected_server_model_id", "_remote_issue", "_model_params")
    saved = {name: getattr(Tts, name) for name in fields}
    registry = (dict(REGISTRY.bindings), dict(REGISTRY.legacy))
    catalog = (dict(TtsModelType._catalog), dict(TtsModelType._specs))
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.LOCAL))
    Tts.init_local_model_type()
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(Tts, name, value)
        REGISTRY.bindings, REGISTRY.legacy = registry
        TtsModelType._catalog, TtsModelType._specs = catalog


@pytest.mark.parametrize("model", [TtsModelType.require_by_id("vibevoice_local"), TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("higgs_v3_audiocpp")])
def test_stable_selection_round_trips_project_and_abr_and_copy(tmp_path, model):
    project = Project(dir_path=str(tmp_path), tts_model_type=model.id)
    assert project.get_tts_model_type() is model
    assert project.save() == ""
    payload = json.loads((tmp_path / "project.json").read_text())
    assert payload["tts_model_type"] == model.id
    assert Project.model_validate(payload).get_tts_model_type() is model
    snapshot = ProjectSerializationUtil.to_snapshot_dict(project)
    assert snapshot["tts_model_type"] == model.id
    dest = Project()
    ProjectTransferUtil.apply_project_settings(dest, project)
    assert dest.tts_model_type == model.id


def test_unknown_and_missing_selection_never_auto_detect_or_disappear():
    Tts._available_local_models = (TtsModelType.require_by_id("vibevoice_local"),)
    unknown = Project(tts_model_type="future_catalog_id")
    assert unknown.get_tts_model_type() is TtsModelType.require_by_id("none")
    assert Tts.bind_project(unknown) is not None
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    assert ProjectSerializationUtil.to_snapshot_dict(unknown)["tts_model_type"] == "future_catalog_id"
    missing = Project.model_validate({})
    assert missing.tts_model_type == "none"
    assert Tts.bind_project(missing) is not None
    assert missing.tts_model_type == "none"


@pytest.mark.parametrize("raw", ["none", "future_catalog_id"])
def test_unselected_support_uses_this_project_not_other_runtime_failure(monkeypatch, raw):
    from tts_audiobook_tool.readiness import get_tts_blockers
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "_remote_issue", "Other project's server is offline")
    project = Project(tts_model_type=raw)
    issues = Tts.get_model_support(project).get_blocking_issues(project, None)
    assert len(issues) == 1
    assert "offline" not in issues[0].verbose
    assert Tts.bind_project(project) is not None
    assert len(get_tts_blockers(project)) == 1


@pytest.mark.parametrize("bad", [None, 7])
def test_selection_requires_a_string(bad):
    with pytest.raises(ValueError):
        Project.model_validate({"tts_model_type": bad})


def test_local_availability_is_independent_of_bound_selection_and_returns_fresh_list():
    assert Tts.get_available_tts_models() == []
    Tts._available_local_models = (TtsModelType.require_by_id("vibevoice_local"),)
    first = Tts.get_available_tts_models()
    first.clear()
    assert Tts.bind_project(Project()) is not None
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    assert Tts.get_available_tts_models() == [TtsModelType.require_by_id("vibevoice_local")]
    assert Tts.bind_project(Project(tts_model_type="vibevoice_local")) is None
    assert Tts.get_active_type() is TtsModelType.require_by_id("vibevoice_local")


def test_multiple_local_capabilities_are_not_arbitrarily_truncated():
    Tts._available_local_models = (TtsModelType.require_by_id("vibevoice_local"), TtsModelType.require_by_id("chatterbox_local"))
    with pytest.raises(RuntimeError, match="More than one"):
        Tts.get_available_tts_models()


@pytest.mark.parametrize("available", [(), (TtsModelType.require_by_id("vibevoice_local"),),
                                       (TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("higgs_v3_audiocpp"))])
def test_fresh_project_initializes_only_sole_available_type(tmp_path, monkeypatch, available):
    monkeypatch.setattr(Tts, "get_available_tts_models", staticmethod(lambda **kwargs: list(available)))
    state = State.for_worker(Prefs())
    assert state.make_and_set_new_project(str(tmp_path / "book")) == ""
    expected = available[0].id if len(available) == 1 else "none"
    assert state.project.tts_model_type == expected
    payload = json.loads((tmp_path / "book" / "project.json").read_text())
    assert payload["tts_model_type"] == expected


def test_remote_metadata_is_initialized_in_local_environment_without_inference_imports():
    model = TtsModelType.require_by_id("higgs_v3_audiocpp")
    project = Project(tts_model_type=model.id)
    assert Tts.get_audio_cpp_definition(model) is not None
    assert REGISTRY.voice_binding(model.id) is not None
    assert Tts.get_info(project).id == model.id
    assert Tts.get_model_support(project).INFO.id == model.id
    assert Tts.bind_project(project) is not None
    assert Tts.get_info(project).id == model.id
    assert "tts_audiobook_tool.tts_models.chatterbox_model" not in sys.modules


def test_sgl_availability_empty_single_failure_and_explicit_refresh(monkeypatch):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    calls = []
    current = RemoteTtsSnapshot(TtsBackendKind.SGL_OMNI)
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda force=False: calls.append(force) or current)
    assert Tts.get_available_tts_models() == []
    current = RemoteTtsSnapshot(TtsBackendKind.SGL_OMNI, candidates=((TtsModelType.require_by_id("auk_sglomni"), "org/auk"),))
    assert Tts.get_available_tts_models(refresh=True) == [TtsModelType.require_by_id("auk_sglomni")]
    current = RemoteTtsSnapshot(TtsBackendKind.SGL_OMNI, issue=RemoteTtsIssue("unavailable", "offline"))
    assert Tts.get_available_tts_models() == []
    assert calls == [False, True, False]


@pytest.mark.parametrize("saved", ["none", "future_catalog_id",
                                   "auk_sglomni", "vibevoice_local"])
def test_reconcile_project_model_sole_local_capability(monkeypatch, saved):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("vibevoice_local"),))
    project = Project(tts_model_type=saved)

    expected = None if saved == "vibevoice_local" else TtsModelType.require_by_id("vibevoice_local")
    assert Tts.reconcile_project_model(project) is expected
    assert project.tts_model_type == "vibevoice_local"
    assert Tts.reconcile_project_model(project) is None
    assert Tts.get_backend_mode() is TtsRuntimeMode.LOCAL


# Both remote backends need matching/nonmatching selections; saved IDs otherwise
# pass through the same reconciliation path regardless of their original backend.
@pytest.mark.parametrize("backend, model, saved", [
    (TtsBackendKind.SGL_OMNI, TtsModelType.require_by_id("auk_sglomni"), "none"),
    (TtsBackendKind.SGL_OMNI, TtsModelType.require_by_id("auk_sglomni"), "future_catalog_id"),
    (TtsBackendKind.SGL_OMNI, TtsModelType.require_by_id("auk_sglomni"), "matching"),
    (TtsBackendKind.AUDIO_CPP, TtsModelType.require_by_id("higgs_v3_audiocpp"), "vibevoice_local"),
    (TtsBackendKind.AUDIO_CPP, TtsModelType.require_by_id("higgs_v3_audiocpp"), "matching"),
])
def test_reconcile_project_model_sole_remote_capability(monkeypatch, backend, model, saved):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    # An installed local capability must not override the process's remote mode.
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("vibevoice_local"),))
    snapshot = RemoteTtsSnapshot(backend, candidates=((model, "server/model"),))
    calls = []
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda force=False: calls.append(force) or snapshot)
    saved_id = model.id if saved == "matching" else saved
    project = Project(tts_model_type=saved_id)

    assert Tts.reconcile_project_model(project) is (None if saved_id == model.id else model)
    assert project.tts_model_type == model.id
    assert Tts.reconcile_project_model(project) is None
    assert calls == [False, False]
    assert Tts.get_backend_mode() is TtsRuntimeMode.REMOTE_CLIENT


@pytest.mark.parametrize("saved, expected", [
    ("none", None),
    ("future_catalog_id", TtsModelType.require_by_id("none")),
    ("vibevoice_local", TtsModelType.require_by_id("none")),
    ("higgs_v3_audiocpp", TtsModelType.require_by_id("none")),
    ("auk_sglomni", None),
    ("fish_s2_sglomni", None),
])
def test_reconcile_project_model_multiple_remote_capabilities(monkeypatch, saved, expected):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    snapshot = RemoteTtsSnapshot(TtsBackendKind.SGL_OMNI, candidates=(
        (TtsModelType.require_by_id("auk_sglomni"), "org/auk"),
        (TtsModelType.require_by_id("fish_s2_sglomni"), "org/fish-s2"),
    ))
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda force=False: snapshot)
    project = Project(tts_model_type=saved)

    assert Tts.reconcile_project_model(project) is expected
    expected_id = "none" if expected is not None and expected.id == "none" else saved
    assert project.tts_model_type == expected_id
    assert Tts.reconcile_project_model(project) is None
    assert project.tts_model_type == expected_id
    assert Tts.get_backend_mode() is TtsRuntimeMode.REMOTE_CLIENT


# Empty local availability clears selections; empty/failed remote discovery
# preserves them. Cover those policies without multiplying every saved ID.
@pytest.mark.parametrize("availability,saved", [
    ("local-empty", "none"),
    ("local-empty", "future_catalog_id"),
    ("local-empty", "vibevoice_local"),
    ("remote-empty", "vibevoice_local"),
    ("remote-offline", "auk_sglomni"),
    ("remote-offline", "none"),
])
def test_reconcile_project_model_zero_capabilities_clears_only_local_selection(monkeypatch, availability, saved):
    mode = TtsRuntimeMode.LOCAL if availability == "local-empty" else TtsRuntimeMode.REMOTE_CLIENT
    monkeypatch.setattr(Tts, "_backend_mode", mode)
    monkeypatch.setattr(Tts, "_available_local_models", ())
    issue = RemoteTtsIssue("unavailable", "offline") if availability == "remote-offline" else None
    snapshot = RemoteTtsSnapshot(TtsBackendKind.SGL_OMNI, issue=issue)
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda force=False: snapshot)
    project = Project(tts_model_type=saved)

    expected_id = "none" if mode is TtsRuntimeMode.LOCAL else saved
    expected_change = TtsModelType.require_by_id("none") if expected_id != saved else None
    assert Tts.reconcile_project_model(project) is expected_change
    assert project.tts_model_type == expected_id
    assert Tts.reconcile_project_model(project) is None
    assert Tts.get_backend_mode() is mode


@pytest.mark.parametrize("mode,available", [
    (TtsRuntimeMode.LOCAL, ()),
    (TtsRuntimeMode.REMOTE_CLIENT, ()),
    (TtsRuntimeMode.REMOTE_CLIENT, (TtsModelType.require_by_id("vibevoice_local"),)),
    (TtsRuntimeMode.LOCAL, (TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("higgs_v3_audiocpp"))),
])
def test_reconcile_project_model_only_queries_and_changes_project_selection(monkeypatch, mode, available):
    monkeypatch.setattr(Tts, "_backend_mode", mode)
    fields = ("_type", "_bound_project_type_id", "_selected_server_model_id", "_binding_issue",
              "_remote_issue", "_configured_runtime", "_model_params", "_backend_mode")
    before = {name: getattr(Tts, name) for name in fields}
    calls = []

    def available_models():
        calls.append(True)
        return list(available)

    def unexpected_side_effect(*args, **kwargs):
        pytest.fail("Reconciliation must not bind, prompt, save, or alter the active runtime")

    monkeypatch.setattr(Tts, "get_available_tts_models", staticmethod(available_models))
    monkeypatch.setattr(Tts, "bind_project", staticmethod(unexpected_side_effect))
    monkeypatch.setattr(Tts, "set_type", staticmethod(unexpected_side_effect))
    monkeypatch.setattr(Tts, "clear_tts_model", staticmethod(unexpected_side_effect))
    monkeypatch.setattr(Project, "save", unexpected_side_effect)
    monkeypatch.setattr("builtins.input", unexpected_side_effect)
    project = Project(tts_model_type="future_catalog_id")
    expected = (available[0] if len(available) == 1 else TtsModelType.require_by_id("none")
                if available or mode is TtsRuntimeMode.LOCAL else None)

    assert Tts.reconcile_project_model(project) is expected
    assert project.tts_model_type == (expected.id if expected is not None else "future_catalog_id")
    assert calls == [True]
    assert {name: getattr(Tts, name) for name in fields} == before


def test_startup_checks_duplicate_local_libraries_before_success(monkeypatch):
    from tts_audiobook_tool.start import Start
    monkeypatch.setattr(Tts, "init_local_model_type", staticmethod(lambda: (TtsModelType.require_by_id("vibevoice_local"), 2)))
    with pytest.raises(SystemExit):
        object.__new__(Start).init_tts_or_exit()


# --- Remembered audio.cpp selection ---

FISH = TtsModelType.require_by_id("fish_s2_audiocpp")
HIGGS = TtsModelType.require_by_id("higgs_v3_audiocpp")
CHATTERBOX_CPP = TtsModelType.require_by_id("chatterbox_audiocpp")


def _remote_snapshot(monkeypatch, backend, candidates):
    snapshot = RemoteTtsSnapshot(backend, candidates=candidates)
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda force=False: snapshot)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)


def test_set_tts_selection_remembers_only_complete_audio_cpp_pairs():
    # Switching away from audio.cpp (or to a type-only legacy selection) keeps
    # the last complete pair; a new complete pair replaces it.
    project = Project()
    project.set_tts_selection(FISH.id, "fish-q8")
    for model_id, server_id in (("vibevoice_local", ""), ("auk_sglomni", ""), (HIGGS.id, ""), ("none", "")):
        project.set_tts_selection(model_id, server_id)
        assert (project.last_audio_cpp_model_type, project.last_audio_cpp_model_id) == (FISH.id, "fish-q8")
    project.set_tts_selection(HIGGS.id, "higgs")
    assert (project.last_audio_cpp_model_type, project.last_audio_cpp_model_id) == (HIGGS.id, "higgs")


def test_remembered_audio_cpp_pair_round_trips_and_seeds_older_projects(tmp_path):
    # A project saved before the fields existed is seeded from its current pair
    # on load, so local-mode reconciliation cannot lose it.
    legacy = Project.model_validate({"tts_model_type": FISH.id, "audio_cpp_model_id": "fish-q8"})
    assert (legacy.last_audio_cpp_model_type, legacy.last_audio_cpp_model_id) == (FISH.id, "fish-q8")
    project = Project(dir_path=str(tmp_path))
    project.set_tts_selection(FISH.id, "fish-q8")
    project.set_tts_selection("vibevoice_local")
    assert project.save() == ""
    payload = json.loads((tmp_path / "project.json").read_text())
    assert (payload["last_audio_cpp_model_type"], payload["last_audio_cpp_model_id"]) == (FISH.id, "fish-q8")
    loaded = Project.model_validate(payload)
    assert (loaded.tts_model_type, loaded.last_audio_cpp_model_id) == ("vibevoice_local", "fish-q8")


def test_local_reconciliation_keeps_remembered_audio_cpp_pair(monkeypatch):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("vibevoice_local"),))
    project = Project(tts_model_type=FISH.id, audio_cpp_model_id="fish-q8")
    assert Tts.reconcile_project_model(project) is TtsModelType.require_by_id("vibevoice_local")
    assert (project.audio_cpp_model_id, project.last_audio_cpp_model_id) == ("", "fish-q8")


@pytest.mark.parametrize("current", ["vibevoice_local", "auk_sglomni", "none", "future_catalog_id"])
def test_reconnecting_to_audio_cpp_restores_remembered_entry(monkeypatch, current):
    # Restoring beats the sole/multiple availability rules, and picks the exact
    # entry even when another entry has the same catalog type.
    from tts_audiobook_tool.tts import TtsSelectionChangeReason
    _remote_snapshot(monkeypatch, TtsBackendKind.AUDIO_CPP,
                     ((FISH, "fish-f16"), (FISH, "fish-q8"), (HIGGS, "higgs")))
    project = Project(last_audio_cpp_model_type=FISH.id, last_audio_cpp_model_id="fish-q8")
    project.set_tts_selection(current)

    change = Tts.reconcile_project_selection(project)

    assert change is not None
    assert (change.model, change.reason) == (FISH, TtsSelectionChangeReason.RESTORED_AUDIO_CPP)
    assert (project.tts_model_type, project.audio_cpp_model_id) == (FISH.id, "fish-q8")
    assert Tts.reconcile_project_selection(project) is None


@pytest.mark.parametrize("candidates, expected", [
    # Remembered ID now serves another type: never adopted.
    (((HIGGS, "fish-q8"), (CHATTERBOX_CPP, "cb")), ("none", "")),
    (((HIGGS, "higgs"), (CHATTERBOX_CPP, "cb")), ("none", "")),
    (((CHATTERBOX_CPP, "cb"),), (CHATTERBOX_CPP.id, "cb")),
])
def test_unavailable_remembered_entry_falls_back_to_ordinary_rules(monkeypatch, candidates, expected):
    from tts_audiobook_tool.tts import TtsSelectionChangeReason
    _remote_snapshot(monkeypatch, TtsBackendKind.AUDIO_CPP, candidates)
    project = Project(tts_model_type="vibevoice_local",
                      last_audio_cpp_model_type=FISH.id, last_audio_cpp_model_id="fish-q8")

    change = Tts.reconcile_project_selection(project)

    assert change is not None and change.reason is TtsSelectionChangeReason.LAST_AUDIO_CPP_UNAVAILABLE
    assert (project.tts_model_type, project.audio_cpp_model_id) == expected


def test_legacy_type_only_audio_cpp_selection_is_not_replaced_by_remembered_entry(monkeypatch):
    _remote_snapshot(monkeypatch, TtsBackendKind.AUDIO_CPP, ((FISH, "fish-q8"), (HIGGS, "higgs")))
    project = Project(tts_model_type=HIGGS.id,
                      last_audio_cpp_model_type=FISH.id, last_audio_cpp_model_id="fish-q8")

    Tts.reconcile_project_selection(project)

    assert (project.tts_model_type, project.audio_cpp_model_id) == (HIGGS.id, "higgs")


@pytest.mark.parametrize("candidates, expected", [
    # Several models: unselected, even though another Fish entry is offered.
    (((FISH, "fish-f16"), (HIGGS, "higgs")), ("none", "")),
    # Sole entry: auto-selected, whatever its type.
    (((HIGGS, "higgs"),), (HIGGS.id, "higgs")),
    # Sole type with several entries: type kept, entry requires selection.
    (((HIGGS, "a"), (HIGGS, "b")), (HIGGS.id, "")),
])
def test_saved_entry_missing_from_reachable_server_is_replaced(monkeypatch, candidates, expected):
    # E.g. after pointing the app at a different audio.cpp server. The old pair
    # stays remembered unless a new complete pair replaces it.
    from tts_audiobook_tool.tts import TtsSelectionChangeReason
    _remote_snapshot(monkeypatch, TtsBackendKind.AUDIO_CPP, candidates)
    project = Project(tts_model_type=FISH.id, audio_cpp_model_id="fish-drama3")

    change = Tts.reconcile_project_selection(project)

    assert change is not None and change.reason is TtsSelectionChangeReason.SAVED_ENTRY_UNAVAILABLE
    assert (project.tts_model_type, project.audio_cpp_model_id) == expected
    assert project.last_audio_cpp_model_id == (expected[1] or "fish-drama3")
    assert Tts.reconcile_project_selection(project) is None


def test_saved_entry_still_offered_by_server_is_kept(monkeypatch):
    _remote_snapshot(monkeypatch, TtsBackendKind.AUDIO_CPP, ((FISH, "fish-drama3"), (HIGGS, "higgs")))
    project = Project(tts_model_type=FISH.id, audio_cpp_model_id="fish-drama3")
    assert Tts.reconcile_project_selection(project) is None


def test_sgl_omni_server_replaces_saved_audio_cpp_entry_and_reconnect_restores_it(monkeypatch):
    # Pointing at SGL-Omni auto-selects its single model; returning to the
    # audio.cpp server restores the remembered entry.
    from tts_audiobook_tool.tts import TtsSelectionChangeReason
    sgl = TtsModelType.require_by_id("auk_sglomni")
    project = Project(tts_model_type=FISH.id, audio_cpp_model_id="fish-drama3")
    _remote_snapshot(monkeypatch, TtsBackendKind.SGL_OMNI, ((sgl, "org/auk"),))
    assert Tts.reconcile_project_model(project) is sgl
    _remote_snapshot(monkeypatch, TtsBackendKind.AUDIO_CPP, ((FISH, "fish-drama3"), (HIGGS, "higgs")))
    change = Tts.reconcile_project_selection(project)
    assert change is not None and change.reason is TtsSelectionChangeReason.RESTORED_AUDIO_CPP
    assert (project.tts_model_type, project.audio_cpp_model_id) == (FISH.id, "fish-drama3")


def test_sgl_omni_server_does_not_restore_audio_cpp_entry(monkeypatch):
    sgl = TtsModelType.require_by_id("auk_sglomni")
    _remote_snapshot(monkeypatch, TtsBackendKind.SGL_OMNI, ((sgl, "org/auk"),))
    project = Project(tts_model_type="vibevoice_local",
                      last_audio_cpp_model_type=FISH.id, last_audio_cpp_model_id="fish-q8")

    change = Tts.reconcile_project_selection(project)

    assert change is not None and change.model is sgl
    assert project.last_audio_cpp_model_id == "fish-q8"
