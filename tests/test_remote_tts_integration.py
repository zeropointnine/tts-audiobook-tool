"""Project-owned remote selection and adapter integration without a live server."""
from __future__ import annotations

import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import (
    RemoteTtsDiscovery, RemoteTtsIssue, RemoteTtsSnapshot,
)
from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

URL = "http://remote.example:8111"
MODEL = TtsModelType.require_by_id("chatterbox_audiocpp")


@pytest.fixture
def remote_mode(monkeypatch):
    fields = ("_backend_mode", "_type", "_configured_runtime", "_config_fingerprint", "_catalog_initialized",
              "_configured_definitions", "_audio_cpp_definitions", "_selected_server_model_id", "_remote_issue",
              "_available_local_models", "_binding_issue", "_bound_project_type_id", "_model_params")
    saved = {name: getattr(Tts, name) for name in fields}
    registry = (dict(REGISTRY.bindings), dict(REGISTRY.legacy), dict(REGISTRY.members))
    catalog = (dict(TtsModelType._catalog), dict(TtsModelType._specs))
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.REMOTE_CLIENT))
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", URL)
    monkeypatch.setattr(AudioCppUtil, "_base_url", URL)
    Tts.init_local_model_type()
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(Tts, name, value)
        REGISTRY.bindings, REGISTRY.legacy, REGISTRY.members = registry
        TtsModelType._catalog, TtsModelType._specs = catalog


def snapshot(*server_ids: str) -> RemoteTtsSnapshot:
    return RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        candidates=tuple((MODEL, model_id) for model_id in server_ids),
    )


def selected_project() -> Project:
    return Project(tts_model_type=MODEL.id)


def test_selected_type_resolves_unique_exact_server_id_and_reuses_adapter(remote_mode, monkeypatch):
    project = selected_project()
    RemoteTtsDiscovery._store(snapshot("operator-defined-id"), URL)
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", lambda *args: pytest.fail("unexpected network probe"))
    assert Tts.bind_project(project) is None
    assert Tts.get_active_type() is MODEL
    assert Tts._selected_server_model_id == "operator-defined-id"
    assert isinstance(Tts.get_model_support(project), AudioCppModelSupport)
    adapter = Tts.get_instance()
    assert isinstance(adapter, AudioCppBackendAdapter)
    assert adapter.server_model_id == "operator-defined-id"
    Tts.bind_project(project)
    assert Tts.get_instance() is adapter


def test_missing_type_preserves_project_and_lists_available_entries(remote_mode, monkeypatch):
    project = selected_project()
    other = TtsModelType.require_by_id("higgs_v3_audiocpp")
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(
        TtsBackendKind.AUDIO_CPP, candidates=((other, "operator-higgs"),)), URL)
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", lambda *args: pytest.fail("unexpected network probe"))
    issue = Tts.bind_project(project)
    assert issue is not None and "No server model matches" in issue.verbose
    assert "operator-higgs" in issue.verbose
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    assert project.tts_model_type == MODEL.id
    assert isinstance(Tts.get_model_support(project), AudioCppModelSupport)


def test_run_start_forces_probe_even_with_fresh_observation(remote_mode, monkeypatch):
    from tts_audiobook_tool import readiness
    RemoteTtsDiscovery._store(snapshot("operator-defined-id"), URL)
    probes = []
    def fake_probe(url, timeout=None):
        probes.append(timeout)
        return snapshot("operator-defined-id")
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", fake_probe)
    readiness.refresh_remote_model_state(selected_project())
    assert probes == [RemoteTtsDiscovery._timeout]


def test_available_types_are_distinct_and_do_not_auto_bind(remote_mode, monkeypatch):
    other = TtsModelType.require_by_id("higgs_v3_audiocpp")
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=(
        (MODEL, "one"), (other, "higgs"), (MODEL, "two"))), URL)
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", lambda *args: pytest.fail("unexpected network probe"))
    assert Tts.get_available_tts_models() == [MODEL, other]
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    project = Project()
    assert Tts.bind_project(project) is not None
    assert project.tts_model_type == "none"


def test_same_type_duplicate_entries_are_ambiguous_without_exact_choice(remote_mode):
    project = selected_project()
    RemoteTtsDiscovery._store(snapshot("operator-one", "operator-two"), URL)
    assert Tts.get_available_tts_models() == [MODEL]
    issue = Tts.bind_project(project)
    assert issue is not None and "Multiple server entries" in issue.verbose
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    assert Tts._selected_server_model_id == ""
    assert project.tts_model_type == MODEL.id


def test_switch_projects_selects_matching_type_not_other_available_entry(remote_mode):
    other = TtsModelType.require_by_id("higgs_v3_audiocpp")
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=(
        (MODEL, "chatterbox"), (other, "higgs"))), URL)
    first, second = selected_project(), Project(tts_model_type=other.id)
    assert Tts.bind_project(first) is None
    assert Tts._selected_server_model_id == "chatterbox"
    assert Tts.bind_project(second) is None
    assert Tts.get_active_type() is other
    assert Tts._selected_server_model_id == "higgs"
    assert first.tts_model_type == MODEL.id and second.tts_model_type == other.id


def test_unsupported_models_issue_preserves_saved_project(remote_mode):
    project = selected_project()
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        issue=RemoteTtsIssue("no_supported_models", "No configured server models match supported catalog variants"),
    ), URL)
    Tts.bind_project(project)
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    assert "No configured server models match supported catalog variants" in Tts._remote_issue
    assert not Tts._selected_server_model_id
    assert project.tts_model_type == MODEL.id


def test_moss_audio_cpp_architectures_bind_their_exact_opaque_server_entries(remote_mode, monkeypatch):
    from tts_audiobook_tool.tts_models.audio_cpp_detection import detect_audio_cpp_models

    models = [
        {"id": "operator-a", "family": "moss_tts_v15", "task": "tts", "mode": "offline"},
        {"id": "operator-b", "family": "moss_tts_local", "task": "clon", "mode": "offline"},
    ]
    candidates = tuple(detect_audio_cpp_models(models))
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(
        TtsBackendKind.AUDIO_CPP, models=tuple(models), candidates=candidates), URL)
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", lambda *args: pytest.fail("unexpected network probe"))
    assert [model.id for model in Tts.get_available_tts_models()] == [
        "moss_delay_audiocpp", "moss_local_audiocpp",
    ]
    for model, server_id in candidates:
        project = Project(tts_model_type=model.id)
        assert Tts.bind_project(project) is None
        assert Tts.get_active_type() is model
        adapter = Tts.get_instance()
        assert isinstance(adapter, AudioCppBackendAdapter)
        assert adapter.server_model_id == server_id
        assert adapter.definition.spec.id == model.id
        Tts.bind_project(project)
        assert Tts.get_instance() is adapter
        assert project.tts_model_type == model.id


@pytest.mark.parametrize("model_id,family", [
    ("moss_delay_audiocpp", "moss_tts_v15"), ("moss_local_audiocpp", "moss_tts_local"),
])
def test_moss_tts_and_clone_server_entries_of_same_family_are_ambiguous(remote_mode, model_id, family):
    from tts_audiobook_tool.tts_models.audio_cpp_detection import detect_audio_cpp_models

    models = [{"id": f"operator-{task}", "family": family, "task": task, "mode": "offline"}
              for task in ("tts", "clon")]
    candidates = tuple(detect_audio_cpp_models(models))
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(
        TtsBackendKind.AUDIO_CPP, models=tuple(models), candidates=candidates), URL)
    model = TtsModelType.require_by_id(model_id)
    assert Tts.get_available_tts_models() == [model]
    project = Project(tts_model_type=model_id)
    issue = Tts.bind_project(project)
    assert issue is not None and "Multiple server entries" in issue.verbose
    assert project.tts_model_type == model_id
    assert Tts.get_active_type().id == "none" and Tts._selected_server_model_id == ""
