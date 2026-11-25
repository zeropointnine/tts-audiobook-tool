"""Generation-facing readiness keeps process compatibility separate from saved metadata."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool import readiness
from tts_audiobook_tool.app_support.remote_tts_discovery import (
    RemoteTtsDiscovery, RemoteTtsIssue, RemoteTtsSnapshot,
)
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode


@pytest.mark.parametrize("remote_issue", [
    None,
    RemoteTtsIssue("unavailable", "Could not connect to the remote TTS server"),
    RemoteTtsIssue("http_error", "Remote TTS server returned HTTP 503"),
])
def test_remote_saved_local_selection_reports_server_and_mode_not_local_settings(monkeypatch, remote_issue):
    # Local LoRA/sample/checkpoint validators cannot repair a remote process;
    # preserve the saved Breeze/remote history while reporting both real causes.
    project = Project(tts_model_type="breeze_tts_2_local",
                      last_audio_cpp_model_type="fireredtts3_audiocpp",
                      last_audio_cpp_model_id="fireredtts3-base")
    before = project.model_dump(exclude={"reason_pauses"})
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "get_model_support",
                        lambda *_: pytest.fail("Must not validate incompatible local parameters/voices/LoRAs"))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot",
                        lambda: RemoteTtsSnapshot(issue=remote_issue))
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh",
                        lambda **_: pytest.fail("Menu blocker formatting must not discover models"))

    issues = readiness.get_tts_blockers(project)

    assert len(issues) == (2 if remote_issue else 1)
    if remote_issue:
        assert issues[0].verbose == remote_issue.message
    mismatch = issues[-1].verbose
    assert "remote mode" in mismatch and "saved local model" in mismatch
    assert "relaunch" in mismatch.lower()
    assert project.model_dump(exclude={"reason_pauses"}) == before


@pytest.mark.parametrize("run_start", [readiness.get_run_blocker_text, readiness.get_tts_preview_blocker_text])
def test_remote_saved_local_selection_run_start_forces_discovery_without_switching(monkeypatch, run_start):
    # Even though binding rejects local selection before discovery, run starts
    # must replace a healthy cached observation with the current outage.
    project = Project(tts_model_type="breeze_tts_2_local")
    state = cast(State, SimpleNamespace(project=project))
    before = project.model_dump(exclude={"reason_pauses"})
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "http://remote.example:8001")
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    RemoteTtsDiscovery._store(RemoteTtsSnapshot())
    calls = []

    def probe(url, timeout=None):
        calls.append((url, timeout))
        return RemoteTtsSnapshot(issue=RemoteTtsIssue("timeout", "Timed out checking the remote TTS server"))

    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", probe)
    monkeypatch.setattr(Tts, "get_model_support",
                        lambda *_: pytest.fail("Local support validation must stay bypassed"))

    text = run_start(state, verbose=True)

    assert calls == [(RemoteTtsDiscovery.get_base_url(), RemoteTtsDiscovery._timeout)]
    assert "Timed out" in text and "remote mode" in text
    assert "relaunch" in text.lower()
    assert Tts.get_active_type().id == "none"
    assert project.model_dump(exclude={"reason_pauses"}) == before


@pytest.mark.parametrize("mode, model_id", [
    (TtsRuntimeMode.LOCAL, "breeze_tts_2_local"),
    (TtsRuntimeMode.REMOTE_CLIENT, "omnivoice_audiocpp"),
])
def test_compatible_selection_keeps_model_validation_and_matching_binding_issue(monkeypatch, mode, model_id):
    # Compatibility filtering must not silence the actual model's prerequisites
    # or its selection-scoped binding failure, nor consult stale remote state locally.
    project = Project(tts_model_type=model_id)
    model_issue = ReadinessIssue("voice", "A compatible model prerequisite")
    binding_issue = ReadinessIssue("TTS model", "A matching binding failure")
    monkeypatch.setattr(Tts, "_backend_mode", mode)
    monkeypatch.setattr(Tts, "_bound_project_type_id", model_id)
    monkeypatch.setattr(Tts, "_bound_project_audio_cpp_model_id", "")
    monkeypatch.setattr(Tts, "_binding_issue", binding_issue)
    calls = []
    support = SimpleNamespace(get_blocking_issues=lambda p, i: calls.append((p, i)) or [model_issue])
    monkeypatch.setattr(Tts, "get_model_support", lambda p: support)
    def cached_snapshot():
        assert mode is TtsRuntimeMode.REMOTE_CLIENT, "Local readiness must not inspect remote discovery"
        return RemoteTtsSnapshot()

    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", cached_snapshot)

    assert readiness.get_tts_blockers(project) == [model_issue, binding_issue]
    assert calls == [(project, None)]


@pytest.mark.parametrize("saved_id", ["none", "future_catalog_id"])
def test_remote_unselected_run_start_reports_fresh_outage_and_own_selection(monkeypatch, saved_id):
    # None/unknown selections also short-circuit binding before discovery; they
    # must report this endpoint's fresh failure, not another project's runtime error.
    project = Project(tts_model_type=saved_id)
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "_remote_issue", "Other project's server is offline")
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "http://remote.example:8001")
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    RemoteTtsDiscovery._store(RemoteTtsSnapshot())
    calls = []

    def probe(url, timeout=None):
        calls.append(timeout)
        return RemoteTtsSnapshot(issue=RemoteTtsIssue("unavailable", "This endpoint is unreachable"))

    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", probe)

    text = readiness.get_tts_preview_blocker_text(state, verbose=True)

    assert calls == [RemoteTtsDiscovery._timeout]
    assert "This endpoint is unreachable" in text
    assert "Other project's" not in text
    assert "saved local model" not in text
    assert ("Select a TTS model" if saved_id == "none" else "Unknown TTS model") in text
    assert project.tts_model_type == saved_id
    assert Tts.get_active_type().id == "none"


def test_remote_cached_discovery_failure_is_not_duplicated_by_model_and_binding(monkeypatch):
    # A compatible remote model can already report the same discovery failure;
    # adding process-level diagnostics must not repeat it in blocker paragraphs.
    project = Project(tts_model_type="omnivoice_audiocpp")
    issue = ReadinessIssue("Remote TTS server", "This endpoint is unreachable")
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "_bound_project_type_id", project.tts_model_type)
    monkeypatch.setattr(Tts, "_bound_project_audio_cpp_model_id", "")
    monkeypatch.setattr(Tts, "_binding_issue", issue)
    monkeypatch.setattr(Tts, "get_model_support", lambda p: SimpleNamespace(
        get_blocking_issues=lambda p, i: [issue]))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        issue=RemoteTtsIssue("unavailable", issue.verbose)))

    assert readiness.get_tts_blockers(project) == [issue]
