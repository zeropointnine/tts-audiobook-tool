"""Runtime status never mistakes retained configuration for an active model."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsIssue, RemoteTtsSnapshot
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
from tts_audiobook_tool.menus.menu_status import MenuStatus
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


@pytest.fixture
def runtime(monkeypatch):
    observation = SimpleNamespace(snapshot=RemoteTtsSnapshot(issue=RemoteTtsIssue("unavailable", "Cannot connect")))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda force=False: observation.snapshot)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: observation.snapshot)
    monkeypatch.setattr(ModelWorker, "get_model_state_blocking", lambda: (None, ""))
    monkeypatch.setattr("tts_audiobook_tool.menus.menu_status._make_memory_text", lambda: "")
    return observation


def make_state(project):
    state = State.for_worker(Prefs())
    state.project = project
    return state


def draw(state, capsys):
    MenuStatus.print_block(state)
    return dict(line.split(":", 1) for line in strip_ansi_codes(capsys.readouterr().out).splitlines() if ":" in line)


@pytest.mark.parametrize("remembered", [False, True])
def test_offline_remote_with_saved_breeze_omits_model_history(runtime, monkeypatch, capsys, remembered):
    # Reproduce the confusing Breeze/LoRA menu, with real reconciliation and binding.
    remote = TtsModelType.require_by_id("fireredtts3_audiocpp")
    project = Project(
        tts_model_type="breeze_tts_2_local",
        last_audio_cpp_model_type=remote.id if remembered else "",
        last_audio_cpp_model_id="firered-base" if remembered else "",
        voice_references=[Voice(loras={"breeze_tts_2_local": VoiceLora(ModelTarget.local("/example/checkpoint"))})],
    )
    before = deepcopy(project.model_dump(exclude={"reason_pauses"}))
    state = make_state(project)
    state.pending_project_load_checks = True
    monkeypatch.setattr("tts_audiobook_tool.sound.audio_meta_util.AudioMetaUtil.get_audio_duration",
                        lambda _: pytest.fail("No local-model duration advice in a remote process"))

    rows = draw(state, capsys)
    assert rows["TTS model"].strip() == "None active (remote mode; server unreachable)"
    assert rows["Voices"].strip() == "1 saved voice"
    assert all("Breeze" not in value and "LoRA" not in value for value in rows.values())
    assert "Last-selected remote model" not in rows
    assert strip_ansi_codes(MenuStatus.make_tts_text(state, include_server_id=True)) == rows["TTS model"].strip()
    assert Tts.get_active_type().id == "none"

    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)
    notice = strip_ansi_codes(capsys.readouterr().out)
    assert notice == "🔔 FYI\nThis project was last used with TTS model Breeze TTS 2 (local).\n\n"
    assert "relaunch" not in notice and "recommended duration" not in notice
    draw(state, capsys)
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)
    assert capsys.readouterr().out == ""
    assert project.model_dump(exclude={"reason_pauses"}) == before


def test_reconnect_then_local_relaunch_preserves_remote_pair(runtime, monkeypatch, capsys):
    # Restoration still works without a history row; local relaunch never probes remote.
    remote = TtsModelType.require_by_id("fireredtts3_audiocpp")
    project = Project(tts_model_type="breeze_tts_2_local", last_audio_cpp_model_type=remote.id,
                      last_audio_cpp_model_id="firered-base")
    state = make_state(project)
    assert "Last-selected remote model" not in draw(state, capsys)
    runtime.snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((remote, "firered-base"),))
    rows = draw(state, capsys)
    assert rows["TTS model"].strip() == f"{remote.value.ui['proper_name']} (audio.cpp)"
    assert "Last-selected remote model" not in rows
    assert (project.tts_model_type, project.audio_cpp_model_id) == (remote.id, "firered-base")
    MenuStatus.show_pending_project_hints(state)
    notice = strip_ansi_codes(capsys.readouterr().out)
    assert notice == "🔔 FYI\nThis project was last used with TTS model Breeze TTS 2 (local).\n\n"

    local = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_available_local_models", (local,))
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda **_: pytest.fail("Local mode must not discover remote"))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: pytest.fail("Local status must not use remote cache"))
    rows = draw(state, capsys)
    assert "Breeze" in rows["TTS model"] and "(local)" in rows["TTS model"]
    assert "Last-selected remote model" not in rows
    assert project.audio_cpp_model_id == ""
    assert (project.last_audio_cpp_model_type, project.last_audio_cpp_model_id) == (remote.id, "firered-base")
    assert Tts.get_active_type() == local
    assert state.pending_binding_notice is None


@pytest.mark.parametrize("saved, remembered_type, remembered_id", [
    ("auk_sglomni", "fireredtts3_audiocpp", "older-entry"),
    ("future-model", "fireredtts3_audiocpp", "remembered-entry"),
    ("breeze_tts_2_local", "breeze_tts_2_local", "invalid-local"),
    ("none", "unknown-remote", "unknown-entry"),
    ("none", "fireredtts3_audiocpp", ""),
])
def test_saved_model_history_never_adds_a_status_row(runtime, capsys, saved, remembered_type, remembered_id):
    # Neither saved remote selections nor remembered/malformed pairs add history to status.
    state = make_state(Project(tts_model_type=saved, last_audio_cpp_model_type=remembered_type,
                               last_audio_cpp_model_id=remembered_id))
    rows = draw(state, capsys)
    assert "None active" in rows["TTS model"]
    assert "server unreachable" in rows["TTS model"]
    assert "Last-selected remote model" not in rows
    assert not any("last good" in value or "successfully used" in value for value in rows.values())


def test_local_mismatch_notice_rearms_only_on_project_load(runtime, capsys):
    # Changing endpoints must not repeat history; assigning/loading a project re-arms it.
    state = make_state(Project(tts_model_type="breeze_tts_2_local"))
    for should_show in (True, False):
        MenuStatus.prepare_tts(state)
        MenuStatus.show_pending_binding_hint(state)
        assert bool(capsys.readouterr().out) == should_show
    RemoteTtsDiscovery.set_base_url("http://another.test")
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_binding_hint(state)
    assert capsys.readouterr().out == ""
    assert state.has_shown_tts_model_hint
    state.project = Project(tts_model_type="breeze_tts_2_local")
    assert not state.has_shown_tts_model_hint
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_binding_hint(state)
    assert strip_ansi_codes(capsys.readouterr().out) == "🔔 FYI\nThis project was last used with TTS model Breeze TTS 2 (local).\n\n"


def test_offline_history_fyi_does_not_repeat_when_server_url_is_fixed(runtime, capsys):
    # Regression: the offline binding notice and later reconciliation share one per-load guard.
    state = make_state(Project(tts_model_type="breeze_tts_2_local"))
    draw(state, capsys)
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)
    assert "Breeze TTS 2 (local)" in strip_ansi_codes(capsys.readouterr().out)
    assert state.has_shown_tts_model_hint

    remote = TtsModelType.require_by_id("fireredtts3_audiocpp")
    RemoteTtsDiscovery.set_base_url("http://working.test")
    runtime.snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((remote, "served"),))
    rows = draw(state, capsys)
    assert rows["TTS model"].strip() == f"{remote.value.ui['proper_name']} (audio.cpp)"
    assert state.project.tts_model_type == remote.id and Tts.get_active_type() == remote
    assert state.pending_tts_model_change is None
    MenuStatus.show_pending_project_hints(state)
    assert capsys.readouterr().out == ""

    # Even a different automatically selected model and re-armed compatibility checks stay silent.
    other = TtsModelType.require_by_id("chatterbox_audiocpp")
    state.pending_project_load_checks = True
    runtime.snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((other, "other"),))
    draw(state, capsys)
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)
    assert capsys.readouterr().out == ""
    assert state.project.tts_model_type == other.id


def test_ambiguity_rearms_independently_after_model_history_was_shown(runtime, capsys):
    # Consuming model-change history must not consume or disable the ambiguity exception.
    remote = TtsModelType.require_by_id("chatterbox_audiocpp")
    state = make_state(Project(tts_model_type="breeze_tts_2_local"))
    runtime.snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((remote, "one"),))
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_project_hints(state)
    assert "Breeze TTS 2 (local)" in strip_ansi_codes(capsys.readouterr().out)
    assert state.has_shown_tts_model_hint

    for iteration in range(2):
        state.project.set_tts_selection(remote.id)  # A legacy/ambiguous selection, not a project reload.
        runtime.snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP,
                                             candidates=((remote, "one"), (remote, "two")))
        MenuStatus.prepare_tts(state)
        MenuStatus.show_pending_project_hints(state)
        notice = strip_ansi_codes(capsys.readouterr().out)
        assert "Several server entries match" in notice and "one, two" in notice
        assert "last used with" not in notice
        MenuStatus.prepare_tts(state)
        MenuStatus.show_pending_project_hints(state)
        assert capsys.readouterr().out == ""
        if iteration == 0:
            runtime.snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((remote, "one"),))
            MenuStatus.prepare_tts(state)
            MenuStatus.show_pending_project_hints(state)
            assert capsys.readouterr().out == ""
    assert state.has_shown_tts_model_hint


def test_local_without_model_has_no_remote_history_or_loaded_qualifiers(runtime, monkeypatch, capsys):
    # A previous worker's resident model and force-CPU setting are not an active binding.
    from tts_audiobook_tool.model_worker_protocol import ModelStateSnapshot

    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_available_local_models", ())
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: pytest.fail("No remote cache locally"))
    monkeypatch.setattr(ModelWorker, "get_model_state_blocking", lambda: (
        ModelStateSnapshot(tts_loaded=True, tts_type_id="breeze_tts_2_local", tts_device="cuda"), ""))
    state = make_state(Project(tts_model_type="breeze_tts_2_local",
                               last_audio_cpp_model_type="fireredtts3_audiocpp", last_audio_cpp_model_id="saved"))
    state.prefs.tts_force_cpu = True
    rows = draw(state, capsys)
    assert rows["TTS model"].strip() == "None active (local mode; no supported model installed)"
    assert "Last-selected remote model" not in rows
    assert "loaded" not in rows["TTS model"] and "cpu" not in rows["TTS model"]
    assert state.project.last_audio_cpp_model_id == "saved"


def test_stale_binding_and_exact_entry_cannot_masquerade_as_active(runtime, monkeypatch):
    # Cached residency/type alone cannot bless another entry or a failed binding.
    remote = TtsModelType.require_by_id("fireredtts3_audiocpp")
    project = Project(tts_model_type=remote.id, audio_cpp_model_id="wanted")
    state = make_state(project)
    runtime.snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((remote, "wanted"),))
    monkeypatch.setattr(Tts, "_type", remote)
    monkeypatch.setattr(Tts, "_binding_issue", None)
    monkeypatch.setattr(Tts, "_selected_server_model_id", "other")
    assert "None active" in strip_ansi_codes(MenuStatus.make_tts_text(state))
    monkeypatch.setattr(Tts, "_selected_server_model_id", "wanted")
    monkeypatch.setattr(Tts, "_binding_issue", ReadinessIssue("TTS model", "Cannot bind"))
    assert "None active" in strip_ansi_codes(MenuStatus.make_tts_text(state))
