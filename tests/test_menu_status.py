from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot, RemoteTtsIssue
from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.main_menu import make_voice_label
from tts_audiobook_tool.menus.menu_status import _make_stt_text
from tts_audiobook_tool.menus.menu_status import MenuStatus
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from project_settings_test_support import set_setting
from tts_audiobook_tool.state import PendingTtsModelChange, State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool import text_util
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.util import COL_DIM, COL_ERROR, COL_MEDIUM


def make_state(model: TtsModelType = TtsModelType.require_by_id("none")) -> State:
    state = object.__new__(State)
    state._prefs = Prefs()
    state._project = Project(dir_path="", tts_model_type=model.id)
    state.pending_tts_model_change = None
    state.pending_project_load_checks = False
    state.has_shown_main_menu = False
    return state


_BACKEND_LABELS = {
    TtsBackendKind.LOCAL: "local",
    TtsBackendKind.SGL_OMNI: "SGL-Omni",
    TtsBackendKind.AUDIO_CPP: "audio.cpp",
}


def _display_name(model: TtsModelType) -> str:
    """Mirror `menu_status._make_model_name()`'s qualified name."""
    label = _BACKEND_LABELS.get(model.value.backend_kind)
    return f"{model.value.ui['proper_name']} ({label})" if label else model.value.ui["proper_name"]


@pytest.fixture(autouse=True)
def clean_remote_state(monkeypatch):
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "")
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_remote_issue", "")
    monkeypatch.setattr(Tts, "_binding_issue", None)
    monkeypatch.setattr(Tts, "_available_local_models", ())
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda force=False: RemoteTtsDiscovery.get_snapshot())
    monkeypatch.setattr(Tts, "_selected_server_model_id", "")
    monkeypatch.setattr(Tts, "bind_project", lambda project, **kwargs: None)
    from tts_audiobook_tool.model_worker import ModelWorker
    from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
    monkeypatch.setattr(ModelWorker, "get_model_state_blocking", lambda: (None, ""))
    monkeypatch.setattr(Tts, "_audio_cpp_definitions", load_audio_cpp_definitions().models)


def test_menu_status_print_block_supports_voice_display_info(capsys, monkeypatch):
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("omnivoice_local"),))
    state = make_state(TtsModelType.require_by_id("omnivoice_local"))
    set_setting(state.project, "omnivoice_voice_file_name", [
        "zzz belle 24a 19s_omnivoice.flac",
        "zzz belle 24b 20s_omnivoice.flac",
    ])

    MenuStatus.print_block(state)

    output = capsys.readouterr().out
    assert f"{COL_DIM}Voice clone: {COL_MEDIUM}zzz belle 24a 19s, +1 more" in output


def test_audio_cpp_omnivoice_voice_line_is_optional(monkeypatch):
    """The audio.cpp OmniVoice entry can run without a voice, like its local twin."""
    from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
    from tts_audiobook_tool.tts import Tts

    model_type = TtsModelType.require_by_id("omnivoice_audiocpp")
    monkeypatch.setattr(Tts, "_audio_cpp_definitions", load_audio_cpp_definitions().models)
    monkeypatch.setattr(Tts, "_available_local_models", (model_type,))
    state = make_state(model_type)

    display = Tts.get_model_support_for_type(model_type).get_voice_display_info(state.project)
    assert display.value.endswith("none")
    assert make_voice_label(state) == "Voice clone"


def test_menus_omit_absent_voice_display_info(monkeypatch, capsys):
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("omnivoice_local"),))
    state = make_state(TtsModelType.require_by_id("omnivoice_local"))
    tts_class = Tts.get_class_for_type(TtsModelType.require_by_id("omnivoice_local"))
    monkeypatch.setattr(
        tts_class, "get_voice_display_info",
        classmethod(lambda cls, project, instance=None: None),
    )

    MenuStatus.print_block(state)

    assert "Voice clone:" not in capsys.readouterr().out
    assert make_voice_label(state) == "Voice clone"


def test_status_block_local_mode_none_shows_tts_model_line(capsys):
    MenuStatus.print_block(make_state())

    lines = [text_util.strip_ansi_codes(line) for line in capsys.readouterr().out.splitlines()]
    tts_lines = [line for line in lines if "TTS model:" in line]
    assert len(tts_lines) == 1
    assert "SGL-Omni" not in tts_lines[0]


@pytest.mark.parametrize("issue_code", [
    "unavailable", "timeout", "http_error", "invalid_response",
    "invalid_health", "unrecognized_server", "invalid_models", "no_models",
])
def test_status_block_remote_failure_preserves_project_model(capsys, monkeypatch, issue_code):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        issue=RemoteTtsIssue(issue_code, "Remote server unavailable")))
    state = make_state(TtsModelType.require_by_id("chatterbox_audiocpp"))
    state._prefs = Prefs(remote_tts_url="http://example.test")

    MenuStatus.print_block(state)

    tts_lines = [line for line in capsys.readouterr().out.splitlines() if "TTS model:" in line]
    assert len(tts_lines) == 1
    assert "Chatterbox" in tts_lines[0]
    assert f"{COL_DIM}(audio.cpp)" in tts_lines[0]
    if issue_code in {"unavailable", "timeout"}:
        assert f"{COL_ERROR}(server unreachable)" in tts_lines[0]
    else:
        assert f"{COL_ERROR}(Remote server unavailable)" in tts_lines[0]
        assert "server unreachable" not in tts_lines[0]
    assert COL_ERROR in tts_lines[0]
    assert state.project.tts_model_type == "chatterbox_audiocpp"


@pytest.mark.parametrize("issue_code", ["unavailable", "timeout"])
def test_status_block_unreachable_unknown_server_omits_server_qualifier(capsys, monkeypatch, issue_code):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        issue=RemoteTtsIssue(issue_code, "Remote server unavailable")))

    MenuStatus.print_block(make_state())

    lines = [text_util.strip_ansi_codes(line) for line in capsys.readouterr().out.splitlines()]
    tts_lines = [line for line in lines if "TTS model:" in line]
    assert len(tts_lines) == 1
    assert tts_lines[0].split(":", 1)[1].strip() == "None (server unreachable)"


def test_status_block_remote_selected_model_uses_gray_backend(capsys, monkeypatch):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("chatterbox_audiocpp"))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "_selected_server_model_id", "my-served-v3")
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        candidates=((TtsModelType.require_by_id("chatterbox_audiocpp"), "my-served-v3"),)))

    MenuStatus.print_block(make_state(TtsModelType.require_by_id("chatterbox_audiocpp")))

    lines = [text_util.strip_ansi_codes(line) for line in capsys.readouterr().out.splitlines()]
    tts_lines = [line for line in lines if "TTS model:" in line]
    assert len(tts_lines) == 1
    assert "Chatterbox (audio.cpp)" in tts_lines[0]
    assert "server model id:" not in tts_lines[0]


@pytest.mark.parametrize("server_backend", [TtsBackendKind.AUDIO_CPP, TtsBackendKind.SGL_OMNI])
def test_status_block_local_selection_with_unsupported_server(capsys, monkeypatch, server_backend):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=server_backend,
        issue=RemoteTtsIssue("no_supported_models", "No configured server models match supported catalog variants")))
    state = make_state(TtsModelType.require_by_id("moss_local"))
    state._prefs = Prefs(remote_tts_url="http://example.test")

    MenuStatus.print_block(state)

    lines = capsys.readouterr().out.splitlines()
    tts_lines = [line for line in lines if "TTS model:" in line]
    assert len(tts_lines) == 1
    backend = _BACKEND_LABELS[server_backend]
    expected_status = f"(server mode; {backend} has no supported models)"
    assert text_util.strip_ansi_codes(tts_lines[0]).split(":", 1)[1].strip() == (
        f"MOSS-TTS (local) {expected_status}"
    )
    assert f"{COL_DIM}(local)" in tts_lines[0]
    assert f"{COL_ERROR}{expected_status}" in tts_lines[0]
    assert not any("TTS server:" in line for line in lines)
    assert state.project.tts_model_type == "moss_local"


@pytest.mark.parametrize("server_backend", [TtsBackendKind.AUDIO_CPP, TtsBackendKind.SGL_OMNI])
def test_server_status_local_selection_is_unavailable(monkeypatch, server_backend):
    from tts_audiobook_tool.menus.menu_status import _make_server_tts_text

    RemoteTtsDiscovery.set_base_url("http://example.test")
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=server_backend))

    output = _make_server_tts_text(make_state(TtsModelType.require_by_id("moss_local")))

    assert text_util.strip_ansi_codes(output) == "MOSS-TTS (local) (unavailable in server mode)"
    assert f"{COL_DIM}(local)" in output
    assert f"{COL_ERROR}(unavailable in server mode)" in output
    assert "\x1b]8;;" not in output


@pytest.mark.parametrize("model_id,server_backend", [
    ("auk_sglomni", TtsBackendKind.AUDIO_CPP),
    ("chatterbox_audiocpp", TtsBackendKind.SGL_OMNI),
])
def test_server_status_mismatched_remote_backend_preserves_model_identity(monkeypatch, model_id, server_backend):
    from tts_audiobook_tool.menus.menu_status import _make_server_tts_text

    model = TtsModelType.require_by_id(model_id)
    RemoteTtsDiscovery.set_base_url("http://example.test")
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=server_backend,
        issue=RemoteTtsIssue("no_supported_models", "No configured server models match supported catalog variants")))

    output = _make_server_tts_text(make_state(model))

    assert text_util.strip_ansi_codes(output) == (
        f"{_display_name(model)} (server mode; {_BACKEND_LABELS[server_backend]} has no supported models)"
    )
    # Do not link the selected backend to a different server's model list.
    assert "\x1b]8;;" not in output


@pytest.mark.parametrize("model_id,models_path", [
    ("chatterbox_audiocpp", "/v1/models?include_session_options=true"),
    ("auk_sglomni", "/v1/models"),
])
@pytest.mark.parametrize("base_url", ["http://example.test/", ""])
def test_server_status_links_backend(monkeypatch, model_id, models_path, base_url):
    from tts_audiobook_tool.menus.menu_status import _make_server_tts_text

    model = TtsModelType.require_by_id(model_id)
    RemoteTtsDiscovery.set_base_url(base_url)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=model.value.backend_kind))

    output = _make_server_tts_text(make_state(model))
    backend = _BACKEND_LABELS[model.value.backend_kind]
    assert text_util.strip_ansi_codes(output) == f"{model.value.ui['proper_name']} ({backend})"
    if base_url:
        link = text_util.make_terminal_hyperlink(
            f"http://example.test{models_path}", backend
        )
        assert f"{COL_DIM}({link})" in output
    else:
        assert "\x1b]8;;" not in output


@pytest.mark.parametrize("models,expected_loaded", [
    (({"id": "bound-server", "loaded": True},), True),
    (({"id": "bound-server", "loaded": False},), False),
    (({"id": "bound-server", "loaded": 1},), False),
    (({"id": "bound-server", "loaded": "true"},), False),
    (({"id": "bound-server", "loaded": None},), False),
    (({"id": "bound-server"},), False),
    (({"id": "other-server", "loaded": True},), False),
    ((), False),
    (({"id": "other-server", "loaded": True}, {"id": "bound-server", "loaded": False}), False),
    (({"id": "other-server", "loaded": False}, {"id": "bound-server", "loaded": True}), True),
])
def test_audio_cpp_status_uses_exact_cached_model_residency(monkeypatch, models, expected_loaded):
    from tts_audiobook_tool.menus.menu_status import _make_server_tts_text

    selected = TtsModelType.require_by_id("breeze_tts_2_audiocpp")
    monkeypatch.setattr(Tts, "_type", selected)
    monkeypatch.setattr(Tts, "_selected_server_model_id", "bound-server")
    RemoteTtsDiscovery.set_base_url("http://example.test")
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP, models=models,
        candidates=((selected, "bound-server"),)))
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda **_: pytest.fail("Status formatting must use cache"))
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", lambda *_args, **_kwargs: pytest.fail("Status formatting must not probe"))

    output = _make_server_tts_text(make_state(selected))

    suffix = ", loaded" if expected_loaded else ""
    assert text_util.strip_ansi_codes(output) == f"Breeze TTS 2 (audio.cpp{suffix})"
    link = text_util.make_terminal_hyperlink(
        "http://example.test/v1/models?include_session_options=true", "audio.cpp",
    )
    assert f"{COL_DIM}({link}{suffix})" in output


@pytest.mark.parametrize("failure", [
    "snapshot_issue", "binding_issue", "different_active_model", "unselected",
    "unknown_selection", "missing_server_id", "unknown_backend", "sgl_omni",
])
def test_server_status_hides_loaded_when_binding_or_snapshot_is_unsuitable(monkeypatch, failure):
    from tts_audiobook_tool.app_types import ReadinessIssue
    from tts_audiobook_tool.menus.menu_status import _make_server_tts_text

    selected = TtsModelType.require_by_id("breeze_tts_2_audiocpp")
    state = make_state(selected)
    monkeypatch.setattr(Tts, "_type", selected)
    monkeypatch.setattr(Tts, "_selected_server_model_id", "bound-server")
    backend_kind = TtsBackendKind.AUDIO_CPP
    issue = None
    if failure == "snapshot_issue":
        issue = RemoteTtsIssue("timeout", "Timed out")
    elif failure == "binding_issue":
        monkeypatch.setattr(Tts, "_binding_issue", ReadinessIssue("TTS model", "Ambiguous binding"))
    elif failure == "different_active_model":
        monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("echo_tts_audiocpp"))
    elif failure == "unselected":
        state.project.tts_model_type = "none"
    elif failure == "unknown_selection":
        state.project.tts_model_type = "unknown-model"
    elif failure == "missing_server_id":
        monkeypatch.setattr(Tts, "_selected_server_model_id", "")
    elif failure == "unknown_backend":
        backend_kind = None
    elif failure == "sgl_omni":
        selected = TtsModelType.require_by_id("auk_sglomni")
        state.project.tts_model_type = selected.id
        monkeypatch.setattr(Tts, "_type", selected)
        backend_kind = TtsBackendKind.SGL_OMNI
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=backend_kind, models=({"id": "bound-server", "loaded": True},),
        candidates=((selected, "bound-server"),), issue=issue))

    output = _make_server_tts_text(state)

    assert ", loaded" not in text_util.strip_ansi_codes(output)
    if failure == "snapshot_issue":
        assert "(server unreachable)" in output
    elif failure == "binding_issue":
        assert "Ambiguous binding" in output


@pytest.mark.parametrize("model", [TtsModelType.require_by_id("chatterbox_local"), TtsModelType.require_by_id("omnivoice_local"),
                                   TtsModelType.require_by_id("moss_local"),
                                   TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("chatterbox_audiocpp"),
                                   TtsModelType.require_by_id("echo_tts_audiocpp")])
@pytest.mark.parametrize("saved", ["none", "mira_local",
                                   "echotts-audio-cpp", "future_catalog_id"])
@pytest.mark.parametrize("dir_path", ["", "/example/book"])
def test_status_auto_selects_sole_model_and_defers_hint_until_menu(
    monkeypatch, capsys, model, saved, dir_path,
):
    state = make_state()
    state.project.tts_model_type = saved
    state.project.dir_path = dir_path
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [model])
    if model.value.backend_kind is not TtsBackendKind.LOCAL:
        monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
        monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
            model.value.backend_kind, candidates=((model, "served-model"),)))
    events = []
    monkeypatch.setattr(Project, "save", lambda project: events.append(("save", project.tts_model_type)) or "")
    monkeypatch.setattr(Tts, "bind_project", lambda project: events.append(("bind", project.tts_model_type)))
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("Model-change notices must not pause"))
    monkeypatch.setattr(Tts, "get_active_type", lambda: model)

    MenuStatus.print_block(state)
    MenuStatus.print_block(state)

    assert state.project.tts_model_type == model.id
    expected = [("save", model.id)] if dir_path else []
    expected.extend([("bind", model.id), ("bind", model.id)])
    assert events == expected
    backend = _BACKEND_LABELS[model.value.backend_kind]
    status_output = capsys.readouterr().out
    assert f"{COL_DIM}({backend})" in status_output
    assert "FYI" not in status_output

    expected_pending = None if saved == "none" else PendingTtsModelChange(saved, model.id)
    assert state.pending_tts_model_change == expected_pending
    MenuStatus.show_pending_tts_model_hint(state, is_first_main_menu=True)
    MenuStatus.show_pending_tts_model_hint(state, is_first_main_menu=True)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    if saved == "none":
        assert output == ""
        assert state.pending_tts_model_change is None
        return
    old_name = (_display_name(TtsModelType.require_by_id("mira_local")) if saved == "mira_local" else
                f"Unknown model: {saved}")
    qualifier = ("sole active audio.cpp" if model.value.backend_kind is TtsBackendKind.AUDIO_CPP
                 else "active")
    assert output == (
        f"🔔 FYI\nThis project was last used with TTS model {old_name};\n"
        f"It will now use the {qualifier} model, {_display_name(model)}\n\n"
    )
    assert state.pending_tts_model_change is None


@pytest.fixture
def sole_remote_model(monkeypatch):
    model = TtsModelType.require_by_id("echo_tts_audiocpp")
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [model])
    monkeypatch.setattr(Tts, "get_active_type", lambda: model)
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("Model-change notices must not pause"))
    return model


def test_model_change_hint_reports_binding_failure(monkeypatch, capsys, sole_remote_model):
    state = make_state(TtsModelType.require_by_id("chatterbox_local"))
    monkeypatch.setattr(Tts, "get_active_type", lambda: TtsModelType.require_by_id("none"))
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_tts_model_hint(state)

    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    assert "last used with TTS model Chatterbox TTS" in output
    assert f"It is now configured to use {sole_remote_model.value.ui['proper_name']}" in output
    assert "runtime is unavailable (see TTS model)" in output
    assert "currently active model" not in output
    assert state.pending_tts_model_change is None


def test_model_change_hint_survives_save_error(monkeypatch, capsys, sole_remote_model):
    state = make_state(TtsModelType.require_by_id("chatterbox_local"))
    state.project.dir_path = "/example/book"
    errors = []
    monkeypatch.setattr(Project, "save", lambda _: "Cannot save project")
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_error", errors.append)
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_tts_model_hint(state)

    assert errors == ["Cannot save project"]
    assert state.project.tts_model_type == sole_remote_model.id
    assert "FYI" in capsys.readouterr().out


def test_model_change_hint_coalesces_to_first_old_and_latest_new(monkeypatch, capsys, sole_remote_model):
    state = make_state(TtsModelType.require_by_id("chatterbox_local"))
    MenuStatus.prepare_tts(state)
    latest = TtsModelType.require_by_id("auk_sglomni")
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [latest])
    monkeypatch.setattr(Tts, "get_active_type", lambda: latest)
    MenuStatus.prepare_tts(state)
    MenuStatus.prepare_tts(state)

    assert state.pending_tts_model_change == PendingTtsModelChange("chatterbox_local", latest.id)
    MenuStatus.show_pending_tts_model_hint(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    assert "last used with TTS model Chatterbox TTS" in output
    assert f"active model, {_display_name(latest)}" in output
    assert sole_remote_model.value.ui["proper_name"] not in output


@pytest.mark.parametrize("reason", ["reverted", "cleared", "explicit-change", "local-mode"])
def test_model_change_hint_discards_cancelled_or_stale_notice(
    monkeypatch, capsys, sole_remote_model, reason,
):
    state = make_state(TtsModelType.require_by_id("auk_sglomni"))
    MenuStatus.prepare_tts(state)
    if reason == "reverted":
        monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [TtsModelType.require_by_id("auk_sglomni")])
        MenuStatus.prepare_tts(state)
    elif reason == "cleared":
        monkeypatch.setattr(Tts, "get_available_tts_models",
                            lambda **kwargs: [TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("fish_s2_sglomni")])
        MenuStatus.prepare_tts(state)
    elif reason == "explicit-change":
        state.project.tts_model_type = "fish_s2_sglomni"
    else:
        monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
        state.has_shown_main_menu = True
    MenuStatus.show_pending_tts_model_hint(state)

    assert capsys.readouterr().out == ""
    assert state.pending_tts_model_change is None


def test_matching_selection_does_not_queue_hint(sole_remote_model):
    state = make_state(sole_remote_model)
    MenuStatus.prepare_tts(state)
    assert state.pending_tts_model_change is None


def test_local_startup_notice_waits_for_first_main_menu(monkeypatch, capsys):
    state = make_state(TtsModelType.require_by_id("echo_tts_audiocpp"))
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("chatterbox_local"),))
    monkeypatch.setattr(Tts, "get_active_type", lambda: TtsModelType.require_by_id("chatterbox_local"))
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_tts_model_hint(state)
    MenuStatus.prepare_tts(state)

    assert state.project.tts_model_type == "chatterbox_local"
    assert state.pending_tts_model_change == PendingTtsModelChange(
        "echo_tts_audiocpp", "chatterbox_local",
    )
    assert capsys.readouterr().out == ""

    state.has_shown_main_menu = True  # MainMenu.on_shown runs before the footer.
    MenuStatus.show_pending_tts_model_hint(state, is_first_main_menu=True)
    MenuStatus.show_pending_tts_model_hint(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    assert output.count("FYI") == 1
    assert "last used with TTS model Echo-TTS" in output
    assert "active model, Chatterbox TTS" in output
    assert state.pending_tts_model_change is None


@pytest.mark.parametrize("saved", [TtsModelType.require_by_id("none"), TtsModelType.require_by_id("echo_tts_audiocpp")])
def test_local_changes_after_first_main_menu_stay_silent(monkeypatch, capsys, saved):
    state = make_state(saved)
    state.has_shown_main_menu = True
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("chatterbox_local"),))
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_tts_model_hint(state)

    assert state.project.tts_model_type == "chatterbox_local"
    assert state.pending_tts_model_change is None
    assert capsys.readouterr().out == ""


def test_matching_local_selection_does_not_queue_startup_notice(monkeypatch):
    state = make_state(TtsModelType.require_by_id("chatterbox_local"))
    monkeypatch.setattr(Tts, "_available_local_models", (TtsModelType.require_by_id("chatterbox_local"),))
    MenuStatus.prepare_tts(state)
    assert state.pending_tts_model_change is None


@pytest.mark.parametrize("reset", [False, True])
def test_project_replacement_or_reset_clears_model_change(monkeypatch, sole_remote_model, reset):
    monkeypatch.setattr(Project, "kill", lambda _: None)
    monkeypatch.setattr(Prefs, "save", lambda _: "")
    state = State.for_worker(Prefs())
    assert state.pending_tts_model_change is None
    state.project = Project(tts_model_type="chatterbox_local")
    MenuStatus.prepare_tts(state)
    assert state.pending_tts_model_change is not None

    if reset:
        state.reset()
    else:
        state.project = Project(tts_model_type=sole_remote_model.id)
    assert state.pending_tts_model_change is None


def test_interactive_state_initializes_without_pending_notice(monkeypatch):
    monkeypatch.setattr(Prefs, "load", lambda: Prefs())
    state = State()
    assert state.pending_tts_model_change is None


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("with_long_text", [False, True])
@pytest.mark.parametrize("saved", ["qwen3tts_local", "omnivoice_local"])
@pytest.mark.parametrize("durations, expected", [
    ([15.01, 15.0, 14.6, None], "is 1 sample for this project that exceeds"),
    ([15.01, 16.0, 15.0, None], "are 2 samples for this project that exceed"),
    ([15.0, 14.6, None], None),
    ([], None),
])
def test_shared_project_load_hints_at_startup_and_runtime(
    monkeypatch, capsys, runtime, with_long_text, saved, durations, expected,
):
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil

    project = Project(tts_model_type=saved, voice_references=[
        {"file_name": f"voice-{index}.flac", "transcript": ""}
        for index in range(len(durations))
    ])
    project.dir_path = "/example/book"
    if with_long_text:
        project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
        project.book.segmentation_settings = project.book.segmentation_settings._replace(
            max_words_per_segment=100,
        )
        project.max_words = 20  # The next import's setting must not affect the FYI.
    original_book = project.book
    original_settings = project.book.segmentation_settings
    monkeypatch.setattr(Prefs, "load", lambda: Prefs(project_dir=project.dir_path))
    monkeypatch.setattr(Prefs, "save", lambda _: "")
    monkeypatch.setattr(Project, "save", lambda _: "")
    monkeypatch.setattr(Project, "kill", lambda _: None)
    monkeypatch.setattr(ProjectLoadUtil, "load_using_dir_path", lambda _: project)
    model = TtsModelType.require_by_id("omnivoice_local")
    monkeypatch.setattr(Tts, "_available_local_models", (model,))
    monkeypatch.setattr(Tts, "get_active_type", lambda: model)
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("Project-load FYIs must not pause"))
    probes = []
    remaining = iter(durations)

    def get_duration(path):
        probes.append(path)
        return next(remaining)

    monkeypatch.setattr(AudioMetaUtil, "get_audio_duration", get_duration)
    if runtime:
        state = State.for_worker(Prefs())
        state.project = Project()
        state.mark_main_menu_shown()
        state.set_existing_project(project.dir_path)
    else:
        state = State()
    assert state.pending_project_load_checks
    MenuStatus.prepare_tts(state)
    MenuStatus.prepare_tts(state)
    assert project.tts_model_type == model.id
    assert "FYI" not in capsys.readouterr().out

    if not runtime:
        MenuStatus.show_pending_project_hints(state)  # Early startup submenu.
        assert probes == []
        assert state.pending_project_load_checks
        assert capsys.readouterr().out == ""
        state.mark_main_menu_shown()  # on_shown precedes footer rendering.
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=not runtime)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    assert ("last used with TTS model Qwen3-TTS" in output) == (saved != model.id)
    if expected:
        assert "The current model's recommended duration for voice clone samples is 15s,\n" in output
        assert f"but there {expected} that value." in output
    else:
        assert "recommended duration" not in output
    if with_long_text:
        assert (
            "This project's text was segmented with a maximum of 100 words per segment,\n"
            "exceeding the current model's recommended maximum of 80.\n"
            "Output accuracy on longer prompts may be degraded."
        ) in output
        if saved != model.id:
            assert output.index("last used with") < output.index("text was segmented")
        if expected:
            assert output.index("recommended duration") < output.index("text was segmented")
        assert project.max_words == 20
    else:
        assert "text was segmented" not in output
    assert output.count("FYI") == int(saved != model.id) + int(expected is not None) + int(with_long_text)
    assert project.book is original_book
    assert project.book.segmentation_settings == original_settings
    assert len(probes) == len(durations)
    assert not state.pending_project_load_checks
    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)
    assert capsys.readouterr().out == ""
    assert len(probes) == len(durations)


@pytest.mark.parametrize("model_id", ["none", "chatterbox_local", "qwen3tts_local"])
def test_project_load_skips_samples_without_duration_recommendation(monkeypatch, capsys, model_id):
    from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil

    state = make_state(TtsModelType.require_by_id(model_id))
    state.pending_project_load_checks = True
    state.project.voice_references = [{"file_name": "voice.flac", "transcript": ""}]
    monkeypatch.setattr(AudioMetaUtil, "get_audio_duration",
                        lambda _: pytest.fail("No recommendation: do not probe sample durations"))
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)
    assert capsys.readouterr().out == ""
    assert not state.pending_project_load_checks


@pytest.mark.parametrize("model_id", ["glm_local", "echo_tts_audiocpp", "higgs_v3_sglomni"])
@pytest.mark.parametrize("offset", [None, -1, 0, 1])
def test_project_load_max_words_uses_model_recommended_upper_bound(
    capsys, model_id, offset,
):
    state = make_state(TtsModelType.require_by_id(model_id))
    state.pending_project_load_checks = True
    state.project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
    limit = Tts.get_model_support(state.project).get_max_words_range_reco(state.project)[1]
    max_words = 0 if offset is None else limit + offset
    state.project.book.segmentation_settings = state.project.book.segmentation_settings._replace(
        max_words_per_segment=max_words,
    )
    state.project.max_words = limit + 100  # Current import setting is irrelevant.

    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)

    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    assert output.count("FYI") == int(offset == 1)
    if offset == 1:
        assert f"maximum of {max_words} words per segment" in output
        assert f"recommended maximum of {limit}." in output
    assert not state.pending_project_load_checks


@pytest.mark.parametrize("model_id, has_text", [
    ("none", True), ("unknown-model", True), ("glm_local", False),
])
def test_project_load_max_words_skips_missing_model_or_text(capsys, model_id, has_text):
    state = make_state()
    state.project.tts_model_type = model_id
    state.pending_project_load_checks = True
    if has_text:
        state.project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
    state.project.book.segmentation_settings = state.project.book.segmentation_settings._replace(
        max_words_per_segment=100,
    )

    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)

    assert capsys.readouterr().out == ""
    assert not state.pending_project_load_checks


@pytest.mark.parametrize("limit", [0, -1])
def test_project_load_max_words_skips_nonpositive_recommendation(monkeypatch, capsys, limit):
    state = make_state(TtsModelType.require_by_id("glm_local"))
    state.pending_project_load_checks = True
    state.project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
    state.project.book.segmentation_settings = state.project.book.segmentation_settings._replace(
        max_words_per_segment=100,
    )
    monkeypatch.setattr(Tts, "get_model_support", lambda _: SimpleNamespace(
        get_max_words_range_reco=lambda _: (0, limit, ""),
    ))

    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)

    assert capsys.readouterr().out == ""
    assert not state.pending_project_load_checks


def test_project_load_max_words_uses_reconciled_model(monkeypatch, capsys):
    state = make_state(TtsModelType.require_by_id("glm_local"))
    state.pending_project_load_checks = True
    state.project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
    state.project.book.segmentation_settings = state.project.book.segmentation_settings._replace(
        max_words_per_segment=60,  # Above Echo's recommendation.
    )
    model = TtsModelType.require_by_id("echo_tts_audiocpp")
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [model])
    monkeypatch.setattr(Tts, "get_active_type", lambda: model)

    MenuStatus.prepare_tts(state)
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)

    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    assert state.project.tts_model_type == model.id
    assert output.count("FYI") == 2
    assert "maximum of 60 words per segment" in output
    assert "recommended maximum of 50." in output
    assert output.index("last used with") < output.index("text was segmented")


def test_project_replacement_rearms_load_checks_and_reset_clears_them(monkeypatch, capsys):
    from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil

    monkeypatch.setattr(Project, "kill", lambda _: None)
    monkeypatch.setattr(Prefs, "save", lambda _: "")
    monkeypatch.setattr(AudioMetaUtil, "get_audio_duration", lambda _: 16.0)
    state = State.for_worker(Prefs())
    state.has_shown_main_menu = True
    for path in ("/example/first", "/example/second"):
        project = Project(tts_model_type="omnivoice_local", voice_references=[
            {"file_name": "voice.flac", "transcript": ""},
        ])
        project.dir_path = path
        project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
        project.book.segmentation_settings = project.book.segmentation_settings._replace(
            max_words_per_segment=100,
        )
        state.project = project
        assert state.pending_project_load_checks
        MenuStatus.show_pending_project_hints(state)
        assert capsys.readouterr().out.count("FYI") == 2
        assert not state.pending_project_load_checks
    project = Project(tts_model_type="omnivoice_local")
    project.dir_path = "/example/third"
    state.project = project
    state.reset()
    assert not state.pending_project_load_checks
    MenuStatus.show_pending_project_hints(state)
    assert capsys.readouterr().out == ""


def test_model_switch_rearms_checks_and_prints_only_compatibility_fyis(monkeypatch, capsys):
    from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil

    monkeypatch.setattr(AudioMetaUtil, "get_audio_duration", lambda _: 16.0)
    model = TtsModelType.require_by_id("omnivoice_local")
    state = make_state(model)
    state.has_shown_main_menu = True
    state.project.dir_path = "/example/book"
    state.project.voice_references = [{"file_name": "voice.flac", "transcript": ""}]
    state.project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
    state.project.book.segmentation_settings = state.project.book.segmentation_settings._replace(
        max_words_per_segment=100,
    )

    # What ProjectMenu.tts_model_menu.on_select leaves behind after a switch.
    state.pending_tts_model_change = None
    state.pending_project_load_checks = True

    MenuStatus.show_pending_project_hints(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    assert output.count("FYI") == 2
    assert "last used with" not in output
    assert "recommended duration for voice clone samples is 15s" in output
    assert "maximum of 100 words per segment" in output
    assert not state.pending_project_load_checks

    # Ordinary redraws after the switch do not repeat the notices.
    MenuStatus.show_pending_project_hints(state)
    assert capsys.readouterr().out == ""


def test_status_clears_and_saves_local_selection_when_no_model_is_available(monkeypatch):
    state = make_state(TtsModelType.require_by_id("mira_local"))
    state.project.dir_path = "/example/book"
    events = []
    monkeypatch.setattr(Project, "save", lambda project: events.append(("save", project.tts_model_type)) or "")
    monkeypatch.setattr(Tts, "bind_project", lambda project: events.append(("bind", project.tts_model_type)))
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("Clearing a local selection must not prompt"))

    MenuStatus.prepare_tts(state)
    MenuStatus.prepare_tts(state)

    assert state.project.tts_model_type == "none"
    assert events == [("save", "none"), ("bind", "none"),
                      ("bind", "none")]


def test_status_clears_unavailable_selection_with_multiple_models(monkeypatch, capsys):
    state = make_state(TtsModelType.require_by_id("mira_local"))
    state.project.dir_path = "/example/book"
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    models = [TtsModelType.require_by_id("chatterbox_audiocpp"), TtsModelType.require_by_id("higgs_v3_audiocpp")]
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        TtsBackendKind.AUDIO_CPP, candidates=tuple((model, model.id) for model in models)))
    saves = []
    monkeypatch.setattr(Project, "save", lambda project: saves.append(project.tts_model_type) or "")
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("Multiple models must not auto-select or pause"))

    MenuStatus.print_block(state)
    MenuStatus.print_block(state)

    assert state.project.tts_model_type == "none"
    assert saves == ["none"]


def test_remote_unselected_project_still_reports_connection_failure(monkeypatch, capsys):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        issue=RemoteTtsIssue("unavailable", "Cannot connect")))

    MenuStatus.print_block(make_state())

    assert f"{COL_ERROR}(server unreachable)" in capsys.readouterr().out


def test_loaded_status_requires_worker_model_to_match_project():
    from tts_audiobook_tool.menus.menu_status import _make_local_tts_text
    from tts_audiobook_tool.model_worker_protocol import ModelStateSnapshot

    state = make_state(TtsModelType.require_by_id("omnivoice_local"))
    wrong = ModelStateSnapshot(tts_loaded=True, tts_type_id="mira_local", tts_device="cuda")
    matching = ModelStateSnapshot(tts_loaded=True, tts_type_id="omnivoice_local", tts_device="cpu")

    assert "loaded" not in _make_local_tts_text(state, wrong)
    assert "cuda" not in _make_local_tts_text(state, wrong)
    assert "cpu, loaded" in _make_local_tts_text(state, matching)


def test_dependent_voice_display_info_overrides_propagate_none(monkeypatch):
    project = Project(dir_path="")
    monkeypatch.setattr(
        TtsBaseModel,
        "get_voice_display_info",
        classmethod(lambda cls, project, instance=None: None),
    )

    assert Qwen3BaseModel.get_voice_display_info(project) is None
    assert IndexTts2BaseModel.get_voice_display_info(project) is None


def test_qwen_custom_voice_status_uses_project_speaker_without_instance():
    project = Project(dir_path="")
    set_setting(project, "qwen3_model_type", "custom_voice")
    set_setting(project, "qwen3_speaker_id", "Ryan")

    display_info = Qwen3BaseModel.get_voice_display_info(project, None)

    assert display_info is not None
    assert display_info.status_prefix == "Speaker"
    assert display_info.main_prefix == "speaker"
    assert display_info.value == "Ryan"


def test_qwen_custom_voice_status_includes_instructions_with_valid_speaker():
    project = Project(dir_path="")
    set_setting(project, "qwen3_model_type", "custom_voice")
    set_setting(project, "qwen3_speaker_id", "Ryan")
    set_setting(project, "qwen3_instructions", "Speak warmly")

    display_info = Qwen3BaseModel.get_voice_display_info(project, None)

    assert display_info is not None
    assert display_info.value == f"Ryan{COL_DIM} + instructions"


def test_qwen_custom_voice_status_requires_speaker_without_instance():
    project = Project(dir_path="")
    set_setting(project, "qwen3_model_type", "custom_voice")

    display_info = Qwen3BaseModel.get_voice_display_info(project, None)

    assert display_info is not None
    assert display_info.status_prefix == "Speaker"
    assert display_info.main_prefix == "speaker"
    assert display_info.value == COL_ERROR + "required"


@pytest.mark.parametrize("dir_path", ["/example/book", ""])
@pytest.mark.parametrize("total_lines", [7272, 0])
def test_text_status_links_generated_count_to_sound_segments(monkeypatch, dir_path, total_lines):
    from tts_audiobook_tool.menus.menu_status import _make_text_text

    state = make_state()
    state.project.dir_path = dir_path
    state.project.book.segmentation_settings = state.project.book.segmentation_settings._replace(
        language_code="en"
    )
    monkeypatch.setattr(Project, "phrase_groups", property(lambda self: range(total_lines)))
    monkeypatch.setattr(state.project.sound_segments, "num_generated", lambda: 100)

    output = _make_text_text(state)

    if total_lines == 0:
        assert output == COL_ERROR + "required"
        return
    assert text_util.strip_ansi_codes(output) == "7272 lines, en (100 generated)"
    if dir_path:
        link = text_util.make_terminal_hyperlink(
            state.project.sound_segments_path, "100 generated", is_file=True
        )
        assert f"{COL_DIM}({link})" in output
    else:
        assert "\x1b]8;;" not in output


def test_stt_status_does_not_repeat_disabled(monkeypatch):
    state = cast(State, SimpleNamespace(
        prefs=SimpleNamespace(
            stt_variant=SttVariant.DISABLED,
            stt_config=SttConfig.CPU_INT8FLOAT32,
        ),
    ))
    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: False)
    monkeypatch.setattr(Stt, "get_variant", lambda: SttVariant.DISABLED)
    monkeypatch.setattr(Stt, "has_instance", lambda: False)

    text = _make_stt_text(state)

    assert text == f"faster-whisper {COL_DIM}(disabled)"
