from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot, RemoteTtsIssue
from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice
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
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.util import COL_DIM, COL_ERROR, ellipsize


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
    voice_line = next(line for line in output.splitlines() if "Voices:" in line)
    assert text_util.strip_ansi_codes(voice_line).split(":", 1)[1].strip() == "zzz belle 24a 19s, +1 more"
    assert "Voice clone:" not in output


# Summary policy depends on `requires_voice`; remote representatives also exercise
# their separate menu-text implementations without multiplying the name layouts.
@pytest.mark.parametrize("model_id, names, expected", [
    ("omnivoice_local", ["first.flac"], "first"),
    ("omnivoice_local", ["first.flac", "second.flac"], "first, +1 more"),
    ("omnivoice_local", ["", "first.flac"], "first, +1 more"),
    ("omnivoice_local", ["", "first.flac", "", "second.flac"], "first, +3 more"),
    ("omnivoice_local", ["first.flac", "", ""], "first, +2 more"),
    ("omnivoice_local", [""], None),
    ("omnivoice_local", ["", ""], None),
    ("glm_local", [""], None),
    ("glm_local", ["", ""], None),
    ("breeze_tts_2_audiocpp", ["first.flac"], "first"),
    ("higgs_v3_sglomni", ["first.flac"], "first"),
])
def test_voices_status_summarizes_all_slots(monkeypatch, capsys, model_id, names, expected):
    model = TtsModelType.require_by_id(model_id)
    state = make_state(model)
    state.project.voice_references = [Voice(name) for name in names]
    original = state.project.voice_references
    # Isolate display from runtime availability/reconciliation.
    monkeypatch.setattr(MenuStatus, "prepare_tts", lambda _: None)
    MenuStatus.print_block(state)
    output = capsys.readouterr().out
    voice_line = next(line for line in output.splitlines() if "Voices:" in line)
    if not any(names) and model.value.requires_voice:
        # No voice has a value and the model needs one: blocking, so error-colored
        expected = "required"
        assert COL_ERROR + expected in voice_line
    elif not any(names):
        # Non-blocking: the model falls back to its built-in voice
        expected = "no sample set"
        assert COL_ERROR not in voice_line
    else:
        assert COL_ERROR not in voice_line
    assert text_util.strip_ansi_codes(voice_line).split(":", 1)[1].strip() == expected
    assert "Voice clone:" not in output
    assert state.project.voice_references is original


def test_voices_status_preserves_sample_hyperlink(tmp_path):
    from tts_audiobook_tool.menus.menu_status import _make_voices_text

    state = make_state(TtsModelType.require_by_id("omnivoice_local"))
    state.project.dir_path = str(tmp_path)
    sample = tmp_path / "voice" / "sample_omnivoice.flac"
    sample.parent.mkdir()
    sample.write_bytes(b"sample")
    state.project.voice_references = [Voice(), Voice(sample.name)]
    link = text_util.make_terminal_hyperlink(str(sample), "sample", is_file=True)
    assert _make_voices_text(state) == f"{link}, +1 more"


def test_status_voices_line_shows_speaker_preset(monkeypatch, capsys):
    """The status block always uses the "Voices" label, including for presets."""
    model_id = "qwen3tts_local"
    state = make_state(TtsModelType.require_by_id(model_id))
    state.project.set_model_setting(model_id, "model_type", "custom_voice")
    state.project.voice_references = [Voice(presets={model_id: "Ryan"})]
    monkeypatch.setattr(MenuStatus, "prepare_tts", lambda _: None)
    MenuStatus.print_block(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    line = next(line for line in output.splitlines() if line.startswith("Voices:"))
    assert line.split(":", 1)[1].strip() == "Ryan"
    assert "Speaker:" not in output


@pytest.mark.parametrize("voices, expected", [
    # A predefined voice takes precedence over its voice's sample
    ([Voice(presets={"pocket_local": "alba"})], "alba"),
    ([Voice("sample_pocket.flac", presets={"pocket_local": "alba"}), Voice()], "alba, +1 more"),
    ([Voice(), Voice(presets={"pocket_local": "jean"})], "jean, +1 more"),
    ([Voice("sample_pocket.flac"), Voice(presets={"pocket_local": "jean"})], "sample, +1 more"),
    ([Voice()], "required"),
])
def test_pocket_status_always_uses_the_voices_line(monkeypatch, capsys, voices, expected):
    state = make_state(TtsModelType.require_by_id("pocket_local"))
    state.project.voice_references = voices
    monkeypatch.setattr(MenuStatus, "prepare_tts", lambda _: None)
    MenuStatus.print_block(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    lines = output.splitlines()
    assert not any(line.startswith("Predefined voice:") for line in lines)
    line = next(line for line in lines if line.startswith("Voices:"))
    assert line.split(":", 1)[1].strip() == expected


@pytest.mark.parametrize("voices, expected", [
    # Instructions count when the voice has no usable sample
    ([Voice(), Voice(instructions={"auk_sglomni": "A calm voice"})], '"A calm voice", +1 more'),
    # A sample takes precedence over instructions
    ([Voice("first.flac", instructions={"auk_sglomni": "A calm voice"})], "first"),
    # Blank instructions are no value
    ([Voice(instructions={"auk_sglomni": "   "}), Voice()], "no sample set"),
    # Instructions for another model are no value
    ([Voice(instructions={"omnivoice_local": "male"})], "no sample set"),
    ([Voice(instructions={"auk_sglomni": "word " * 20})], '"' + ellipsize("word " * 19 + "word", 40) + '"'),
])
def test_voices_status_counts_instructions(monkeypatch, voices, expected):
    from tts_audiobook_tool.menus.menu_status import _make_voices_text

    state = make_state(TtsModelType.require_by_id("auk_sglomni"))
    state.project.voice_references = voices
    text = _make_voices_text(state)
    assert text_util.strip_ansi_codes(text) == expected
    assert COL_ERROR not in text


def test_voices_status_instructions_do_not_satisfy_a_required_voice():
    from tts_audiobook_tool.menus.menu_status import _make_voices_text

    model = TtsModelType.require_by_id("glm_local")
    assert model.value.requires_voice
    state = make_state(model)
    state.project.voice_references = [Voice(instructions={"glm_local": "A calm voice"})]
    assert _make_voices_text(state) == COL_ERROR + "required"


@pytest.mark.parametrize("voices, expected", [
    # VoiceDesign can't clone: its voices are defined by their instructions
    ([Voice(instructions={"qwen3tts_local": "sleepy"}),
      Voice(instructions={"qwen3tts_local": "a cute young girl"})], '"sleepy", +1 more'),
    ([Voice(), Voice(instructions={"qwen3tts_local": "sleepy"})], '"sleepy", +1 more'),
    # No instructions: an arbitrary voice (non-blocking), not a required sample
    ([Voice()], "no instructions set"),
    ([Voice("sample_qwen3.flac")], "no instructions set"),
])
def test_qwen_voice_design_status_counts_instructions(voices, expected):
    from tts_audiobook_tool.menus.menu_status import _make_voices_text

    model = TtsModelType.require_by_id("qwen3tts_local")
    assert model.value.requires_voice
    state = make_state(model)
    state.project.set_model_setting("qwen3tts_local", "model_type", "voice_design")
    state.project.voice_references = voices
    text = _make_voices_text(state)
    assert text_util.strip_ansi_codes(text) == expected
    assert COL_ERROR not in text


def test_audio_cpp_omnivoice_voice_line_is_optional(monkeypatch):
    """The audio.cpp OmniVoice entry can run without a voice, like its local twin."""
    from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
    from tts_audiobook_tool.tts import Tts

    model_type = TtsModelType.require_by_id("omnivoice_audiocpp")
    monkeypatch.setattr(Tts, "_audio_cpp_definitions", load_audio_cpp_definitions().models)
    monkeypatch.setattr(Tts, "_available_local_models", (model_type,))
    state = make_state(model_type)

    from tts_audiobook_tool.menus.menu_status import _make_voices_text

    assert _make_voices_text(state) == "no sample set"
    assert make_voice_label(state) == "Voices"


def test_status_block_local_mode_none_shows_tts_model_line(capsys):
    MenuStatus.print_block(make_state())

    lines = [text_util.strip_ansi_codes(line) for line in capsys.readouterr().out.splitlines()]
    tts_lines = [line for line in lines if "TTS model:" in line]
    assert len(tts_lines) == 1
    assert "SGL-Omni" not in tts_lines[0]


# Only unavailable/timeout ("server unreachable") vs any other code (issue message) matter.
@pytest.mark.parametrize("issue_code", ["unavailable", "timeout", "http_error"])
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


def test_status_block_unreachable_unknown_server_omits_server_qualifier(capsys, monkeypatch):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        issue=RemoteTtsIssue("unavailable", "Remote server unavailable")))

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
    (({"id": "bound-server", "loaded": 1},), False),  # truthy non-bool is not loaded
    (({"id": "other-server", "loaded": True},), False),  # id must match
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


# The loaded qualifier names the resident model file (Windows or POSIX server path),
# ellipsized to 30 chars.
@pytest.mark.parametrize("path, expected", [
    ("/models/qwen3-tts-base.gguf", "loaded, qwen3-tts-base.gguf"),
    ("C:\\models\\qwen3-tts-base.gguf", "loaded, qwen3-tts-base.gguf"),
    ("/m/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf", "loaded, qwen3-tts-12hz-1.7b-customv..."),
    (None, "loaded"),
])
def test_audio_cpp_status_shows_loaded_model_filename(monkeypatch, path, expected):
    from tts_audiobook_tool.menus.menu_status import _make_server_tts_text

    selected = TtsModelType.require_by_id("qwen3tts_audiocpp")
    monkeypatch.setattr(Tts, "_type", selected)
    monkeypatch.setattr(Tts, "_selected_server_model_id", "bound-server")
    entry = {"id": "bound-server", "loaded": True}
    if path is not None:
        entry["path"] = path
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP, models=(entry,),
        candidates=((selected, "bound-server"),)))

    output = text_util.strip_ansi_codes(_make_server_tts_text(make_state(selected)))

    assert output == f"Qwen3-TTS (audio.cpp, {expected})"


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


# Reconciliation itself is tested in test_project_tts_selection. Here only the backend
# (local mode vs remote wording), saved none / known / unknown
# (hint or not, old-name copy) and dir_path (save or not) matter.
@pytest.mark.parametrize("model_id, saved, dir_path", [
    ("chatterbox_local", "none", ""),
    ("chatterbox_local", "mira_local", "/example/book"),
    ("omnivoice_local", "future_catalog_id", ""),
    ("moss_local", "none", ""),  # distinct target-dependent menu text
    ("auk_sglomni", "none", "/example/book"),
    ("auk_sglomni", "mira_local", ""),
    ("echo_tts_audiocpp", "none", ""),
    ("echo_tts_audiocpp", "mira_local", "/example/book"),
    ("chatterbox_audiocpp", "echotts-audio-cpp", "/example/book"),
])
def test_status_auto_selects_sole_model_and_defers_hint_until_menu(
    monkeypatch, capsys, model_id, saved, dir_path,
):
    model = TtsModelType.require_by_id(model_id)
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
    # The notice explains where the sole model came from: the server or this environment.
    current = _display_name(model)
    reason = (f"This environment has {current} installed, so it will use that."
              if model.value.backend_kind is TtsBackendKind.LOCAL
              else f"The server is serving {current}, so it will use that.")
    assert output == f"🔔 FYI\nThis project was last used with TTS model {old_name}.\n{reason}\n\n"
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
    assert output == (
        "🔔 FYI\nThis project was last used with TTS model Chatterbox TTS (local).\n"
        f"It has been switched to {_display_name(sole_remote_model)}, "
        "but that model can't be used yet (see Model settings).\n\n"
    )
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
    assert f"The server is serving {_display_name(latest)}, so it will use that." in output
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
        monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
        monkeypatch.setattr(Tts, "get_available_tts_models", lambda **kwargs: [])
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
    assert "This environment has Chatterbox TTS (local) installed, so it will use that." in output
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


# Pairwise rows: every duration case runs with both values of each flag, and the three
# flags together cover all 8 combinations. The first row raises all three FYIs.
_ONE_EXCEEDS = "is 1 sample for this project that exceeds"
_TWO_EXCEED = "are 2 samples for this project that exceed"
@pytest.mark.parametrize("runtime, with_long_text, saved, durations, expected", [
    (False, True, "qwen3tts_local", [15.01, 15.0, 14.6, None], _ONE_EXCEEDS),
    (True, False, "omnivoice_local", [15.01, 15.0, 14.6, None], _ONE_EXCEEDS),
    (False, True, "omnivoice_local", [15.01, 16.0, 15.0, None], _TWO_EXCEED),
    (True, False, "qwen3tts_local", [15.01, 16.0, 15.0, None], _TWO_EXCEED),
    (False, False, "qwen3tts_local", [15.0, 14.6, None], None),
    (True, True, "omnivoice_local", [15.0, 14.6, None], None),
    (True, True, "qwen3tts_local", [], None),
    (False, False, "omnivoice_local", [], None),
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
    state.project.voice_references = [Voice("voice.flac")]
    monkeypatch.setattr(AudioMetaUtil, "get_audio_duration",
                        lambda _: pytest.fail("No recommendation: do not probe sample durations"))
    MenuStatus.show_pending_project_hints(state, is_first_main_menu=True)
    assert capsys.readouterr().out == ""
    assert not state.pending_project_load_checks


# Comparison boundaries run on one model; SGL also has its own limit accessor.
@pytest.mark.parametrize("model_id,offset", [
    ("glm_local", None), ("glm_local", -1), ("glm_local", 0), ("glm_local", 1),
    ("higgs_v3_sglomni", 1),
])
def test_project_load_max_words_uses_model_recommended_upper_bound(capsys, model_id, offset):
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
    state.project.voice_references = [Voice("voice.flac")]
    state.project.phrase_groups = [PhraseGroup([Phrase("Short text.", Reason.PARAGRAPH)])]
    state.project.book.segmentation_settings = state.project.book.segmentation_settings._replace(
        max_words_per_segment=100,
    )

    # What ModelSelectMenu.menu.on_select leaves behind after a switch.
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


@pytest.mark.parametrize("runtime_load, saved", [
    (False, "echo_tts_audiocpp"),
    (False, "mira_local"),
    (False, "auk_sglomni"),
    (False, "future_catalog_id"),
    (False, "none"),
    (True, "echo_tts_audiocpp"),  # runtime load still announces
])
def test_status_clears_unavailable_selection_with_multiple_models(monkeypatch, capsys, runtime_load, saved):
    # Startup/runtime loads announce only unavailable audio.cpp selections; other old types stay silent.
    state = make_state()
    state.project.tts_model_type = saved
    state.has_shown_main_menu = runtime_load
    state.pending_project_load_checks = True
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
    assert saves == ([] if saved == "none" else ["none"])
    assert "FYI" not in capsys.readouterr().out
    should_notify = saved == "echo_tts_audiocpp"
    assert state.pending_tts_model_change == (
        PendingTtsModelChange(saved, "none") if should_notify else None
    )

    MenuStatus.show_pending_project_hints(state, is_first_main_menu=not runtime_load)
    MenuStatus.show_pending_project_hints(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)
    old_name = (f"Unknown model: {saved}" if saved == "future_catalog_id" else
                _display_name(TtsModelType.get_by_id(saved)))
    assert output == ("" if not should_notify else (
        f"🔔 FYI\nThis project was last used with TTS model {old_name},\nwhich is currently unavailable.\n"
        "Select a TTS model in Model settings > TTS model.\n\n"
    ))
    assert state.pending_tts_model_change is None


@pytest.mark.parametrize("case", [
    "restored", "restored_from_none", "restored_unbound", "restored_unbound_from_none",
    "unavailable_multiple", "unavailable_sole", "unavailable_sole_from_none", "unavailable_sole_unbound",
])
def test_remembered_audio_cpp_selection_notices(monkeypatch, capsys, case):
    # Reconnecting to audio.cpp after using a local model restores (or reports
    # failing to restore) the project's last-used audio.cpp entry, deferred to
    # the end of the next complete menu like other model-change notices.
    # From an unselected project the "last used" line is omitted, so the next
    # sentence must name the project rather than start with "It"/"Its".
    fish = TtsModelType.require_by_id("fish_s2_audiocpp")
    chatterbox = TtsModelType.require_by_id("chatterbox_audiocpp")
    higgs = TtsModelType.require_by_id("higgs_v3_audiocpp")
    state = make_state()
    state.project.set_tts_selection(fish.id, "fish-q8")
    state.project.set_tts_selection("none" if case.endswith("_from_none") else "chatterbox_local")
    state.project.dir_path = "/example/book"
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    candidates = (((chatterbox, "cb"), (higgs, "higgs")) if case == "unavailable_multiple" else
                  ((chatterbox, "cb"),) if case.startswith("unavailable_sole") else
                  ((chatterbox, "cb"), (fish, "fish-q8")))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        TtsBackendKind.AUDIO_CPP, candidates=candidates))
    monkeypatch.setattr(Project, "save", lambda _: "")
    monkeypatch.setattr(Tts, "get_active_type", lambda: (
        TtsModelType.require_by_id("none") if "_unbound" in case
        else TtsModelType.get_by_id(state.project.tts_model_type)))
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_enter_to_continue",
                        lambda *_: pytest.fail("Model-change notices must not pause"))

    MenuStatus.print_block(state)
    MenuStatus.print_block(state)
    assert "FYI" not in capsys.readouterr().out
    MenuStatus.show_pending_project_hints(state)
    MenuStatus.show_pending_project_hints(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)

    old_line = "This project was last used with TTS model Chatterbox TTS (local)"
    remembered = f"{fish.value.ui['proper_name']} (audio.cpp; fish-q8)"
    is_sole = case.startswith("unavailable_sole")
    expected_selection = (("none", "") if case == "unavailable_multiple" else
                          (chatterbox.id, "cb") if is_sole else (fish.id, "fish-q8"))
    assert (state.project.tts_model_type, state.project.audio_cpp_model_id) == expected_selection
    assert (state.project.last_audio_cpp_model_type, state.project.last_audio_cpp_model_id) == (
        (chatterbox.id, "cb") if is_sole else (fish.id, "fish-q8"))
    sole = _display_name(chatterbox)
    unusable = "but that model can't be used yet (see Model settings)."
    expected = {
        "restored": f"{old_line}.\nIt will switch back to its last-used audio.cpp model, {remembered}.",
        "restored_from_none": f"This project will switch back to its last-used audio.cpp model, {remembered}.",
        "restored_unbound": (f"{old_line}.\nIt has been switched back to its last-used audio.cpp model, "
                             f"{remembered},\n{unusable}"),
        "restored_unbound_from_none": (f"This project has been switched back to its last-used audio.cpp model, "
                                       f"{remembered},\n{unusable}"),
        "unavailable_multiple": (f"{old_line}.\nIts last-used audio.cpp model, {remembered},\n"
                                 "is currently unavailable.\nSelect a TTS model in Model settings > TTS model."),
        "unavailable_sole": (f"{old_line}.\nIts last-used audio.cpp model, {remembered}, is not on this server.\n"
                             f"The server is serving {sole}, so it will use that."),
        "unavailable_sole_from_none": (f"This project's last-used audio.cpp model, {remembered}, "
                                       f"is not on this server.\nThe server is serving {sole}, so it will use that."),
        "unavailable_sole_unbound": (f"{old_line}.\nIts last-used audio.cpp model, {remembered}, "
                                     f"is not on this server.\nThis project has been switched to {sole}, {unusable}"),
    }[case]
    assert output == f"🔔 FYI\n{expected}\n\n"


@pytest.mark.parametrize("sole", [True, False])
def test_saved_entry_missing_from_new_server_announces_replacement(monkeypatch, capsys, sole):
    # Eg, after changing the server URL: the saved entry is replaced (sole
    # entry, else unselected) with the usual model-change FYI, naming entries.
    fish = TtsModelType.require_by_id("fish_drama3_audiocpp")
    chatterbox = TtsModelType.require_by_id("chatterbox_audiocpp")
    higgs = TtsModelType.require_by_id("higgs_v3_audiocpp")
    state = make_state()
    state.project.set_tts_selection(fish.id, "fish-drama3")
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    candidates = ((chatterbox, "cb"),) if sole else ((chatterbox, "cb"), (higgs, "higgs"))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        TtsBackendKind.AUDIO_CPP, candidates=candidates))
    monkeypatch.setattr(Tts, "get_active_type", lambda: TtsModelType.get_by_id(state.project.tts_model_type))

    MenuStatus.print_block(state)
    capsys.readouterr()
    MenuStatus.show_pending_project_hints(state)
    output = text_util.strip_ansi_codes(capsys.readouterr().out)

    old_name = f"{fish.value.ui['proper_name']} (audio.cpp; fish-drama3)"
    if sole:
        assert output == (f"🔔 FYI\nThis project was last used with TTS model {old_name}.\n"
                          f"The server is serving {chatterbox.value.ui['proper_name']} (audio.cpp; cb), "
                          "so it will use that.\n\n")
    else:
        assert output == (f"🔔 FYI\nThis project was last used with TTS model {old_name},\nwhich is currently "
                          "unavailable.\nSelect a TTS model in Model settings > TTS model.\n\n")
    assert state.project.last_audio_cpp_model_id == ("cb" if sole else "fish-drama3")


def test_binding_issue_is_a_terse_status_plus_one_time_fyi(monkeypatch, capsys):
    # The status row keeps only a short qualifier while the issue persists; the
    # full explanation is shown once, and again only after the issue clears.
    from tts_audiobook_tool.tts import TtsBindingIssue
    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    state = make_state(model)
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    snapshot = RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((model, "one"), (model, "two")))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    issue = TtsBindingIssue("TTS model", "Multiple server entries match the selected TTS model (one, two). ...",
                            kind="ambiguous_entry", status="requires entry selection", server_ids=("one", "two"))
    current = {"issue": issue}

    def bind(project, **kwargs):
        Tts._binding_issue = current["issue"]
        return current["issue"]
    monkeypatch.setattr(Tts, "bind_project", bind)
    name = f"{model.value.ui['proper_name']} (audio.cpp)"
    notice = (f"🔔 FYI\nSeveral server entries match this project's TTS model, {name}:\n"
              "one, two.\nSelect one in Model settings > TTS model.\n\n")

    def draw() -> tuple[str, str]:
        MenuStatus.print_block(state)
        status = text_util.strip_ansi_codes(capsys.readouterr().out)
        MenuStatus.show_pending_project_hints(state)
        return status, text_util.strip_ansi_codes(capsys.readouterr().out)

    status, hint = draw()
    assert "(requires entry selection)" in status and "Multiple server entries" not in status
    assert hint == notice
    assert draw()[1] == ""
    current["issue"] = None
    assert draw()[1] == ""
    current["issue"] = issue
    assert draw()[1] == notice


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


@pytest.mark.parametrize("model_type, speaker_id", [
    ("custom_voice", ""), ("custom_voice", "Nobody"), ("voice_design", "Ryan"), ("base", "Nobody"),
])
def test_qwen_readiness_does_not_require_speaker_without_instance(model_type, speaker_id):
    # A missing speaker is a per-voice pre-flight problem, not project-wide readiness
    project = Project(dir_path="", voice_references=[Voice(presets={"qwen3tts_local": speaker_id})])
    set_setting(project, "qwen3_model_type", model_type)

    assert not any("speaker" in issue.short for issue in Qwen3BaseModel.get_blocking_issues(project, None))


class _StubQwen3(Qwen3BaseModel):
    def __init__(self, speakers):
        self._speakers = speakers

    def kill(self):
        pass

    def generate_using_project(self, *args, **kwargs):
        raise AssertionError("not generating in readiness test")

    @property
    def model_type(self):
        return "custom_voice"

    @property
    def supported_speakers(self):
        return self._speakers


@pytest.mark.parametrize("speakers, presets, invalid", [
    (["Ryan", "Vivian"], ["Ryan", ""], ""),
    (["Ryan", "Vivian"], ["Ryan", "Nobody"], "Voice 2 (Nobody)"),
    (["Ryan"], ["Nobody"], ""),  # single-speaker checkpoints always use their only speaker
])
def test_qwen_custom_voice_readiness_flags_speakers_the_checkpoint_lacks(speakers, presets, invalid):
    project = Project(dir_path="", voice_references=[
        Voice(presets={"qwen3tts_local": preset} if preset else {}) for preset in presets
    ])
    set_setting(project, "qwen3_model_type", "custom_voice")

    issues = Qwen3BaseModel.get_blocking_issues(project, _StubQwen3(speakers))

    flagged = [issue for issue in issues if issue.short == "valid speaker id"]
    assert bool(flagged) == bool(invalid)
    if invalid:
        assert invalid in flagged[0].verbose


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
