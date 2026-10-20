"""CosyVoice3 (SGL-Omni) Mode decisions, checked at every entry point together.

The SGL-Omni server infers its prompt mode from which request fields are
present and rejects a transcript combined with instructions. The app-side Mode
(never sent) decides, through `CosyVoice3SglOmniBehavior`, whether the
transcript is required/sent and whether the selected voice's instructions are
sent. These tests drive the real menu, readiness, voice pre-flight, server
startup check and request builder from one project, mirroring
`test_cosyvoice3_audio_cpp_entry_points.py`.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.menus.menu_util import MenuUtil, get_string_from
from tts_audiobook_tool.menus.model.model_configured_sgl_omni_menu import ModelConfiguredSglOmniMenu
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.server.server import is_missing_required_transcript
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.sgl_omni_behavior import SglOmniModelBehavior, get_sgl_omni_behavior
from tts_audiobook_tool.tts_models.sgl_omni_behavior_cosyvoice3 import CosyVoice3SglOmniBehavior
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredModelSupport, SglOmniBackendAdapter
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

MODEL_ID = "cosyvoice3_sglomni"
HF_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TRANSCRIPT = "Reference transcript."


@pytest.fixture
def definition(monkeypatch):
    # conftest's initialize_tts_type_for_tests restores Tts, catalog and registry state.
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.REMOTE_CLIENT))
    Tts.init_local_model_type()
    model_type = TtsModelType.require_by_id(MODEL_ID)
    Tts._type = model_type
    Tts._selected_server_model_id = HF_ID
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.SGL_OMNI, candidates=((model_type, HF_ID),)))
    return Tts.get_configured_definition(model_type)


def _project(tmp_path, mode: str | None, transcript: str = "", instructions: str = "Speak warmly.") -> Project:
    sf.write(tmp_path / "a.flac", np.full(2400, .2), 24000, format="FLAC")
    project = Project(dir_path=str(tmp_path), tts_model_type=MODEL_ID,
                      voice_references=[{"file_name": "a.flac", "transcript": transcript}])
    if mode is not None:
        project.set_model_setting(MODEL_ID, "template_name", mode)
    project.voice_references[0].instructions[MODEL_ID] = instructions
    return project


def _send(monkeypatch, definition, project, **kwargs) -> tuple[object, list[dict]]:
    sent: list[dict] = []
    sound = Sound(np.zeros(1, dtype=np.float32), 24000)
    monkeypatch.setattr(SglOmniUtil, "get_base_url", staticmethod(lambda: "http://example.test"))
    monkeypatch.setattr(SoundUtil, "make_audio_data_uri", staticmethod(lambda path: "data:reference"))
    monkeypatch.setattr(SglOmniUtil, "generate_concurrent",
                        lambda url, payloads, **_: sent.extend(payloads) or [sound for _ in payloads])
    monkeypatch.setattr(SglOmniUtil, "generate_streaming",
                        lambda url, payload, **_: sent.append(payload) or sound)
    adapter = SglOmniBackendAdapter(definition, ConfiguredModelSupport(definition))
    return adapter.generate_using_project(project, ["hi"], **kwargs), sent


def test_definition_declares_mode_choice_and_voice_instructions(definition):
    assert definition.voice_instructions
    assert set(definition.choices) == {"template_name"}
    choice = definition.choices["template_name"]
    assert choice.default == "zero_shot"
    assert [c.value for c in choice.choices] == ["zero_shot", "cross_lingual", "instruct"]
    # The Mode is app-side only: it never becomes a request parameter.
    assert "template_name" not in definition.parameters
    assert "template_name" not in definition.request_defaults
    assert isinstance(ConfiguredModelSupport(definition).behavior, CosyVoice3SglOmniBehavior)
    assert "instructions_message" in definition.spec.ui


@pytest.mark.parametrize("mode, uses_transcript", [
    ("zero_shot", True),
    ("cross_lingual", False),
    ("instruct", False),
])
def test_mode_decisions_agree_across_entry_points(definition, monkeypatch, tmp_path, mode, uses_transcript):
    # A valid voice sample with an empty transcript: only Zero-shot reads the
    # transcript, so only Zero-shot may block on it, invoke Whisper or refuse
    # server startup.
    project = _project(tmp_path, mode)
    state = SimpleNamespace(project=project, prefs=SimpleNamespace(stt_variant=None, stt_config={}))
    support = ConfiguredModelSupport(definition)

    # Instructions are per-voice data, never a global menu control.
    labels = [get_string_from(state, item.label) for item in ModelConfiguredSglOmniMenu.make_items(state, definition)]
    assert labels[0].startswith("Mode ")
    assert not any("instructions" in label.lower() for label in labels)
    assert support.can_instructions(project) is (mode == "instruct")
    assert support.expects_voice_instructions(project) is (mode == "instruct")

    assert [issue for issue in support.get_blocking_issues(project) if issue.short != "SGL-Omni server"] == []

    assert Tts.requires_reference_transcript(project) is uses_transcript
    assert VoiceMenuShared.get_voice_problems(state) == (
        ["Voice file a.flac has no transcript"] if uses_transcript else [])

    transcribed = []
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text",
                        lambda *args: transcribed.append(args) or ("", "stt broke"))
    monkeypatch.setattr(voice_menu_shared, "printt", lambda *_: None)
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", lambda *_: None)
    assert VoiceMenuShared.validate_voices(state) is not uses_transcript
    assert bool(transcribed) is uses_transcript

    assert is_missing_required_transcript(project) is uses_transcript


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["zero_shot", "cross_lingual", "instruct"])
def test_request_never_combines_transcript_and_instructions(definition, monkeypatch, tmp_path, mode, streaming):
    # Both a transcript and instructions are stored; the Mode picks at most one.
    project = _project(tmp_path, mode, transcript=TRANSCRIPT, instructions="  Speak warmly.  ")
    kwargs = {"on_stream_chunk": lambda _: None} if streaming else {}
    result, sent = _send(monkeypatch, definition, project, **kwargs)
    assert not isinstance(result, str)
    assert len(sent) == 1
    payload = sent[0]
    assert "template_name" not in payload
    assert payload["references"] == [
        {"audio_path": "data:reference", "text": TRANSCRIPT} if mode == "zero_shot" else {"audio_path": "data:reference"}]
    if mode == "instruct":
        assert payload["instructions"] == "Speak warmly."  # stripped, as the server does
    else:
        assert "instructions" not in payload


def test_instruct_with_blank_instructions_sends_neither(definition, monkeypatch, tmp_path):
    project = _project(tmp_path, "instruct", transcript=TRANSCRIPT, instructions="   ")
    result, sent = _send(monkeypatch, definition, project)
    assert not isinstance(result, str)
    assert "instructions" not in sent[0] and sent[0]["references"] == [{"audio_path": "data:reference"}]


def test_instructions_follow_the_selected_voice(definition, monkeypatch, tmp_path):
    for name in ("a.flac", "b.flac"):
        sf.write(tmp_path / name, np.full(2400, .2), 24000, format="FLAC")
    project = Project(dir_path=str(tmp_path), tts_model_type=MODEL_ID, voice_references=[
        {"file_name": "a.flac", "transcript": ""}, {"file_name": "b.flac", "transcript": ""}])
    project.set_model_setting(MODEL_ID, "template_name", "instruct")
    project.voice_references[0].instructions[MODEL_ID] = "Speak warmly."
    project.voice_references[1].instructions[MODEL_ID] = "Speak quickly."
    _, sent = _send(monkeypatch, definition, project, voice_selection_index=1)
    assert sent[0]["instructions"] == "Speak quickly."


def test_default_mode_preserves_existing_zero_shot_request(definition, monkeypatch, tmp_path):
    # Unset Mode reads as Zero-shot: existing projects send what they did before.
    project = _project(tmp_path, None, transcript=TRANSCRIPT)
    assert project.get_model_setting(MODEL_ID, "template_name") == "zero_shot"
    _, sent = _send(monkeypatch, definition, project)
    assert sent[0]["references"] == [{"audio_path": "data:reference", "text": TRANSCRIPT}]
    assert "instructions" not in sent[0]


def test_invalid_mode_blocks_and_falls_back_to_the_static_requirements(definition, tmp_path):
    project = _project(tmp_path, "sft")
    support = ConfiguredModelSupport(definition)
    assert [issue.short for issue in support.get_blocking_issues(project)] == ["template_name"]
    assert Tts.requires_reference_transcript(project) is True
    assert support.can_instructions(project) is True
    assert support.expects_voice_instructions(project) is False


def test_mode_item_saves_via_option_submenu_and_default_resets(definition, monkeypatch, tmp_path):
    project = _project(tmp_path, None)
    monkeypatch.setattr(Project, "save", lambda self: "")
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_menu_shared.print_feedback", lambda *_, **__: None)
    state = SimpleNamespace(project=project)
    item = ModelConfiguredSglOmniMenu.make_items(state, definition)[0]
    assert "Zero-shot" in get_string_from(state, item.label)
    picks = iter(["instruct", "zero_shot"])
    offered = {}

    def fake_options_menu(**kwargs):
        offered.update(kwargs)
        kwargs["on_select"](next(picks))

    monkeypatch.setattr(MenuUtil, "options_menu", fake_options_menu)
    item.handler(state, item)
    assert offered["labels"] == ["Zero-shot", "Cross-lingual", "Instruct"]
    assert project.get_model_setting(MODEL_ID, "template_name") == "instruct"
    assert "Instruct" in get_string_from(state, item.label)
    item.handler(state, item)
    stored = project.model_settings.models.get(MODEL_ID, {}).get("parameters", {})
    assert "template_name" not in stored  # Selecting the default leaves the setting unset.


def test_behavior_registry_and_required_parameter_check(definition):
    assert type(get_sgl_omni_behavior(replace(definition, spec=definition.spec._replace(id="custom")))) \
        is SglOmniModelBehavior
    with pytest.raises(ValueError, match="requires catalog parameter 'template_name'"):
        CosyVoice3SglOmniBehavior(replace(definition, choices={}))
