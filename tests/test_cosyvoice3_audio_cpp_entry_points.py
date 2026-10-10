"""CosyVoice3 (audio.cpp) Mode decisions, checked at every entry point together.

The behavior layer decides per Mode whether the transcript is required and
whether Instructions apply. These tests drive the real menu, readiness,
voice pre-flight, server startup check and request builder from one project,
so an entry point that bypasses the behavior (eg reading the static transcript
binding) shows up as a disagreement.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.server.server import is_missing_required_transcript
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport

MODEL_ID = "cosyvoice3_audiocpp"
TRANSCRIPT_ERROR = "Voice clone transcript required when a voice clone sample is supplied"


@pytest.mark.parametrize("mode, uses_transcript", [
    ("zero_shot", True),
    ("cross_lingual", False),
    ("instruct", False),
])
def test_mode_decisions_agree_across_entry_points(monkeypatch, tmp_path, mode, uses_transcript):
    # A valid voice sample with an empty transcript: only Zero-shot reads the
    # transcript, so only Zero-shot may block on it, invoke Whisper, refuse
    # server startup or refuse to build the request.
    sf.write(tmp_path / "a.flac", np.full(2400, .2), 24000, format="FLAC")
    project = Project(dir_path=str(tmp_path), tts_model_type=MODEL_ID,
                      voice_references=[{"file_name": "a.flac", "transcript": ""}])
    project.set_model_setting(MODEL_ID, "template_name", mode)
    project.set_model_setting(MODEL_ID, "instruction", "Speak warmly.")
    state = SimpleNamespace(project=project, prefs=SimpleNamespace(stt_variant=None, stt_config={}))
    definition = Tts._audio_cpp_definitions[MODEL_ID]
    support = AudioCppModelSupport(definition)

    # Menu: Instructions appear only in Instruct mode.
    labels = [get_string_from(state, item.label) for item in ModelAudioCppMenu.make_items(state, definition)]
    assert any(label.startswith("Instructions ") for label in labels) == (mode == "instruct")

    # Readiness: no settings blockers in any mode (Instruct has its instruction).
    assert [issue for issue in support.get_blocking_issues(project) if issue.short != "audio.cpp server"] == []

    # The shared runtime query, and each entry point that consumes it.
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

    # Request: Zero-shot refuses without a transcript; other modes send none,
    # and only Instruct sends the stored instruction.
    sent = []
    monkeypatch.setattr(AudioCppUtil, "generate", lambda url, payload, print_request=False: sent.append(payload)
                        or SimpleNamespace(data=np.zeros(1, dtype=np.float32), sr=24000))
    result = AudioCppBackendAdapter(definition, support, "cosyvoice3-q8").generate_using_project(project, ["hi"])
    if uses_transcript:
        assert result == TRANSCRIPT_ERROR and not sent
    else:
        assert not isinstance(result, str)
        assert "reference_text" not in sent[0]
        assert sent[0]["options"]["template_name"] == mode
        assert ("instruction" in sent[0]["options"]) == (mode == "instruct")


def test_invalid_mode_falls_back_to_the_static_transcript_requirement(tmp_path):
    # With an unreadable Mode the behavior cannot be asked; the catalog's static
    # flag (transcript required) applies, and readiness reports the bad value.
    project = Project(dir_path=str(tmp_path), tts_model_type=MODEL_ID)
    project.set_model_setting(MODEL_ID, "template_name", "sft")
    assert Tts.requires_reference_transcript(project) is True
