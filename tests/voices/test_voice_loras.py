"""`Voice.loras` storage (per-voice LoRA adapters, keyed by TTS model id) and its project helpers."""
import json

import pytest

from tts_audiobook_tool.app_types.voice import LORA_STRENGTH_MAX, LORA_STRENGTH_MIN, Voice, VoiceLora
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil

BREEZE = "breeze_tts_2_local"
TARGET = "/runs/fw-leda/lora/checkpoint-step-000500"


def test_empty_map_and_default_strength_are_not_written():
    assert "loras" not in Voice("a.flac").to_dict()
    data = Voice("a.flac", loras={BREEZE: VoiceLora(TARGET)}).to_dict()
    assert data["loras"] == {BREEZE: {"target": TARGET}}


def test_loras_round_trip():
    voice = Voice("a.flac", loras={BREEZE: VoiceLora(TARGET, 0.75), "other": VoiceLora("owner/repo")})
    assert Voice.from_dict(voice.to_dict()) == voice
    assert voice.lora_for(BREEZE) == VoiceLora(TARGET, 0.75)
    assert voice.lora_for("missing") is None


@pytest.mark.parametrize("stored", [None, 3, ["x"], "owner/repo"])
def test_malformed_map_loads_empty(stored):
    voice = Voice.from_dict({"file_name": "a.flac", "loras": stored})
    assert voice is not None and voice.loras == {}


def test_bad_entries_are_dropped_and_strength_is_defaulted_or_clamped():
    # A bad entry loses only itself; a bad strength keeps the LoRA at the default strength
    voice = Voice.from_dict({"file_name": "a.flac", "loras": {
        "no_target": {"strength": 0.5},
        "empty_target": {"target": "  "},
        "not_object": "owner/repo",
        "bad_strength": {"target": "a", "strength": "high"},
        "bool_strength": {"target": "b", "strength": True},
        "too_low": {"target": "c", "strength": 0.0},
        "too_high": {"target": "d", "strength": 9},
    }})
    assert voice is not None
    assert voice.loras == {
        "bad_strength": VoiceLora("a"), "bool_strength": VoiceLora("b"),
        "too_low": VoiceLora("c", LORA_STRENGTH_MIN), "too_high": VoiceLora("d", LORA_STRENGTH_MAX),
    }


@pytest.mark.parametrize("kwargs", [{"target": ""}, {"target": "a", "strength": 5.0}, {"target": "a", "strength": True}])
def test_voice_lora_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        VoiceLora(**kwargs)


def test_voice_does_not_alias_caller_dict_and_stays_hashable():
    source = {BREEZE: VoiceLora(TARGET)}
    voice = Voice("a.flac", loras=source)
    source.clear()
    assert voice.lora_for(BREEZE) is not None
    hash(voice)


def test_set_voice_lora_changes_only_that_model_and_reverts_on_failed_save(monkeypatch, tmp_path):
    # Exercise real storage in a disposable project, including rollback after a failed save.
    project = Project(dir_path=str(tmp_path), tts_model_type=BREEZE,
                      voice_references=[Voice("a.flac", loras={"other": VoiceLora("x")})])
    stored_path = tmp_path / "project.json"
    assert ProjectVoiceUtil.set_voice_lora_at_index_and_save(project, 0, BREEZE, VoiceLora(TARGET)) == ""
    assert project.voice_references[0].loras == {"other": VoiceLora("x"), BREEZE: VoiceLora(TARGET)}
    assert json.loads(stored_path.read_text())["voice_references"][0]["loras"] == {
        "other": {"target": "x"}, BREEZE: {"target": TARGET},
    }
    assert ProjectVoiceUtil.set_voice_lora_at_index_and_save(project, 0, BREEZE, None) == ""
    assert project.voice_references[0].loras == {"other": VoiceLora("x")}
    assert json.loads(stored_path.read_text())["voice_references"][0]["loras"] == {"other": {"target": "x"}}
    saved_bytes = stored_path.read_bytes()

    monkeypatch.setattr(Project, "save", lambda self: "disk full")
    assert ProjectVoiceUtil.set_voice_lora_at_index_and_save(project, 0, BREEZE, VoiceLora(TARGET)) == "disk full"
    assert project.voice_references[0].loras == {"other": VoiceLora("x")}
    assert stored_path.read_bytes() == saved_bytes


def test_lora_is_active_only_for_a_model_that_takes_per_voice_loras():
    # Mira stores the entry (it's keyed by model id) but doesn't use it
    voice = Voice(loras={BREEZE: VoiceLora(TARGET), "mira_local": VoiceLora(TARGET)})
    breeze = Project(tts_model_type=BREEZE, voice_references=[voice])
    mira = Project(tts_model_type="mira_local", voice_references=[voice])
    assert ProjectVoiceUtil.active_voice_lora(breeze, voice) == VoiceLora(TARGET)
    assert ProjectVoiceUtil.active_voice_lora(mira, voice) is None


def test_a_voice_with_only_a_lora_has_a_value():
    # A sample-less, instruction-less voice is "builtin" unless its LoRA fixes the speaker
    project = Project(tts_model_type=BREEZE, voice_references=[Voice(loras={BREEZE: VoiceLora(TARGET)}), Voice()])
    with_lora, without = (ProjectVoiceUtil.resolve_voice_source(project, v) for v in project.voice_references)
    assert with_lora.kind == "lora" and with_lora.has_value and "fw-leda-checkpoint" in with_lora.label
    assert without.kind == "builtin"
    # A sample still takes precedence as the voice's source
    sampled = Voice("a.flac", "Hi.", loras={BREEZE: VoiceLora(TARGET)})
    assert ProjectVoiceUtil.resolve_voice_source(project, sampled).kind == "sample"
