"""
Sample-less voices: a project always has at least one voice; Add creates a
default voice; a voice's sample is set (or replaced) from its own submenu.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from tts_audiobook_tool.app_types import RoomReverb, Sound, SttVariant
from tts_audiobook_tool.app_types.voice import Voice, VoiceCrop
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.server import server
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def model(model_id="glm_local") -> TtsModelType:
    return TtsModelType.require_by_id(model_id)


def sound(seconds=3.0) -> Sound:
    return Sound(np.full(int(APP_SAMPLE_RATE * seconds), 0.1, dtype=np.float32), APP_SAMPLE_RATE)


def make_project(tmp_path, voices, model_id="glm_local") -> Project:
    project = Project(dir_path=str(tmp_path), tts_model_type=model_id, voice_references=voices)
    return project


# --- ProjectVoiceUtil ---

def test_remove_and_clear_leave_one_default_voice(tmp_path):
    project = make_project(tmp_path, [Voice("a.flac", "t", pan=0.5)])
    ProjectVoiceUtil.remove_voice_at_index_and_save(project, model(), 0)
    assert project.voice_references == [Voice()]
    project.voice_references = [Voice("a.flac"), Voice("b.flac")]
    ProjectVoiceUtil.clear_voice_and_save(project, model())
    assert project.voice_references == [Voice()]
    project.set_model_setting("glm_local", "file_name", [])
    assert project.voice_references == [Voice()]


def test_add_voice_appends_default_voice(tmp_path):
    project = make_project(tmp_path, [Voice("a.flac", "t")])
    assert ProjectVoiceUtil.add_voice_and_save(project) == (1, "")
    assert project.voice_references == [Voice("a.flac", "t"), Voice()]


def test_append_import_replaces_sole_default_voice_only(tmp_path):
    project = make_project(tmp_path, [])
    assert not ProjectVoiceUtil.set_voice_and_save(project, sound(), "a", "A", model(), append=True)
    assert [v.file_name for v in project.voice_references] == ["a.flac"]
    # A sole voice with values is not a placeholder: append after it.
    project.voice_references = [Voice(pan=0.5)]
    assert not ProjectVoiceUtil.set_voice_and_save(project, sound(), "b", "B", model(), append=True)
    assert [v.file_name for v in project.voice_references] == ["", "b.flac"]


def test_used_voice_file_names_ignore_sample_less_voices(tmp_path):
    project = make_project(tmp_path, [Voice(), Voice("a.flac")])
    assert ProjectVoiceUtil.get_used_voice_file_names(project, None) == {"a.flac"}
    assert ProjectVoiceUtil.has_any_voice_sample(project)
    assert ProjectVoiceUtil.get_voice_sample_count(project, model()) == 1
    project.voice_references = [Voice(), Voice()]
    assert not ProjectVoiceUtil.has_any_voice_sample(project)
    assert not any(ProjectVoiceUtil.resolve_voice_source(project, voice).has_value
                   for voice in project.voice_references)


def test_set_sample_resets_trim_clears_transcript_and_keeps_effects(tmp_path):
    crops = tmp_path / "voice" / "crops"
    crops.mkdir(parents=True)
    (crops / "old-trim.flac").write_bytes(b"trim")
    voice = Voice(
        "old.flac", "old text", VoiceCrop("old-trim.flac", 0.0, 3.0, "trim text"),
        room_reverb=RoomReverb.SMALL_DRR17, pan=-0.3, instructions={"glm_local": "warm"},
    )
    project = make_project(tmp_path, [voice])
    assert ProjectVoiceUtil.set_voice_sample_at_index_and_save(project, 0, sound(), "new", model()) == ""
    assert project.voice_references == [replace(voice, file_name="new.flac", transcript="", crop=None)]
    assert (tmp_path / "voice" / "new.flac").is_file()
    assert not (crops / "old-trim.flac").exists()


@pytest.mark.parametrize("shared", ["none", "crop", "original", "secondary"])
def test_clear_sample_keeps_slot_settings_and_other_inputs(tmp_path, shared):
    crops = tmp_path / "voice" / "crops"
    crops.mkdir(parents=True)
    source = tmp_path / "voice" / "old.flac"
    source.write_bytes(b"original")
    trimmed = crops / "trim.flac"
    trimmed.write_bytes(b"trim")
    voice = Voice(
        "old.flac", "old text", VoiceCrop("trim.flac", 0.0, 3.0, "trim text"),
        room_reverb=RoomReverb.SMALL_DRR17, pan=-0.3, presets={"pocket_local": "alba"},
        instructions={"glm_local": "warm", "pocket_local": "slow"},
    )
    other = Voice("other.flac", "other text")
    if shared == "crop":
        other = replace(other, crop=voice.crop)
    elif shared == "original":
        other = replace(other, file_name="crops/trim.flac")
    project = make_project(tmp_path, [voice, other])
    if shared == "secondary":
        project.set_model_setting("indextts2_local", "emo_voice", "crops/trim.flac")
    assert ProjectVoiceUtil.clear_voice_sample_at_index_and_save(project, 0) == ""
    assert project.voice_references == [replace(voice, file_name="", transcript="", crop=None), other]
    assert source.read_bytes() == b"original"
    assert trimmed.exists() == (shared != "none")
    # Saved data retains the sample-less slot and all its other settings.
    loaded = Project.model_validate_json((tmp_path / "project.json").read_text())
    assert loaded.voice_references == project.voice_references


def test_clear_sample_save_failure_preserves_voice_and_trim(tmp_path, monkeypatch):
    crops = tmp_path / "voice" / "crops"
    crops.mkdir(parents=True)
    trimmed = crops / "trim.flac"
    trimmed.write_bytes(b"trim")
    voice = Voice("old.flac", "text", VoiceCrop("trim.flac", 0.0, 3.0, "trim text"))
    project = make_project(tmp_path, [voice])
    original = project.voice_references
    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    assert ProjectVoiceUtil.clear_voice_sample_at_index_and_save(project, 0) == "disk full"
    assert project.voice_references is original
    assert trimmed.read_bytes() == b"trim"


@pytest.mark.parametrize("index", [-1, 1])
def test_clear_sample_invalid_index_does_not_save(tmp_path, monkeypatch, index):
    project = make_project(tmp_path, [Voice("old.flac")])
    monkeypatch.setattr(Project, "save", lambda _: pytest.fail("must not save"))
    with pytest.raises(IndexError):
        ProjectVoiceUtil.clear_voice_sample_at_index_and_save(project, index)
    assert project.voice_references == [Voice("old.flac")]


def test_set_sample_reuses_own_name_but_not_other_voices_names(tmp_path):
    project = make_project(tmp_path, [Voice("a.flac", "t"), Voice()])
    # Voice 2 cannot take voice 1's name
    assert ProjectVoiceUtil.set_voice_sample_at_index_and_save(project, 1, sound(), "a", model()) == ""
    assert [v.file_name for v in project.voice_references] == ["a.flac", "a_2.flac"]
    # Re-setting a voice's own name overwrites in place
    assert ProjectVoiceUtil.set_voice_sample_at_index_and_save(project, 0, sound(), "a", model()) == ""
    assert [v.file_name for v in project.voice_references] == ["a.flac", "a_2.flac"]


def test_set_sample_keeps_pocket_predefined_voice(tmp_path):
    project = make_project(tmp_path, [Voice(presets={"pocket_local": "alba"})], model_id="pocket_local")
    assert ProjectVoiceUtil.set_voice_sample_at_index_and_save(project, 0, sound(), "a", model("pocket_local")) == ""
    voice = project.voice_references[0]
    assert voice.file_name == "a.flac" and voice.presets == {"pocket_local": "alba"}


def test_set_sample_save_failure_restores_voice(tmp_path, monkeypatch):
    project = make_project(tmp_path, [Voice("a.flac", "t")])
    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    assert ProjectVoiceUtil.set_voice_sample_at_index_and_save(project, 0, sound(), "b", model()) == "disk full"
    assert project.voice_references == [Voice("a.flac", "t")]


def test_server_transcript_check_ignores_sample_less_first_voice(monkeypatch):
    monkeypatch.setattr(Tts, "requires_reference_transcript", lambda _: True)
    project = Project(tts_model_type="glm_local")
    assert server.is_missing_required_transcript(project) is False
    project.voice_references = [Voice("a.flac", "")]
    assert server.is_missing_required_transcript(project) is True


# --- Set voice sample (menu flow) ---

def setup_flow(tmp_path, monkeypatch, voices, *, sidecar="", stt=("", "")):
    path = tmp_path / "in" / "clip.wav"
    path.parent.mkdir()
    path.write_bytes(b"sound")
    if sidecar:
        path.with_suffix(".txt").write_text(sidecar, encoding="utf-8")
    project = make_project(tmp_path, voices)
    prefs = SimpleNamespace(last_voice_dir="", save=lambda: None, stt_variant=SttVariant.DISABLED, import_max_silence_gap=None)
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    events: list = []
    monkeypatch.setattr(VoiceMenuShared, "ask_voice_file", lambda *_: str(path))
    monkeypatch.setattr(SoundFileUtil, "load", lambda *_: sound())
    monkeypatch.setattr(voice_menu_shared.SoundPipeline, "apply_voice_clone_post_processing", lambda value, **_: value)
    monkeypatch.setattr(voice_menu_shared.PlaySoundUtil, "play_sound_async", lambda *_: None)
    monkeypatch.setattr(voice_menu_shared.hints, "show_hint_if_necessary", lambda *_, **__: None)
    monkeypatch.setattr(voice_menu_shared.hints, "show_hints_if_necessary", lambda *_, **__: None)
    monkeypatch.setattr(voice_menu_shared.ask, "ask_enter_to_continue", lambda *_: events.append(("enter",)))
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", lambda message: events.append(("error", message)))
    monkeypatch.setattr(voice_menu_shared, "print_feedback", lambda *args, **_: events.append(("feedback", args[0])))

    def transcribe(_state, _sound):
        # The transcript is already cleared when transcription starts.
        events.append(("transcribing", project.voice_references[0].transcript))
        return stt

    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", transcribe)
    return state, project, events


def test_set_sample_fills_transcript_from_speech_to_text(tmp_path, monkeypatch):
    state, project, events = setup_flow(
        tmp_path, monkeypatch, [Voice("old.flac", "old text", pan=0.5)], stt=("New words.", ""),
    )
    VoiceMenuShared.ask_and_set_voice_sample(state, model(), 0)
    assert project.voice_references == [Voice("clip.flac", "New words.", pan=0.5)]
    assert events == [("transcribing", ""), ("feedback", "Voice file saved"), ("enter",)]


def test_set_sample_uses_sidecar_transcript(tmp_path, monkeypatch):
    state, project, events = setup_flow(tmp_path, monkeypatch, [Voice()], sidecar="Sidecar words.")
    VoiceMenuShared.ask_and_set_voice_sample(state, model(), 0)
    assert project.voice_references == [Voice("clip.flac", "Sidecar words.")]
    # A sidecar transcript skips speech-to-text but still waits for Enter,
    # like a transcribed one, so its printed text stays readable
    assert events == [("feedback", "Voice file saved"), ("enter",)]


def test_set_sample_keeps_sample_when_speech_to_text_fails(tmp_path, monkeypatch):
    state, project, events = setup_flow(
        tmp_path, monkeypatch, [Voice("old.flac", "old text")], stt=("", "STT unavailable"),
    )
    VoiceMenuShared.ask_and_set_voice_sample(state, model(), 0)
    assert project.voice_references == [Voice("clip.flac", "")]
    assert events[-1] == ("error", "Voice file saved, but its transcript is missing: STT unavailable")
    assert VoiceMenuShared.is_voice_transcript_missing(project, model(), 0)


def test_set_sample_validation_failure_saves_nothing(tmp_path, monkeypatch):
    state, project, events = setup_flow(tmp_path, monkeypatch, [Voice("old.flac", "old text")])
    monkeypatch.setattr(SoundFileUtil, "load", lambda *_: sound(1.0))
    monkeypatch.setattr(Project, "save", lambda _: pytest.fail("must not save"))
    VoiceMenuShared.ask_and_set_voice_sample(state, model(), 0)
    assert project.voice_references == [Voice("old.flac", "old text")]
    assert events == [("error", "Sound file must be at least 2 seconds")]


@pytest.mark.parametrize("model_id", ["dots_local", "breeze_tts_2_audiocpp"])
def test_voice_tag_of_sample_less_voice_among_sampled_ones_is_not_the_primary(model_id):
    project = Project(tts_model_type=model_id, voice_references=[Voice("suzie.flac", "t"), Voice()])
    assert Tts.get_voice_tag_for_selection_index(project, 0) == "suzie"
    assert Tts.get_voice_tag_for_selection_index(project, 1) == "none"
    # The project itself is untouched
    assert project.voice_references == [Voice("suzie.flac", "t"), Voice()]


@pytest.mark.parametrize("model_id, flagged", [("glm_local", True), ("moss_local", False)])
def test_missing_transcript_is_flagged_only_when_the_model_requires_one(tmp_path, monkeypatch, model_id, flagged):
    from tts_audiobook_tool import text_util
    project = make_project(tmp_path, [Voice("a.flac", "")], model_id)
    monkeypatch.setattr(VoiceMenuShared, "get_voice_sample_duration", lambda *_: None)
    label, _ = VoiceMenuShared.make_voice_sample_item_label(project, model(model_id), 0)
    assert ("missing transcript" in text_util.strip_ansi_codes(label)) == flagged
