"""Per-voice volume (`Voice.volume`) storage and lookup."""
import math

import pytest

from tts_audiobook_tool.app_types.voice import Voice, volume_to_db
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.project_support.voice_spatial_policy import VoiceSpatialPolicy


def test_default_volume_is_not_written_and_other_values_round_trip():
    assert "volume" not in Voice("a.flac").to_dict()
    data = Voice("a.flac", volume=70).to_dict()
    assert data["volume"] == 70
    assert Voice.from_dict(data) == Voice("a.flac", volume=70)


@pytest.mark.parametrize("stored, loaded", [
    (70, 70), (30, 30), (12, 30), (250, 100), (64.6, 65),
    ("70", 100), (True, 100), (None, 100), (math.nan, 100), (math.inf, 100),
])
def test_stored_volume_is_clamped_or_default_when_malformed(stored, loaded):
    voice = Voice.from_dict({"file_name": "a.flac", "volume": stored})
    assert voice is not None and voice.volume == loaded


def test_missing_volume_loads_default():
    voice = Voice.from_dict({"file_name": "a.flac"})
    assert voice is not None and voice.volume == 100


@pytest.mark.parametrize("volume", [29, 101, 70.0, True])
def test_voice_rejects_invalid_volume(volume):
    with pytest.raises(ValueError):
        Voice("a.flac", volume=volume)


def test_volume_is_amplitude_percent():
    assert volume_to_db(100) == 0
    assert volume_to_db(50) == pytest.approx(-6.02, abs=0.01)


def test_set_volume_at_index_edits_only_that_voice(monkeypatch):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac"}, {"file_name": "b.flac", "pan": 0.3},
    ])
    monkeypatch.setattr(Project, "save", lambda _: "")
    assert ProjectVoiceUtil.set_voice_volume_at_index_and_save(project, 1, 60) == ""
    assert [v.volume for v in project.voice_references] == [100, 60]
    assert project.voice_references[1].pan == 0.3
    with pytest.raises(IndexError):
        ProjectVoiceUtil.set_voice_volume_at_index_and_save(project, 2, 60)


@pytest.mark.parametrize("index, volume", [(-1, 80), (0, 80), (1, 50), (99, 80)])
def test_segments_use_their_voices_volume_with_generation_fallback(index, volume):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        Voice("a.flac", volume=80), Voice("b.flac", volume=50),
    ])
    policy = VoiceSpatialPolicy.from_project(project)
    assert policy.volume_for_voice_index(index) == volume
    assert policy.volume_db_for_voice_index(index) == pytest.approx(volume_to_db(volume))


@pytest.mark.parametrize("volumes, summary", [([70, 70], 70), ([70, 100], None), ([], 100)])
def test_volume_summary(volumes, summary):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        Voice(f"{i}.flac", volume=volume) for i, volume in enumerate(volumes)
    ])
    assert VoiceSpatialPolicy.from_project(project).volume_summary == summary
