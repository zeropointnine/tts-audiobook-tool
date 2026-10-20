"""Per-voice stereo position (`Voice.pan`) storage."""
import math

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


def test_centered_pan_is_not_written_and_off_center_round_trips():
    assert "pan" not in Voice("a.flac").to_dict()
    data = Voice("a.flac", pan=-0.3).to_dict()
    assert data["pan"] == -0.3
    assert Voice.from_dict(data) == Voice("a.flac", pan=-0.3)


@pytest.mark.parametrize("stored, loaded", [
    (0.5, 0.5), (-1, -1.0), (1.7, 1.0), (-3, -1.0),
    ("0.5", 0.0), (True, 0.0), (None, 0.0), (math.nan, 0.0), (math.inf, 0.0),
])
def test_stored_pan_is_clamped_or_centered_when_malformed(stored, loaded):
    voice = Voice.from_dict({"file_name": "a.flac", "pan": stored})
    assert voice is not None and voice.pan == loaded


def test_missing_pan_loads_centered():
    voice = Voice.from_dict({"file_name": "a.flac"})
    assert voice is not None and voice.pan == 0.0


@pytest.mark.parametrize("pan", [1.01, -2, math.nan])
def test_voice_rejects_invalid_pan(pan):
    with pytest.raises(ValueError):
        Voice("a.flac", pan=pan)


def test_set_pan_at_index_edits_only_that_voice(monkeypatch):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac"}, {"file_name": "b.flac", "room_reverb": "small_drr22"},
    ])
    monkeypatch.setattr(Project, "save", lambda _: "")
    assert ProjectVoiceUtil.set_voice_pan_at_index_and_save(project, 1, 0.4) == ""
    assert [v.pan for v in project.voice_references] == [0.0, 0.4]
    assert project.voice_references[1].room_reverb.id == "small_drr22"
    with pytest.raises(IndexError):
        ProjectVoiceUtil.set_voice_pan_at_index_and_save(project, 2, 0.4)
