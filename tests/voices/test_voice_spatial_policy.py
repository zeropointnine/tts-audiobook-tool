from dataclasses import replace
import pytest

from tts_audiobook_tool.app_types import RoomReverb, RoomReverbStereo
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.voice_spatial_policy import VoiceSpatialPolicy


def project(voice_count=2,
            reverb=RoomReverb.OFF, spread=RoomReverbStereo.MONO,
            model="chatterbox_local"):
    # Every voice gets the same room, so the output's room doesn't depend on voice
    result = Project(
        tts_model_type=model,
        voice_references=[Voice(f"voice{i}.flac", room_reverb=reverb) for i in range(voice_count)],
    )
    result.room_reverb_stereo = spread
    return result


# Uniform rooms make voice identity irrelevant, every non-OFF reverb and every stereo spread
# behave alike, and voice counts >= 1 behave alike, so these cover every class.
@pytest.mark.parametrize("voice_count,reverb,spread,channels", [
    pytest.param(0, RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, 1,
                 id="no-voices-means-no-reverb"),
    pytest.param(1, RoomReverb.OFF, RoomReverbStereo.ALL_W100, 1,
                 id="reverb-off"),
    pytest.param(1, RoomReverb.SMALL_DRR22, RoomReverbStereo.MONO, 1,
                 id="mono-spread"),
    pytest.param(1, RoomReverb.SMALL_DRR22, RoomReverbStereo.LATE_W100, 2,
                 id="single-voice-late-spread"),
    pytest.param(2, RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, 2,
                 id="two-voices-all-spread"),
])
def test_project_channel_matrix(voice_count, reverb, spread, channels):
    policy = VoiceSpatialPolicy.from_project(project(voice_count, reverb=reverb, spread=spread))
    assert policy.requested_channels == channels
    assert all(policy.pan_for_voice_index(index) == 0 for index in (-1, 0, 1, 2, 99))


def pans_project(pans, model="chatterbox_local"):
    return Project(tts_model_type=model, voice_references=[
        Voice(f"voice{i}.flac", pan=pan) for i, pan in enumerate(pans)
    ])


@pytest.mark.parametrize("pans,index,pan", [
    ([], 1, 0), ([0, .25], -1, 0), ([0, .25], 1, .25), ([0, .25], 99, .25), ([0, .25, 0], 99, 0),
])
def test_segments_use_their_voices_position_clamped_like_generation(pans, index, pan):
    assert VoiceSpatialPolicy.from_project(pans_project(pans)).pan_for_voice_index(index) == pan


@pytest.mark.parametrize("pans,expected,channels", [
    ([0, -.3], [0, 0, -.3], 2),
    ([0, 0], [0, 0, 0], 1),
    ([.4, -.3], [.4, .4, -.3], 2),  # unassigned (-1) lines use voice 0's position
])
def test_position_and_stereo_request(pans, expected, channels):
    policy = VoiceSpatialPolicy.from_project(pans_project(pans))
    assert [policy.pan_for_voice_index(i) for i in (-1, 0, 1)] == expected
    assert policy.requested_channels == channels


def test_positions_do_not_depend_on_the_selected_models_clone_support():
    # Existing audio keeps its voices' positions, e.g. concat with no model selected
    policy = VoiceSpatialPolicy.from_project(pans_project([0, .5], model="none"))
    assert policy.voice_count == 0
    assert policy.pan_for_voice_index(1) == .5
    assert policy.requested_channels == 2


def test_pocket_preset_voices_keep_their_positions():
    project = pans_project([.5, -.5], model="pocket_local")
    project.voice_references = [replace(voice, presets={"pocket_local": "alba"})
                                for voice in project.voice_references]
    policy = VoiceSpatialPolicy.from_project(project)
    assert [policy.pan_for_voice_index(i) for i in (0, 1)] == [.5, -.5]
    assert policy.requested_channels == 2


@pytest.mark.parametrize("pans,summary", [
    ([.3, .3], .3),
    ([.3, 0], None),
    ([], 0),
])
def test_stereo_position_summary(pans, summary):
    assert VoiceSpatialPolicy.from_project(pans_project(pans)).stereo_position_summary == summary


def test_snapshot_is_independent_of_later_project_changes_and_range():
    source = project(reverb=RoomReverb.MEDIUM_DRR21, spread=RoomReverbStereo.ALL_W100)
    policy = VoiceSpatialPolicy.from_project(source)
    source.voice_references = [Voice("voice0.flac")]
    assert policy.requested_channels == 2
    assert policy.pan_for_voice_index(0) == 0
    assert policy.pan_for_voice_index(1) == 0


def voices_project(reverbs, spread=RoomReverbStereo.MONO, model="chatterbox_local"):
    result = Project(tts_model_type=model, voice_references=[
        Voice(f"voice{i}.flac", room_reverb=reverb) for i, reverb in enumerate(reverbs)
    ])
    result.room_reverb_stereo = spread
    return result


SMALL, MEDIUM, OFF = RoomReverb.SMALL_DRR22, RoomReverb.MEDIUM_DRR21, RoomReverb.OFF


@pytest.mark.parametrize("index,expected", [(-1, SMALL), (1, OFF), (99, MEDIUM)])
def test_segments_use_their_voices_room(index, expected):
    policy = VoiceSpatialPolicy.from_project(voices_project([SMALL, OFF, MEDIUM]))
    assert policy.room_reverb_for_voice_index(index) is expected


def test_no_voices_means_no_room_but_pocket_preset_voices_keep_theirs():
    assert VoiceSpatialPolicy.from_project(voices_project([])).room_reverb_for_voice_index(0) is OFF
    project = voices_project([MEDIUM, SMALL], spread=RoomReverbStereo.ALL_W100, model="pocket_local")
    project.voice_references = [replace(voice, presets={"pocket_local": "alba"})
                                for voice in project.voice_references]
    policy = VoiceSpatialPolicy.from_project(project)
    assert [policy.room_reverb_for_voice_index(i) for i in (0, 1)] == [MEDIUM, SMALL]
    assert policy.requested_channels == 2


def test_rooms_do_not_depend_on_the_selected_models_clone_support():
    # Existing audio keeps its voices' rooms, e.g. concat with no model selected
    policy = VoiceSpatialPolicy.from_project(voices_project([OFF, MEDIUM], model="none"))
    assert policy.room_reverb_for_voice_index(1) is MEDIUM


@pytest.mark.parametrize("reverbs,channels", [
    ([OFF, SMALL], 2),  # any voice with a room
    ([SMALL, OFF], 2),
    ([OFF, OFF], 1),
])
def test_stereo_spread_requested_when_a_voice_has_a_room(reverbs, channels):
    project = voices_project(reverbs, spread=RoomReverbStereo.LATE_W100)
    assert VoiceSpatialPolicy.from_project(project).requested_channels == channels


@pytest.mark.parametrize("reverbs,summary", [
    ([SMALL, SMALL], SMALL),
    ([SMALL, OFF], None),
    ([], OFF),
])
def test_room_reverb_summary(reverbs, summary):
    assert VoiceSpatialPolicy.from_project(voices_project(reverbs)).room_reverb_summary is summary
