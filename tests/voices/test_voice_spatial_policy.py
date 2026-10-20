from dataclasses import replace
import pytest

from tts_audiobook_tool.app_types import RoomReverb, RoomReverbStereo, VoiceSelectMode
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.voice_spatial_policy import VoiceSpatialPolicy


def project(voice_count=2, mode=VoiceSelectMode.USER_DEFINED,
            reverb=RoomReverb.OFF, spread=RoomReverbStereo.MONO,
            model="chatterbox_local"):
    # Every voice gets the same room, so the output's room doesn't depend on voice
    result = Project(
        tts_model_type=model,
        voice_references=[Voice(f"voice{i}.flac", room_reverb=reverb) for i in range(voice_count)],
    )
    result.voice_select_mode = mode
    result.room_reverb_stereo = spread
    return result


# Uniform rooms make mode irrelevant, every non-OFF reverb and every stereo spread
# behave alike, and voice counts >= 1 behave alike, so these cover every class.
@pytest.mark.parametrize("voice_count,mode,reverb,spread,channels", [
    pytest.param(0, VoiceSelectMode.USER_DEFINED, RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, 1,
                 id="no-voices-means-no-reverb"),
    pytest.param(1, VoiceSelectMode.USER_DEFINED, RoomReverb.OFF, RoomReverbStereo.ALL_W100, 1,
                 id="reverb-off"),
    pytest.param(1, VoiceSelectMode.USER_DEFINED, RoomReverb.SMALL_DRR22, RoomReverbStereo.MONO, 1,
                 id="mono-spread"),
    pytest.param(1, VoiceSelectMode.USER_DEFINED, RoomReverb.SMALL_DRR22, RoomReverbStereo.LATE_W100, 2,
                 id="single-voice-late-spread"),
    pytest.param(2, VoiceSelectMode.AUTO_ADVANCE, RoomReverb.MEDIUM_DRR21, RoomReverbStereo.ALL_W100, 2,
                 id="all-spread-auto-advance"),
    pytest.param(2, VoiceSelectMode.DISABLED, RoomReverb.SMALL_DRR22, RoomReverbStereo.LATE_W100, 2,
                 id="late-spread-disabled"),
])
def test_project_channel_matrix(voice_count, mode, reverb, spread, channels):
    policy = VoiceSpatialPolicy.from_project(project(voice_count, mode=mode, reverb=reverb, spread=spread))
    assert policy.requested_channels == channels
    assert all(policy.pan_for_voice_index(index) == 0 for index in (-1, 0, 1, 2, 99))


def pans_project(pans, mode=VoiceSelectMode.USER_DEFINED, model="chatterbox_local"):
    result = Project(tts_model_type=model, voice_references=[
        Voice(f"voice{i}.flac", pan=pan) for i, pan in enumerate(pans)
    ])
    result.voice_select_mode = mode
    return result


@pytest.mark.parametrize("pans,index,pan", [
    ([], 1, 0), ([0, .25], -1, 0), ([0, .25], 1, .25), ([0, .25], 99, .25), ([0, .25, 0], 99, 0),
])
def test_user_defined_segments_use_their_voices_position_clamped_like_generation(pans, index, pan):
    assert VoiceSpatialPolicy.from_project(pans_project(pans)).pan_for_voice_index(index) == pan


@pytest.mark.parametrize("mode,pans,expected,channels", [
    (VoiceSelectMode.USER_DEFINED, [0, -.3], [0, 0, -.3], 2),
    (VoiceSelectMode.USER_DEFINED, [0, 0], [0, 0, 0], 1),
    (VoiceSelectMode.DISABLED, [.4, -.3], [.4, .4, .4], 2),   # voice 0 speaks every line
    (VoiceSelectMode.DISABLED, [0, -.3], [0, 0, 0], 1),
    (VoiceSelectMode.AUTO_ADVANCE, [.4, -.3], [0, 0, 0], 1),  # identity unknown: centered
])
def test_position_by_mode_and_stereo_request(mode, pans, expected, channels):
    policy = VoiceSpatialPolicy.from_project(pans_project(pans, mode=mode))
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


@pytest.mark.parametrize("mode,pans,summary", [
    (VoiceSelectMode.USER_DEFINED, [.3, .3], .3),
    (VoiceSelectMode.USER_DEFINED, [.3, 0], None),
    (VoiceSelectMode.DISABLED, [.3, 0], .3),
    (VoiceSelectMode.AUTO_ADVANCE, [.3, .3], 0),
    (VoiceSelectMode.USER_DEFINED, [], 0),
])
def test_stereo_position_summary(mode, pans, summary):
    assert VoiceSpatialPolicy.from_project(pans_project(pans, mode=mode)).stereo_position_summary == summary


def test_snapshot_is_independent_of_later_project_changes_and_range():
    source = project(reverb=RoomReverb.MEDIUM_DRR21, spread=RoomReverbStereo.ALL_W100)
    policy = VoiceSpatialPolicy.from_project(source)
    source.voice_references = [Voice("voice0.flac")]
    source.voice_select_mode = VoiceSelectMode.DISABLED
    assert policy.requested_channels == 2
    assert policy.pan_for_voice_index(0) == 0
    assert policy.pan_for_voice_index(1) == 0


def voices_project(reverbs, mode=VoiceSelectMode.USER_DEFINED, spread=RoomReverbStereo.MONO,
                   model="chatterbox_local"):
    result = Project(tts_model_type=model, voice_references=[
        Voice(f"voice{i}.flac", room_reverb=reverb) for i, reverb in enumerate(reverbs)
    ])
    result.voice_select_mode = mode
    result.room_reverb_stereo = spread
    return result


SMALL, MEDIUM, OFF = RoomReverb.SMALL_DRR22, RoomReverb.MEDIUM_DRR21, RoomReverb.OFF


@pytest.mark.parametrize("index,expected", [(-1, SMALL), (1, OFF), (99, MEDIUM)])
def test_user_defined_segments_use_their_voices_room(index, expected):
    policy = VoiceSpatialPolicy.from_project(voices_project([SMALL, OFF, MEDIUM]))
    assert policy.room_reverb_for_voice_index(index) is expected


@pytest.mark.parametrize("mode", [VoiceSelectMode.DISABLED, VoiceSelectMode.AUTO_ADVANCE])
def test_other_modes_use_voice_zeros_room(mode):
    policy = VoiceSpatialPolicy.from_project(voices_project([OFF, MEDIUM], mode=mode))
    assert [policy.room_reverb_for_voice_index(i) for i in (-1, 0, 1)] == [OFF, OFF, OFF]
    policy = VoiceSpatialPolicy.from_project(voices_project([MEDIUM, OFF], mode=mode))
    assert [policy.room_reverb_for_voice_index(i) for i in (-1, 0, 1)] == [MEDIUM] * 3


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


@pytest.mark.parametrize("mode,reverbs,channels", [
    (VoiceSelectMode.USER_DEFINED, [OFF, SMALL], 2),  # any voice with a room
    (VoiceSelectMode.USER_DEFINED, [OFF, OFF], 1),
    (VoiceSelectMode.AUTO_ADVANCE, [OFF, SMALL], 1),  # only voice 0 is heard
    (VoiceSelectMode.AUTO_ADVANCE, [SMALL, OFF], 2),
    (VoiceSelectMode.DISABLED, [OFF, SMALL], 1),
])
def test_stereo_spread_requested_when_a_heard_voice_has_a_room(mode, reverbs, channels):
    project = voices_project(reverbs, mode=mode, spread=RoomReverbStereo.LATE_W100)
    assert VoiceSpatialPolicy.from_project(project).requested_channels == channels


@pytest.mark.parametrize("mode,reverbs,summary", [
    (VoiceSelectMode.USER_DEFINED, [SMALL, SMALL], SMALL),
    (VoiceSelectMode.USER_DEFINED, [SMALL, OFF], None),
    (VoiceSelectMode.AUTO_ADVANCE, [SMALL, OFF], SMALL),
    (VoiceSelectMode.USER_DEFINED, [], OFF),
])
def test_room_reverb_summary(mode, reverbs, summary):
    assert VoiceSpatialPolicy.from_project(voices_project(reverbs, mode=mode)).room_reverb_summary is summary
