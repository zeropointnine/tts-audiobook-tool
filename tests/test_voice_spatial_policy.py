from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_types import RoomReverb, RoomReverbStereo, VoiceSelectMode
from tts_audiobook_tool.project_support.voice_spatial_policy import (
    VoiceSpatialPolicy, pan_for_voice_index,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def project(voice_count=2, mode=VoiceSelectMode.USER_DEFINED,
            reverb=RoomReverb.OFF, spread=RoomReverbStereo.MONO,
            model="chatterbox_local"):
    return SimpleNamespace(
        get_tts_model_type=lambda: TtsModelType.require_by_id(model),
        voice_references=[{"file_name": f"voice{i}.flac"} for i in range(voice_count)],
        voice_select_mode=mode, room_reverb=reverb, room_reverb_stereo=spread,
    )


@pytest.mark.parametrize("index", [-1, 0, 1, 2, 99])
def test_all_voice_slots_are_centered(index):
    assert pan_for_voice_index(index) == 0


@pytest.mark.parametrize("voice_count", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", list(VoiceSelectMode))
@pytest.mark.parametrize("reverb", list(RoomReverb))
@pytest.mark.parametrize("spread", list(RoomReverbStereo))
def test_project_channel_matrix(voice_count, mode, reverb, spread):
    policy = VoiceSpatialPolicy.from_project(project(voice_count, mode=mode, reverb=reverb, spread=spread))
    stereo = reverb is not RoomReverb.OFF and spread.is_stereo
    assert policy.requested_channels == (2 if stereo else 1)
    assert all(policy.pan_for_voice_index(index) == 0 for index in [-1, 0, 1, 2, 99])


@pytest.mark.parametrize("voice_count,index,pan", [
    (0, 1, 0), (1, 1, 0), (2, -1, 0), (2, 0, 0), (2, 1, .25),
    (2, 99, .25), (3, 99, 0), (3, 1, .25),
])
def test_effective_voice_clamps_like_generation(monkeypatch, voice_count, index, pan):
    # Exercise the dormant resolution path without enabling any app voice pan.
    monkeypatch.setattr(
        "tts_audiobook_tool.project_support.voice_spatial_policy.pan_for_voice_index",
        lambda index: .25 if index == 1 else 0.0,
    )
    assert VoiceSpatialPolicy.from_project(project(voice_count)).pan_for_voice_index(index) == pan


@pytest.mark.parametrize("mode", [VoiceSelectMode.DISABLED, VoiceSelectMode.AUTO_ADVANCE])
def test_unresolved_or_disabled_phrase_identity_stays_centered(mode):
    policy = VoiceSpatialPolicy.from_project(project(mode=mode))
    assert policy.pan_for_voice_index(1) == 0
    assert policy.requested_channels == 1


def test_model_without_voice_binding_ignores_reference_list():
    policy = VoiceSpatialPolicy.from_project(project(model="none"))
    assert policy.voice_count == 0
    assert policy.requested_channels == 1
    assert policy.pan_for_voice_index(1) == 0


def test_snapshot_is_independent_of_later_project_changes_and_range():
    source = project(reverb=RoomReverb.MEDIUM_DRR21, spread=RoomReverbStereo.ALL_W100)
    policy = VoiceSpatialPolicy.from_project(source)
    source.voice_references.clear()
    source.voice_select_mode = VoiceSelectMode.DISABLED
    source.room_reverb = RoomReverb.OFF
    assert policy.requested_channels == 2
    assert policy.pan_for_voice_index(0) == 0
    assert policy.pan_for_voice_index(1) == 0
