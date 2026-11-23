"""Local OmniVoice per-voice LoRA capabilities, tags and generation scheduling (no model library)."""
import pytest

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.generate_util import get_voice_switch_keys
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.tts_models.voice_variation_warning import get_voice_variation_warnings

MODEL_ID = "omnivoice_local"


def make_project(*voices: Voice) -> Project:
    return Project(tts_model_type=MODEL_ID, voice_references=list(voices))


def test_per_voice_loras_do_not_change_model_load_params():
    # Only the base checkpoint is a load parameter; changing an adapter or its strength
    # must keep the loaded model and switch weights in place instead.
    project = make_project(Voice())
    project.set_model_setting(MODEL_ID, "target", "owner/custom-base")
    before = Tts.get_model_params_using_project(project)
    project.voice_references = [Voice(loras={MODEL_ID: VoiceLora("owner/adapter", 0.5)})]
    assert Tts.get_model_params_using_project(project) == before
    assert before["omnivoice_target"] == "owner/custom-base"
    assert "omnivoice_lora_target" not in before
    assert OmniVoiceBaseModel.voice_lora_mode(project) == "merged"


def test_voice_tags_follow_each_voices_lora_over_its_sample():
    project = make_project(
        Voice("plain.flac"),
        Voice("other.flac", loras={MODEL_ID: VoiceLora("wjcwqc/PaigeVoice/lora")}),
    )
    sample_tag = Tts.get_voice_tag_for_selection_index(project, 0)
    lora_tag = Tts.get_voice_tag_for_selection_index(project, 1)
    assert lora_tag != sample_tag and "PaigeVoice" in lora_tag
    assert OmniVoiceBaseModel.get_voice_tag(project) == sample_tag
    project.voice_references.reverse()
    assert OmniVoiceBaseModel.get_voice_tag(project) == lora_tag


def test_only_lora_voices_have_a_fixed_speaker_and_switch_keys():
    # The shared variation and scheduling rules apply to OmniVoice without a
    # model-wide fixed-default-voice flag suppressing notes for unrelated voices.
    project = make_project(Voice(), Voice(loras={MODEL_ID: VoiceLora("owner/adapter", 0.5)}))
    project.phrase_groups = [
        PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=index)
        for index in (0, 1)
    ]
    notes = get_voice_variation_warnings(project, [0, 1])
    assert len(notes) == 1 and notes[0].endswith("(Voice 1)")
    assert get_voice_switch_keys(project, [0, 1]) == {0: None, 1: ("owner/adapter", 0.5)}


def test_a_stored_omnivoice_lora_without_strength_loads_at_full_strength():
    # Strength is written only when not 1, so a project entry without one means 1.0
    voice = Voice.from_dict({"file_name": "", "loras": {MODEL_ID: {"target": "owner/adapter"}}})
    assert voice is not None and voice.lora_for(MODEL_ID) == VoiceLora("owner/adapter", 1.0)


@pytest.mark.parametrize("model_id, expected", [
    ("omnivoice_local", "no sample or LoRA set"),
    ("breeze_tts_2_local", "no sample or LoRA set"),
    ("vibevoice_local", "no sample set"),  # its LoRA is model-wide, not a voice value
])
def test_a_voice_without_a_value_names_the_lora_as_an_option(model_id, expected):
    # With per-voice LoRAs a LoRA alone gives a voice its speaker, so the hint names both
    assert ProjectVoiceUtil.make_no_voice_value_text(Project(tts_model_type=model_id)) == expected
