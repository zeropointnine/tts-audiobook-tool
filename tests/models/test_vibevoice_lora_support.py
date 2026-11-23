"""Local VibeVoice per-voice LoRA capabilities, tags and generation scheduling (no model library)."""
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
from tts_audiobook_tool.generate_util import get_voice_switch_keys
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.tts_models.voice_variation_warning import get_voice_variation_warnings

MODEL_ID = "vibevoice_local"


def make_project(*voices: Voice) -> Project:
    return Project(tts_model_type=MODEL_ID, voice_references=list(voices))


def test_per_voice_loras_do_not_change_model_load_params():
    # The model-wide LoRA used to be a load parameter (changing it reloaded the model);
    # per-voice adapters are switched in place, so only the base checkpoint remains one.
    project = make_project(Voice())
    project.set_model_setting(MODEL_ID, "target", "owner/custom-base")
    before = Tts.get_model_params_using_project(project)
    project.voice_references = [Voice(loras={MODEL_ID: VoiceLora(ModelTarget.hf("owner/adapter"), 0.5)})]
    assert Tts.get_model_params_using_project(project) == before
    assert before["vibevoice_target"] == "owner/custom-base"
    assert "vibevoice_lora_path" not in before
    assert VibeVoiceBaseModel.voice_lora_mode(project) == "merged"


def test_voice_tags_follow_each_voices_lora_over_its_sample():
    project = make_project(
        Voice("plain.flac"),
        Voice("other.flac", loras={MODEL_ID: VoiceLora(ModelTarget.hf("vibevoice-community/klett"))}),
    )
    sample_tag = Tts.get_voice_tag_for_selection_index(project, 0)
    lora_tag = Tts.get_voice_tag_for_selection_index(project, 1)
    assert lora_tag != sample_tag and "klett" in lora_tag
    assert VibeVoiceBaseModel.get_voice_tag(project) == sample_tag
    project.voice_references.reverse()
    assert VibeVoiceBaseModel.get_voice_tag(project) == lora_tag


def test_only_lora_voices_have_a_fixed_speaker_and_switch_keys():
    # A LoRA no longer suppresses the "voices may vary" note project-wide: only the voice
    # that has one gets a fixed speaker, and voices are scheduled by their LoRA.
    project = make_project(Voice(), Voice(loras={MODEL_ID: VoiceLora(ModelTarget.hf("owner/adapter"), 0.5)}))
    project.phrase_groups = [
        PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=index)
        for index in (0, 1)
    ]
    notes = get_voice_variation_warnings(project, [0, 1])
    assert len(notes) == 1 and notes[0].endswith("(Voice 1)")
    assert get_voice_switch_keys(project, [0, 1]) == {0: None, 1: (ModelTarget.hf("owner/adapter"), 0.5)}
