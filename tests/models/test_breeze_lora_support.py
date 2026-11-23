"""Breeze TTS 2 per-voice LoRA project support: no load-time LoRA params, per-voice voice tags,
variation notes, run pre-flight and LoRA-aware scheduling keys (no model library)."""
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
from tts_audiobook_tool.generate_util import get_voice_lora_blocker_text, get_voice_switch_keys
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel
from tts_audiobook_tool.tts_models.voice_variation_warning import get_voice_variation_warnings

MODEL_ID = "breeze_tts_2_local"
LEDA = ModelTarget.local("/runs/fw-leda/lora/checkpoint-step-000500")


def lora_voice(target: ModelTarget = LEDA, strength: float = 1.0, **kwargs) -> Voice:
    return Voice(loras={MODEL_ID: VoiceLora(target, strength)}, **kwargs)


def make_project(*voices: Voice, voice_of_line: tuple[int, ...] = ()) -> Project:
    project = Project(tts_model_type=MODEL_ID, voice_references=list(voices))
    project.phrase_groups = [PhraseGroup([Phrase("Hello there.", Reason.SENTENCE)], voice_index=v)
                             for v in voice_of_line]
    return project


def test_loras_are_not_model_load_params():
    # Switched per call, so a LoRA change must never reload the model
    params = Tts.get_model_params_using_project(make_project(lora_voice()))
    assert not any(key.startswith("breeze_lora") for key in params)
    assert BreezeBaseModel.voice_lora_mode(make_project()) == "merged"


def test_voice_tag_names_each_voices_lora_over_its_sample():
    project = make_project(Voice("voice-1.flac", "Hi."), lora_voice(file_name="voice-2.flac", transcript="Hi."))
    sample_tag = Tts.get_voice_tag_for_selection_index(project, 0)
    lora_tag = Tts.get_voice_tag_for_selection_index(project, 1)
    assert "fw_leda_checkpoint" in lora_tag and "fw_leda" not in sample_tag
    # The project-level tag follows voice 1
    assert "fw_leda" not in BreezeBaseModel.get_voice_tag(project)
    assert "fw_leda_checkpoint" in BreezeBaseModel.get_voice_tag(make_project(lora_voice()))


def test_a_lora_voice_gets_no_voice_variation_note():
    # A LoRA fixes the speaker, even for a voice with only instructions
    project = make_project(Voice(), lora_voice(), lora_voice(instructions={MODEL_ID: "Calm."}))
    notes = get_voice_variation_warnings(project, [0, 1, 2])
    assert len(notes) == 1 and notes[0].endswith("(Voice 1)")


def test_lora_preflight_checks_only_the_jobs_lora_voices(monkeypatch):
    # Voices without a LoRA (or unused by the job) are never handed to the model
    checked = []

    class Instance:
        def prepare_voice_loras(self, project, voice_indices):
            checked.append(list(voice_indices))
            return ["Voice 2 LoRA (x): bad"]

    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: Instance()))
    project = make_project(Voice(), lora_voice(), lora_voice(ModelTarget.local("other")), voice_of_line=(0, 1, 0))
    assert get_voice_lora_blocker_text(project, [0, 1, 2]) == "Voice 2 LoRA (x): bad"
    assert checked == [[1]]
    assert get_voice_lora_blocker_text(project, [0]) == ""
    assert checked == [[1]]


def test_switch_keys_only_when_the_job_mixes_loras():
    # One key per voice slot: (target, strength), or None for no LoRA; no keys when nothing would switch
    project = make_project(Voice(), lora_voice(), lora_voice(strength=0.5), voice_of_line=(0, 1, 2, 1))
    assert get_voice_switch_keys(project, [0, 1, 2, 3]) == {0: None, 1: (LEDA, 1.0), 2: (LEDA, 0.5)}
    assert get_voice_switch_keys(project, [1, 3]) is None
    same = make_project(lora_voice(), lora_voice(), voice_of_line=(0, 1))
    assert get_voice_switch_keys(same, [0, 1]) is None
    mira = Project(tts_model_type="mira_local", voice_references=[Voice(), lora_voice()])
    assert get_voice_switch_keys(mira, []) is None
