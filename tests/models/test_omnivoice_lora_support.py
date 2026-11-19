"""OmniVoice LoRA project support: model params, blocking issue and voice tag (no model library)."""
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel


class _Instance(OmniVoiceBaseModel):
    def __init__(self, has_lora: bool):
        self._has = has_lora

    @property
    def has_lora(self) -> bool:
        return self._has

    def generate_using_project(self, *args, **kwargs):
        raise NotImplementedError


def make_project(lora_target: str) -> Project:
    project = Project(tts_model_type="omnivoice_local", voice_references=[{"file_name": "voice-1.flac"}])
    project.set_model_setting("omnivoice_local", "lora_target", lora_target)
    return project


def test_lora_target_reaches_model_params():
    # A changed target is part of the params snapshot, so the model reloads
    project = make_project("owner/repo/lora")
    assert Tts.get_model_params_using_project(project)["omnivoice_lora_target"] == "owner/repo/lora"


def test_unloaded_lora_blocks_generation():
    def blocking_names(project, instance):
        return [issue.short for issue in OmniVoiceBaseModel.get_blocking_issues(project, instance)]

    assert "valid LoRA" in blocking_names(make_project("owner/repo"), _Instance(has_lora=False))
    assert "valid LoRA" not in blocking_names(make_project("owner/repo"), _Instance(has_lora=True))
    assert "valid LoRA" not in blocking_names(make_project(""), _Instance(has_lora=False))


def test_voice_tag_names_the_lora_over_the_sample():
    sample_tag = OmniVoiceBaseModel.get_voice_tag(make_project(""))
    lora_tag = OmniVoiceBaseModel.get_voice_tag(make_project("wjcwqc/PaigeVoice/lora"))
    assert lora_tag != sample_tag
    assert "PaigeVoice" in lora_tag
