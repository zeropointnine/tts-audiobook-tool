"""Breeze TTS 2 LoRA project support: model params, reload triggers, blocking issue and voice tag (no model library)."""
import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel

MODEL_ID = "breeze_tts_2_local"


class _Instance(BreezeBaseModel):
    def __init__(self, has_lora: bool):
        self._has = has_lora

    @property
    def has_lora(self) -> bool:
        return self._has

    def generate_using_project(self, *args, **kwargs):
        raise NotImplementedError

    def kill(self) -> None:
        pass


def make_project(lora_target: str, strength: float = -1.0) -> Project:
    project = Project(tts_model_type=MODEL_ID, voice_references=[{"file_name": "voice-1.flac", "transcript": "Hi."}])
    project.set_model_setting(MODEL_ID, "lora_target", lora_target)
    project.set_model_setting(MODEL_ID, "lora_strength", strength)
    return project


def test_lora_target_and_strength_reach_model_params():
    # Unset strength (-1) resolves to the catalog default 1.0
    params = Tts.get_model_params_using_project(make_project("/runs/x/lora/checkpoint-step-000500"))
    assert params["breeze_lora_target"] == "/runs/x/lora/checkpoint-step-000500"
    assert params["breeze_lora_strength"] == 1.0
    assert Tts.get_model_params_using_project(make_project("x", 0.5))["breeze_lora_strength"] == 0.5


@pytest.mark.parametrize("change", [{"breeze_lora_target": "other"}, {"breeze_lora_strength": 0.5}])
def test_changed_lora_target_or_strength_reloads_the_model(monkeypatch, change):
    # Both are load-time parameters (the adapter is merged into the weights), so either change discards the model
    import tts_audiobook_tool.tts as tts_module
    cleared = []
    monkeypatch.setattr(Tts, "_model_params", {})
    monkeypatch.setattr(Tts, "clear_tts_model", staticmethod(lambda: cleared.append(True)))
    monkeypatch.setattr(tts_module, "current_role", lambda: None)
    base = Tts.get_model_params_using_project(make_project("x"))
    Tts.set_model_params(base)
    cleared.clear()
    Tts.set_model_params({**base})
    assert cleared == []
    Tts.set_model_params({**base, **change})
    assert cleared == [True]


def test_unloaded_lora_blocks_generation():
    def blocking_names(project, instance):
        return [issue.short for issue in BreezeBaseModel.get_blocking_issues(project, instance)]

    assert "valid LoRA" in blocking_names(make_project("owner/repo"), _Instance(has_lora=False))
    assert "valid LoRA" not in blocking_names(make_project("owner/repo"), _Instance(has_lora=True))
    assert "valid LoRA" not in blocking_names(make_project(""), _Instance(has_lora=False))


def test_lora_gives_a_fixed_default_voice():
    assert BreezeBaseModel.has_fixed_default_voice(make_project("owner/repo"))
    assert not BreezeBaseModel.has_fixed_default_voice(make_project(""))


def test_voice_tag_names_the_lora_over_the_sample():
    sample_tag = BreezeBaseModel.get_voice_tag(make_project(""))
    lora_tag = BreezeBaseModel.get_voice_tag(make_project("/runs/fw-leda/lora/checkpoint-step-000500"))
    assert lora_tag != sample_tag
    assert "fw_leda_checkpoint" in lora_tag
