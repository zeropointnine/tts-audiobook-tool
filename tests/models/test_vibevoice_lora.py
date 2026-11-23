"""VibeVoice LoRA merge (`vibevoice_lora`): adapter layout, key mapping, and per-voice switching.

Runs in venv-vv (needs torch and safetensors; the wrapper tests need the vibevoice library).
The PEFT-format validation itself is shared with OmniVoice (`peft_lora`, test_omnivoice_lora).
"""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models import vibevoice_lora
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, apply_merge

RANK = 2
MODEL_ID = "vibevoice_local"
Q = "model.language_model.layers.0.self_attn.q_proj.weight"


class TinyVibeVoice(torch.nn.Module):
    """Same module paths as the parts of VibeVoice an adapter touches (the trainer wraps
    only `model.language_model` with PEFT), plus a head an adapter must not reach."""

    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 6, bias=False)
        layer.self_attn.k_proj = torch.nn.Linear(4, 6, bias=False)
        language_model = torch.nn.Module()
        language_model.layers = torch.nn.ModuleList([layer])
        self.model = torch.nn.Module()
        self.model.language_model = language_model
        self.model.prediction_head = torch.nn.Linear(4, 4, bias=False)


def make_model() -> TinyVibeVoice:
    torch.manual_seed(0)
    return TinyVibeVoice().to(torch.bfloat16)


def adapter_tensors(scale: float = 1.0) -> dict:
    # Keys relative to the language model, as the trainer's PeftModel writes them
    g = torch.Generator().manual_seed(1)
    q = "base_model.model.layers.0.self_attn.q_proj"
    return {
        f"{q}.lora_A.weight": torch.randn(RANK, 4, generator=g),
        f"{q}.lora_B.weight": torch.randn(6, RANK, generator=g) * scale,
    }


def write_adapter(directory, tensors=None, *, bin_format=False) -> ModelTarget:
    directory.mkdir(parents=True, exist_ok=True)
    config = {"peft_type": "LORA", "r": RANK, "lora_alpha": 4, "use_rslora": False,
              "use_dora": False, "bias": "none", "target_modules": ["q_proj"]}
    (directory / "adapter_config.json").write_text(json.dumps(config))
    tensors = adapter_tensors() if tensors is None else tensors
    if bin_format:
        torch.save(tensors, str(directory / "adapter_model.bin"))
    else:
        save_file(tensors, str(directory / "adapter_model.safetensors"))
    return ModelTarget.local(str(directory))


def state_of(model) -> dict:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def expected_q(before: dict, tensors: dict, strength: float = 1.0) -> torch.Tensor:
    a = tensors["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    return (before[Q].float() + (b @ a) * (strength * 4 / RANK)).to(torch.bfloat16)


@pytest.mark.parametrize("subdir,bin_format", [("", False), ("lora", True)])
def test_language_model_adapter_merges_under_the_language_model(tmp_path, subdir, bin_format):
    # The adapter's language-model-relative keys land on `model.language_model.*`; the
    # directory may be the adapter itself or the trainer's output dir (adapter in lora/),
    # with safetensors or an older .bin. Untouched weights stay bit-identical.
    write_adapter(tmp_path / subdir if subdir else tmp_path, bin_format=bin_format)
    model = make_model()
    before = state_of(model)
    plan = vibevoice_lora.load_lora(model, ModelTarget.local(str(tmp_path)), 0.5)
    assert plan.target_names == {Q}
    after = state_of(model)
    assert torch.equal(after[Q], expected_q(before, adapter_tensors(), 0.5))
    for name in before.keys() - {Q}:
        assert torch.equal(after[name], before[name])


def test_directory_without_an_adapter_is_a_load_error(tmp_path):
    (tmp_path / "lora").mkdir()
    (tmp_path / "lora" / "adapter_config.json").write_text("{}")
    with pytest.raises(LoraLoadError, match="adapter_model.safetensors"):
        vibevoice_lora.resolve_adapter(ModelTarget.local(str(tmp_path)))


def test_whole_model_adapter_is_rejected_unchanged(tmp_path):
    # An adapter keyed for another module tree can't be mapped; reject it before merging
    model = make_model()
    before = state_of(model)
    tensors = {**adapter_tensors(), "model.prediction_head.weight": torch.ones(4, 4)}
    with pytest.raises(LoraLoadError, match="VibeVoice-finetuning"):
        vibevoice_lora.load_lora(model, write_adapter(tmp_path, tensors))
    assert all(torch.equal(state_of(model)[name], value) for name, value in before.items())


def make_wrapper(base_dir):
    pytest.importorskip("vibevoice")
    from tts_audiobook_tool.app_types import DeviceType
    from tts_audiobook_tool.tts_models.vibe_voice_model import VibeVoiceModel

    model = VibeVoiceModel.__new__(VibeVoiceModel)
    model.model = make_model()  # type: ignore[assignment]
    model.processor = SimpleNamespace()  # type: ignore[assignment]
    model._device_type = DeviceType.CPU
    base_dir.mkdir()
    save_file(state_of(model.model), str(base_dir / "model.safetensors"))
    model._loras = VibeVoiceModel._make_lora_switcher(model.model, base_dir)
    return model


def test_vibevoice_activates_each_selected_voices_lora_before_generation(tmp_path, monkeypatch):
    # One loaded model serves adapter, no-adapter and different-strength voices; a rejected
    # adapter blocks that voice's inference without touching the merged weights.
    from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
    from tts_audiobook_tool.project import Project

    model = make_wrapper(tmp_path / "base")
    base = state_of(model.model)
    write_adapter(tmp_path / "adapter" / "lora")
    bad = write_adapter(tmp_path / "bad", {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(RANK, 4)})
    project = Project(tts_model_type=MODEL_ID, voice_references=[
        Voice(loras={MODEL_ID: VoiceLora(ModelTarget.local(str(tmp_path / "adapter")))}), Voice(),
        Voice(loras={MODEL_ID: VoiceLora(ModelTarget.local(str(tmp_path / "adapter")), 0.5)}),
        Voice(loras={MODEL_ID: VoiceLora(bad)}),
    ])
    calls = []

    def generate(**kwargs):
        calls.append((state_of(model.model), kwargs["lora"]))
        return []

    monkeypatch.setattr(model, "generate", generate)
    for index, strength in ((0, 1.0), (1, None), (2, 0.5)):
        assert model.generate_using_project(project, ["Hello."], voice_selection_index=index) == []
        weights, lora_label = calls[-1]
        expected = base[Q] if strength is None else expected_q(base, adapter_tensors(), strength)
        assert torch.equal(weights[Q], expected)
        assert (lora_label == "") == (strength is None)
    error = model.generate_using_project(project, ["Hello."], voice_selection_index=3)
    assert isinstance(error, str) and "voice 4" in error and "Incomplete LoRA pair" in error
    assert len(calls) == 3 and torch.equal(state_of(model.model)[Q], expected_q(base, adapter_tensors(), 0.5))
    # The run preflight reports only the used voices' unusable adapters
    problems = model.prepare_voice_loras(project, [0, 1, 3])
    assert len(problems) == 1 and problems[0].startswith("Voice 4 LoRA") and "Incomplete" in problems[0]
    model.end_generation_run()
    assert model._loras is not None and model._loras.base.mode == "undecided"
    model.kill()
    assert model._loras is None


def test_model_loading_and_base_restores_share_one_resolved_snapshot(tmp_path, monkeypatch):
    # A hub target is resolved once, and the processor, the model and the base-weight
    # source all read that one directory; LoRAs are no longer a constructor argument.
    pytest.importorskip("vibevoice")
    import huggingface_hub
    from tts_audiobook_tool.app_types import DeviceType
    from tts_audiobook_tool.tts_models import vibe_voice_model

    downloaded, loaded, sources = [], [], []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **_: downloaded.append(repo_id) or str(tmp_path))
    monkeypatch.setattr(vibe_voice_model.VibeVoiceProcessor, "from_pretrained",
                        lambda path: loaded.append(("processor", path)) or object())
    model = make_model()
    monkeypatch.setattr(vibe_voice_model.VibeVoiceForConditionalGenerationInference, "from_pretrained",
                        lambda path, **kwargs: loaded.append(("model", path)) or model)
    sentinel = object()
    monkeypatch.setattr(vibe_voice_model.VibeVoiceModel, "_make_lora_switcher",
                        staticmethod(lambda value, path: sources.append((value, str(path))) or sentinel))
    wrapper = vibe_voice_model.VibeVoiceModel(DeviceType.CPU, model_target=ModelTarget.hf("owner/base"))
    assert downloaded == ["owner/base"]
    assert loaded == [("processor", str(tmp_path)), ("model", str(tmp_path))]
    assert sources == [(model, str(tmp_path))]
    assert wrapper._loras is sentinel
