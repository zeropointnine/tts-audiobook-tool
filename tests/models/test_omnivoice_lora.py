"""OmniVoice LoRA merge (`omnivoice_lora`): arithmetic, and all-or-nothing rejection.

Runs in venv-omni (needs torch and safetensors). Parity with peft's own merge on
real adapters is checked separately by `testx/omnivoice_lora_peft_parity.py`.
"""
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

from tts_audiobook_tool.tts_models import omnivoice_lora
from tts_audiobook_tool.tts_models.omnivoice_lora import LoraLoadError, load_lora

RANK = 2


class TinyOmniVoice(torch.nn.Module):
    """Same module paths as the parts of OmniVoice an adapter touches."""

    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 6, bias=False)
        layer.self_attn.k_proj = torch.nn.Linear(4, 6, bias=False)
        self.llm = torch.nn.Module()
        self.llm.layers = torch.nn.ModuleList([layer])
        self.audio_embeddings = torch.nn.Embedding(5, 4)
        self.audio_heads = torch.nn.Linear(4, 5, bias=False)


def make_model() -> TinyOmniVoice:
    torch.manual_seed(0)
    return TinyOmniVoice().half()


def adapter_tensors() -> dict:
    g = torch.Generator().manual_seed(1)
    q = "base_model.model.llm.layers.0.self_attn.q_proj"
    return {
        f"{q}.lora_A.weight": torch.randn(RANK, 4, generator=g),
        f"{q}.lora_B.weight": torch.randn(6, RANK, generator=g),
        "base_model.model.audio_heads.weight": torch.randn(5, 4, generator=g),
    }


def write_adapter(directory, tensors=None, **config_overrides) -> str:
    config = {"peft_type": "LORA", "r": RANK, "lora_alpha": 4, "use_rslora": False,
              "use_dora": False, "bias": "none", "lora_bias": False, "rank_pattern": {},
              "alpha_pattern": {}, "target_parameters": None, **config_overrides}
    (directory / "adapter_config.json").write_text(json.dumps(config))
    save_file(adapter_tensors() if tensors is None else tensors, str(directory / "adapter_model.safetensors"))
    return str(directory)


def state_of(model) -> dict:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize("use_rslora,expected_scale", [(False, 4 / RANK), (True, 4 / math.sqrt(RANK))])
def test_merge_adds_scaled_delta_rounded_once_and_copies_replacements(tmp_path, use_rslora, expected_scale):
    # W' = fp16(fp32(W) + scale * B @ A); replacements are copied (rounded to
    # the model dtype); untouched weights stay bit-identical
    model = make_model()
    before = state_of(model)
    tensors = adapter_tensors()
    plan = load_lora(model, write_adapter(tmp_path, use_rslora=use_rslora))

    assert plan.scale == pytest.approx(expected_scale)
    a = tensors["base_model.model.llm.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.llm.layers.0.self_attn.q_proj.lora_B.weight"]
    name = "llm.layers.0.self_attn.q_proj.weight"
    expected = (before[name].float() + (b @ a) * expected_scale).half()
    after = state_of(model)
    assert torch.equal(after[name], expected)
    assert torch.equal(after["audio_heads.weight"], tensors["base_model.model.audio_heads.weight"].half())
    for untouched in ("llm.layers.0.self_attn.k_proj.weight", "audio_embeddings.weight"):
        assert torch.equal(after[untouched], before[untouched])


def _q(suffix: str) -> str:
    return f"base_model.model.llm.layers.0.self_attn.q_proj.{suffix}"


def _without(key: str):
    return lambda tensors: {k: v for k, v in tensors.items() if k != key}


def _with(key: str, value):
    return lambda tensors: {**tensors, key: value}


@pytest.mark.parametrize("config,edit,message", [
    ({"use_dora": True}, None, "use_dora"),
    ({"rank_pattern": {"q_proj": 4}}, None, "rank_pattern"),
    ({"peft_type": "LOHA"}, None, "LOHA"),
    ({"r": 3}, None, "Shape mismatch"),
    ({}, _without(_q("lora_B.weight")), "Incomplete LoRA pair"),
    ({}, _with("llm.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(RANK, 4)), "Unexpected tensor"),
    ({}, _with("base_model.model.llm.embed_tokens.lora_embedding_A", torch.ones(RANK, 4)),
     "Unsupported adapter tensor"),
    ({}, _with("base_model.model.not_a_module.weight", torch.ones(2, 2)), "doesn't have"),
    ({}, _with("base_model.model.audio_heads.weight", torch.ones(4, 4)), "Shape mismatch"),
])
def test_rejected_adapter_leaves_model_unchanged(tmp_path, config, edit, message):
    # Validation happens before any weight is written, so an adapter is
    # rejected as a whole even when its other tensors would merge fine
    model = make_model()
    before = state_of(model)
    tensors = edit(adapter_tensors()) if edit else None
    with pytest.raises(LoraLoadError, match=message):
        load_lora(model, write_adapter(tmp_path, tensors, **config))
    after = state_of(model)
    assert all(torch.equal(after[name], value) for name, value in before.items())


def test_missing_adapter_file_is_a_load_error(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(LoraLoadError, match="adapter_model.safetensors"):
        omnivoice_lora.resolve_adapter_dir(str(tmp_path))


def test_omnivoice_model_keeps_base_model_when_lora_is_rejected(tmp_path, monkeypatch):
    # A rejected adapter is reported (has_lora False + reason) instead of
    # failing the model load, matching VibeVoice; generation is then blocked
    # by get_blocking_issues
    pytest.importorskip("omnivoice")
    from tts_audiobook_tool.tts_models.omnivoice_model import OmniVoiceModel

    model = OmniVoiceModel.__new__(OmniVoiceModel)
    model._model = make_model()  # type: ignore[assignment]
    model._load_lora(write_adapter(tmp_path, use_dora=True))
    assert not model.has_lora
    assert "use_dora" in model.lora_error

    model._load_lora(write_adapter(tmp_path))
    assert model.has_lora
