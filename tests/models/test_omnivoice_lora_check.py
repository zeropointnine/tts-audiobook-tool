"""Model-free OmniVoice LoRA check (`omnivoice_lora_check`): the voice menu's check, against the
shapes the base model's config.json implies instead of the loaded model.

Runs in venv-base (numpy only).
"""
import glob
import json
import math
import os

import numpy as np
import pytest

from lora_file_test_support import write_safetensors
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.lora_files import ZERO_LORA_WARNING, SafetensorsFile
from tts_audiobook_tool.tts_models.omnivoice_lora_check import check_adapter, expected_shapes

RANK = 2
# A tiny Qwen3-style base: hidden 8, 2 heads of 4 (q 8), 1 kv head (k/v 4), mlp 12, 2 layers
BASE_CONFIG = {
    "model_type": "omnivoice", "num_audio_codebook": 2, "audio_vocab_size": 3,
    "llm_config": {"model_type": "qwen3", "hidden_size": 8, "intermediate_size": 12, "num_attention_heads": 2,
                   "num_key_value_heads": 1, "head_dim": 4, "num_hidden_layers": 2, "vocab_size": 10},
}
Q = "base_model.model.llm.layers.1.self_attn.q_proj"
K = "base_model.model.llm.layers.0.self_attn.k_proj"


def write_base(directory, config=None) -> ModelTarget:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(BASE_CONFIG if config is None else config))
    return ModelTarget.local(str(directory))


def adapter_tensors(b_scale=0.01) -> dict:
    rng = np.random.default_rng(0)
    return {
        f"{Q}.lora_A.weight": rng.standard_normal((RANK, 8)).astype(np.float32),
        f"{Q}.lora_B.weight": (rng.standard_normal((8, RANK)) * b_scale).astype(np.float32),
        f"{K}.lora_A.weight": rng.standard_normal((RANK, 8)).astype(np.float32),
        f"{K}.lora_B.weight": (rng.standard_normal((4, RANK)) * b_scale).astype(np.float32),
        "base_model.model.audio_heads.modules_to_save.default.weight": np.ones((6, 8), np.float32),
    }


def write_adapter(directory, tensors=None, **config_overrides) -> ModelTarget:
    directory.mkdir(exist_ok=True)
    config = {"peft_type": "LORA", "r": RANK, "lora_alpha": 4, "use_dora": False, "bias": "none", **config_overrides}
    (directory / "adapter_config.json").write_text(json.dumps(config))
    write_safetensors(directory / "adapter_model.safetensors", adapter_tensors() if tensors is None else tensors)
    return ModelTarget.local(str(directory))


def test_valid_adapter_passes_against_the_base_config(tmp_path):
    result = check_adapter(write_adapter(tmp_path / "adapter"), write_base(tmp_path / "base"), 0.5)
    assert result.error == "" and result.warnings == ()


def edited(**changes):
    return lambda tensors: {**{k: v for k, v in tensors.items() if k not in changes}, **{
        k: v for k, v in changes.items() if v is not None}}


@pytest.mark.parametrize("edit,message", [
    (edited(**{f"{Q}.lora_B.weight": None}), "Incomplete LoRA pair"),
    # k_proj is (4, 8) in this base: a B shaped for q_proj fits the rank but not the base
    (edited(**{f"{K}.lora_B.weight": np.ones((8, RANK), np.float32)}), "Shape mismatch"),
    (edited(**{"base_model.model.audio_heads.modules_to_save.default.weight": np.ones((7, 8), np.float32)}),
     "Shape mismatch for audio_heads.weight"),
    (edited(**{"base_model.model.llm.layers.2.mlp.up_proj.lora_A.weight": np.ones((RANK, 8), np.float32),
               "base_model.model.llm.layers.2.mlp.up_proj.lora_B.weight": np.ones((12, RANK), np.float32)}),
     "which the model doesn't have"),
    (edited(**{"base_model.model.llm.norm.lora_A.weight": np.ones((RANK, 8), np.float32),
               "base_model.model.llm.norm.lora_B.weight": np.ones((8, RANK), np.float32)}), "is not a matrix"),
    (edited(**{"llm.layers.0.self_attn.q_proj.weight": np.ones((8, 8), np.float32)}), "Unexpected tensor"),
    (edited(**{"base_model.model.audio_heads.modules_to_save.default.weight": np.ones((6, 8), np.int32)}),
     "Non-float adapter tensor"),
    (edited(**{f"{Q}.lora_A.weight": np.full((RANK, 8), math.inf, np.float32)}), "NaN or infinite"),
    (lambda tensors: {}, "no tensors"),
])
def test_bad_tensors_are_rejected(tmp_path, edit, message):
    target = write_adapter(tmp_path / "adapter", edit(adapter_tensors()))
    assert message in check_adapter(target, write_base(tmp_path / "base")).error


def test_unsupported_config_is_rejected(tmp_path):
    target = write_adapter(tmp_path / "adapter", use_dora=True)
    assert "use_dora" in check_adapter(target, write_base(tmp_path / "base")).error


def test_untrained_adapter_is_accepted_with_a_warning(tmp_path):
    target = write_adapter(tmp_path / "adapter", adapter_tensors(b_scale=0.0))
    result = check_adapter(target, write_base(tmp_path / "base"))
    assert result.error == "" and result.warnings == (ZERO_LORA_WARNING,)


@pytest.mark.parametrize("config", [None, {"llm_config": {"model_type": "llama"}}])
def test_unreadable_or_unknown_base_config_warns_and_checks_internal_consistency(tmp_path, config):
    # Without the base's shapes the factors need only agree with each other and the rank;
    # the run's check against the loaded model stays authoritative
    base = tmp_path / "base"
    if config is None:
        base.mkdir()
        base_target = ModelTarget.local(str(base))
    else:
        base_target = write_base(base, config)
    result = check_adapter(write_adapter(tmp_path / "adapter"), base_target)
    assert result.error == "" and len(result.warnings) == 1 and "checked against the model" in result.warnings[0]

    bad = edited(**{f"{K}.lora_B.weight": np.ones((4, RANK + 1), np.float32)})(adapter_tensors())
    assert "Shape mismatch" in check_adapter(write_adapter(tmp_path / "bad", bad), base_target).error


def test_qwen2_layout_has_qkv_biases_and_no_qk_norms():
    shapes = expected_shapes({**BASE_CONFIG, "llm_config": {**BASE_CONFIG["llm_config"], "model_type": "qwen2"}})
    assert shapes is not None
    assert shapes["llm.layers.0.self_attn.k_proj.bias"] == (4,) and "llm.layers.0.self_attn.o_proj.bias" not in shapes
    assert "llm.layers.0.self_attn.q_norm.weight" not in shapes


def test_expected_shapes_equal_cached_omnivoice_checkpoints():
    # The config-derived table is exactly the checkpoint's tensors, for every OmniVoice
    # snapshot in the local hf cache (stock and fine-tuned); skipped when there are none
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    snapshots = [path for path in glob.glob(os.path.join(hub, "models--*", "snapshots", "*"))
                 if os.path.isfile(os.path.join(path, "model.safetensors"))
                 and os.path.isfile(os.path.join(path, "config.json"))
                 and '"omnivoice"' in open(os.path.join(path, "config.json")).read()]
    if not snapshots:
        pytest.skip("no OmniVoice checkpoint in the hf cache")
    for path in snapshots:
        config = json.loads(open(os.path.join(path, "config.json")).read())
        actual = {key: info.shape for key, info in SafetensorsFile(os.path.join(path, "model.safetensors")).tensors.items()}
        assert expected_shapes(config) == actual, path
