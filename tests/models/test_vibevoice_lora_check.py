"""Model-free VibeVoice LoRA check (`vibevoice_lora_check`): the voice menu's check, against the
language model shapes the base model's config.json implies instead of the loaded model.

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
from tts_audiobook_tool.tts_models.vibevoice_lora_check import (
    BIN_WEIGHTS_WARNING, LANGUAGE_MODEL_PREFIX, check_adapter, expected_shapes,
)

RANK = 2
# A tiny Qwen2 decoder: hidden 8, 2 heads of 4 (q 8), 1 kv head (k/v 4), mlp 12, 2 layers
BASE_CONFIG = {
    "model_type": "vibevoice",
    "decoder_config": {"model_type": "qwen2", "hidden_size": 8, "intermediate_size": 12, "num_attention_heads": 2,
                       "num_key_value_heads": 1, "num_hidden_layers": 2, "vocab_size": 10},
}
# Keys relative to the language model, as VibeVoice-finetuning's PeftModel writes them
Q = "base_model.model.layers.1.self_attn.q_proj"
K = "base_model.model.layers.0.self_attn.k_proj"


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
    }


def write_adapter(directory, tensors=None, *, bin_format=False, **config_overrides) -> ModelTarget:
    directory.mkdir(parents=True, exist_ok=True)
    config = {"peft_type": "LORA", "r": RANK, "lora_alpha": 4, "use_dora": False, "bias": "none", **config_overrides}
    (directory / "adapter_config.json").write_text(json.dumps(config))
    if bin_format:
        # Never read by the check (a pickle needs torch)
        (directory / "adapter_model.bin").write_bytes(b"not read")
    else:
        write_safetensors(directory / "adapter_model.safetensors", adapter_tensors() if tensors is None else tensors)
    return ModelTarget.local(str(directory))


@pytest.mark.parametrize("subdir", ["", "lora"])
def test_valid_adapter_passes_against_the_base_config(tmp_path, subdir):
    # The adapter may be the directory itself or sit in the trainer's lora/ subdirectory
    write_adapter(tmp_path / "adapter" / subdir if subdir else tmp_path / "adapter")
    result = check_adapter(ModelTarget.local(str(tmp_path / "adapter")), write_base(tmp_path / "base"), 0.5)
    assert result.error == "" and result.warnings == ()


def edited(**changes):
    return lambda tensors: {**{k: v for k, v in tensors.items() if k not in changes}, **{
        k: v for k, v in changes.items() if v is not None}}


@pytest.mark.parametrize("edit,message", [
    (edited(**{f"{Q}.lora_B.weight": None}), "Incomplete LoRA pair"),
    # k_proj is (4, 8) in this base: a B shaped for q_proj fits the rank but not the base
    (edited(**{f"{K}.lora_B.weight": np.ones((8, RANK), np.float32)}), "Shape mismatch"),
    (edited(**{"base_model.model.layers.2.mlp.up_proj.lora_A.weight": np.ones((RANK, 8), np.float32),
               "base_model.model.layers.2.mlp.up_proj.lora_B.weight": np.ones((12, RANK), np.float32)}),
     "which the model doesn't have"),
    (edited(**{"base_model.model.embed_tokens.modules_to_save.default.weight": np.ones((11, 8), np.float32)}),
     f"Shape mismatch for {LANGUAGE_MODEL_PREFIX}embed_tokens.weight"),
    # A whole-model key (eg the diffusion head) isn't a language model adapter
    (edited(**{"model.prediction_head.weight": np.ones((4, 4), np.float32)}), "VibeVoice-finetuning"),
    (edited(**{f"{Q}.lora_A.weight": np.full((RANK, 8), math.nan, np.float32)}), "NaN or infinite"),
    (edited(**{f"{K}.lora_B.weight": np.full((4, RANK), -math.inf, np.float32)}), "NaN or infinite"),
    (lambda tensors: {}, "no tensors"),
])
def test_bad_tensors_are_rejected(tmp_path, edit, message):
    target = write_adapter(tmp_path / "adapter", edit(adapter_tensors()))
    assert message in check_adapter(target, write_base(tmp_path / "base")).error


def test_qkv_bias_replacement_fits_the_qwen2_layout(tmp_path):
    # Qwen2 has q/k/v biases, so a modules_to_save replacement of one is a valid target
    tensors = {**adapter_tensors(), "base_model.model.layers.0.self_attn.k_proj.bias": np.zeros(4, np.float32)}
    result = check_adapter(write_adapter(tmp_path / "adapter", tensors), write_base(tmp_path / "base"))
    assert result.error == ""


def test_unsupported_config_is_rejected(tmp_path):
    target = write_adapter(tmp_path / "adapter", use_dora=True)
    assert "use_dora" in check_adapter(target, write_base(tmp_path / "base")).error


def test_untrained_adapter_is_accepted_with_a_warning(tmp_path):
    target = write_adapter(tmp_path / "adapter", adapter_tensors(b_scale=0.0))
    result = check_adapter(target, write_base(tmp_path / "base"))
    assert result.error == "" and result.warnings == (ZERO_LORA_WARNING,)


def test_directory_without_an_adapter_is_rejected(tmp_path):
    (tmp_path / "adapter" / "lora").mkdir(parents=True)
    (tmp_path / "adapter" / "lora" / "adapter_config.json").write_text("{}")
    result = check_adapter(ModelTarget.local(str(tmp_path / "adapter")), write_base(tmp_path / "base"))
    assert "adapter_model.safetensors" in result.error


def test_bin_adapter_gets_a_config_check_and_a_warning(tmp_path):
    # An older pickled adapter can't be read torch-free: only its config is checked here, and
    # the run's check against the loaded model covers its weights
    base = write_base(tmp_path / "base")
    result = check_adapter(write_adapter(tmp_path / "adapter", bin_format=True), base)
    assert result.error == "" and result.warnings == (BIN_WEIGHTS_WARNING,)
    bad = write_adapter(tmp_path / "bad", bin_format=True, use_dora=True)
    assert "use_dora" in check_adapter(bad, base).error


@pytest.mark.parametrize("config", [None, {"decoder_config": {"model_type": "llama"}}])
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


def test_expected_shapes_equal_cached_vibevoice_language_models():
    # The config-derived table is exactly the checkpoint's language model tensors, for every
    # VibeVoice snapshot in the local hf cache (names from the shard index; shapes too when the
    # shards are downloaded); skipped when there are none
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    snapshots = [path for path in glob.glob(os.path.join(hub, "models--*", "snapshots", "*"))
                 if os.path.isfile(os.path.join(path, "model.safetensors.index.json"))
                 and os.path.isfile(os.path.join(path, "config.json"))
                 and json.loads(open(os.path.join(path, "config.json")).read()).get("model_type") == "vibevoice"]
    if not snapshots:
        pytest.skip("no VibeVoice checkpoint in the hf cache")
    for path in snapshots:
        expected = expected_shapes(json.loads(open(os.path.join(path, "config.json")).read()))
        assert expected is not None, path
        weight_map = json.loads(open(os.path.join(path, "model.safetensors.index.json")).read())["weight_map"]
        assert {key for key in weight_map if key.startswith(LANGUAGE_MODEL_PREFIX)} == set(expected), path
        shards = {os.path.join(path, name) for name in weight_map.values()}
        if all(os.path.isfile(shard) for shard in shards):
            actual = {key: info.shape for shard in shards for key, info in SafetensorsFile(shard).tensors.items()
                      if key.startswith(LANGUAGE_MODEL_PREFIX)}
            assert actual == expected, path
