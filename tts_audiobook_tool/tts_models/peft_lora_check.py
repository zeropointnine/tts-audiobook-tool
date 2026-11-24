"""Torch-free, model-free check of a plain PEFT LoRA adapter's weights, shared by the voice menu's
checks for the PEFT-format models (`omnivoice_lora_check`, `vibevoice_lora_check`).

The adapter's tensors are checked as the merge planner (`peft_lora.plan_merge`) checks them: key
layout, complete A/B pairs, float dtypes, plus values (NaN/inf rejects; all-zero B factors warn).
Where the planner compares against the loaded model's parameters, this compares against the
shapes the base model's `config.json` implies (each model's `expected_shapes`, built on
`qwen_language_model_shapes`). Without them (a base whose config can't be read or isn't
recognized), the factors need only agree with each other and the rank.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.lora_files import (
    LoraLoadError, SafetensorsFile, TensorInfo, check_values, read_json_object,
)
from tts_audiobook_tool.tts_models.peft_lora_config import (
    PeftLoraConfig, check_replacement_name, parameter_name, split_pair,
)

BASE_CONFIG_FILE_NAME = "config.json"
# Appended to a warning when the adapter can't be fully checked here
CHECKED_LATER = "the adapter will be checked against the model when it loads"

# Language model types whose parameter layout `qwen_language_model_shapes` knows
_QWEN_TYPES = ("qwen3", "qwen2")

Shapes = dict[str, tuple[int, ...]]


def positive_int(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(key)
    return value


def qwen_language_model_shapes(llm: object, prefix: str) -> Shapes | None:
    """Parameter name -> shape of a Qwen3 / Qwen2 language model (`Qwen*Model`, no lm_head) with
    this config, its names under `prefix` (eg "llm."). None when it isn't a recognized config
    (another model type, or missing dimensions)."""
    if not isinstance(llm, dict) or llm.get("model_type") not in _QWEN_TYPES:
        return None
    try:
        hidden = positive_int(llm, "hidden_size")
        intermediate = positive_int(llm, "intermediate_size")
        heads = positive_int(llm, "num_attention_heads")
        layers = positive_int(llm, "num_hidden_layers")
        vocab = positive_int(llm, "vocab_size")
        kv_heads = positive_int(llm, "num_key_value_heads") if llm.get("num_key_value_heads") is not None else heads
        head_dim = positive_int(llm, "head_dim") if llm.get("head_dim") is not None else hidden // heads
    except ValueError:
        return None
    # Qwen2 always has q/k/v biases; Qwen3 has them (and o's) only with attention_bias
    qwen3 = llm.get("model_type") == "qwen3"
    biases = ("q", "k", "v", "o") if qwen3 and llm.get("attention_bias") else ("q", "k", "v") if not qwen3 else ()

    q_out, kv_out = heads * head_dim, kv_heads * head_dim
    shapes: Shapes = {
        f"{prefix}embed_tokens.weight": (vocab, hidden),
        f"{prefix}norm.weight": (hidden,),
    }
    projections = {
        "self_attn.q_proj": (q_out, hidden), "self_attn.k_proj": (kv_out, hidden),
        "self_attn.v_proj": (kv_out, hidden), "self_attn.o_proj": (hidden, q_out),
        "mlp.gate_proj": (intermediate, hidden), "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }
    for index in range(layers):
        layer = f"{prefix}layers.{index}."
        for name, shape in projections.items():
            shapes[f"{layer}{name}.weight"] = shape
        for name in biases:
            shapes[f"{layer}self_attn.{name}_proj.bias"] = projections[f"self_attn.{name}_proj"][:1]
        shapes[f"{layer}input_layernorm.weight"] = (hidden,)
        shapes[f"{layer}post_attention_layernorm.weight"] = (hidden,)
        if qwen3:
            shapes[f"{layer}self_attn.q_norm.weight"] = (head_dim,)
            shapes[f"{layer}self_attn.k_norm.weight"] = (head_dim,)
    return shapes


def read_base_shapes(
        base_target: ModelTarget, expected_shapes: Callable[[dict], Shapes | None], model_name: str,
) -> tuple[Shapes | None, str]:
    """(`expected_shapes` of the base's config.json, "") or (None, a warning). Only config.json is
    read (downloaded for an hf base, if not cached)."""
    try:
        directory = base_target.resolve_local_dir([BASE_CONFIG_FILE_NAME])
        config = read_json_object(os.path.join(directory, BASE_CONFIG_FILE_NAME), "the base model's config.json")
    except Exception as e:
        return None, f"Couldn't read the base model's config ({e}); {CHECKED_LATER}"
    shapes = expected_shapes(config)
    if shapes is None:
        return None, f"The base model's config.json isn't a recognized {model_name} config; {CHECKED_LATER}"
    return shapes, ""


def check_weights(
        file: SafetensorsFile,
        config: PeftLoraConfig,
        expected: Shapes | None,
        *,
        param_prefix: str = "",
        foreign_key_hint: str = "",
) -> list[str]:
    """Checks the adapter's tensors against `expected` (None: internal consistency only); raises
    `LoraLoadError`, else returns warnings. `param_prefix` and `foreign_key_hint` are as for
    `peft_lora.plan_merge`."""
    halves: dict[str, dict[str, tuple[str, TensorInfo]]] = {}
    replacements = 0
    for key, info in file.tensors.items():
        name = parameter_name(key, param_prefix, foreign_key_hint)
        pair = split_pair(name)
        if pair:
            halves.setdefault(pair[0], {})[pair[1]] = (key, info)
            continue
        check_replacement_name(key, name)
        if expected is not None:
            shape = expected.get(name)
            if shape is None:
                raise LoraLoadError(f"Adapter replaces {name}, which the model doesn't have")
            if info.shape != shape:
                raise LoraLoadError(f"Shape mismatch for {name}: adapter {info.shape}, model {shape}")
        if not info.is_float:
            raise LoraLoadError(f"Non-float adapter tensor: {key}")
        replacements += 1

    rank = config.rank
    b_keys = []
    for name, half in halves.items():
        if "A" not in half or "B" not in half:
            raise LoraLoadError(f"Incomplete LoRA pair for {name}")
        (_, a), (b_key, b) = half["A"], half["B"]
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise LoraLoadError(f"LoRA target {name} is not a matrix")
        if expected is not None:
            shape = expected.get(name)
            if shape is None:
                raise LoraLoadError(f"Adapter targets {name}, which the model doesn't have")
            if len(shape) != 2:
                raise LoraLoadError(f"LoRA target {name} is not a matrix")
            out_features, in_features = shape
            model_text = f"model {shape}"
        else:
            # Without the base's shapes, the factors need only agree with each other and the rank
            out_features, in_features = b.shape[0], a.shape[1]
            model_text = "expected A (r, in), B (out, r)"
        if a.shape != (rank, in_features) or b.shape != (out_features, rank):
            raise LoraLoadError(f"Shape mismatch for {name}: A {a.shape}, B {b.shape}, {model_text}, r={rank}")
        if not a.is_float or not b.is_float:
            raise LoraLoadError(f"Non-float LoRA factors for {name}")
        b_keys.append(b_key)

    if not halves and not replacements:
        raise LoraLoadError("Adapter has no tensors")
    return check_values(file, list(file.tensors), b_keys)
