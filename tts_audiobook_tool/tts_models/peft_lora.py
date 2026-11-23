"""Plans the merge of a plain PEFT LoRA adapter (`adapter_config.json` + adapter weights)
into a loaded model, without peft. Format-level logic shared by the PEFT-format models
(`omnivoice_lora`, `vibevoice_lora`); merging itself is `lora_merge.apply_merge`.

Tensor keys carry PEFT's `base_model.model.` prefix, relative to the module the adapter
was trained on; `param_prefix` maps them onto the loaded model's parameter names
(eg VibeVoice trains on its language model only: `model.language_model.`):

    base_model.model.<module>.lora_{A,B}[.default].weight   LoRA pairs
    base_model.model.<module>[.modules_to_save.default].{weight,bias}
                                                             full replacements (modules_to_save)

Merging computes `W + strength * (alpha / r) * B @ A` (alpha / sqrt(r) for rsLoRA).
Replacements are copied at strength 1, else blended from the base: `W + strength * (R - W)`.

Only plain LoRA is supported; anything else (DoRA, per-module rank or alpha patterns,
LoRA biases, embedding LoRA, other PEFT types) is rejected rather than merged
approximately. Everything is validated before any weight changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import torch

from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan

ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILE_NAME = "adapter_model.safetensors"
# Older peft versions (and `save_pretrained(safe_serialization=False)`) write a pickle
ADAPTER_WEIGHTS_BIN_FILE_NAME = "adapter_model.bin"

KEY_PREFIX = "base_model.model."
_PAIR_SUFFIXES = {
    ".lora_A.weight": "A", ".lora_B.weight": "B",
    ".lora_A.default.weight": "A", ".lora_B.default.weight": "B",
}
# Config fields that change what the adapter computes, with the value plain LoRA has
_PLAIN_LORA_CONFIG = {
    "use_dora": False,
    "fan_in_fan_out": False,
    "use_qalora": False,
    "lora_bias": False,
    "bias": "none",
    "rank_pattern": {},
    "alpha_pattern": {},
    "layer_replication": None,
    "target_parameters": None,
    "trainable_token_indices": None,
    "alora_invocation_tokens": None,
    "use_bdlora": None,
    "arrow_config": None,
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter_tensors(weights_path: str) -> dict[str, torch.Tensor]:
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weights_path)
    value = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        raise LoraLoadError(f"{os.path.basename(weights_path)} is not a tensor state dict")
    return value


def plan_merge(
        model: torch.nn.Module,
        adapter_dir: str,
        strength: float = 1.0,
        *,
        weights_file_name: str = ADAPTER_WEIGHTS_FILE_NAME,
        param_prefix: str = "",
        foreign_key_hint: str = "",
) -> LoraMergePlan:
    """Reads and validates an adapter without changing `model`. `foreign_key_hint` is appended
    to the error for a tensor without PEFT's key prefix."""
    try:
        return _plan_merge(model, adapter_dir, strength, weights_file_name, param_prefix, foreign_key_hint)
    except LoraLoadError:
        raise
    except Exception as e:
        raise LoraLoadError(f"Couldn't read adapter: {e}") from e


def _plan_merge(
        model: torch.nn.Module,
        adapter_dir: str,
        strength: float,
        weights_file_name: str,
        param_prefix: str,
        foreign_key_hint: str,
) -> LoraMergePlan:
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or strength <= 0):
        raise LoraLoadError(f"Invalid LoRA strength: {strength!r}")
    strength = float(strength)

    with open(os.path.join(adapter_dir, ADAPTER_CONFIG_FILE_NAME), encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise LoraLoadError("Adapter config is not a JSON object")
    peft_type = config.get("peft_type", "LORA")
    if peft_type != "LORA":
        raise LoraLoadError(f"Unsupported adapter type: {peft_type} (only LoRA is supported)")
    for name, plain in _PLAIN_LORA_CONFIG.items():
        value = config.get(name, plain)
        # Unset collections are written as null or empty depending on the peft version
        if value != plain and not (plain in (None, {}) and not value):
            raise LoraLoadError(f"Unsupported adapter feature: {name} = {value!r}")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise LoraLoadError(f"Invalid LoRA rank in adapter config: {rank!r}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise LoraLoadError(f"Invalid lora_alpha in adapter config: {alpha!r}")
    alpha = float(alpha)
    scale = strength * alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)

    params = dict(model.named_parameters())
    plan = LoraMergePlan(rank=rank, alpha=alpha, scale=scale, replacement_strength=strength)
    halves: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_adapter_tensors(os.path.join(adapter_dir, weights_file_name)).items():
        if not key.startswith(KEY_PREFIX):
            hint = f"; {foreign_key_hint}" if foreign_key_hint else ""
            raise LoraLoadError(f"Unexpected tensor {key}{hint}")
        name = param_prefix + key.removeprefix(KEY_PREFIX).replace(".modules_to_save.default", "")
        suffix = next((s for s in _PAIR_SUFFIXES if name.endswith(s)), None)
        if suffix:
            halves.setdefault(name.removesuffix(suffix) + ".weight", {})[_PAIR_SUFFIXES[suffix]] = value
            continue
        if "lora_" in name or not name.endswith((".weight", ".bias")):
            raise LoraLoadError(f"Unsupported adapter tensor: {key}")
        target = params.get(name)
        if target is None:
            raise LoraLoadError(f"Adapter replaces {name}, which the model doesn't have")
        if tuple(value.shape) != tuple(target.shape):
            raise LoraLoadError(
                f"Shape mismatch for {name}: adapter {tuple(value.shape)}, model {tuple(target.shape)}")
        if not value.is_floating_point():
            raise LoraLoadError(f"Non-float adapter tensor: {key}")
        # A copied replacement is rounded now, as the merge would (halving an fp32 one's RAM while
        # the plan is cached); a blended one keeps its precision, so the blend rounds only once
        plan.replacements[name] = value.to(target.dtype) if strength == 1.0 else value

    for name, half in halves.items():
        if "A" not in half or "B" not in half:
            raise LoraLoadError(f"Incomplete LoRA pair for {name}")
        a, b = half["A"], half["B"]
        target = params.get(name)
        if target is None:
            raise LoraLoadError(f"Adapter targets {name}, which the model doesn't have")
        if target.ndim != 2 or a.ndim != 2 or b.ndim != 2:
            raise LoraLoadError(f"LoRA target {name} is not a matrix")
        out_features, in_features = target.shape
        if tuple(a.shape) != (rank, in_features) or tuple(b.shape) != (out_features, rank):
            raise LoraLoadError(
                f"Shape mismatch for {name}: A {tuple(a.shape)}, B {tuple(b.shape)}, "
                f"model {tuple(target.shape)}, r={rank}")
        if not a.is_floating_point() or not b.is_floating_point():
            raise LoraLoadError(f"Non-float LoRA factors for {name}")
        plan.pairs[name] = (a, b)

    if not plan.pairs and not plan.replacements:
        raise LoraLoadError("Adapter has no tensors")
    return plan


def make_identity(weights_path: str, plan: LoraMergePlan) -> str:
    """`PreparedLora.identity`: the weights file's checksum plus the scaling metadata, which
    lives outside the tensor file (including rsLoRA's sqrt(r)), so a change to either remerges."""
    return (f"{sha256_file(weights_path)}:{plan.rank}:{plan.alpha!r}:{plan.scale!r}:"
            f"{plan.replacement_strength!r}")
