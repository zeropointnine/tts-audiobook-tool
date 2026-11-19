"""Applies an OmniVoice LoRA adapter by merging it into the loaded model's weights.

Upstream OmniVoice trains adapters with PEFT and applies them with
`PeftModel.from_pretrained(model, dir).merge_and_unload()`
(`omnivoice.utils.lora.load_lora_adapter`, git master only). This module does
the same merge by hand so that peft is not a dependency;
`testx/omnivoice_lora_peft_parity.py` checks that the two produce the same
weights (they are bit-identical in fp16).

An adapter directory holds `adapter_config.json` and `adapter_model.safetensors`.
Tensor keys are relative to the whole `OmniVoice` module:

    base_model.model.llm.layers.N.<module>.<proj>.lora_{A,B}.weight   (LoRA pairs)
    base_model.model.audio_embeddings.weight                          (full replacements,
    base_model.model.audio_heads.weight                                PEFT modules_to_save)

Merging computes `W + (alpha / r) * B @ A` (alpha / sqrt(r) for rsLoRA) in fp32
and rounds once to the model dtype, and copies the replacements over.

Everything is validated before any weight changes, so an adapter that is
rejected (`LoraLoadError`) leaves the model untouched. Only plain LoRA as the
upstream recipe writes it is supported; anything else (DoRA, per-module rank or
alpha patterns, LoRA biases, embedding LoRA, other PEFT types, adapters for
another base) is rejected rather than merged approximately. The merge cannot be
undone in place: applying a different adapter requires reloading the model.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch

ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILE_NAME = "adapter_model.safetensors"
ADAPTER_FILE_NAMES = [ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME]

_KEY_PREFIX = "base_model.model."
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


class LoraLoadError(Exception):
    """The adapter could not be resolved or does not fit the model; the model is unchanged."""


@dataclass
class LoraMergePlan:
    """A validated adapter, ready to merge. Tensors are on the CPU."""
    rank: int
    alpha: float
    scale: float
    # Model parameter name -> (A, B)
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    # Model parameter name -> replacement value
    replacements: dict[str, torch.Tensor] = field(default_factory=dict)

    def describe(self) -> str:
        return (f"{len(self.pairs)} LoRA layers (r={self.rank}, alpha={self.alpha:g}), "
                f"{len(self.replacements)} replaced")


def resolve_adapter_dir(target: str) -> str:
    """A local directory, an hf repo id, or `owner/repo/sub/dir`; downloads only the adapter files."""
    from tts_audiobook_tool import target_util
    try:
        directory = target_util.resolve_to_local_dir(target, ADAPTER_FILE_NAMES)
    except Exception as e:
        raise LoraLoadError(f"Couldn't get adapter files for {target}: {e}") from e
    for name in ADAPTER_FILE_NAMES:
        if not os.path.isfile(os.path.join(directory, name)):
            raise LoraLoadError(f"Missing {name} in {target}")
    return directory


def plan_merge(model: torch.nn.Module, adapter_dir: str) -> LoraMergePlan:
    """Reads and validates an adapter against `model` without changing it."""
    try:
        return _plan_merge(model, adapter_dir)
    except LoraLoadError:
        raise
    except Exception as e:
        raise LoraLoadError(f"Couldn't read adapter: {e}") from e


def _plan_merge(model: torch.nn.Module, adapter_dir: str) -> LoraMergePlan:
    from safetensors.torch import load_file

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
    scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)

    params = dict(model.named_parameters())
    plan = LoraMergePlan(rank=rank, alpha=alpha, scale=scale)
    halves: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(os.path.join(adapter_dir, ADAPTER_WEIGHTS_FILE_NAME)).items():
        if not key.startswith(_KEY_PREFIX):
            raise LoraLoadError(
                f"Unexpected tensor {key}; expected an adapter for the whole OmniVoice model, "
                "as written by OmniVoice's LoRA training")
        name = key.removeprefix(_KEY_PREFIX).replace(".modules_to_save.default", "")
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
        plan.replacements[name] = value

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


@torch.no_grad()
def apply_merge(model: torch.nn.Module, plan: LoraMergePlan) -> None:
    """Merges a validated plan into `model` in place. An exception here can leave
    the model partially merged, so the caller must discard it."""
    params = dict(model.named_parameters())
    for name, value in plan.replacements.items():
        target = params[name]
        target.copy_(value.to(device=target.device, dtype=target.dtype))
    for name, (a, b) in plan.pairs.items():
        target = params[name]
        # The delta is computed on the CPU in fp32, so the result doesn't
        # depend on GPU matmul settings (eg TF32)
        delta = (b.float() @ a.float()) * plan.scale
        merged = target.float() + delta.to(target.device)
        target.copy_(merged.to(target.dtype))


def load_lora(model: torch.nn.Module, target: str) -> LoraMergePlan:
    """
    Resolves, validates and merges an adapter into `model`. Raises `LoraLoadError`
    (model unchanged) when the adapter can't be used; any other exception means
    the merge failed partway and the model must be discarded.
    """
    plan = plan_merge(model, resolve_adapter_dir(target))
    apply_merge(model, plan)
    return plan

