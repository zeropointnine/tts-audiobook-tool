"""Applies a Breeze TTS 2 LoRA adapter (Instavar format) by merging it into the loaded model's weights.

Adapters come from Instavar's trainer (https://github.com/instavar/breeze-tts2-finetuning),
which uses its own dependency-free LoRA wrapper, not PEFT, so peft is not needed
here either. Two directory forms are accepted:

    Adapter export / release (local directory or hf repo, optionally `owner/repo/sub/dir`):
        adapter_config.json   Instavar manifest: schema_version 1, artifact_type
                              "breeze_lora_adapter", base_model {id, revision, files: name -> sha256},
                              adapter {file, sha256}, lora {variant, rank, alpha, seed}
        <adapter.file>        the weights (normally adapter.safetensors)

    Training checkpoint (local directory only), eg `.../lora/checkpoint-step-000500`:
        adapter.safetensors
        checkpoint-receipt.json   schema_version 1, status "five_role_checkpoint_complete",
                                  roles {"adapter.safetensors": sha256, ...}
        ../run-config.json        the run's rank and alpha (the checkpoint itself doesn't record them)

Weight keys are module names of `BreezeForConditionalGeneration` plus `.lora_A` / `.lora_B`
(eg `backbone_model.layers.0.self_attn.q_proj.lora_A`, `lm_head.lora_B`); A is (rank, in), B is (out, rank).
Scaling is plain LoRA, `alpha / rank`, times the app's strength multiplier.

A manifest pins the base model's files by sha256; any difference rejects the adapter (the revision
string is not compared: the hub revision can move on without the weights changing). A training
checkpoint pins nothing, so only its keys and shapes are checked against the model.

Everything is validated before any weight changes, so a rejected adapter (`LoraLoadError`) leaves the
model untouched. Must not import `breeze_tts`. The file-level half is torch-free (`breeze_lora_source`),
shared with the voice menu's model-free check (`breeze_lora_check`).

Per-voice switching (`lora_switch`) restores and re-merges weights in place, so it is safe after the CFG
depth graph has been captured: parameter storage never moves, and the graph's fp32 copy of
`depth_decoder.codebooks_head` can't go stale because that head (a `BreezeCodebooksHead`, not a
linear layer) is never a valid LoRA target.
"""
from __future__ import annotations

import math

import torch

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan, apply_merge, check_finite
from tts_audiobook_tool.tts_models.lora_switch import PreparedLora
# File-level handling lives in the torch-free `breeze_lora_source`
from tts_audiobook_tool.tts_models.breeze_lora_source import (
    PAIR_SUFFIXES, BreezeAdapterSource, resolve_adapter, verify_base,
)


def plan_merge(model: torch.nn.Module, source: BreezeAdapterSource, strength: float = 1.0) -> LoraMergePlan:
    """Reads and validates the adapter's tensors against `model` without changing it."""
    try:
        return _plan_merge(model, source, strength)
    except LoraLoadError:
        raise
    except Exception as e:
        raise LoraLoadError(f"Couldn't read adapter: {e}") from e


def _plan_merge(model: torch.nn.Module, source: BreezeAdapterSource, strength: float) -> LoraMergePlan:
    from safetensors.torch import load_file

    if not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength <= 0:
        raise LoraLoadError(f"Invalid LoRA strength: {strength!r}")
    rank = source.rank

    halves: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(source.weights_path).items():
        suffix = next((s for s in PAIR_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            raise LoraLoadError(f"Unsupported adapter tensor: {key} (expected Instavar .lora_A / .lora_B factors)")
        halves.setdefault(key.removesuffix(suffix), {})[PAIR_SUFFIXES[suffix]] = value

    modules = dict(model.named_modules())
    # Parameter storage -> number of names that share it (tied weights)
    storage_counts: dict[int, int] = {}
    for _, param in model.named_parameters(remove_duplicate=False):
        storage_counts[param.data_ptr()] = storage_counts.get(param.data_ptr(), 0) + 1

    plan = LoraMergePlan(rank=rank, alpha=source.alpha, scale=strength * source.alpha / rank)
    for module_name, half in halves.items():
        if "A" not in half or "B" not in half:
            raise LoraLoadError(f"Incomplete LoRA pair for {module_name}")
        module = modules.get(module_name)
        if module is None:
            raise LoraLoadError(f"Adapter targets {module_name}, which the model doesn't have")
        if not isinstance(module, torch.nn.Linear):
            raise LoraLoadError(f"LoRA target {module_name} is not a linear layer")
        weight = module.weight
        if storage_counts.get(weight.data_ptr(), 0) > 1:
            raise LoraLoadError(f"LoRA target {module_name} shares its weight with another layer")
        a, b = half["A"], half["B"]
        if a.ndim != 2 or b.ndim != 2:
            raise LoraLoadError(f"LoRA factors for {module_name} are not matrices")
        out_features, in_features = weight.shape
        if tuple(a.shape) != (rank, in_features) or tuple(b.shape) != (out_features, rank):
            raise LoraLoadError(
                f"Shape mismatch for {module_name}: A {tuple(a.shape)}, B {tuple(b.shape)}, "
                f"model {tuple(weight.shape)}, r={rank}")
        if not a.is_floating_point() or not b.is_floating_point():
            raise LoraLoadError(f"Non-float LoRA factors for {module_name}")
        check_finite({f"{module_name}.lora_A": a, f"{module_name}.lora_B": b})
        plan.pairs[f"{module_name}.weight"] = (a, b)

    if not plan.pairs:
        raise LoraLoadError("Adapter has no tensors")
    return plan


def prepare_lora(model: torch.nn.Module, model_dir: str, target: ModelTarget, strength: float = 1.0) -> PreparedLora:
    """
    Resolves and validates an adapter against `model` and its base files (`model_dir`) without changing
    the model, for `lora_switch.LoraSwitcher`. Raises `LoraLoadError` when the adapter can't be used.
    """
    source = resolve_adapter(target)
    if source.base_files:
        verify_base(model_dir, source)
    plan = plan_merge(model, source, strength)
    # Rank/alpha live in JSON, not the tensor file; changing them must invalidate
    # an already merged adapter too. Strength is tracked separately in LoraKey.
    identity = f"{source.weights_sha256}:{source.rank}:{source.alpha!r}"
    return PreparedLora(plan=plan, identity=identity, display=target.display_name())


def load_lora(
        model: torch.nn.Module, model_dir: str, target: ModelTarget, strength: float = 1.0
) -> tuple[LoraMergePlan, BreezeAdapterSource]:
    """
    Resolves, validates and merges an adapter into `model` (`model_dir`: the loaded base model's files,
    for a manifest's base checksums). Raises `LoraLoadError` (model unchanged) when the adapter can't
    be used; any other exception means the merge failed partway and the model must be discarded.
    """
    source = resolve_adapter(target)
    if source.base_files:
        verify_base(model_dir, source)
    plan = plan_merge(model, source, strength)
    apply_merge(model, plan)
    return plan, source
