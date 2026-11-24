"""Model-agnostic pieces of merging a LoRA adapter into loaded weights.

Format-specific loaders (`omnivoice_lora`, `breeze_lora`) read and validate an
adapter into a `LoraMergePlan` without touching the model, then `apply_merge()`
folds it in. Merging computes `W + scale * B @ A` in fp32 (on the CPU, or on the
weight's device with TF32 off) and rounds once to the model dtype; full
replacements are copied, or at a `replacement_strength` below/above 1 blended from
the base toward the replacement. A merge can't
be subtracted back out exactly (bf16/fp16 rounding), so switching adapters means
restoring the base weights first: see `lora_switch`.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal

import torch

# Re-exported: defined torch-free, for the model-free checks
from tts_audiobook_tool.tts_models.lora_files import LoraLoadError, nonfinite_message


def check_finite(tensors: dict[str, torch.Tensor]) -> None:
    """Rejects adapter tensors when any holds a NaN or infinity: merged, they would
    silently corrupt the weights (eg silence or noise) rather than fail."""
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise LoraLoadError(nonfinite_message(name))


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
    # How far each replaced weight moves from its base value toward the replacement:
    # `base + s * (replacement - base)`, ie a scale on the full-rank change the replacement
    # implies (the counterpart of LoRA strength, which is folded into `scale`). 1 copies it.
    replacement_strength: float = 1.0

    def describe(self) -> str:
        return (f"{len(self.pairs)} LoRA layers (r={self.rank}, alpha={self.alpha:g}), "
                f"{len(self.replacements)} replaced")

    @property
    def target_names(self) -> set[str]:
        """Every model parameter the merge writes."""
        return set(self.pairs) | set(self.replacements)


@contextmanager
def _tf32_disabled() -> Iterator[None]:
    """Full-fp32 CUDA matmuls, so a device-side delta doesn't depend on global matmul settings."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def apply_merge(
        model: torch.nn.Module, plan: LoraMergePlan, delta_device: Literal["cpu", "target"] = "cpu",
) -> None:
    """Merges a validated plan into `model` in place (`copy_`, so parameter storage
    doesn't move). An exception here can leave the model partially merged, so the
    caller must discard it (or restore its base weights). With a `replacement_strength`
    other than 1, replaced weights must hold their base values (pristine or restored
    weights; `lora_switch` restores before every merge), since the blend starts there.

    `delta_device` is where `B @ A` is computed in fp32: "cpu" (slow for large
    adapters, but independent of GPU matmul settings) or "target", the weight's own
    device with TF32 disabled (much faster on CUDA; results can differ from "cpu" in
    the last bits, so a given caller should stick to one)."""
    params = dict(model.named_parameters())
    for name, value in plan.replacements.items():
        target = params[name]
        if plan.replacement_strength == 1.0:
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            continue
        # In fp32, rounded once, as for LoRA deltas
        base = target.float()
        blended = base + (value.to(target.device).float() - base) * plan.replacement_strength
        target.copy_(blended.to(target.dtype))
        del base, blended
    with _tf32_disabled():
        for name, (a, b) in plan.pairs.items():
            target = params[name]
            device = torch.device("cpu") if delta_device == "cpu" else target.device
            delta = (b.to(device).float() @ a.to(device).float()) * plan.scale
            merged = target.float() + delta.to(target.device)
            target.copy_(merged.to(target.dtype))
            del delta, merged
