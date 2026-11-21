"""Model-agnostic pieces of merging a LoRA adapter into loaded weights.

Format-specific loaders (`omnivoice_lora`, `breeze_lora`) read and validate an
adapter into a `LoraMergePlan` without touching the model, then `apply_merge()`
folds it in. Merging computes `W + scale * B @ A` in fp32 on the CPU and rounds
once to the model dtype. The merge cannot be undone in place: applying a
different adapter requires reloading the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


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
