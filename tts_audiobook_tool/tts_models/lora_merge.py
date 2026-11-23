"""Model-agnostic pieces of merging a LoRA adapter into loaded weights.

Format-specific loaders (`omnivoice_lora`, `breeze_lora`) read and validate an
adapter into a `LoraMergePlan` without touching the model, then `apply_merge()`
folds it in. Merging computes `W + scale * B @ A` in fp32 (on the CPU, or on the
weight's device with TF32 off) and rounds once to the model dtype. A merge can't
be subtracted back out exactly (bf16/fp16 rounding), so switching adapters means
restoring the base weights first: see `lora_switch`.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal

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
    caller must discard it (or restore its base weights).

    `delta_device` is where `B @ A` is computed in fp32: "cpu" (slow for large
    adapters, but independent of GPU matmul settings) or "target", the weight's own
    device with TF32 disabled (much faster on CUDA; results can differ from "cpu" in
    the last bits, so a given caller should stick to one)."""
    params = dict(model.named_parameters())
    for name, value in plan.replacements.items():
        target = params[name]
        target.copy_(value.to(device=target.device, dtype=target.dtype))
    with _tf32_disabled():
        for name, (a, b) in plan.pairs.items():
            target = params[name]
            device = torch.device("cpu") if delta_device == "cpu" else target.device
            delta = (b.to(device).float() @ a.to(device).float()) * plan.scale
            merged = target.float() + delta.to(target.device)
            target.copy_(merged.to(target.dtype))
            del delta, merged
