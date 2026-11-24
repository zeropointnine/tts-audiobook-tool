"""Applies a VibeVoice LoRA adapter by merging it into the loaded model's weights.

Adapters come from the community trainer (https://github.com/voicepowered-ai/VibeVoice-finetuning,
also `vibevoice.finetune`), which wraps only the Qwen2 language model with PEFT, so the adapter's
keys are relative to `model.language_model` (eg `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`).
It is merged by hand (`peft_lora`), not with peft: unmerged PEFT layers cost ~40% more time per
generated frame (two language model passes per frame, CFG), while a merged adapter costs nothing.

An adapter directory holds `adapter_config.json` plus `adapter_model.safetensors` (or an older
`adapter_model.bin`), either directly or in a `lora/` subdirectory (the trainer's output and
checkpoint directories); that torch-free layout handling lives in `vibevoice_lora_check`, the
voice menu's model-free check. The trainer can also save a fine-tuned diffusion head and
connectors next to the adapter (`diffusion_head/`, `acoustic_connector/`, `semantic_connector/`);
like the app's earlier PEFT loading, those are not applied.

Everything is validated before any weight changes, including that no adapter tensor holds a NaN
or infinity. Per-voice adapters are switched by restoring clean base weights and remerging
(`lora_switch`).
"""
from __future__ import annotations

import torch

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models import peft_lora
from tts_audiobook_tool.tts_models.lora_merge import LoraMergePlan, apply_merge
from tts_audiobook_tool.tts_models.lora_switch import PreparedLora
# The torch-free adapter-directory handling, shared with the model-free menu check
from tts_audiobook_tool.tts_models.vibevoice_lora_check import (
    FOREIGN_KEY_HINT, LANGUAGE_MODEL_PREFIX, VibeVoiceAdapter, resolve_adapter,
)


def plan_merge(model: torch.nn.Module, adapter: VibeVoiceAdapter, strength: float = 1.0) -> LoraMergePlan:
    """Reads and validates an adapter against the whole VibeVoice model without changing it."""
    plan = peft_lora.plan_merge(
        model, adapter.directory, strength, weights_file_name=adapter.weights_file_name,
        param_prefix=LANGUAGE_MODEL_PREFIX, foreign_key_hint=FOREIGN_KEY_HINT,
    )
    peft_lora.check_plan_finite(plan)
    return plan


def prepare_lora(model: torch.nn.Module, target: ModelTarget, strength: float = 1.0) -> PreparedLora:
    """Validates an adapter for per-voice switching, without changing the loaded weights."""
    adapter = resolve_adapter(target)
    plan = plan_merge(model, adapter, strength)
    return PreparedLora(plan, peft_lora.make_identity(adapter.weights_path, plan), target.display_name())


def load_lora(model: torch.nn.Module, target: ModelTarget, strength: float = 1.0) -> LoraMergePlan:
    """Resolves, validates and merges an adapter into `model` as per-voice switching does.
    For checks and probes. Raises `LoraLoadError` (model unchanged) when the adapter can't be used."""
    plan = plan_merge(model, resolve_adapter(target), strength)
    apply_merge(model, plan, delta_device="target")
    return plan
