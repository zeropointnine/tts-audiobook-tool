"""Applies a VibeVoice LoRA adapter by merging it into the loaded model's weights.

Adapters come from the community trainer (https://github.com/voicepowered-ai/VibeVoice-finetuning,
also `vibevoice.finetune`), which wraps only the Qwen2 language model with PEFT, so the adapter's
keys are relative to `model.language_model` (eg `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`).
It is merged by hand (`peft_lora`), not with peft: unmerged PEFT layers cost ~40% more time per
generated frame (two language model passes per frame, CFG), while a merged adapter costs nothing.

An adapter directory holds `adapter_config.json` plus `adapter_model.safetensors` (or an older
`adapter_model.bin`), either directly or in a `lora/` subdirectory (the trainer's output and
checkpoint directories). The trainer can also save a fine-tuned diffusion head and connectors
next to the adapter (`diffusion_head/`, `acoustic_connector/`, `semantic_connector/`); like the
app's earlier PEFT loading, those are not applied.

Per-voice adapters are switched by restoring clean base weights and remerging (`lora_switch`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models import peft_lora
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan, apply_merge
from tts_audiobook_tool.tts_models.lora_switch import PreparedLora
from tts_audiobook_tool.tts_models.peft_lora import (
    ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_BIN_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME,
)

# The trainer saves the adapter here, inside its output and checkpoint directories
LORA_SUBDIR = "lora"
# Where the adapter's keys (relative to the language model) live in the whole model
LANGUAGE_MODEL_PREFIX = "model.language_model."

_WEIGHTS_FILE_NAMES = [ADAPTER_WEIGHTS_FILE_NAME, ADAPTER_WEIGHTS_BIN_FILE_NAME]
_FILE_NAMES = [ADAPTER_CONFIG_FILE_NAME, *_WEIGHTS_FILE_NAMES]
# Files that make a local directory (or hf repo) acceptable (`ModelTarget.exist_test` markers)
LOCAL_FILE_MARKERS = [ADAPTER_CONFIG_FILE_NAME, f"{LORA_SUBDIR}/{ADAPTER_CONFIG_FILE_NAME}"]

_FOREIGN_KEY_HINT = ("expected a language model adapter, as written by VibeVoice-finetuning "
                     "(voicepowered-ai/VibeVoice-finetuning)")


@dataclass(frozen=True)
class VibeVoiceAdapter:
    directory: str
    weights_file_name: str

    @property
    def weights_path(self) -> str:
        return os.path.join(self.directory, self.weights_file_name)


def find_adapter(directory: str) -> VibeVoiceAdapter | None:
    """The adapter in `directory` itself, else in its `lora/` subdirectory; None if neither has one."""
    for candidate in (directory, os.path.join(directory, LORA_SUBDIR)):
        if not os.path.isfile(os.path.join(candidate, ADAPTER_CONFIG_FILE_NAME)):
            continue
        for name in _WEIGHTS_FILE_NAMES:
            if os.path.isfile(os.path.join(candidate, name)):
                return VibeVoiceAdapter(candidate, name)
    return None


def resolve_adapter(target: ModelTarget) -> VibeVoiceAdapter:
    """A local directory, an hf repo, or a directory inside one; downloads only the adapter files."""
    try:
        names = [*_FILE_NAMES, *(f"{LORA_SUBDIR}/{name}" for name in _FILE_NAMES)]
        directory = target.resolve_local_dir(names)
    except Exception as e:
        raise LoraLoadError(f"Couldn't get adapter files for {target.value}: {e}") from e
    adapter = find_adapter(directory)
    if adapter is None:
        raise LoraLoadError(f"No {ADAPTER_CONFIG_FILE_NAME} with {ADAPTER_WEIGHTS_FILE_NAME} "
                            f"in {target.value} (or its {LORA_SUBDIR}/ subdirectory)")
    return adapter


def plan_merge(model: torch.nn.Module, adapter: VibeVoiceAdapter, strength: float = 1.0) -> LoraMergePlan:
    """Reads and validates an adapter against the whole VibeVoice model without changing it."""
    return peft_lora.plan_merge(
        model, adapter.directory, strength, weights_file_name=adapter.weights_file_name,
        param_prefix=LANGUAGE_MODEL_PREFIX, foreign_key_hint=_FOREIGN_KEY_HINT,
    )


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
