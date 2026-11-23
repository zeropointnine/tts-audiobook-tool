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

Merging computes `W + strength * (alpha / r) * B @ A` (alpha / sqrt(r) for
rsLoRA) in fp32 on the weight's device (TF32 off) and rounds once to the model
dtype. Replacements are copied over at strength 1, else blended from the base:
`W + strength * (R - W)`.

Everything is validated before any weight changes, so an adapter that is
rejected (`LoraLoadError`) leaves the model untouched. Only plain LoRA as the
upstream recipe writes it is supported; anything else (DoRA, per-module rank or
alpha patterns, LoRA biases, embedding LoRA, other PEFT types, adapters for
another base) is rejected rather than merged approximately. Per-voice adapters
are switched by restoring clean base weights and remerging (`lora_switch`), not
by subtracting a rounded delta.

Strength (`VoiceLora.strength`, 1 when the project has none) scales the adapter's
whole change to the weights: the low-rank delta, and the full-rank change each
modules_to_save replacement implies (`R - W`; the upstream recipe also retrains
the audio embeddings and heads). Scaling only the LoRA would pair a partial
low-rank change with fully fine-tuned embeddings and heads, which were trained
together. The replacements' implied change is only meaningful against the base
the adapter was trained on; PEFT adapters don't record which (a trainer-local
`base_model_name_or_path`, no checksums), so that can't be verified.
The PEFT-format parts live in `peft_lora` (shared with VibeVoice), the
format-independent ones in `lora_merge`.
"""
from __future__ import annotations

import os

import torch

from tts_audiobook_tool.app_types.model_target import ModelTarget
# Re-exported: callers and tests import these from here
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan, apply_merge
from tts_audiobook_tool.tts_models.lora_switch import PreparedLora
from tts_audiobook_tool.tts_models import peft_lora
from tts_audiobook_tool.tts_models.peft_lora import ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME

ADAPTER_FILE_NAMES = [ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME]

_FOREIGN_KEY_HINT = "expected an adapter for the whole OmniVoice model, as written by OmniVoice's LoRA training"


def resolve_adapter_dir(target: ModelTarget) -> str:
    """A local directory, an hf repo, or a directory inside one; downloads only the adapter files."""
    try:
        directory = target.resolve_local_dir(ADAPTER_FILE_NAMES)
    except Exception as e:
        raise LoraLoadError(f"Couldn't get adapter files for {target.value}: {e}") from e
    for name in ADAPTER_FILE_NAMES:
        if not os.path.isfile(os.path.join(directory, name)):
            raise LoraLoadError(f"Missing {name} in {target.value}")
    return directory


def plan_merge(model: torch.nn.Module, adapter_dir: str, strength: float = 1.0) -> LoraMergePlan:
    """Reads and validates an adapter without changing `model`."""
    return peft_lora.plan_merge(model, adapter_dir, strength, foreign_key_hint=_FOREIGN_KEY_HINT)


def prepare_lora(model: torch.nn.Module, target: ModelTarget, strength: float = 1.0) -> PreparedLora:
    """Validates an adapter for per-voice switching, without changing the loaded weights."""
    directory = resolve_adapter_dir(target)
    plan = plan_merge(model, directory, strength)
    identity = peft_lora.make_identity(os.path.join(directory, ADAPTER_WEIGHTS_FILE_NAME), plan)
    return PreparedLora(plan, identity, target.display_name())


def load_lora(model: torch.nn.Module, target: ModelTarget, strength: float = 1.0) -> LoraMergePlan:
    """
    Resolves, validates and merges an adapter into `model`, computing the delta as
    per-voice switching does (`lora_switch`). For checks and probes; the app
    switches adapters with `prepare_lora`. Raises `LoraLoadError` (model unchanged)
    when the adapter can't be used; any other exception means the merge failed
    partway and the model must be discarded.
    """
    plan = plan_merge(model, resolve_adapter_dir(target), strength)
    apply_merge(model, plan, delta_device="target")
    return plan
