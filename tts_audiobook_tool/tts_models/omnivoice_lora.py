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
The format-independent parts live in `lora_merge`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import torch

# Re-exported: callers and tests import these from here
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan, apply_merge
from tts_audiobook_tool.tts_models.lora_switch import PreparedLora

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


def plan_merge(model: torch.nn.Module, adapter_dir: str, strength: float = 1.0) -> LoraMergePlan:
    """Reads and validates an adapter without changing `model`."""
    try:
        return _plan_merge(model, adapter_dir, strength)
    except LoraLoadError:
        raise
    except Exception as e:
        raise LoraLoadError(f"Couldn't read adapter: {e}") from e


def _plan_merge(model: torch.nn.Module, adapter_dir: str, strength: float) -> LoraMergePlan:
    from safetensors.torch import load_file

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


def prepare_lora(model: torch.nn.Module, target: str, strength: float = 1.0) -> PreparedLora:
    """Validates an adapter for per-voice switching, without changing the loaded weights."""
    from tts_audiobook_tool.target_util import make_target_display_name

    directory = resolve_adapter_dir(target)
    plan = plan_merge(model, directory, strength)
    digest = hashlib.sha256()
    with open(os.path.join(directory, ADAPTER_WEIGHTS_FILE_NAME), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    # Scaling metadata lives outside the tensor file (including rsLoRA's sqrt(r));
    # remerge when any of it changes even if the weights file is unchanged.
    identity = f"{digest.hexdigest()}:{plan.rank}:{plan.alpha!r}:{plan.scale!r}:{plan.replacement_strength!r}"
    return PreparedLora(plan, identity, make_target_display_name(target))


def load_lora(model: torch.nn.Module, target: str, strength: float = 1.0) -> LoraMergePlan:
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

