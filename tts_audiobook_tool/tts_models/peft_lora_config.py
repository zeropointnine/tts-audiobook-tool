"""Torch-free parts of reading a plain PEFT LoRA adapter (see `peft_lora`): the config
(`adapter_config.json`, plain LoRA only) and the mapping of tensor keys onto model parameter
names. Shared by the merge planner (`peft_lora`) and the model-free menu checks
(`peft_lora_check`).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from tts_audiobook_tool.tts_models.lora_files import LoraLoadError

ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILE_NAME = "adapter_model.safetensors"
# Older peft versions (and `save_pretrained(safe_serialization=False)`) write a pickle
ADAPTER_WEIGHTS_BIN_FILE_NAME = "adapter_model.bin"

KEY_PREFIX = "base_model.model."
PAIR_SUFFIXES = {
    ".lora_A.weight": "A", ".lora_B.weight": "B",
    ".lora_A.default.weight": "A", ".lora_B.default.weight": "B",
}
# Config fields that change what the adapter computes, with the value plain LoRA has
PLAIN_LORA_CONFIG = {
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


@dataclass(frozen=True)
class PeftLoraConfig:
    rank: int
    alpha: float
    use_rslora: bool

    def scale(self, strength: float) -> float:
        """The LoRA delta's multiplier: strength * alpha / r (alpha / sqrt(r) for rsLoRA)."""
        return strength * self.alpha / (math.sqrt(self.rank) if self.use_rslora else self.rank)


def read_config(adapter_dir: str) -> PeftLoraConfig:
    """Reads and validates `adapter_config.json`: plain LoRA only. Raises `LoraLoadError`
    for an unsupported adapter; an unreadable file raises the underlying error."""
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG_FILE_NAME), encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise LoraLoadError("Adapter config is not a JSON object")
    peft_type = config.get("peft_type", "LORA")
    if peft_type != "LORA":
        raise LoraLoadError(f"Unsupported adapter type: {peft_type} (only LoRA is supported)")
    for name, plain in PLAIN_LORA_CONFIG.items():
        value = config.get(name, plain)
        # Unset collections are written as null or empty depending on the peft version
        if value != plain and not (plain in (None, {}) and not value):
            raise LoraLoadError(f"Unsupported adapter feature: {name} = {value!r}")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise LoraLoadError(f"Invalid LoRA rank in adapter config: {rank!r}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise LoraLoadError(f"Invalid lora_alpha in adapter config: {alpha!r}")
    return PeftLoraConfig(rank=rank, alpha=float(alpha), use_rslora=bool(config.get("use_rslora")))


def parameter_name(key: str, param_prefix: str = "", foreign_key_hint: str = "") -> str:
    """The model parameter name a tensor key maps to (a LoRA factor keeps its suffix; see
    `split_pair`). `foreign_key_hint` is appended to the error for a key without PEFT's prefix."""
    if not key.startswith(KEY_PREFIX):
        hint = f"; {foreign_key_hint}" if foreign_key_hint else ""
        raise LoraLoadError(f"Unexpected tensor {key}{hint}")
    return param_prefix + key.removeprefix(KEY_PREFIX).replace(".modules_to_save.default", "")


def split_pair(name: str) -> tuple[str, str] | None:
    """For a LoRA factor's name, (the target weight's name, "A" or "B"); else None."""
    suffix = next((s for s in PAIR_SUFFIXES if name.endswith(s)), None)
    if suffix is None:
        return None
    return name.removesuffix(suffix) + ".weight", PAIR_SUFFIXES[suffix]


def check_replacement_name(key: str, name: str) -> None:
    """A non-factor tensor must be a whole weight or bias (a `modules_to_save` replacement)."""
    if "lora_" in name or not name.endswith((".weight", ".bias")):
        raise LoraLoadError(f"Unsupported adapter tensor: {key}")
