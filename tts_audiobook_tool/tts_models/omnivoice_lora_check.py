"""Model-free check of an OmniVoice LoRA adapter (PEFT format, see `omnivoice_lora`), for the
voice menu: run in the main process, torch-free, without loading the model or reading its weights.

The adapter's own files are fully checked as the merge planner (`peft_lora.plan_merge`) checks
them (plain-LoRA config; the tensors by `peft_lora_check`). Where the planner compares against the
loaded model's parameters, this compares against the shapes the base model's `config.json` implies
(`expected_shapes`, a Qwen3 language model plus OmniVoice's audio embeddings and heads; for the
stock checkpoint, exactly its tensors). A base whose config can't be read or isn't recognized
only gets a warning, and the adapter's internal consistency is checked instead.

This is less strict than the planner: the base checkpoint itself isn't read, so eg a custom base
with a nonstandard layout passes here. Every run still prepares each voice's LoRA against the
loaded model (`prepare_voice_loras`) before generating, so such an adapter is rejected then,
before any weight changes.
"""
from __future__ import annotations

import os

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.lora_files import LoraCheck, LoraLoadError, SafetensorsFile, check_strength
from tts_audiobook_tool.tts_models.peft_lora_check import (
    Shapes, check_weights, positive_int, qwen_language_model_shapes, read_base_shapes,
)
from tts_audiobook_tool.tts_models.peft_lora_config import (
    ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME, read_config,
)

ADAPTER_FILE_NAMES = [ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME]

FOREIGN_KEY_HINT = "expected an adapter for the whole OmniVoice model, as written by OmniVoice's LoRA training"


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


def expected_shapes(config: dict) -> Shapes | None:
    """Parameter name -> shape of an OmniVoice model with this `config.json` (OmniVoice's
    checkpoint keys equal its parameter names). None when the config isn't a recognized
    OmniVoice config (eg another language model type, or missing dimensions)."""
    shapes = qwen_language_model_shapes(config.get("llm_config"), "llm.")
    if shapes is None:
        return None
    try:
        codebooks = positive_int(config, "num_audio_codebook")
        audio_vocab = positive_int(config, "audio_vocab_size")
    except ValueError:
        return None
    (hidden,) = shapes["llm.norm.weight"]
    return {
        "codebook_layer_offsets": (codebooks,),
        "audio_embeddings.weight": (codebooks * audio_vocab, hidden),
        "audio_heads.weight": (codebooks * audio_vocab, hidden),
        **shapes,
    }


def check_adapter(target: ModelTarget, base_target: ModelTarget, strength: float = 1.0) -> LoraCheck:
    """Checks an adapter for the OmniVoice base `base_target` without loading the model."""
    try:
        check_strength(strength)
        directory = resolve_adapter_dir(target)
        config = read_config(directory)
        file = SafetensorsFile(os.path.join(directory, ADAPTER_WEIGHTS_FILE_NAME))
        expected, base_warning = read_base_shapes(base_target, expected_shapes, "OmniVoice")
        warnings = [base_warning] if base_warning else []
        warnings += check_weights(file, config, expected, foreign_key_hint=FOREIGN_KEY_HINT)
        return LoraCheck(warnings=tuple(warnings))
    except LoraLoadError as e:
        return LoraCheck(error=str(e))
    except Exception as e:
        return LoraCheck(error=f"Couldn't read adapter: {type(e).__name__}: {e}")
