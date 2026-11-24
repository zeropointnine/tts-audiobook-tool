"""Model-free check of a VibeVoice LoRA adapter (PEFT format, see `vibevoice_lora`), for the voice
menu: run in the main process, torch-free, without loading the model or reading its weights.
Also the torch-free adapter-directory handling (layout, resolution), shared with `vibevoice_lora`.

The adapter's own files are checked as the merge planner (`vibevoice_lora.plan_merge`) checks
them (plain-LoRA config; the tensors by `peft_lora_check`). Where the planner compares against the
loaded model's parameters, this compares against the shapes the base model's `config.json` implies
(`expected_shapes`: its `decoder_config`, a Qwen2 language model, under `model.language_model.`;
for the stock 1.5B and 7B checkpoints, exactly their language model tensors). A base whose config
can't be read or isn't recognized only gets a warning, and the adapter's internal consistency is
checked instead. An older `adapter_model.bin` (a pickle, which needs torch) gets only its config
checked, with a warning.

Every run still prepares each voice's LoRA against the loaded model (`prepare_voice_loras`) before
generating, which is the authoritative check, before any weight changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.lora_files import LoraCheck, LoraLoadError, SafetensorsFile, check_strength
from tts_audiobook_tool.tts_models.peft_lora_check import (
    CHECKED_LATER, Shapes, check_weights, qwen_language_model_shapes, read_base_shapes,
)
from tts_audiobook_tool.tts_models.peft_lora_config import (
    ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_BIN_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME, read_config,
)

# The trainer saves the adapter here, inside its output and checkpoint directories
LORA_SUBDIR = "lora"
# Where the adapter's keys (relative to the language model) live in the whole model
LANGUAGE_MODEL_PREFIX = "model.language_model."

_WEIGHTS_FILE_NAMES = [ADAPTER_WEIGHTS_FILE_NAME, ADAPTER_WEIGHTS_BIN_FILE_NAME]
_FILE_NAMES = [ADAPTER_CONFIG_FILE_NAME, *_WEIGHTS_FILE_NAMES]
# Files that make a local directory (or hf repo) acceptable (`ModelTarget.exist_test` markers)
LOCAL_FILE_MARKERS = [ADAPTER_CONFIG_FILE_NAME, f"{LORA_SUBDIR}/{ADAPTER_CONFIG_FILE_NAME}"]

FOREIGN_KEY_HINT = ("expected a language model adapter, as written by VibeVoice-finetuning "
                    "(voicepowered-ai/VibeVoice-finetuning)")

BIN_WEIGHTS_WARNING = (f"{ADAPTER_WEIGHTS_BIN_FILE_NAME} (an older format) can't be checked yet; "
                       f"only its config was, and {CHECKED_LATER}")


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


def expected_shapes(config: dict) -> Shapes | None:
    """Parameter name -> shape of the language model of a VibeVoice model with this `config.json`
    (VibeVoice's checkpoint keys equal its parameter names), the only part an adapter can reach:
    the trainer wraps only `model.language_model`, so eg `lm_head` (tied or not) is out of reach.
    None when the config isn't a recognized VibeVoice config."""
    return qwen_language_model_shapes(config.get("decoder_config"), LANGUAGE_MODEL_PREFIX)


def check_adapter(target: ModelTarget, base_target: ModelTarget, strength: float = 1.0) -> LoraCheck:
    """Checks an adapter for the VibeVoice base `base_target` without loading the model."""
    try:
        check_strength(strength)
        adapter = resolve_adapter(target)
        config = read_config(adapter.directory)
        if adapter.weights_file_name != ADAPTER_WEIGHTS_FILE_NAME:
            return LoraCheck(warnings=(BIN_WEIGHTS_WARNING,))
        file = SafetensorsFile(adapter.weights_path)
        expected, base_warning = read_base_shapes(base_target, expected_shapes, "VibeVoice")
        warnings = [base_warning] if base_warning else []
        warnings += check_weights(file, config, expected, param_prefix=LANGUAGE_MODEL_PREFIX,
                                  foreign_key_hint=FOREIGN_KEY_HINT)
        return LoraCheck(warnings=tuple(warnings))
    except LoraLoadError as e:
        return LoraCheck(error=str(e))
    except Exception as e:
        return LoraCheck(error=f"Couldn't read adapter: {type(e).__name__}: {e}")
