"""Model-free check of an OmniVoice LoRA adapter (PEFT format, see `omnivoice_lora`), for the
voice menu: run in the main process, torch-free, without loading the model or reading its weights.

The adapter's own files are fully checked as the merge planner (`peft_lora.plan_merge`) checks
them: plain-LoRA config, key layout, complete A/B pairs, float dtypes, and in addition values
(NaN/inf rejects; all-zero B factors warn). Where the planner compares against the loaded model's
parameters, this compares against the shapes the base model's `config.json` implies
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
from tts_audiobook_tool.tts_models.lora_files import (
    LoraCheck, LoraLoadError, SafetensorsFile, TensorInfo, check_strength, check_values, read_json_object,
)
from tts_audiobook_tool.tts_models.peft_lora_config import (
    ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME, check_replacement_name, parameter_name,
    read_config, split_pair,
)

ADAPTER_FILE_NAMES = [ADAPTER_CONFIG_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME]
BASE_CONFIG_FILE_NAME = "config.json"

FOREIGN_KEY_HINT = "expected an adapter for the whole OmniVoice model, as written by OmniVoice's LoRA training"

# Language model types whose parameter layout `expected_shapes` knows
_QWEN_TYPES = ("qwen3", "qwen2")


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


def _positive_int(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(key)
    return value


def expected_shapes(config: dict) -> dict[str, tuple[int, ...]] | None:
    """Parameter name -> shape of an OmniVoice model with this `config.json` (OmniVoice's
    checkpoint keys equal its parameter names). None when the config isn't a recognized
    OmniVoice config (eg another language model type, or missing dimensions)."""
    llm = config.get("llm_config")
    if not isinstance(llm, dict) or llm.get("model_type") not in _QWEN_TYPES:
        return None
    try:
        hidden = _positive_int(llm, "hidden_size")
        intermediate = _positive_int(llm, "intermediate_size")
        heads = _positive_int(llm, "num_attention_heads")
        layers = _positive_int(llm, "num_hidden_layers")
        vocab = _positive_int(llm, "vocab_size")
        kv_heads = _positive_int(llm, "num_key_value_heads") if llm.get("num_key_value_heads") is not None else heads
        head_dim = _positive_int(llm, "head_dim") if llm.get("head_dim") is not None else hidden // heads
        codebooks = _positive_int(config, "num_audio_codebook")
        audio_vocab = _positive_int(config, "audio_vocab_size")
    except ValueError:
        return None
    # Qwen2 always has q/k/v biases; Qwen3 has them (and o's) only with attention_bias
    qwen3 = llm.get("model_type") == "qwen3"
    biases = ("q", "k", "v", "o") if qwen3 and llm.get("attention_bias") else ("q", "k", "v") if not qwen3 else ()

    q_out, kv_out = heads * head_dim, kv_heads * head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "codebook_layer_offsets": (codebooks,),
        "audio_embeddings.weight": (codebooks * audio_vocab, hidden),
        "audio_heads.weight": (codebooks * audio_vocab, hidden),
        "llm.embed_tokens.weight": (vocab, hidden),
        "llm.norm.weight": (hidden,),
    }
    projections = {
        "self_attn.q_proj": (q_out, hidden), "self_attn.k_proj": (kv_out, hidden),
        "self_attn.v_proj": (kv_out, hidden), "self_attn.o_proj": (hidden, q_out),
        "mlp.gate_proj": (intermediate, hidden), "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }
    for index in range(layers):
        prefix = f"llm.layers.{index}."
        for name, shape in projections.items():
            shapes[f"{prefix}{name}.weight"] = shape
        for name in biases:
            shapes[f"{prefix}self_attn.{name}_proj.bias"] = projections[f"self_attn.{name}_proj"][:1]
        shapes[f"{prefix}input_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}post_attention_layernorm.weight"] = (hidden,)
        if qwen3:
            shapes[f"{prefix}self_attn.q_norm.weight"] = (head_dim,)
            shapes[f"{prefix}self_attn.k_norm.weight"] = (head_dim,)
    return shapes


def read_base_shapes(base_target: ModelTarget) -> tuple[dict[str, tuple[int, ...]] | None, str]:
    """(`expected_shapes` of the base's config.json, "") or (None, a warning). Only config.json is
    read (downloaded for an hf base, if not cached)."""
    later = "the adapter will be checked against the model when it loads"
    try:
        directory = base_target.resolve_local_dir([BASE_CONFIG_FILE_NAME])
        config = read_json_object(os.path.join(directory, BASE_CONFIG_FILE_NAME), "the base model's config.json")
    except Exception as e:
        return None, f"Couldn't read the base model's config ({e}); {later}"
    shapes = expected_shapes(config)
    if shapes is None:
        return None, f"The base model's config.json isn't a recognized OmniVoice config; {later}"
    return shapes, ""


def check_adapter(target: ModelTarget, base_target: ModelTarget, strength: float = 1.0) -> LoraCheck:
    """Checks an adapter for the OmniVoice base `base_target` without loading the model."""
    try:
        return _check_adapter(target, base_target, strength)
    except LoraLoadError as e:
        return LoraCheck(error=str(e))
    except Exception as e:
        return LoraCheck(error=f"Couldn't read adapter: {type(e).__name__}: {e}")


def _check_adapter(target: ModelTarget, base_target: ModelTarget, strength: float) -> LoraCheck:
    check_strength(strength)
    directory = resolve_adapter_dir(target)
    config = read_config(directory)
    file = SafetensorsFile(os.path.join(directory, ADAPTER_WEIGHTS_FILE_NAME))
    expected, base_warning = read_base_shapes(base_target)
    warnings = [base_warning] if base_warning else []

    halves: dict[str, dict[str, tuple[str, TensorInfo]]] = {}
    replacements = 0
    for key, info in file.tensors.items():
        name = parameter_name(key, foreign_key_hint=FOREIGN_KEY_HINT)
        pair = split_pair(name)
        if pair:
            halves.setdefault(pair[0], {})[pair[1]] = (key, info)
            continue
        check_replacement_name(key, name)
        if expected is not None:
            shape = expected.get(name)
            if shape is None:
                raise LoraLoadError(f"Adapter replaces {name}, which the model doesn't have")
            if info.shape != shape:
                raise LoraLoadError(f"Shape mismatch for {name}: adapter {info.shape}, model {shape}")
        if not info.is_float:
            raise LoraLoadError(f"Non-float adapter tensor: {key}")
        replacements += 1

    rank = config.rank
    b_keys = []
    for name, half in halves.items():
        if "A" not in half or "B" not in half:
            raise LoraLoadError(f"Incomplete LoRA pair for {name}")
        (a_key, a), (b_key, b) = half["A"], half["B"]
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise LoraLoadError(f"LoRA target {name} is not a matrix")
        if expected is not None:
            shape = expected.get(name)
            if shape is None:
                raise LoraLoadError(f"Adapter targets {name}, which the model doesn't have")
            if len(shape) != 2:
                raise LoraLoadError(f"LoRA target {name} is not a matrix")
            out_features, in_features = shape
            model_text = f"model {shape}"
        else:
            # Without the base's shapes, the factors need only agree with each other and the rank
            out_features, in_features = b.shape[0], a.shape[1]
            model_text = "expected A (r, in), B (out, r)"
        if a.shape != (rank, in_features) or b.shape != (out_features, rank):
            raise LoraLoadError(f"Shape mismatch for {name}: A {a.shape}, B {b.shape}, {model_text}, r={rank}")
        if not a.is_float or not b.is_float:
            raise LoraLoadError(f"Non-float LoRA factors for {name}")
        b_keys.append(b_key)

    if not halves and not replacements:
        raise LoraLoadError("Adapter has no tensors")
    warnings += check_values(file, list(file.tensors), b_keys)
    return LoraCheck(warnings=tuple(warnings))
