"""Applies a Breeze TTS 2 LoRA adapter (Instavar format) by merging it into the loaded model's weights.

Adapters come from Instavar's trainer (https://github.com/instavar/breeze-tts2-finetuning),
which uses its own dependency-free LoRA wrapper, not PEFT, so peft is not needed
here either. Two directory forms are accepted:

    Adapter export / release (local directory or hf repo, optionally `owner/repo/sub/dir`):
        adapter_config.json   Instavar manifest: schema_version 1, artifact_type
                              "breeze_lora_adapter", base_model {id, revision, files: name -> sha256},
                              adapter {file, sha256}, lora {variant, rank, alpha, seed}
        <adapter.file>        the weights (normally adapter.safetensors)

    Training checkpoint (local directory only), eg `.../lora/checkpoint-step-000500`:
        adapter.safetensors
        checkpoint-receipt.json   schema_version 1, status "five_role_checkpoint_complete",
                                  roles {"adapter.safetensors": sha256, ...}
        ../run-config.json        the run's rank and alpha (the checkpoint itself doesn't record them)

Weight keys are module names of `BreezeForConditionalGeneration` plus `.lora_A` / `.lora_B`
(eg `backbone_model.layers.0.self_attn.q_proj.lora_A`, `lm_head.lora_B`); A is (rank, in), B is (out, rank).
Scaling is plain LoRA, `alpha / rank`, times the app's strength multiplier.

A manifest pins the base model's files by sha256; any difference rejects the adapter (the revision
string is not compared: the hub revision can move on without the weights changing). A training
checkpoint pins nothing, so only its keys and shapes are checked against the model.

Everything is validated before any weight changes, so a rejected adapter (`LoraLoadError`) leaves the
model untouched. Must not import `breeze_tts`.

Per-voice switching (`lora_switch`) restores and re-merges weights in place, so it is safe after the CFG
depth graph has been captured: parameter storage never moves, and the graph's fp32 copy of
`depth_decoder.codebooks_head` can't go stale because that head (a `BreezeCodebooksHead`, not a
linear layer) is never a valid LoRA target.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

import torch

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan, apply_merge
from tts_audiobook_tool.tts_models.lora_switch import PreparedLora

ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
CHECKPOINT_WEIGHTS_FILE_NAME = "adapter.safetensors"
CHECKPOINT_RECEIPT_FILE_NAME = "checkpoint-receipt.json"
RUN_CONFIG_FILE_NAME = "run-config.json"
# Files that make a local directory look like either form (`target_util.exist_test` markers)
LOCAL_FILE_MARKERS = [ADAPTER_CONFIG_FILE_NAME, CHECKPOINT_WEIGHTS_FILE_NAME]

BASE_MODEL_ID = "BreezeBlue/Breeze-TTS-2"
ARTIFACT_TYPE = "breeze_lora_adapter"
CHECKPOINT_COMPLETE_STATUS = "five_role_checkpoint_complete"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PAIR_SUFFIXES = {".lora_A": "A", ".lora_B": "B"}

UNRECOGNIZED_MESSAGE = (
    "Not a Breeze LoRA: expected adapter_config.json (Instavar adapter export) "
    "or adapter.safetensors + checkpoint-receipt.json (Instavar training checkpoint)"
)


@dataclass(frozen=True)
class BreezeAdapterSource:
    """A resolved, file-level-validated adapter (weights not yet read)."""
    kind: str  # "manifest" | "checkpoint"
    weights_path: str
    rank: int
    alpha: float
    # Base model file name -> sha256 (manifest only)
    base_files: dict[str, str] | None = None
    # The verified checksum of the weights file
    weights_sha256: str = ""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value.lower()))


def _read_json_object(path: str, what: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
    except Exception as e:
        raise LoraLoadError(f"Couldn't read {what}: {e}") from e
    if not isinstance(value, dict):
        raise LoraLoadError(f"{what} is not a JSON object")
    return value


def _rank_and_alpha(rank: object, alpha: object, where: str) -> tuple[int, float]:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise LoraLoadError(f"Invalid LoRA rank in {where}: {rank!r}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise LoraLoadError(f"Invalid LoRA alpha in {where}: {alpha!r}")
    return rank, float(alpha)


def read_manifest_dir(directory: str) -> BreezeAdapterSource:
    """Validates an adapter export / release directory (manifest + weights)."""
    manifest = _read_json_object(os.path.join(directory, ADAPTER_CONFIG_FILE_NAME), ADAPTER_CONFIG_FILE_NAME)
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        if "peft_type" in manifest:
            raise LoraLoadError("This is a PEFT adapter; only Instavar-format Breeze LoRAs are supported")
        raise LoraLoadError(f"{ADAPTER_CONFIG_FILE_NAME} is not a Breeze LoRA manifest "
                            f"(artifact_type {manifest.get('artifact_type')!r})")
    if manifest.get("schema_version") != 1:
        raise LoraLoadError(f"Unsupported manifest schema_version: {manifest.get('schema_version')!r}")
    base, adapter, lora = manifest.get("base_model"), manifest.get("adapter"), manifest.get("lora")
    if not all(isinstance(item, dict) for item in (base, adapter, lora)):
        raise LoraLoadError("Manifest is missing its base_model, adapter or lora section")
    assert isinstance(base, dict) and isinstance(adapter, dict) and isinstance(lora, dict)

    base_id = base.get("id")
    if not isinstance(base_id, str) or base_id.lower() != BASE_MODEL_ID.lower():
        raise LoraLoadError(f"Adapter is for base model {base_id!r}, not {BASE_MODEL_ID}")
    base_files = base.get("files")
    if not isinstance(base_files, dict) or not base_files:
        raise LoraLoadError("Manifest pins no base model files")
    for name, digest in base_files.items():
        parts = str(name).replace("\\", "/").split("/")
        if not name or os.path.isabs(name) or ".." in parts:
            raise LoraLoadError(f"Unsafe base model file name in manifest: {name!r}")
        if not _is_sha256(digest):
            raise LoraLoadError(f"Invalid checksum for base model file {name!r}")

    file_name = adapter.get("file")
    if (not isinstance(file_name, str) or os.path.basename(file_name) != file_name
            or not file_name.endswith(".safetensors")):
        raise LoraLoadError(f"Invalid adapter file name in manifest: {file_name!r}")
    weights_path = os.path.join(directory, file_name)
    if not os.path.isfile(weights_path):
        raise LoraLoadError(f"Missing {file_name}")
    expected = adapter.get("sha256")
    if not _is_sha256(expected):
        raise LoraLoadError("Manifest has no valid adapter checksum")
    assert isinstance(expected, str)
    if sha256_file(weights_path) != expected.lower():
        raise LoraLoadError(f"{file_name} doesn't match the manifest checksum (corrupt or modified)")

    rank, alpha = _rank_and_alpha(lora.get("rank"), lora.get("alpha"), ADAPTER_CONFIG_FILE_NAME)
    return BreezeAdapterSource(
        kind="manifest", weights_path=weights_path, rank=rank, alpha=alpha,
        base_files={str(name): str(digest).lower() for name, digest in base_files.items()},
        weights_sha256=expected.lower(),
    )


def read_checkpoint_dir(directory: str) -> BreezeAdapterSource:
    """Validates a training checkpoint directory; rank and alpha come from the run's `run-config.json`."""
    directory = os.path.normpath(directory)
    receipt = _read_json_object(os.path.join(directory, CHECKPOINT_RECEIPT_FILE_NAME), CHECKPOINT_RECEIPT_FILE_NAME)
    if receipt.get("schema_version") != 1:
        raise LoraLoadError(f"Unsupported checkpoint receipt schema_version: {receipt.get('schema_version')!r}")
    if receipt.get("status") != CHECKPOINT_COMPLETE_STATUS:
        raise LoraLoadError(f"Checkpoint is incomplete (status {receipt.get('status')!r})")
    roles = receipt.get("roles")
    expected = roles.get(CHECKPOINT_WEIGHTS_FILE_NAME) if isinstance(roles, dict) else None
    if not _is_sha256(expected):
        raise LoraLoadError(f"Checkpoint receipt has no checksum for {CHECKPOINT_WEIGHTS_FILE_NAME}")
    assert isinstance(expected, str)
    weights_path = os.path.join(directory, CHECKPOINT_WEIGHTS_FILE_NAME)
    if sha256_file(weights_path) != expected.lower():
        raise LoraLoadError(f"{CHECKPOINT_WEIGHTS_FILE_NAME} doesn't match the checkpoint receipt (corrupt or modified)")

    run_config_path = os.path.join(os.path.dirname(directory), RUN_CONFIG_FILE_NAME)
    if not os.path.isfile(run_config_path):
        raise LoraLoadError(f"Can't determine the LoRA's alpha: no {RUN_CONFIG_FILE_NAME} next to the checkpoint "
                            "(export the checkpoint as an adapter instead)")
    run_config = _read_json_object(run_config_path, RUN_CONFIG_FILE_NAME)
    rank, alpha = _rank_and_alpha(run_config.get("rank"), run_config.get("alpha"), RUN_CONFIG_FILE_NAME)
    return BreezeAdapterSource(kind="checkpoint", weights_path=weights_path, rank=rank, alpha=alpha,
                               weights_sha256=expected.lower())


def read_local_dir(directory: str) -> BreezeAdapterSource:
    """Either form, by which files the directory holds (a manifest takes precedence)."""
    if os.path.isfile(os.path.join(directory, ADAPTER_CONFIG_FILE_NAME)):
        return read_manifest_dir(directory)
    if (os.path.isfile(os.path.join(directory, CHECKPOINT_WEIGHTS_FILE_NAME))
            and os.path.isfile(os.path.join(directory, CHECKPOINT_RECEIPT_FILE_NAME))):
        return read_checkpoint_dir(directory)
    raise LoraLoadError(UNRECOGNIZED_MESSAGE)


def resolve_adapter(target: ModelTarget) -> BreezeAdapterSource:
    """A local directory (either form), or an hf repo (or a directory inside one) holding a manifest.
    Downloads only the manifest and the weights file it names."""
    try:
        if target.is_local:
            if not os.path.isdir(target.value):
                raise LoraLoadError(f"No such directory: {target.value}")
            return read_local_dir(target.value)
        try:
            directory = target.resolve_local_dir([ADAPTER_CONFIG_FILE_NAME])
            if not os.path.isfile(os.path.join(directory, ADAPTER_CONFIG_FILE_NAME)):
                raise LoraLoadError(f"No {ADAPTER_CONFIG_FILE_NAME} in {target.value} "
                                    "(only adapter exports can be loaded from huggingface)")
            file_name = _read_json_object(
                os.path.join(directory, ADAPTER_CONFIG_FILE_NAME), ADAPTER_CONFIG_FILE_NAME
            ).get("adapter", {}).get("file")
            if isinstance(file_name, str) and os.path.basename(file_name) == file_name:
                directory = target.resolve_local_dir([ADAPTER_CONFIG_FILE_NAME, file_name])
        except LoraLoadError:
            raise
        except Exception as e:
            raise LoraLoadError(f"Couldn't get adapter files for {target.value}: {e}") from e
        return read_manifest_dir(directory)
    except LoraLoadError:
        raise
    except Exception as e:
        raise LoraLoadError(f"Couldn't read adapter: {e}") from e


def verify_base(model_dir: str, source: BreezeAdapterSource) -> None:
    """Checks the manifest's pinned base files against the loaded model's directory.
    A file in the hf cache that links to `blobs/<sha256>` is identified by that name (hf names
    LFS blobs by their sha256), so the multi-GB shards needn't be re-hashed; anything else is hashed."""
    for name, expected in (source.base_files or {}).items():
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise LoraLoadError(f"Base model file {name} is missing, so the adapter's base can't be verified")
        real = os.path.realpath(path)
        blob_name = os.path.basename(real)
        if os.path.basename(os.path.dirname(real)) == "blobs" and _is_sha256(blob_name):
            observed = blob_name.lower()
        else:
            observed = sha256_file(path)
        if observed != expected:
            raise LoraLoadError(f"Adapter was trained on a different Breeze TTS 2 base ({name} differs)")


def plan_merge(model: torch.nn.Module, source: BreezeAdapterSource, strength: float = 1.0) -> LoraMergePlan:
    """Reads and validates the adapter's tensors against `model` without changing it."""
    try:
        return _plan_merge(model, source, strength)
    except LoraLoadError:
        raise
    except Exception as e:
        raise LoraLoadError(f"Couldn't read adapter: {e}") from e


def _plan_merge(model: torch.nn.Module, source: BreezeAdapterSource, strength: float) -> LoraMergePlan:
    from safetensors.torch import load_file

    if not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength <= 0:
        raise LoraLoadError(f"Invalid LoRA strength: {strength!r}")
    rank = source.rank

    halves: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(source.weights_path).items():
        suffix = next((s for s in _PAIR_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            raise LoraLoadError(f"Unsupported adapter tensor: {key} (expected Instavar .lora_A / .lora_B factors)")
        halves.setdefault(key.removesuffix(suffix), {})[_PAIR_SUFFIXES[suffix]] = value

    modules = dict(model.named_modules())
    # Parameter storage -> number of names that share it (tied weights)
    storage_counts: dict[int, int] = {}
    for _, param in model.named_parameters(remove_duplicate=False):
        storage_counts[param.data_ptr()] = storage_counts.get(param.data_ptr(), 0) + 1

    plan = LoraMergePlan(rank=rank, alpha=source.alpha, scale=strength * source.alpha / rank)
    for module_name, half in halves.items():
        if "A" not in half or "B" not in half:
            raise LoraLoadError(f"Incomplete LoRA pair for {module_name}")
        module = modules.get(module_name)
        if module is None:
            raise LoraLoadError(f"Adapter targets {module_name}, which the model doesn't have")
        if not isinstance(module, torch.nn.Linear):
            raise LoraLoadError(f"LoRA target {module_name} is not a linear layer")
        weight = module.weight
        if storage_counts.get(weight.data_ptr(), 0) > 1:
            raise LoraLoadError(f"LoRA target {module_name} shares its weight with another layer")
        a, b = half["A"], half["B"]
        if a.ndim != 2 or b.ndim != 2:
            raise LoraLoadError(f"LoRA factors for {module_name} are not matrices")
        out_features, in_features = weight.shape
        if tuple(a.shape) != (rank, in_features) or tuple(b.shape) != (out_features, rank):
            raise LoraLoadError(
                f"Shape mismatch for {module_name}: A {tuple(a.shape)}, B {tuple(b.shape)}, "
                f"model {tuple(weight.shape)}, r={rank}")
        if not a.is_floating_point() or not b.is_floating_point():
            raise LoraLoadError(f"Non-float LoRA factors for {module_name}")
        plan.pairs[f"{module_name}.weight"] = (a, b)

    if not plan.pairs:
        raise LoraLoadError("Adapter has no tensors")
    return plan


def prepare_lora(model: torch.nn.Module, model_dir: str, target: ModelTarget, strength: float = 1.0) -> PreparedLora:
    """
    Resolves and validates an adapter against `model` and its base files (`model_dir`) without changing
    the model, for `lora_switch.LoraSwitcher`. Raises `LoraLoadError` when the adapter can't be used.
    """
    source = resolve_adapter(target)
    if source.base_files:
        verify_base(model_dir, source)
    plan = plan_merge(model, source, strength)
    # Rank/alpha live in JSON, not the tensor file; changing them must invalidate
    # an already merged adapter too. Strength is tracked separately in LoraKey.
    identity = f"{source.weights_sha256}:{source.rank}:{source.alpha!r}"
    return PreparedLora(plan=plan, identity=identity, display=target.display_name())


def load_lora(
        model: torch.nn.Module, model_dir: str, target: ModelTarget, strength: float = 1.0
) -> tuple[LoraMergePlan, BreezeAdapterSource]:
    """
    Resolves, validates and merges an adapter into `model` (`model_dir`: the loaded base model's files,
    for a manifest's base checksums). Raises `LoraLoadError` (model unchanged) when the adapter can't
    be used; any other exception means the merge failed partway and the model must be discarded.
    """
    source = resolve_adapter(target)
    if source.base_files:
        verify_base(model_dir, source)
    plan = plan_merge(model, source, strength)
    apply_merge(model, plan)
    return plan, source
