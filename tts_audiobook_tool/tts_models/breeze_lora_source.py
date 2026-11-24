"""Torch-free file-level handling of Breeze TTS 2 LoRA adapters (Instavar format): recognizing
either directory form, reading its manifest or checkpoint receipt and verifying checksums,
and the base-file pins (see `breeze_lora` for the formats). Shared by the merge loader
(`breeze_lora`) and the model-free menu check (`breeze_lora_check`).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.lora_files import LoraLoadError, read_json_object, sha256_file

ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
CHECKPOINT_WEIGHTS_FILE_NAME = "adapter.safetensors"
CHECKPOINT_RECEIPT_FILE_NAME = "checkpoint-receipt.json"
RUN_CONFIG_FILE_NAME = "run-config.json"
# Files that make a local directory look like either form (`ModelTarget.exist_test` markers)
LOCAL_FILE_MARKERS = [ADAPTER_CONFIG_FILE_NAME, CHECKPOINT_WEIGHTS_FILE_NAME]

BASE_MODEL_ID = "BreezeBlue/Breeze-TTS-2"
ARTIFACT_TYPE = "breeze_lora_adapter"
CHECKPOINT_COMPLETE_STATUS = "five_role_checkpoint_complete"

_SHA256 = re.compile(r"[0-9a-f]{64}")
# Weight key suffix -> factor (eg `lm_head.lora_A`: A is (rank, in), B is (out, rank))
PAIR_SUFFIXES = {".lora_A": "A", ".lora_B": "B"}

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


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value.lower()))


def _rank_and_alpha(rank: object, alpha: object, where: str) -> tuple[int, float]:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise LoraLoadError(f"Invalid LoRA rank in {where}: {rank!r}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise LoraLoadError(f"Invalid LoRA alpha in {where}: {alpha!r}")
    return rank, float(alpha)


def read_manifest_dir(directory: str) -> BreezeAdapterSource:
    """Validates an adapter export / release directory (manifest + weights)."""
    manifest = read_json_object(os.path.join(directory, ADAPTER_CONFIG_FILE_NAME), ADAPTER_CONFIG_FILE_NAME)
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
    receipt = read_json_object(os.path.join(directory, CHECKPOINT_RECEIPT_FILE_NAME), CHECKPOINT_RECEIPT_FILE_NAME)
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
    run_config = read_json_object(run_config_path, RUN_CONFIG_FILE_NAME)
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
            file_name = read_json_object(
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
