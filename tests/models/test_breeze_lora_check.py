"""Model-free Breeze TTS 2 LoRA check (`breeze_lora_check`): the voice menu's check, against the
pinned base's generated tables instead of the loaded model and its files.

Runs in venv-base (numpy only).
"""
import hashlib
import json
import math

import numpy as np
import pytest

from lora_file_test_support import bf16, write_safetensors
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models import breeze_lora_targets
from tts_audiobook_tool.tts_models.breeze_lora_check import check_adapter, target_shape
from tts_audiobook_tool.tts_models.lora_files import ZERO_LORA_WARNING

RANK = 2
Q = "backbone_model.layers.0.self_attn.q_proj"  # (2048, 2048)
HEAD = "text_encoder_proj"  # (2048, 1152)
CONFIG_SHA = breeze_lora_targets.BASE_FILE_SHA256["config.json"]


def pair(module, out_features, in_features, b_scale=0.01) -> dict:
    rng = np.random.default_rng(0)
    return {f"{module}.lora_A": bf16(rng.standard_normal((RANK, in_features))),
            f"{module}.lora_B": bf16(rng.standard_normal((out_features, RANK)) * b_scale)}


def adapter_tensors() -> dict:
    return {**pair(Q, 2048, 2048), **pair(HEAD, 2048, 1152)}


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_manifest_dir(directory, tensors=None, base_files=None) -> ModelTarget:
    directory.mkdir(parents=True, exist_ok=True)
    write_safetensors(directory / "adapter.safetensors", adapter_tensors() if tensors is None else tensors)
    manifest = {
        "schema_version": 1, "artifact_type": "breeze_lora_adapter",
        "base_model": {"id": "BreezeBlue/Breeze-TTS-2", "revision": "anything",
                       "files": base_files or {"config.json": CONFIG_SHA}},
        "adapter": {"file": "adapter.safetensors", "sha256": sha256(directory / "adapter.safetensors")},
        "lora": {"variant": "backbone_depth_projection", "rank": RANK, "alpha": 4.0, "seed": 42},
    }
    (directory / "adapter_config.json").write_text(json.dumps(manifest))
    return ModelTarget.local(str(directory))


def write_checkpoint_dir(run_root, tensors=None) -> ModelTarget:
    checkpoint = run_root / "checkpoint-step-000500"
    checkpoint.mkdir(parents=True)
    write_safetensors(checkpoint / "adapter.safetensors", adapter_tensors() if tensors is None else tensors)
    receipt = {"schema_version": 1, "status": "five_role_checkpoint_complete",
               "roles": {"adapter.safetensors": sha256(checkpoint / "adapter.safetensors")}}
    (checkpoint / "checkpoint-receipt.json").write_text(json.dumps(receipt))
    (run_root / "run-config.json").write_text(json.dumps({"rank": RANK, "alpha": 4.0}))
    return ModelTarget.local(str(checkpoint))


def test_table_is_for_the_pinned_revision():
    # The tables must be regenerated (testx/breeze_lora_targets_gen.py) whenever the pin moves
    from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel
    assert breeze_lora_targets.BASE_REVISION == BreezeBaseModel.HF_REVISION


@pytest.mark.parametrize("form", ["manifest", "checkpoint"])
def test_valid_adapter_passes_without_warnings(tmp_path, form):
    target = write_manifest_dir(tmp_path / "export") if form == "manifest" else write_checkpoint_dir(tmp_path / "run")
    assert check_adapter(target, 0.5).error == ""
    assert check_adapter(target, 0.5).warnings == ()


@pytest.mark.parametrize("name,expected", [
    (Q, (2048, 2048)),
    ("backbone_model.layers.27.mlp.down_proj", (2048, 6144)),
    ("lm_head", (2052, 2048)),
    ("backbone_model.layers.28.mlp.down_proj", None),  # past the last layer
    ("backbone_model.layers.01.mlp.down_proj", None),  # not a module name the model has
    ("depth_decoder.codebooks_head", None),  # not a linear layer
    ("backbone_model.embed_tokens.embed_audio_tokens", None),  # tied embedding
])
def test_target_shapes_come_from_the_pinned_table(name, expected):
    assert target_shape(name) == expected


@pytest.mark.parametrize("edit,message", [
    (lambda t: {k: v for k, v in t.items() if k != f"{Q}.lora_B"}, "Incomplete LoRA pair"),
    (lambda t: {**t, "lm_head.weight": np.ones((4, 4), np.float32)}, "Unsupported adapter tensor"),
    (lambda t: {**t, f"{Q}.lora_A": bf16(np.ones((3, 2048)))}, "Shape mismatch"),
    (lambda t: {**t, **pair("depth_decoder.codebooks_head", 2051, 1024)}, "isn't a layer of Breeze TTS 2"),
    (lambda t: {**t, f"{Q}.lora_A": np.ones((RANK, 2048), np.int32)}, "Non-float"),
    (lambda t: {**t, f"{Q}.lora_A": bf16(np.full((RANK, 2048), math.nan))}, "NaN or infinite"),
    (lambda t: {}, "no tensors"),
])
def test_bad_tensors_are_rejected(tmp_path, edit, message):
    result = check_adapter(write_manifest_dir(tmp_path / "export", edit(adapter_tensors())))
    assert message in result.error


def test_untrained_adapter_is_accepted_with_a_warning(tmp_path):
    # B starts at zero in LoRA training, so a step-0 checkpoint is valid but does nothing
    tensors = {**pair(Q, 2048, 2048, b_scale=0.0), **pair(HEAD, 2048, 1152, b_scale=0.0)}
    result = check_adapter(write_checkpoint_dir(tmp_path / "run", tensors))
    assert result.error == "" and result.warnings == (ZERO_LORA_WARNING,)


@pytest.mark.parametrize("base_files,message", [
    ({"config.json": "0" * 64}, "different Breeze TTS 2 base (config.json differs)"),
    ({"model-00003-of-00003.safetensors": "0" * 64}, "isn't part of the Breeze TTS 2 model"),
])
def test_manifest_base_pins_are_checked_against_the_pinned_revision(tmp_path, base_files, message):
    # A matching pin (as in the valid case) passes without the base model's files present
    assert message in check_adapter(write_manifest_dir(tmp_path / "export", base_files=base_files)).error


def test_file_level_problems_are_reported_not_raised(tmp_path):
    (tmp_path / "latest.json").write_text("{}")
    assert "Not a Breeze LoRA" in check_adapter(ModelTarget.local(str(tmp_path))).error
    assert "Invalid LoRA strength" in check_adapter(ModelTarget.local(str(tmp_path)), 0).error
