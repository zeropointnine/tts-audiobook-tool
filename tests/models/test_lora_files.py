"""Torch-free safetensors reading and value scans (`lora_files`), used by the model-free LoRA checks.

Runs in venv-base (numpy only); the interop test also needs torch and safetensors (eg venv-breeze).
"""
import json
import math
import struct

import numpy as np
import pytest

from lora_file_test_support import bf16, write_safetensors
from tts_audiobook_tool.tts_models import lora_files
from tts_audiobook_tool.tts_models.lora_files import LoraLoadError, SafetensorsFile, check_values


def test_header_gives_dtype_shape_and_skips_metadata(tmp_path):
    path = tmp_path / "a.safetensors"
    write_safetensors(path, {"x": np.zeros((2, 3), np.float32), "y": bf16(np.ones(4)), "z": np.arange(3)})
    file = SafetensorsFile(str(path))
    assert {k: (v.dtype, v.shape, v.is_float) for k, v in file.tensors.items()} == {
        "x": ("F32", (2, 3), True), "y": ("BF16", (4,), True), "z": ("I64", (3,), False)}


@pytest.mark.parametrize("kind,bad", [
    (np.float16, math.nan), ("bf16", math.inf), (np.float32, -math.inf), (np.float64, math.nan),
])
def test_nonfinite_values_are_found_by_bit_pattern(tmp_path, kind, bad):
    # Exponent-all-ones detection works for every float width, bf16 included (no numpy bf16 type)
    values = np.array([0.5, -2.0, bad, 3.0])
    tensor = bf16(values) if kind == "bf16" else values.astype(kind)
    clean = bf16(values[:2]) if kind == "bf16" else values[:2].astype(kind)
    path = tmp_path / "a.safetensors"
    write_safetensors(path, {"bad": tensor, "clean": clean})
    file = SafetensorsFile(str(path))
    assert file.has_nonfinite("bad") and not file.has_nonfinite("clean")
    with pytest.raises(LoraLoadError, match="bad contains NaN or infinite"):
        check_values(file, ["clean", "bad"], [])


def test_scanning_crosses_chunk_boundaries(tmp_path, monkeypatch):
    # A bad value past the first chunk is still found; large tensors are scanned in pieces
    monkeypatch.setattr(lora_files, "_SCAN_CHUNK", 3)
    values = np.zeros(10, np.float32)
    values[7] = math.nan
    write_safetensors(tmp_path / "a.safetensors", {"x": values})
    assert SafetensorsFile(str(tmp_path / "a.safetensors")).has_nonfinite("x")


def test_all_zero_b_factors_warn_and_negative_zero_counts_as_zero(tmp_path):
    path = tmp_path / "a.safetensors"
    write_safetensors(path, {"b1": np.array([0.0, -0.0], np.float16), "b2": bf16([0.0, -0.0]),
                             "b3": np.array([0.0, 1e-30], np.float32)})
    file = SafetensorsFile(str(path))
    assert check_values(file, ["b1", "b2"], ["b1", "b2"]) == [lora_files.ZERO_LORA_WARNING]
    # One trained (non-zero) factor is enough to make the adapter do something
    assert check_values(file, ["b1", "b3"], ["b1", "b3"]) == []


@pytest.mark.parametrize("content", [b"", b"\x05\x00\x00\x00\x00\x00\x00\x00{}", b"\xff" * 8 + b"{}",
                                     b"\x02\x00\x00\x00\x00\x00\x00\x00[]"])
def test_malformed_files_are_load_errors(tmp_path, content):
    (tmp_path / "a.safetensors").write_bytes(content)
    with pytest.raises(LoraLoadError, match="Couldn't read a.safetensors"):
        SafetensorsFile(str(tmp_path / "a.safetensors"))


def test_entry_whose_data_overruns_the_file_is_rejected(tmp_path):
    path = tmp_path / "a.safetensors"
    write_safetensors(path, {"x": np.zeros(4, np.float32)})
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(LoraLoadError, match="invalid header entry for x"):
        SafetensorsFile(str(path))


def _write_raw(path, entries: dict, payload: bytes) -> None:
    header = json.dumps({k: {"dtype": "F32", "shape": [(end - start) // 4], "data_offsets": [start, end]}
                         for k, (start, end) in entries.items()}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


@pytest.mark.parametrize("entries,payload,error", [
    ({"a": (0, 8), "b": (4, 8)}, 8, "invalid data offsets for b"),
    ({"a": (0, 4), "b": (8, 12)}, 12, "invalid data offsets for b"),
    ({"a": (4, 8)}, 8, "invalid data offsets for a"),
    ({"a": (0, 4)}, 8, "doesn't cover the file"),
    ({}, 4, "doesn't cover the file"),
])
def test_layout_must_tile_the_data_section(tmp_path, entries, payload, error):
    # Overlap, gaps and trailing bytes each pass the per-entry checks, but the safetensors loader
    # rejects them, so the model-free check must too
    path = tmp_path / "a.safetensors"
    _write_raw(path, entries, b"\0" * payload)
    with pytest.raises(LoraLoadError, match=error):
        SafetensorsFile(str(path))


def test_zero_size_tensors_and_any_header_order_are_valid_layouts(tmp_path):
    path = tmp_path / "a.safetensors"
    _write_raw(path, {"b": (4, 8), "empty": (4, 4), "a": (0, 4)}, b"\0" * 8)
    assert set(SafetensorsFile(str(path)).tensors) == {"a", "b", "empty"}


@pytest.mark.parametrize("dtype", ["F8_E4M3", "F8_E5M2"])
def test_float8_tensors_are_rejected(tmp_path, dtype):
    # Float8 values aren't scanned for NaN/inf (and torch.isfinite can't handle F8_E4M3), so
    # rather than pass unchecked, the value check refuses them
    path = tmp_path / "a.safetensors"
    write_safetensors(path, {"x": (dtype, np.array([0x7C, 0x00], np.uint8))})
    file = SafetensorsFile(str(path))
    assert file.tensors["x"].is_float
    with pytest.raises(LoraLoadError, match=f"x has unsupported dtype {dtype}"):
        check_values(file, ["x"], ["x"])


def test_reads_files_written_by_safetensors(tmp_path):
    # Interop with the real writer, including its __metadata__ and bf16 layout
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    a = torch.tensor([[1.0, float("nan")]], dtype=torch.bfloat16)
    save_file({"a": a, "b": torch.zeros(3, 2, dtype=torch.float16)}, str(tmp_path / "a.safetensors"),
              metadata={"format": "pt"})
    file = SafetensorsFile(str(tmp_path / "a.safetensors"))
    assert file.tensors["a"].shape == (1, 2) and file.tensors["b"].dtype == "F16"
    assert file.has_nonfinite("a") and not file.has_nonfinite("b") and file.is_all_zero("b")
