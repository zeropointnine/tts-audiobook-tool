"""Torch-free LoRA adapter file handling, shared by the merge loaders (`breeze_lora`, `peft_lora`)
and the voice menu's model-free checks (`breeze_lora_check`, `peft_lora_check`), which
run in the main process: no torch, safetensors or model library is imported here.

A safetensors file is an 8-byte little-endian header length, a JSON header (tensor name ->
dtype, shape and data offsets, plus an optional `__metadata__`), then the raw tensor bytes.
`SafetensorsFile` reads the header and scans tensor values (NaN/inf, all-zero) with numpy,
by bit pattern, so bf16 needs no special support.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np


class LoraLoadError(Exception):
    """The adapter could not be resolved or does not fit the model; the model is unchanged."""


@dataclass(frozen=True)
class LoraCheck:
    """A model-free adapter check: `error` ("" when usable) and non-blocking `warnings`."""
    error: str = ""
    warnings: tuple[str, ...] = ()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str, what: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
    except Exception as e:
        raise LoraLoadError(f"Couldn't read {what}: {e}") from e
    if not isinstance(value, dict):
        raise LoraLoadError(f"{what} is not a JSON object")
    return value


def check_strength(strength: object) -> float:
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or strength <= 0):
        raise LoraLoadError(f"Invalid LoRA strength: {strength!r}")
    return float(strength)


# safetensors float dtype -> (unsigned view of the same width, exponent mask, sign mask).
# A value is NaN or infinite exactly when all its exponent bits are set (IEEE 754 and bfloat16).
_FLOAT_BITS: dict[str, tuple[str, int, int]] = {
    "F16": ("<u2", 0x7C00, 0x8000),
    "BF16": ("<u2", 0x7F80, 0x8000),
    "F32": ("<u4", 0x7F800000, 0x80000000),
    "F64": ("<u8", 0x7FF0000000000000, 0x8000000000000000),
}
_DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}
# Scan this many elements at a time, to bound memory on large tensors
_SCAN_CHUNK = 1 << 22


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    # Absolute byte range in the file
    start: int
    end: int

    @property
    def is_float(self) -> bool:
        return self.dtype in _FLOAT_BITS or self.dtype.startswith("F8_")


class SafetensorsFile:
    """A safetensors file's header, read and validated up front; values are read on demand."""

    MAX_HEADER_BYTES = 100 * 1024 * 1024

    def __init__(self, path: str):
        self.path = path
        name = os.path.basename(path)
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                prefix = f.read(8)
                if len(prefix) < 8:
                    raise ValueError("file is too short")
                length = int.from_bytes(prefix, "little")
                if length > min(size - 8, self.MAX_HEADER_BYTES):
                    raise ValueError("invalid header length")
                header = json.loads(f.read(length))
        except Exception as e:
            raise LoraLoadError(f"Couldn't read {name}: not a valid safetensors file ({e})") from e
        if not isinstance(header, dict):
            raise LoraLoadError(f"Couldn't read {name}: invalid safetensors header")
        data_start = 8 + length
        self.tensors: dict[str, TensorInfo] = {}
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            try:
                dtype, shape, (start, end) = entry["dtype"], tuple(entry["shape"]), entry["data_offsets"]
                if not all(isinstance(n, int) and n >= 0 for n in (*shape, start, end)):
                    raise ValueError
                expected = math.prod(shape) * _DTYPE_SIZES[dtype]
                if end - start != expected or data_start + end > size:
                    raise ValueError
            except Exception:
                raise LoraLoadError(f"Couldn't read {name}: invalid header entry for {key}") from None
            self.tensors[key] = TensorInfo(dtype, shape, data_start + start, data_start + end)
        # As the safetensors loader requires: the tensors tile the data section exactly, with no
        # overlap, gap or trailing bytes (zero-size tensors sort before any at the same start)
        cursor = data_start
        for key, info in sorted(self.tensors.items(), key=lambda item: (item[1].start, item[1].end)):
            if info.start != cursor:
                raise LoraLoadError(f"Couldn't read {name}: invalid data offsets for {key}")
            cursor = info.end
        if cursor != size:
            raise LoraLoadError(f"Couldn't read {name}: tensor data doesn't cover the file")

    def _bit_chunks(self, key: str):
        info = self.tensors[key]
        view, _, _ = _FLOAT_BITS[info.dtype]
        count = (info.end - info.start) // np.dtype(view).itemsize
        if count == 0:
            return
        values = np.memmap(self.path, dtype=view, mode="r", offset=info.start, shape=(count,))
        for begin in range(0, count, _SCAN_CHUNK):
            yield values[begin:begin + _SCAN_CHUNK]

    def has_nonfinite(self, key: str) -> bool:
        """Whether a float tensor (F16, BF16, F32, F64) holds a NaN or infinity. False for non-float
        dtypes; raises `LoraLoadError` for float8, which isn't supported (not scanned here, and the
        runtime finiteness check can't handle F8_E4M3)."""
        info = self.tensors[key]
        if info.is_float and info.dtype not in _FLOAT_BITS:
            raise LoraLoadError(
                f"Adapter tensor {key} has unsupported dtype {info.dtype} (float8 adapters aren't supported)")
        if info.dtype not in _FLOAT_BITS:
            return False
        _, exponent, _ = _FLOAT_BITS[info.dtype]
        mask = np.array(exponent, dtype=_FLOAT_BITS[info.dtype][0])
        return any(bool(np.any((chunk & mask) == mask)) for chunk in self._bit_chunks(key))

    def is_all_zero(self, key: str) -> bool:
        """Whether a float tensor (F16, BF16, F32, F64) is entirely (+/-) zero. False for other dtypes."""
        info = self.tensors[key]
        if info.dtype not in _FLOAT_BITS:
            return False
        view, _, sign = _FLOAT_BITS[info.dtype]
        magnitude = np.array(~sign & ((1 << (8 * np.dtype(view).itemsize)) - 1), dtype=view)
        return not any(bool(np.any(chunk & magnitude)) for chunk in self._bit_chunks(key))


def nonfinite_message(name: str) -> str:
    return f"Adapter tensor {name} contains NaN or infinite values (corrupt or diverged training)"


ZERO_LORA_WARNING = ("All of the adapter's LoRA B factors are zero, so its LoRA layers change nothing "
                     "(eg an untrained step-0 checkpoint)")


def check_values(file: SafetensorsFile, keys: list[str], b_keys: list[str]) -> list[str]:
    """Raises `LoraLoadError` for a NaN/inf in any of `keys`; returns the all-zero warning
    when there are LoRA B factors (`b_keys`) and every one is zero."""
    for key in keys:
        if file.has_nonfinite(key):
            raise LoraLoadError(nonfinite_message(key))
    if b_keys and all(file.is_all_zero(key) for key in b_keys):
        return [ZERO_LORA_WARNING]
    return []
