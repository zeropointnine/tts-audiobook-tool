"""Writes safetensors files with numpy only, for the torch-free LoRA check tests (venv-base)."""
from __future__ import annotations

import json
import struct

import numpy as np

_DTYPES = {np.dtype("float16"): "F16", np.dtype("float32"): "F32", np.dtype("float64"): "F64",
           np.dtype("int64"): "I64", np.dtype("int32"): "I32"}


def bf16(values) -> tuple[str, np.ndarray]:
    """A bfloat16 tensor: the upper half of each float32's bits (rounded toward zero)."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32) >> 16
    return "BF16", bits.astype(np.uint16)


def write_safetensors(path, tensors: dict) -> None:
    """`tensors`: name -> numpy array, or (safetensors dtype, raw-bits array) eg from `bf16`."""
    header, chunks, offset = {}, [], 0
    for name, value in tensors.items():
        dtype, array = value if isinstance(value, tuple) else (_DTYPES[np.asarray(value).dtype], np.asarray(value))
        data = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))
