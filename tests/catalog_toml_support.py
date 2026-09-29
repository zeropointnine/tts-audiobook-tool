"""Dependency-free TOML fixtures for tests that mutate the shipped model catalog.

This intentionally writes a small TOML subset (scalars, arrays, inline tables)
rather than introducing a runtime or test dependency just to serialize fixtures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import tomllib
from typing import Any


def read_catalog(path: Path) -> dict[str, Any]:
    """Read a shipped or test catalog without involving the production loader."""
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _key(name: str) -> str:
    # Quoted keys also work for bare-key names and preserve punctuation.
    return json.dumps(name, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite floats are not supported in catalog fixtures")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(key)} = {_value(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"unsupported TOML fixture value: {value!r}")


def catalog_to_toml(catalog: dict[str, Any]) -> str:
    """Serialize a catalog fixture; reject types TOML cannot express."""
    return "\n".join(f"{_key(key)} = {_value(value)}" for key, value in catalog.items()) + "\n"


def write_catalog(path: Path, catalog: dict[str, Any]) -> Path:
    path.write_text(catalog_to_toml(catalog), encoding="utf-8")
    return path
