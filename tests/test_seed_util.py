"""Random limits preserve native ranges and use inclusive maximum semantics."""
import ast
from pathlib import Path
import sys

import pytest

from tts_audiobook_tool.constants import SEED_MAX
from tts_audiobook_tool.seed_util import get_random_seed_max, resolve_max_random_seed


@pytest.mark.parametrize("limits, expected", [
    ((-1,), -1), ((-1, -1), -1), ((10, -1), 10),
    ((10, 20), 10), ((20, 10), 10), ((0, 10), 0),
])
def test_resolve_max_random_seed_intersects_only_configured_caps(limits, expected):
    assert resolve_max_random_seed(*limits) == expected


@pytest.mark.parametrize("invalid", [True, False, -2, 1.5, "10", None])
def test_random_seed_limits_reject_invalid_values(invalid):
    with pytest.raises(ValueError, match="max_random_seed"):
        resolve_max_random_seed(-1, invalid)
    with pytest.raises(ValueError, match="max_random_seed"):
        get_random_seed_max(SEED_MAX, invalid)


@pytest.mark.parametrize("native_max, minimum", [
    (SEED_MAX - 1, 0),  # most local models and SGL-Omni: randrange(SEED_MAX)
    (2**32 - 1, 0),  # audio.cpp: randrange(2**32)
    (sys.maxsize, 1),  # Higgs: randint(1, sys.maxsize)
])
def test_unbounded_and_larger_caps_preserve_native_range(native_max, minimum):
    assert get_random_seed_max(native_max, min_seed=minimum) == native_max
    assert get_random_seed_max(native_max, native_max + 10, min_seed=minimum) == native_max


@pytest.mark.parametrize("cap", [0, 1, 2147483647])
def test_caps_are_inclusive(cap):
    assert get_random_seed_max(2**32 - 1, cap) == cap


def test_positive_only_range_cannot_accept_zero_cap():
    with pytest.raises(ValueError, match="at least 1"):
        get_random_seed_max(sys.maxsize, 0, min_seed=1)
    assert get_random_seed_max(sys.maxsize, 1, min_seed=1) == 1


def test_every_generation_wrapper_accepts_the_optional_limit():
    # Inspect interfaces without importing every model's separate inference
    # dependencies into one environment. Behavioral tests exercise real wrappers.
    root = Path(__file__).resolve().parents[1] / "tts_audiobook_tool"
    paths = [root / "tts.py", *sorted((root / "tts_models").glob("*_model.py")),
             root / "tts_models" / "audio_cpp_configured.py",
             root / "tts_models" / "sgl_omni_configured.py"]
    checked = 0
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef) or node.name != "generate_using_project":
                continue
            args = node.args.args
            assert args[-1].arg == "max_random_seed", path.name
            assert ast.unparse(args[-1].annotation) == "int", path.name
            assert ast.literal_eval(node.args.defaults[-1]) == -1, path.name
            checked += 1
    assert checked == 18  # dispatcher, base, 13 local models, placeholder, 2 remotes
