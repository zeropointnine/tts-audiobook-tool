"""Local-model detection honors `local_module_exclude` (eg a Breeze TTS 2 venv also contains `qwen_tts`)."""

import subprocess
import sys
import types

import pytest

import launch
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def _fake_find_spec(present: set[str]):
    return lambda name: types.ModuleType(name) if name in present else None


def test_qwen_tts_alone_detects_qwen3tts(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", _fake_find_spec({"qwen_tts"}))

    tts_type, matches = Tts.init_local_model_type()

    assert tts_type is TtsModelType.require_by_id("qwen3tts_local")
    assert matches == 1


def test_breeze_venv_detects_only_breeze(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", _fake_find_spec({"qwen_tts", "breeze_tts"}))

    tts_type, matches = Tts.init_local_model_type()

    assert tts_type is TtsModelType.require_by_id("breeze_tts_2_local")
    assert matches == 1


@pytest.fixture
def isolated_venv(monkeypatch, tmp_path):
    """Run the launcher's subprocess probe against fake packages only (no site-packages)."""
    run = subprocess.run
    monkeypatch.setattr(launch, "get_venv_python", lambda path: sys.executable)

    def run_probe(command, **kwargs):
        assert command[:2] == [sys.executable, "-c"]
        code = f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n" + command[2]
        return run([sys.executable, "-S", "-c", code], **kwargs)

    monkeypatch.setattr(launch.subprocess, "run", run_probe)

    def add_packages(*names: str) -> str:
        for name in names:
            package = tmp_path / name
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
        return str(tmp_path)

    return add_packages


def test_launcher_probe_qwen_tts_alone(isolated_venv):
    qwen_name = TtsModelType.require_by_id("qwen3tts_local").value.ui["proper_name"]
    names, count = launch.probe_venv(isolated_venv("qwen_tts"))
    assert (names, count) == ([qwen_name], 1)


def test_launcher_probe_breeze_venv_detects_only_breeze(isolated_venv):
    breeze_name = TtsModelType.require_by_id("breeze_tts_2_local").value.ui["proper_name"]
    names, count = launch.probe_venv(isolated_venv("qwen_tts", "breeze_tts"))
    assert (names, count) == ([breeze_name], 1)
