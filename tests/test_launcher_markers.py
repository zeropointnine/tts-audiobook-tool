"""Compatibility coverage for remote-client venv detection in the launcher."""

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import launch


NEW_MARKER = "tts_audiobook_tool_remote_client_marker"
LEGACY_MARKER = "tts_audiobook_tool_sgl_omni_marker"


@pytest.fixture
def marker_venv(monkeypatch, tmp_path):
    """Run the launcher's probe with isolated marker packages and no site-packages."""
    run = subprocess.run
    monkeypatch.setattr(launch, "get_venv_python", lambda path: sys.executable)

    def run_probe(command, **kwargs):
        assert command[:2] == [sys.executable, "-c"]
        code = f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n" + command[2]
        return run([sys.executable, "-S", "-c", code], **kwargs)

    monkeypatch.setattr(launch.subprocess, "run", run_probe)
    return tmp_path


@pytest.mark.parametrize(
    "markers, expected",
    [
        ((), False),
        ((NEW_MARKER,), True),
        ((LEGACY_MARKER,), True),
        ((NEW_MARKER, LEGACY_MARKER), True),
    ],
)
def test_has_remote_client_marker(marker_venv, markers, expected):
    for marker in markers:
        package = marker_venv / marker
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")

    assert launch.has_remote_client_marker(str(marker_venv)) is expected


@pytest.mark.parametrize("present_marker", [NEW_MARKER, LEGACY_MARKER])
def test_unreadable_other_marker_does_not_hide_remote_client(monkeypatch, present_marker):
    monkeypatch.setattr(launch, "get_venv_python", lambda path: sys.executable)

    def find_spec(name):
        if name == present_marker:
            return ModuleType(name)
        raise OSError("unreadable")

    monkeypatch.setattr("importlib.util.find_spec", find_spec)

    def run_probe(command, **kwargs):
        with pytest.raises(SystemExit) as result:
            exec(command[2], {})
        return SimpleNamespace(returncode=result.value.code)

    monkeypatch.setattr(launch.subprocess, "run", run_probe)
    assert launch.has_remote_client_marker("client-venv")


def test_has_remote_client_marker_without_python_is_false(monkeypatch):
    monkeypatch.setattr(launch, "get_venv_python", lambda path: None)
    assert not launch.has_remote_client_marker("missing-venv")


def test_both_markers_produce_one_remote_client_entry(marker_venv, monkeypatch):
    for marker in (NEW_MARKER, LEGACY_MARKER):
        package = marker_venv / marker
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")

    path = str(marker_venv)
    monkeypatch.setattr(launch, "find_venvs", lambda base_dir: [path])
    monkeypatch.setattr(launch, "probe_venv", lambda venv: ([], 0))
    assert launch.build_venv_list(path) == [
        (path, marker_venv.name, [launch.REMOTE_CLIENT_DISPLAY_MODEL]),
    ]
