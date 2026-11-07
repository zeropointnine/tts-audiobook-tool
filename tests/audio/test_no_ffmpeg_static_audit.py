"""
Static audit of app-owned code for system-FFmpeg dependencies.

The app's own audio I/O uses only wheel-provided libraries (PyAV, soundfile).
These tests scan the production package's source so that a reintroduced
FFmpeg command line, audioread/pydub/ffmpeg-python use, `librosa.load` file
decoding (whose audioread fallback probes `ffmpeg`), or a torchaudio/TorchCodec
import (which load system FFmpeg) is caught even when no test exercises it.

Model adapters (`tts_models/`) are excluded: model-side loading is audited per
model (see docs-dev/audio-io.md).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "tts_audiobook_tool"
EXCLUDED_DIRS = {"tts_models"}

# An executable name as a whole string literal, or the start of a command line
_EXECUTABLE = re.compile(r"^(ffmpeg|ffprobe|ffplay)(\.exe)?(\s|$)")  # not prose ("FFmpeg ...")
_FORBIDDEN_MODULES = {"audioread", "pydub", "ffmpeg", "torchaudio", "torchcodec"}


def app_sources() -> list[Path]:
    return sorted(
        path for path in PACKAGE.rglob("*.py")
        if not EXCLUDED_DIRS.intersection(path.relative_to(PACKAGE).parts)
    )


def findings(path: Path, root: Path = PACKAGE.parent) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    relative = path.relative_to(root)
    found = []

    def add(node: ast.AST, message: str) -> None:
        # Built only for findings: formatting a location for every node dominated the scan time.
        found.append(f"{relative}:{getattr(node, 'lineno', '?')}: {message}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _EXECUTABLE.match(node.value):
            add(node, f"FFmpeg executable {node.value!r}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_MODULES:
                    add(node, f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in _FORBIDDEN_MODULES:
                add(node, f"from {node.module} import ...")
        elif (
            isinstance(node, ast.Attribute) and node.attr == "load"
            and isinstance(node.value, ast.Name) and node.value.id == "librosa"
        ):
            add(node, "librosa.load (audioread fallback)")
    return found


def test_scan_covers_the_app_owned_audio_modules() -> None:
    # Guards against the audit silently scanning nothing after a move/rename
    names = {path.relative_to(PACKAGE).as_posix() for path in app_sources()}
    for expected in (
        "concat_util.py", "stt.py", "sound/audio_decoder.py", "sound/audio_writers.py",
        "sound/sound_file_util.py", "enhance/enhance_alignment.py", "enhance/enhance_flow.py",
    ):
        assert expected in names


def test_app_code_has_no_system_ffmpeg_dependency() -> None:
    # One scan of the whole package; a failure lists every offending file:line.
    found = [finding for path in app_sources() for finding in findings(path)]
    assert found == [], "\n".join(found)


def test_audit_detects_each_kind_of_finding(tmp_path: Path) -> None:
    # The audit's patterns, on a sample with one of each forbidden construct
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import subprocess, librosa\n"
        "from pydub import AudioSegment\n"
        "import torchaudio\n"
        "subprocess.run(['ffmpeg', '-version'])\n"
        "subprocess.run('ffprobe -i x', shell=True)\n"
        "librosa.load('x.mp3')\n"
        "name = 'ffmpeg_util.py'  # not an executable\n",
        encoding="utf-8",
    )
    kinds = sorted(line.split(": ", 1)[1] for line in findings(sample, root=tmp_path))
    assert kinds == sorted([
        "from pydub import ...",
        "import torchaudio",
        "FFmpeg executable 'ffmpeg'",
        "FFmpeg executable 'ffprobe -i x'",
        "librosa.load (audioread fallback)",
    ])
