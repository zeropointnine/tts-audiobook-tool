"""Loaded-FFmpeg-library report: name classification, build checks, and a real PyAV load."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import (
    FfmpegLibReport,
    ffmpeg_lib_family,
    libs_from_paths,
    pyav_bundled_lib_dirs,
)

BUNDLED = "/venv/lib/python3.12/site-packages/av.libs"


@pytest.mark.parametrize(
    ("path", "family"),
    [
        ("/usr/lib/x86_64-linux-gnu/libavcodec.so.61.19.101", "avcodec"),
        (f"{BUNDLED}/libavformat-4639def2.so.62.12.102", "avformat"),
        (f"{BUNDLED}/libswresample-f0bc92fc.so.6.3.102", "swresample"),
        ("/usr/lib/libavutil.so", "avutil"),
        (r"C:\venv\Lib\site-packages\av.libs\avcodec-62-5e6b81c2.dll", "avcodec"),
        (r"C:\ffmpeg\bin\SWSCALE-9.dll", "swscale"),
        ("/venv/site-packages/av/.dylibs/libavfilter.11.dylib", "avfilter"),
    ],
)
def test_ffmpeg_libraries_are_classified(path: str, family: str) -> None:
    assert ffmpeg_lib_family(path) == family


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Windows\System32\avrt.dll",  # Windows multimedia scheduler, not FFmpeg
        "/usr/lib/x86_64-linux-gnu/libavahi-client.so.3",
        f"{BUNDLED}/libx264-d6533a8d.so.165",
        "/venv/site-packages/torio/lib/libtorio_ffmpeg6.so",
        "/venv/site-packages/av/codec/codec.abi3.so",
        "[heap]",
        "",
    ],
)
def test_non_ffmpeg_libraries_are_ignored(path: str) -> None:
    assert ffmpeg_lib_family(path) is None


def test_report_flags_system_libraries_next_to_bundled_ones() -> None:
    # A second build (system FFmpeg via torchaudio, say) next to PyAV's copy
    # is exactly what the check must catch.
    system = "/usr/lib/x86_64-linux-gnu/libavcodec.so.61"
    report = FfmpegLibReport(
        libs_from_paths([f"{BUNDLED}/libavcodec-1.so.62", f"{BUNDLED}/libavutil-2.so.60", system, system]),
        [BUNDLED],
    )
    assert not report.single_build
    assert not report.only_bundled
    assert [lib.path for lib in report.outside_bundled] == [system]


def test_report_accepts_one_bundled_build() -> None:
    report = FfmpegLibReport(
        libs_from_paths([f"{BUNDLED}/libavcodec-1.so.62", f"{BUNDLED}/libavutil-2.so.60", "/usr/lib/libc.so.6"]),
        [BUNDLED],
    )
    assert report.single_build and report.only_bundled
    assert [lib.family for lib in report.libs] == ["avcodec", "avutil"]


def test_no_libraries_loaded_is_a_clean_report() -> None:
    report = FfmpegLibReport([], [BUNDLED])
    assert report.single_build and report.only_bundled
    assert "No FFmpeg libraries loaded" in report.describe()


def _report_in_fresh_process(imports: str) -> dict:
    # Other tests import torchaudio, which loads system FFmpeg; a fresh
    # interpreter keeps the report about the given imports only.
    script = (
        f"{imports}\n"
        "import json\n"
        "from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import ffmpeg_lib_report\n"
        "r = ffmpeg_lib_report()\n"
        "print(json.dumps({'families': [l.family for l in r.libs], 'paths': [l.path for l in r.libs],\n"
        "    'only_bundled': r.only_bundled, 'single_build': r.single_build, 'describe': r.describe()}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


requires_memory_maps = pytest.mark.skipif(
    sys.platform == "darwin", reason="psutil memory_maps() is unavailable on macOS"
)


@requires_memory_maps
def test_importing_pyav_loads_only_its_bundled_ffmpeg() -> None:
    if not pyav_bundled_lib_dirs():
        pytest.skip("PyAV is not installed from a wheel with bundled FFmpeg")
    report = _report_in_fresh_process("import av")
    assert {"avcodec", "avformat", "avutil", "swresample"} <= set(report["families"]), report["describe"]
    assert report["only_bundled"], report["describe"]
    assert report["single_build"], report["describe"]
    assert all(os.path.isfile(path) for path in report["paths"])


@requires_memory_maps
def test_system_ffmpeg_loaded_next_to_pyav_is_reported() -> None:
    # torchaudio 2.8's import-time backend probing loads system FFmpeg when
    # present: the mixed-build situation Milestone B must eliminate.
    pytest.importorskip("torchaudio")
    if not pyav_bundled_lib_dirs():
        pytest.skip("PyAV is not installed from a wheel with bundled FFmpeg")
    report = _report_in_fresh_process("import av\nimport torchaudio")
    if report["only_bundled"]:
        pytest.skip("torchaudio did not load a system FFmpeg here")
    assert not report["single_build"], report["describe"]
