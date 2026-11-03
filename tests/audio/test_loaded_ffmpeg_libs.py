"""Loaded-FFmpeg-library report: name classification, build checks, and a real PyAV load."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tts_audiobook_tool.system_support.loaded_ffmpeg_libs as ffmpeg_libs

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


def test_importing_pyav_loads_only_its_bundled_ffmpeg() -> None:
    if not pyav_bundled_lib_dirs():
        pytest.skip("PyAV is not installed from a wheel with bundled FFmpeg")
    report = _report_in_fresh_process("import av")
    assert {"avcodec", "avformat", "avutil", "swresample"} <= set(report["families"]), report["describe"]
    assert report["only_bundled"], report["describe"]
    assert report["single_build"], report["describe"]
    assert all(os.path.isfile(path) for path in report["paths"])


def test_system_ffmpeg_loaded_next_to_pyav_is_reported() -> None:
    # torchaudio 2.8's import-time probing can load system FFmpeg next to PyAV.
    # Coexistence is tolerated; the diagnostic must still identify it.
    pytest.importorskip("torchaudio")
    if not pyav_bundled_lib_dirs():
        pytest.skip("PyAV is not installed from a wheel with bundled FFmpeg")
    report = _report_in_fresh_process("import av\nimport torchaudio")
    if report["only_bundled"]:
        pytest.skip("torchaudio did not load a system FFmpeg here")
    assert not report["single_build"], report["describe"]


def test_darwin_image_enumeration_uses_process_symbols_and_handles_null_names(monkeypatch) -> None:
    # Exercise ctypes wiring without requiring a Mac or loading another dylib.
    paths = [b"/venv/av/.dylibs/libavcodec.62.dylib", None, os.fsencode("/tmp/café/libavutil.60.dylib")]
    process = SimpleNamespace(
        _dyld_image_count=Mock(return_value=len(paths)),
        _dyld_get_image_name=Mock(side_effect=lambda index: paths[index]),
    )
    load = Mock(return_value=process)
    monkeypatch.setattr(ffmpeg_libs.ctypes, "CDLL", load)

    assert ffmpeg_libs._darwin_loaded_image_paths() == [os.fsdecode(paths[0]), os.fsdecode(paths[2])]
    load.assert_called_once_with(None)
    assert process._dyld_image_count.argtypes == []
    assert process._dyld_image_count.restype is ffmpeg_libs.ctypes.c_uint32
    assert process._dyld_get_image_name.argtypes == [ffmpeg_libs.ctypes.c_uint32]
    assert process._dyld_get_image_name.restype is ffmpeg_libs.ctypes.c_char_p


@pytest.mark.parametrize("explicit_pid", [False, True])
def test_darwin_current_process_uses_dyld_instead_of_psutil(monkeypatch, explicit_pid) -> None:
    # The explicit current PID must work too, but psutil has no macOS memory_maps.
    monkeypatch.setattr(ffmpeg_libs.sys, "platform", "darwin")
    monkeypatch.setattr(ffmpeg_libs, "_darwin_loaded_image_paths", lambda: [
        "/venv/av/.dylibs/libavcodec.62.dylib", "/usr/lib/libSystem.B.dylib",
    ])
    process = Mock(side_effect=AssertionError("must not call psutil on macOS"))
    monkeypatch.setattr(ffmpeg_libs.psutil, "Process", process)

    libs = ffmpeg_libs.loaded_ffmpeg_libs(os.getpid() if explicit_pid else None)

    assert [(lib.family, lib.path) for lib in libs] == [("avcodec", "/venv/av/.dylibs/libavcodec.62.dylib")]
    process.assert_not_called()


def test_darwin_other_pid_is_explicitly_unsupported(monkeypatch) -> None:
    # Never silently report self when the caller requests a different process.
    monkeypatch.setattr(ffmpeg_libs.sys, "platform", "darwin")
    with pytest.raises(NotImplementedError, match="only the current process"):
        ffmpeg_libs.loaded_ffmpeg_libs(os.getpid() + 1)


def test_runtime_monitor_does_not_import_ffmpeg_consumers() -> None:
    # The check must not create the very condition it is trying to detect.
    report = _report_in_fresh_process(
        "from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import FfmpegLibMonitor\n"
        "import sys\n"
        "FfmpegLibMonitor().check('fresh process')\n"
        "assert not {'av', 'torchaudio', 'torio', 'torchcodec'} & sys.modules.keys()"
    )
    assert report["paths"] == [], report["describe"]
