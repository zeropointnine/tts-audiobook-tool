"""Passive mixed-FFmpeg warnings must never become a model/audio policy guard."""

from __future__ import annotations

import logging
import os

import pytest

import tts_audiobook_tool.system_support.loaded_ffmpeg_libs as ffmpeg_libs
from tts_audiobook_tool.l import L
from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import (
    FfmpegLibMonitor,
    FfmpegLibReport,
    libs_from_paths,
)

BUNDLED = "/venv/site-packages/av.libs"
BUNDLED_CODEC = f"{BUNDLED}/libavcodec-123abc.so.62"
SYSTEM_CODEC = "/usr/lib/libavcodec.so.60"
OTHER_CODEC = "/opt/another/libavcodec.so.61"


def _report(*paths: str) -> FfmpegLibReport:
    return FfmpegLibReport(libs_from_paths(paths), [BUNDLED])


@pytest.mark.parametrize("paths", [
    (),
    (BUNDLED_CODEC, f"{BUNDLED}/libavutil-abcdef.so.60"),
    (SYSTEM_CODEC, "/usr/lib/libavutil.so.58"),
])
def test_monitor_does_not_warn_for_empty_or_single_builds(monkeypatch, caplog, paths) -> None:
    # System-only is not a coexistence warning, even though it isn't PyAV's bundle.
    monkeypatch.setattr(ffmpeg_libs, "ffmpeg_lib_report", lambda: _report(*paths))
    with caplog.at_level(logging.DEBUG):
        FfmpegLibMonitor().check("startup")
    assert not caplog.records


def test_monitor_keeps_checking_and_deduplicates_distinct_suspicious_snapshots(monkeypatch, caplog) -> None:
    # Lazy imports after clean startup must be caught; old states must not spam.
    mixed = _report(BUNDLED_CODEC, SYSTEM_CODEC)
    reverse = FfmpegLibReport(list(reversed(mixed.libs)), [BUNDLED])
    reports = iter([
        _report(), _report(SYSTEM_CODEC), mixed, reverse,
        _report(BUNDLED_CODEC, SYSTEM_CODEC, OTHER_CODEC), mixed,
    ])
    monkeypatch.setattr(ffmpeg_libs, "ffmpeg_lib_report", lambda: next(reports))
    monitor = FfmpegLibMonitor()
    with caplog.at_level(logging.WARNING):
        for context in ("startup", "system only", "first decode", "repeat", "new backend", "old state"):
            assert monitor.check(context) is None

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert all("FFMPEG_MULTIPLE_BUILDS" in message for message in messages)
    assert all(f"pid={os.getpid()}" in message for message in messages)
    assert all("may be benign" in message and "no operation is blocked" in message for message in messages)
    assert all(BUNDLED_CODEC in message and SYSTEM_CODEC in message for message in messages)
    assert "context='first decode'" in messages[0]
    assert "context='new backend'" in messages[1]
    assert OTHER_CODEC in messages[1]


def test_monitor_catches_duplicate_families_even_inside_one_directory(monkeypatch, caplog) -> None:
    # Two differently named versions can live in the same directory.
    report = _report(BUNDLED_CODEC, f"{BUNDLED}/libavcodec-456def.so.61")
    monkeypatch.setattr(ffmpeg_libs, "ffmpeg_lib_report", lambda: report)
    with caplog.at_level(logging.WARNING):
        FfmpegLibMonitor().check("same directory")
    assert len(caplog.records) == 1
    assert "FFMPEG_MULTIPLE_BUILDS" in caplog.text


@pytest.mark.parametrize("exception", [
    ffmpeg_libs.psutil.AccessDenied(os.getpid()),
    NotImplementedError("unsupported image enumeration"),
    OSError("dyld unavailable"),
])
def test_monitor_enumeration_failures_are_nonfatal_deduplicated_and_retried(monkeypatch, caplog, exception) -> None:
    # A failed snapshot is unknown, not clean; later successful checks still warn.
    calls = []

    def report():
        calls.append(True)
        if len(calls) < 3:
            raise exception
        return _report(BUNDLED_CODEC, SYSTEM_CODEC)

    monkeypatch.setattr(ffmpeg_libs, "ffmpeg_lib_report", report)
    monitor = FfmpegLibMonitor()
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            monitor.check("operation")
    assert len(calls) == 3
    assert sum("FFMPEG_LIBRARY_CHECK_UNAVAILABLE" in record.getMessage() for record in caplog.records) == 1
    assert sum("FFMPEG_MULTIPLE_BUILDS" in record.getMessage() for record in caplog.records) == 1


def test_monitor_logging_failure_cannot_break_operations_or_suppress_later_warning(monkeypatch, caplog) -> None:
    # Even a broken logger must not promote diagnostics to an inference failure.
    monkeypatch.setattr(ffmpeg_libs, "ffmpeg_lib_report", lambda: _report(BUNDLED_CODEC, SYSTEM_CODEC))
    monitor = FfmpegLibMonitor()

    def broken_logger(message):
        raise RuntimeError("log destination unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(L, "w", broken_logger)
        patch.setattr(L, "d", broken_logger)
        monitor.check("broken logging")

    with caplog.at_level(logging.WARNING):
        monitor.check("logging recovered")
    assert len(caplog.records) == 1
    assert "context='logging recovered'" in caplog.text
