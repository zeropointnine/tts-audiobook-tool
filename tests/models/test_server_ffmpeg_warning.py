"""Server diagnostics cover lazy synthesis imports without sockets or inference."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import tts_audiobook_tool.server.server as server_module
import tts_audiobook_tool.system_support.loaded_ffmpeg_libs as ffmpeg_libs
from tts_audiobook_tool.tts import Tts


class StopWorker(Exception):
    pass


def _mixed_libs():
    return ffmpeg_libs.libs_from_paths([
        "/venv/av.libs/libavcodec-123abc.so.62", "/usr/lib/libavcodec.so.60",
    ])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("generation_fails", [False, True])
def test_server_checks_after_synthesis_without_changing_cleanup_or_errors(
    monkeypatch, caplog, streaming, generation_fails,
):
    # Both synthesis modes may load a decoder lazily, including before failure.
    server = server_module.Server.__new__(server_module.Server)
    calls = []
    loaded = []
    prompt = SimpleNamespace(
        phrase_group=SimpleNamespace(text="hello"), can_eager=False,
        use_tts_streaming=streaming, is_prompt_start=False,
    )

    class Queue:
        def __init__(self):
            self.first = True

        def get(self):
            if not self.first:
                raise StopWorker
            self.first = False
            return 0, 0, prompt

        def task_done(self):
            calls.append("done")

    def generate(*args):
        calls.append("generate")
        loaded.extend(_mixed_libs())
        if generation_fails:
            raise ValueError("original synthesis failure")

    def report():
        calls.append("scan")
        return ffmpeg_libs.FfmpegLibReport(list(loaded), ["/venv/av.libs"])

    server._queue = Queue()
    server._generation_id = 0
    server._tts_ready = threading.Event()
    server._tts_ready.set()
    server._audio_stream = SimpleNamespace(get_seconds_left=lambda: 0)
    server._project = object()
    server._ffmpeg_monitor = ffmpeg_libs.FfmpegLibMonitor()
    server.generate_streaming_output = generate
    server.generate_non_streaming_output = generate
    monkeypatch.setattr(Tts, "get_info", lambda *args: SimpleNamespace(can_stream=True))
    monkeypatch.setattr(ffmpeg_libs, "ffmpeg_lib_report", report)

    if generation_fails:
        with pytest.raises(ValueError, match="original synthesis failure"):
            server._worker()
    else:
        with pytest.raises(StopWorker):
            server._worker()

    assert calls == ["generate", "scan", "done"]
    assert server._prompt_currently_inferencing == ""
    assert server._tts_ready.is_set()
    assert sum("FFMPEG_MULTIPLE_BUILDS" in record.getMessage() for record in caplog.records) == 1
    assert "server after synthesis" in caplog.text


@pytest.mark.parametrize("initialization_fails", [False, True])
def test_server_checks_model_initialization_even_before_fatal_exit(monkeypatch, caplog, initialization_fails):
    # No prompt need arrive for the startup state to reach the persistent log.
    server = server_module.Server.__new__(server_module.Server)
    server._project = object()
    server._tts_ready = threading.Event()
    server._is_initializing = True
    server._ffmpeg_monitor = ffmpeg_libs.FfmpegLibMonitor()
    loaded = []
    model = object()

    def initialize():
        loaded.extend(_mixed_libs())
        if initialization_fails:
            raise ValueError("original initialization failure")
        return model

    class HttpServer:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def serve_forever(self):
            pass

    def hard_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(server_module, "ThreadingHTTPServer", HttpServer)
    monkeypatch.setattr(server_module.threading, "Thread", lambda *, target, daemon: SimpleNamespace(start=target))
    monkeypatch.setattr(server_module, "get_blocking_issues_error", lambda *args: "")
    monkeypatch.setattr(server_module, "get_run_warnings", lambda *args: [])
    monkeypatch.setattr(server_module, "print_feedback", lambda *args: None)
    monkeypatch.setattr(server_module.os, "_exit", hard_exit)
    monkeypatch.setattr(Tts, "get_instance", initialize)
    monkeypatch.setattr(Tts, "get_instance_if_exists", lambda: model)
    monkeypatch.setattr(ffmpeg_libs, "ffmpeg_lib_report", lambda: ffmpeg_libs.FfmpegLibReport(
        list(loaded), ["/venv/av.libs"],
    ))

    if initialization_fails:
        with pytest.raises(SystemExit) as exc:
            server.run()
        assert exc.value.code == 1
        assert not server._tts_ready.is_set()
        assert "server model initialization failed" in caplog.text
    else:
        server.run()
        assert server._tts_ready.is_set()
        assert not server._is_initializing
    assert sum("FFMPEG_MULTIPLE_BUILDS" in record.getMessage() for record in caplog.records) == 1
