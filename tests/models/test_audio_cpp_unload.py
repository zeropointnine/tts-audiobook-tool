"""Offline HTTP contract tests for synchronous audio.cpp bulk unloading."""
import httpx
import pytest

from tts_audiobook_tool.app_support import audio_cpp_util
from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil


@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"unloaded": ["unsupported-server-model"]}),
    httpx.Response(200, json={"unloaded": []}),
    httpx.Response(204),
])
def test_unload_posts_without_body_and_waits_for_response(monkeypatch, response):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(audio_cpp_util.httpx, "post", post)

    assert AudioCppUtil.unload_all_models(" http://server.test/api/// ") is None
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "http://server.test/api/v1/tasks/unload_all_models"
    assert set(kwargs) == {"timeout"}
    timeout = kwargs["timeout"]
    assert timeout.connect == 5.0
    assert timeout.write == 30.0
    assert timeout.pool == 5.0
    assert timeout.read is None


@pytest.mark.parametrize(("response", "expected"), [
    (httpx.Response(503, json={"error": {"message": "Server is busy", "type": "server_busy"}}),
     "Server is busy"),
    (httpx.Response(500, text="Teardown failed"),
     "audio.cpp unload failed (HTTP 500): Teardown failed"),
    (httpx.Response(403), "audio.cpp unload failed (HTTP 403): Forbidden"),
    (httpx.Response(302), "audio.cpp unload failed (HTTP 302): Found"),
    (httpx.Response(500, json={"error": {"message": " "}}),
     'audio.cpp unload failed (HTTP 500): {"error":{"message":" "}}'),
    (httpx.Response(500, json=["unexpected"]),
     'audio.cpp unload failed (HTTP 500): ["unexpected"]'),
])
def test_unload_returns_server_errors_without_retry(monkeypatch, response, expected):
    calls = []
    monkeypatch.setattr(audio_cpp_util.httpx, "post", lambda *a, **kw: calls.append(a) or response)

    assert AudioCppUtil.unload_all_models("http://server.test") == expected
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    httpx.ConnectError("Connection refused"),
    httpx.ConnectTimeout("Connection timed out"),
    httpx.ReadError("Connection closed"),
    httpx.InvalidURL("Invalid port"),
    ValueError("Invalid URL"),
])
def test_unload_returns_transport_and_url_errors(monkeypatch, error):
    def post(*args, **kwargs):
        raise error

    monkeypatch.setattr(audio_cpp_util.httpx, "post", post)

    assert AudioCppUtil.unload_all_models("http://server.test") == f"audio.cpp unload failed: {error}"


@pytest.mark.parametrize("url", ["", "  ", "/"])
def test_unload_requires_base_url_without_network(monkeypatch, url):
    monkeypatch.setattr(audio_cpp_util.httpx, "post", lambda *a, **kw: pytest.fail("Must not request"))

    assert AudioCppUtil.unload_all_models(url) == "audio.cpp server URL is not configured"
