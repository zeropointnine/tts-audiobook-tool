"""Remote probing stays explicit and cannot guess a backend or model family from an ID."""

from __future__ import annotations

import sys
from types import ModuleType

import httpx
import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.tts_models import audio_cpp_detection
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


AUDIO_HEALTH = {"status": "ok", "backend": "cpu", "models": 1, "ui": False, "ui_management": False}
AUDIO_MODEL = {"id": "operator-defined-name", "family": "chatterbox", "task": "clon", "mode": "offline",
               "session_options": {"chatterbox.multilingual_t3": "v3"}}


@pytest.fixture(autouse=True)
def clean_discovery(monkeypatch):
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "")


def mock_server(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))


def test_audio_probe_query_snapshot_cache_and_force(monkeypatch):
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json=AUDIO_HEALTH)
        assert request.url.path == "/v1/models"
        assert request.url.params.get("include_session_options") == "true"
        return httpx.Response(200, json={"data": [AUDIO_MODEL]})

    mock_server(monkeypatch, respond)
    monkeypatch.setattr(audio_cpp_detection, "detect_audio_cpp_models", lambda models: [(TtsModelType.require_by_id("chatterbox_local"), models[0]["id"])])
    RemoteTtsDiscovery.set_base_url("http://example.test:8000/")
    assert RemoteTtsDiscovery.get_base_url() == "http://example.test:8000"
    assert RemoteTtsDiscovery.get_snapshot().issue.code == "not_probed"
    assert not requests
    snapshot = RemoteTtsDiscovery.refresh()
    assert snapshot.backend_kind is TtsBackendKind.AUDIO_CPP
    assert snapshot.models == (AUDIO_MODEL,)
    assert snapshot.candidates == ((TtsModelType.require_by_id("chatterbox_local"), "operator-defined-name"),)
    assert snapshot.issue is None
    assert RemoteTtsDiscovery.refresh() is snapshot
    assert RemoteTtsDiscovery.get_snapshot() is snapshot
    assert len(requests) == 2
    refreshed = RemoteTtsDiscovery.refresh(force=True)
    assert refreshed is not snapshot
    assert len(requests) == 4
    RemoteTtsDiscovery.set_base_url("http://other.test")
    assert RemoteTtsDiscovery.get_snapshot().issue.code == "not_probed"
    RemoteTtsDiscovery.set_base_url("http://example.test:8000")
    assert RemoteTtsDiscovery.get_snapshot() is refreshed


@pytest.mark.parametrize("health", [
    {"status": "ok"},
    {"status": "ok", "backend": "cpu", "models": True, "ui": False, "ui_management": False},
    {"status": "ok", "backend": "", "models": 1, "ui": False, "ui_management": False},
    {"status": "unknown"},
])
def test_unrecognized_health_never_fetches_models(monkeypatch, health):
    paths = []
    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=health)
    mock_server(monkeypatch, respond)
    RemoteTtsDiscovery.set_base_url("http://example.test")
    snapshot = RemoteTtsDiscovery.refresh()
    assert snapshot.backend_kind is None
    assert snapshot.issue.code == "unrecognized_server"
    assert paths == ["/health"]


def test_sgl_probe_needs_nonempty_structured_models(monkeypatch):
    # Stub the separately owned detector to test only discovery's wiring.
    detector = ModuleType("tts_audiobook_tool.tts_models.sgl_omni_detection")
    calls = []
    detector.detect_sgl_omni_models = lambda entries: calls.append(entries) or [(TtsModelType.require_by_id("higgs_v3_sglomni"), entries[0]["id"])]
    monkeypatch.setitem(sys.modules, detector.__name__, detector)
    payload = {"data": [{"id": "org/model"}]}
    def respond(request):
        assert not request.url.params
        return httpx.Response(200, json={"status": "healthy"} if request.url.path == "/health" else payload)
    mock_server(monkeypatch, respond)
    RemoteTtsDiscovery.set_base_url("http://example.test")
    snapshot = RemoteTtsDiscovery.refresh()
    assert snapshot.backend_kind is TtsBackendKind.SGL_OMNI
    assert snapshot.candidates == ((TtsModelType.require_by_id("higgs_v3_sglomni"), "org/model"),)
    assert calls == [payload["data"]]
    payload["data"] = [{"id": ""}]
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "invalid_models"
    payload["data"] = []
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "no_models"
    payload["data"] = [{"id": "org/one"}, {"id": "org/two"}]
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "invalid_models"
    payload.clear()
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "invalid_models"


def test_network_failure_is_cached_until_forced(monkeypatch):
    calls = []
    def respond(request):
        calls.append(request)
        raise httpx.ConnectError("offline")
    mock_server(monkeypatch, respond)
    RemoteTtsDiscovery.set_base_url("http://example.test")
    assert RemoteTtsDiscovery.refresh().issue.code == "unavailable"
    assert RemoteTtsDiscovery.refresh().issue.code == "unavailable"
    assert len(calls) == 1
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "unavailable"
    assert len(calls) == 2


@pytest.mark.parametrize("url", [
    "http://localhost:bad",
    "http://localhost:99999",
    "http://[broken",
    "http://",
    "ftp://example.test",
    "http://exa mple.test",
    "http://localhost:\n8000",
])
def test_malformed_url_is_a_cached_recoverable_issue_without_network(monkeypatch, url):
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("Malformed URL must not probe"))
    RemoteTtsDiscovery.set_base_url(url)

    snapshot = RemoteTtsDiscovery.refresh()

    assert snapshot.issue.code == "invalid_url"
    assert "Invalid remote TTS server URL" in snapshot.issue.message
    assert RemoteTtsDiscovery.refresh() is snapshot
    assert RemoteTtsDiscovery.get_snapshot() is snapshot
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "invalid_url"
    RemoteTtsDiscovery.set_base_url("http://valid.example")
    assert RemoteTtsDiscovery.get_snapshot().issue.code == "not_probed"


def test_invalid_url_from_httpx_request_is_recoverable(monkeypatch):
    def respond(request):
        raise httpx.InvalidURL("Malformed request URL")
    mock_server(monkeypatch, respond)
    RemoteTtsDiscovery.set_base_url("http://example.test")

    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "invalid_url"


@pytest.mark.parametrize("url", [
    "http://localhost:8000", "https://example.test/api", "http://[::1]:8000",
])
def test_url_validation_is_side_effect_free(monkeypatch, url):
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("Validation must not probe"))
    RemoteTtsDiscovery.set_base_url("http://original.test")

    assert RemoteTtsDiscovery.validate_url(url) is None
    assert RemoteTtsDiscovery.get_base_url() == "http://original.test"
    assert RemoteTtsDiscovery._snapshots == {}


def test_observation_is_reused_within_ttl_and_reprobed_after(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(RemoteTtsDiscovery, "_now", staticmethod(lambda: now[0]))
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=AUDIO_HEALTH if request.url.path == "/health" else {"data": [AUDIO_MODEL]})
    mock_server(monkeypatch, respond)
    monkeypatch.setattr(audio_cpp_detection, "detect_audio_cpp_models",
                        lambda models: [(TtsModelType.require_by_id("chatterbox_local"), models[0]["id"])])
    RemoteTtsDiscovery.set_base_url("http://example.test")

    first = RemoteTtsDiscovery.refresh()
    assert len(requests) == 2

    now[0] += RemoteTtsDiscovery._ttl_seconds - 0.1
    assert RemoteTtsDiscovery.refresh() is first
    assert len(requests) == 2, "within the TTL the observation must be reused"

    now[0] += 0.2
    second = RemoteTtsDiscovery.refresh()
    assert second is not first
    assert len(requests) == 4, "after the TTL the next refresh re-probes"


def test_failure_backoff_outlasts_the_success_ttl(monkeypatch):
    now = [2000.0]
    monkeypatch.setattr(RemoteTtsDiscovery, "_now", staticmethod(lambda: now[0]))
    calls = []
    def respond(request):
        calls.append(request)
        raise httpx.ConnectError("offline")
    mock_server(monkeypatch, respond)
    RemoteTtsDiscovery.set_base_url("http://example.test")

    assert RemoteTtsDiscovery.refresh().issue.code == "unavailable"
    assert len(calls) == 1

    now[0] += RemoteTtsDiscovery._ttl_seconds + 1.0
    assert RemoteTtsDiscovery.refresh().issue.code == "unavailable"
    assert len(calls) == 1, "a cached failure must not be retried on the success TTL"

    now[0] += RemoteTtsDiscovery._failure_ttl_seconds
    assert RemoteTtsDiscovery.refresh().issue.code == "unavailable"
    assert len(calls) == 2, "the failure backoff window must eventually expire"


def test_forced_probe_uses_the_full_timeout_and_revalidation_the_short_one(monkeypatch):
    timeouts = []
    def respond(request):
        return httpx.Response(200, json={"status": "ok", "backend": "cpu", "models": 0, "ui": False, "ui_management": False})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: (
        timeouts.append(kwargs.get("timeout")),
        real_client(transport=httpx.MockTransport(respond), **kwargs))[1])
    RemoteTtsDiscovery.set_base_url("http://example.test")

    RemoteTtsDiscovery.refresh(force=True)
    assert timeouts == [RemoteTtsDiscovery._timeout], "a forced probe waits on the real timeout"

    monkeypatch.setattr(RemoteTtsDiscovery, "_now", staticmethod(lambda: 1e12))
    RemoteTtsDiscovery.refresh()
    assert timeouts[-1] is RemoteTtsDiscovery._revalidate_timeout, "revalidation fails fast"


@pytest.mark.parametrize("session_options", [
    {"chatterbox.multilingual_t3": "v2"},
    {"chatterbox.multilingual_t3": "v3"},
    {},
    None,
])
def test_audio_probe_accepts_server_selected_chatterbox_version(monkeypatch, session_options):
    model = {**AUDIO_MODEL, "session_options": session_options}
    if session_options is None:
        del model["session_options"]

    def respond(request):
        return httpx.Response(200, json=AUDIO_HEALTH if request.url.path == "/health" else {"data": [model]})
    mock_server(monkeypatch, respond)
    RemoteTtsDiscovery.set_base_url("http://example.test")

    snapshot = RemoteTtsDiscovery.refresh()

    assert snapshot.backend_kind is TtsBackendKind.AUDIO_CPP
    assert snapshot.models == (model,)
    assert snapshot.candidates == ((TtsModelType.require_by_id("chatterbox_audiocpp"), model["id"]),)
    assert snapshot.issue is None


@pytest.mark.parametrize("session_options", [
    {"chatterbox.multilingual_t3": "v2"},
    {"chatterbox.multilingual_t3": "v3"},
    {},
])
def test_recognized_backend_with_no_supported_model_reports_issue(monkeypatch, session_options):
    model = {**AUDIO_MODEL, "session_options": session_options}
    def respond(request):
        return httpx.Response(200, json=AUDIO_HEALTH if request.url.path == "/health" else {"data": [model]})
    mock_server(monkeypatch, respond)
    monkeypatch.setattr(audio_cpp_detection, "detect_audio_cpp_models", lambda _models: [])
    RemoteTtsDiscovery.set_base_url("http://example.test")
    snapshot = RemoteTtsDiscovery.refresh()
    assert snapshot.backend_kind is TtsBackendKind.AUDIO_CPP
    assert snapshot.models == (model,)
    assert snapshot.candidates == ()
    assert snapshot.issue.code == "no_supported_models"


def test_audio_model_metadata_and_session_options_must_be_valid(monkeypatch):
    model = dict(AUDIO_MODEL)
    def respond(request):
        return httpx.Response(200, json=AUDIO_HEALTH if request.url.path == "/health" else {"data": [model]})
    mock_server(monkeypatch, respond)
    RemoteTtsDiscovery.set_base_url("http://example.test")
    model["family"] = 7
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "invalid_models"
    model["family"] = "chatterbox"
    model["session_options"] = ["v3"]
    assert RemoteTtsDiscovery.refresh(force=True).issue.code == "invalid_models"


@pytest.mark.parametrize("session_options", [
    {"chatterbox.multilingual_t3": "v2"},
    {"chatterbox.multilingual_t3": "v3"},
    {"multilingual_t3": "v3"},
    {},
    None,
])
def test_audio_detector_accepts_server_selected_chatterbox_version(session_options):
    model = {**AUDIO_MODEL, "id": "operator-defined-name", "session_options": session_options}
    if session_options is None:
        del model["session_options"]
    assert audio_cpp_detection.detect_audio_cpp_models([model]) == [
        (TtsModelType.require_by_id("chatterbox_audiocpp"), "operator-defined-name")
    ]


@pytest.mark.parametrize("unsupported", [
    {"family": "chatterbox_turbo"},
    {"task": "tts"},
    {"mode": "streaming"},
])
def test_audio_detector_chatterbox_still_requires_offline_clone_family(unsupported):
    assert audio_cpp_detection.detect_audio_cpp_models([{**AUDIO_MODEL, **unsupported}]) == []


def test_audio_detector_matches_higgs_without_session_options():
    higgs = {"id": "higgs-audio-tts", "family": "higgs_audio_tts", "task": "tts", "mode": "offline",
             "session_options": {}}
    assert audio_cpp_detection.detect_audio_cpp_models([higgs]) == [
        (TtsModelType.require_by_id("higgs_v3_audiocpp"), "higgs-audio-tts")
    ]
    # Declared family/task/mode still identifies the family, so a mismatch is not a candidate.
    assert audio_cpp_detection.detect_audio_cpp_models([{**higgs, "task": "clon"}]) == []
    assert audio_cpp_detection.detect_audio_cpp_models([{**higgs, "family": "chatterbox"}]) == []
    assert audio_cpp_detection.detect_audio_cpp_models([AUDIO_MODEL, higgs]) == [
        (TtsModelType.require_by_id("chatterbox_audiocpp"), "operator-defined-name"),
        (TtsModelType.require_by_id("higgs_v3_audiocpp"), "higgs-audio-tts"),
    ]


def test_audio_detector_matches_breeze_tts_and_clone_routes():
    breeze = {"id": "breeze-clone", "family": "breeze_tts", "task": "clon", "mode": "offline",
              "session_options": {}}
    tts = {**breeze, "id": "breeze-tts", "task": "tts"}
    handle = TtsModelType.require_by_id("breeze_tts_2_audiocpp")
    # Keep existing clone configurations and discover reference-less TTS ones.
    assert audio_cpp_detection.detect_audio_cpp_models([breeze, tts]) == [
        (handle, "breeze-clone"), (handle, "breeze-tts"),
    ]
    assert audio_cpp_detection.detect_audio_cpp_models([{**breeze, "mode": "streaming"}]) == []
    assert audio_cpp_detection.detect_audio_cpp_models([{**tts, "mode": "streaming"}]) == []
    assert audio_cpp_detection.detect_audio_cpp_models([{**breeze, "task": "asr"}]) == []


def test_audio_detector_matches_echo_clone_route_only():
    echo = {"id": "echo-tts", "family": "echo_tts", "task": "clon", "mode": "offline",
            "session_options": {}}
    assert audio_cpp_detection.detect_audio_cpp_models([echo]) == [
        (TtsModelType.require_by_id("echo_tts_audiocpp"), "echo-tts")
    ]
    # Anything but the offline clone route is a different family entry.
    assert audio_cpp_detection.detect_audio_cpp_models([{**echo, "task": "tts"}]) == []
    assert audio_cpp_detection.detect_audio_cpp_models([{**echo, "mode": "streaming"}]) == []


def test_audio_detector_matches_omnivoice_offline_route_only():
    # OmniVoice's one `tts` route serves clone, voice design and auto voice;
    # which one a request gets is decided by the request body, so matching is
    # still the session identity alone.
    omnivoice = {"id": "omnivoice", "family": "omnivoice", "task": "tts", "mode": "offline",
                 "session_options": {}}
    assert audio_cpp_detection.detect_audio_cpp_models([omnivoice]) == [
        (TtsModelType.require_by_id("omnivoice_audiocpp"), "omnivoice")
    ]
    assert audio_cpp_detection.detect_audio_cpp_models([{**omnivoice, "family": "chatterbox"}]) == []
    assert audio_cpp_detection.detect_audio_cpp_models([{**omnivoice, "mode": "streaming"}]) == []


# DISABLED (audio.cpp GLM-TTS): the catalog entry is commented out, so this
# detector test has no declared entry to match. Restore it with the entry.
# def test_audio_detector_matches_glm_under_either_advertised_task():
#     # audio.cpp advertises the task token its operator configured, and this
#     # family's two advertised routes are the same reference-conditioned path,
#     # so both spellings select the one catalog entry.
#     for task in ("tts", "clon"):
#         glm = {"id": "my-glm-server-entry", "family": "glm_tts", "task": task, "mode": "offline",
#                "session_options": {}}
#         assert audio_cpp_detection.detect_audio_cpp_models([glm]) == [
#             (TtsModelType.require_by_id("glm_tts_audiocpp"), "my-glm-server-entry")
#         ]
#     # A server entry whose ID merely mentions the model proves nothing; the
#     # declared family/task/mode has to identify a supported route.
#     for unsupported in (
#         {"id": "glm_tts_zeroshot", "family": "glm"},
#         {"id": "my-glm-server-entry", "family": "glm_tts", "task": "vdes"},
#         {"id": "my-glm-server-entry", "family": "glm_tts", "task": "tts", "mode": "streaming"},
#     ):
#         model = {"family": "glm_tts", "task": "tts", "mode": "offline", "session_options": {}, **unsupported}
#         assert audio_cpp_detection.detect_audio_cpp_models([model]) == []


def test_audio_detector_treats_a_disabled_family_as_unsupported():
    """While the GLM-TTS entry is commented out, its server entries are not candidates."""
    for task in ("tts", "clon"):
        glm = {"id": "my-glm-server-entry", "family": "glm_tts", "task": task, "mode": "offline",
               "session_options": {}}
        assert audio_cpp_detection.detect_audio_cpp_models([glm]) == []


def test_audio_detector_matches_catalog_metadata_not_id(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.toml"
    catalog.write_text('''[[models]]
id = "chatterbox_local"
backend_kind = "audio_cpp"
[models.audio_cpp.match]
family = "chatterbox"
task = "clon"
mode = "offline"
[models.audio_cpp.match.session_options]
"chatterbox.multilingual_t3" = "v3"
''', encoding="utf-8")
    monkeypatch.setattr(audio_cpp_detection, "CATALOG_PATH", catalog)
    monkeypatch.setattr(TtsModelType, "_specs", {
        **TtsModelType._specs,
        "chatterbox_local": TtsModelType.require_by_id("chatterbox_local").value._replace(backend_kind=TtsBackendKind.AUDIO_CPP),
    })
    entries = [
        {**AUDIO_MODEL, "id": "this-says-v2-but-is-v3"},
        {**AUDIO_MODEL, "id": "this-says-v3-but-is-v2", "session_options": {}},
        {**AUDIO_MODEL, "id": "unsupported-option", "session_options": {"chatterbox.multilingual_t3": "v4"}},
        {**AUDIO_MODEL, "id": "invalid-options", "session_options": {"chatterbox.multilingual_t3": True}},
        {**AUDIO_MODEL, "id": "wrong-task", "task": "tts"},
    ]
    assert audio_cpp_detection.detect_audio_cpp_models(entries) == [
        (TtsModelType.require_by_id("chatterbox_local"), "this-says-v2-but-is-v3"),
    ]
