"""Offline tests for the advisory audio.cpp cold-model initialization notice."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest

from tts_audiobook_tool.app_support import audio_cpp_util
from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.constants import COL_DIM_ITALICS


BASE_URL = "http://audio-server:8080"
MODEL_ID = "server-model"
DESCRIPTION = "Example TTS"
MODEL_PATH = "/models/example.gguf"


def cold_model(**overrides):
    return {"id": MODEL_ID, "loaded": False, "path": MODEL_PATH, **overrides}


def response_for(payload):
    response = Mock(spec=httpx.Response)
    response.json.return_value = payload
    return response


@pytest.fixture
def notice_io(monkeypatch):
    """Keep all requests offline and reject unrelated generation/discovery work."""
    get = Mock(name="httpx.get")
    print_init = Mock(name="print_init")
    forbidden = {
        "generate": Mock(name="generate"),
        "check_readiness": Mock(name="check_readiness"),
        "set_base_url": Mock(name="set_base_url"),
        "post": Mock(name="httpx.post"),
        "client": Mock(name="httpx.Client"),
    }
    monkeypatch.setattr(audio_cpp_util.httpx, "get", get)
    monkeypatch.setattr(audio_cpp_util, "print_init", print_init)
    for name in ("generate", "check_readiness", "set_base_url"):
        monkeypatch.setattr(AudioCppUtil, name, forbidden[name])
    monkeypatch.setattr(audio_cpp_util.httpx, "post", forbidden["post"])
    monkeypatch.setattr(audio_cpp_util.httpx, "Client", forbidden["client"])
    # The explicit URL, not the shared configured URL, must drive this request.
    monkeypatch.setattr(AudioCppUtil, "_base_url", "http://do-not-rebind:9090")
    yield SimpleNamespace(get=get, print_init=print_init)
    for mock in forbidden.values():
        mock.assert_not_called()
    assert AudioCppUtil._base_url == "http://do-not-rebind:9090"


def invoke():
    return AudioCppUtil.print_model_init_if_unloaded(BASE_URL, MODEL_ID, DESCRIPTION)


def assert_silent_payload(notice_io, payload):
    response = response_for(payload)
    notice_io.get.return_value = response
    assert invoke() is None
    notice_io.get.assert_called_once_with(BASE_URL + "/v1/models", timeout=2.0)
    response.raise_for_status.assert_called_once_with()
    response.json.assert_called_once_with()
    notice_io.print_init.assert_not_called()


@pytest.mark.parametrize("trailing_slashes", ["", "/", "///"])
@pytest.mark.parametrize("non_first", [False, True])
def test_cold_exact_model_prints_verbatim_path(notice_io, trailing_slashes, non_first):
    path = "  /models/Example Model.GGUF\t "
    models = [cold_model(path=path)]
    if non_first:
        models.insert(0, cold_model(id="other-model", path="/models/wrong.gguf"))
    response = response_for({"data": models})
    notice_io.get.return_value = response

    assert AudioCppUtil.print_model_init_if_unloaded(
        BASE_URL + trailing_slashes, MODEL_ID, DESCRIPTION,
    ) is None

    notice_io.get.assert_called_once_with(BASE_URL + "/v1/models", timeout=2.0)
    response.raise_for_status.assert_called_once_with()
    response.json.assert_called_once_with()
    notice_io.print_init.assert_called_once_with(
        f"audio.cpp will load {DESCRIPTION}\n{COL_DIM_ITALICS}server path: {path}"
    )


@pytest.mark.parametrize(
    "loaded", [True, 0, 1, "false", None, [], {}],
    ids=["loaded", "zero", "one", "string-false", "null", "list", "dict"],
)
def test_loaded_must_be_exactly_false(notice_io, loaded):
    assert_silent_payload(notice_io, {"data": [cold_model(loaded=loaded)]})


def test_missing_loaded_is_silent(notice_io):
    model = cold_model()
    del model["loaded"]
    assert_silent_payload(notice_io, {"data": [model]})


@pytest.mark.parametrize(
    "payload",
    [
        None, [], "invalid", 0, True,
        {}, {"models": [cold_model()]},
        {"data": None}, {"data": {}}, {"data": "invalid"}, {"data": 0},
        {"data": []}, {"data": [None, "invalid", 0, [], True]},
    ],
    ids=[
        "null-root", "list-root", "str-root", "number-root", "bool-root",
        "missing-data", "wrong-data-key", "null-data", "dict-data", "str-data",
        "number-data", "empty-data", "non-dict-models",
    ],
)
def test_malformed_json_shapes_are_silent(notice_io, payload):
    assert_silent_payload(notice_io, payload)


@pytest.mark.parametrize(
    "models",
    [
        [cold_model(id="other-model")],
        [cold_model(id=MODEL_ID + "-extra")],
        [cold_model(id="prefix-" + MODEL_ID)],
        [cold_model(id=MODEL_ID.upper())],
        [cold_model(id=None)],
        [{"loaded": False, "path": MODEL_PATH}],
        [cold_model(), cold_model()],
        [cold_model(), cold_model(loaded=True)],
        [cold_model(loaded=True), cold_model()],
        [cold_model(), {"id": MODEL_ID}],
    ],
    ids=[
        "unmatched", "suffix", "prefix", "case-sensitive", "null-id", "missing-id",
        "duplicate-cold", "duplicate-cold-loaded", "duplicate-loaded-cold",
        "duplicate-missing-state",
    ],
)
def test_unmatched_or_duplicate_ids_are_silent(notice_io, models):
    assert_silent_payload(notice_io, {"data": models})


@pytest.mark.parametrize(
    "path", [None, False, 0, 1, [], {}, "", " ", "\t\n"],
    ids=["null", "bool", "zero", "number", "list", "dict", "empty", "space", "whitespace"],
)
def test_invalid_or_blank_path_is_silent(notice_io, path):
    assert_silent_payload(notice_io, {"data": [cold_model(path=path)]})


def test_missing_path_is_silent(notice_io):
    model = cold_model()
    del model["path"]
    assert_silent_payload(notice_io, {"data": [model]})


@pytest.mark.parametrize(
    "error",
    [
        httpx.TimeoutException("timed out"),
        httpx.ConnectError("connection failed"),
        httpx.InvalidURL("invalid URL"),
    ],
    ids=["timeout", "http-error", "invalid-url"],
)
def test_request_errors_are_silent(notice_io, error, capsys):
    notice_io.get.side_effect = error
    assert invoke() is None
    notice_io.get.assert_called_once_with(BASE_URL + "/v1/models", timeout=2.0)
    notice_io.print_init.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_http_status_error_is_silent_and_json_is_not_read(notice_io, capsys):
    response = response_for({"data": [cold_model()]})
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server unavailable", request=httpx.Request("GET", BASE_URL + "/v1/models"),
        response=httpx.Response(503),
    )
    notice_io.get.return_value = response

    assert invoke() is None

    notice_io.get.assert_called_once_with(BASE_URL + "/v1/models", timeout=2.0)
    response.raise_for_status.assert_called_once_with()
    response.json.assert_not_called()
    notice_io.print_init.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_invalid_json_value_error_is_silent(notice_io, capsys):
    response = response_for(None)
    response.json.side_effect = ValueError("invalid JSON")
    notice_io.get.return_value = response

    assert invoke() is None

    notice_io.get.assert_called_once_with(BASE_URL + "/v1/models", timeout=2.0)
    response.raise_for_status.assert_called_once_with()
    response.json.assert_called_once_with()
    notice_io.print_init.assert_not_called()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("base_url,model_id", [("", MODEL_ID), (BASE_URL, ""), ("", "")])
def test_empty_config_skips_network(notice_io, base_url, model_id):
    assert AudioCppUtil.print_model_init_if_unloaded(base_url, model_id, DESCRIPTION) is None
    notice_io.get.assert_not_called()
    notice_io.print_init.assert_not_called()


def test_each_invocation_requests_current_state_to_notice_reloads(notice_io):
    responses = [
        response_for({"data": [cold_model()]}),
        response_for({"data": [cold_model(loaded=True)]}),
        response_for({"data": [cold_model(path="/models/reloaded.gguf")]}),
    ]
    notice_io.get.side_effect = responses

    for _ in responses:
        assert invoke() is None

    assert notice_io.get.call_args_list == [
        call(BASE_URL + "/v1/models", timeout=2.0),
    ] * 3
    for response in responses:
        response.raise_for_status.assert_called_once_with()
        response.json.assert_called_once_with()
    assert notice_io.print_init.call_args_list == [
        call(f"audio.cpp will load {DESCRIPTION}\n{COL_DIM_ITALICS}server path: {MODEL_PATH}"),
        call(f"audio.cpp will load {DESCRIPTION}\n{COL_DIM_ITALICS}server path: /models/reloaded.gguf"),
    ]
