from types import SimpleNamespace
import httpx
import pytest
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts

from tts_audiobook_tool.app_types import ReadinessIssue, SttVariant
from tts_audiobook_tool.conversation.conversation_types import ChatInputMode
from tts_audiobook_tool.textual import conversation_app
from tts_audiobook_tool.textual.conversation_app import (
    ConversationAppResult,
    ConversationAppStatus,
    run_conversation_app,
)
from tts_audiobook_tool import readiness


@pytest.fixture(autouse=True)
def clean_binding(monkeypatch):
    monkeypatch.setattr(Tts, "_binding_issue", None)


def make_state(mode: ChatInputMode):
    return SimpleNamespace(
        prefs=SimpleNamespace(
            chat_input_mode=mode,
            llm_url="http://llm",
            stt_variant=SttVariant.DISABLED,
        ),
        project=Project(tts_model_type="vibevoice_local"),
    )


def test_text_readiness_does_not_probe_microphone_or_require_stt(monkeypatch) -> None:
    state = make_state(ChatInputMode.TEXT)
    monkeypatch.setattr(
        readiness.SoundInputDeviceInfo,
        "get_check_error",
        lambda: (_ for _ in ()).throw(AssertionError("microphone probed")),
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts.Tts.get_model_support",
        lambda project: SimpleNamespace(get_blocking_issues=lambda project, voice: []),
    )

    assert readiness.get_chat_blockers(state) == []


def test_echo_readiness_does_not_require_llm_endpoint(monkeypatch) -> None:
    state = make_state(ChatInputMode.TEXT)
    state.prefs.chat_echo_override = True
    state.prefs.llm_url = ""
    monkeypatch.setattr(
        "tts_audiobook_tool.tts.Tts.get_model_support",
        lambda project: SimpleNamespace(get_blocking_issues=lambda project, voice: []),
    )

    assert readiness.get_chat_blockers(state) == []


def test_microphone_readiness_retains_input_requirements(monkeypatch) -> None:
    state = make_state(ChatInputMode.MIC_ENTER)
    monkeypatch.setattr(
        readiness.SoundInputDeviceInfo, "get_check_error", lambda: "no mic"
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts.Tts.get_model_support",
        lambda project: SimpleNamespace(get_blocking_issues=lambda project, voice: []),
    )

    issues = readiness.get_chat_blockers(state)
    assert [issue.short for issue in issues] == [
        "microphone",
        "Whisper model enabled",
    ]


def test_runner_checks_textual_before_readiness(monkeypatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(conversation_app, "can_textual", lambda: False)
    monkeypatch.setattr(
        conversation_app.readiness,
        "refresh_remote_model_state",
        lambda project: pytest.fail("Unavailable Textual must not refresh"),
    )
    monkeypatch.setattr(
        conversation_app.readiness,
        "get_chat_blockers",
        lambda state: (_ for _ in ()).throw(AssertionError("readiness called")),
    )
    monkeypatch.setattr(conversation_app.ask, "ask_error", errors.append)

    result = run_conversation_app(make_state(ChatInputMode.TEXT))

    assert result.status == ConversationAppStatus.UNAVAILABLE
    assert errors == [result.message]


def test_runner_shows_blockers_in_app_without_console_hints(monkeypatch) -> None:
    state = make_state(ChatInputMode.TEXT)
    expected = ConversationAppResult(ConversationAppStatus.COMPLETED)
    observed: list[tuple[str, ...]] = []

    class FakeApp:
        _exception = None

        def __init__(self, state, *, blocker_messages):
            observed.append(blocker_messages)
            self.runtime = SimpleNamespace(close=lambda: None)

        def run(self, *, inline):
            assert inline is False
            return expected

    monkeypatch.setattr(conversation_app, "can_textual", lambda: True)
    monkeypatch.setattr(conversation_app.readiness, "refresh_remote_model_state", lambda project: None)
    monkeypatch.setattr(
        conversation_app.readiness,
        "get_chat_blockers",
        lambda state: [ReadinessIssue("model", "model unavailable")],
    )
    monkeypatch.setattr(
        conversation_app.app_hint_util,
        "show_pre_inference_hints",
        lambda prefs, project: (_ for _ in ()).throw(AssertionError("hints called")),
    )
    monkeypatch.setattr(conversation_app, "ConversationTextualApp", FakeApp)

    assert run_conversation_app(state) == expected
    assert observed == [("model unavailable",)]


def _install_runner_prerequisites(monkeypatch) -> None:
    monkeypatch.setattr(conversation_app, "can_textual", lambda: True)
    monkeypatch.setattr(conversation_app.readiness, "refresh_remote_model_state", lambda project: None)
    monkeypatch.setattr(
        conversation_app.readiness, "get_chat_blockers", lambda state: []
    )
    monkeypatch.setattr(
        conversation_app.app_hint_util,
        "show_pre_inference_hints",
        lambda prefs, project: True,
    )


def test_runner_classifies_thrown_exception_and_closes_runtime(monkeypatch) -> None:
    _install_runner_prerequisites(monkeypatch)
    errors: list[str] = []
    closed: list[bool] = []

    class FakeApp:
        _exception = None

        def __init__(self, state, *, blocker_messages):
            self.runtime = SimpleNamespace(close=lambda: closed.append(True))

        def run(self, *, inline):
            raise RuntimeError("boom")

    monkeypatch.setattr(conversation_app, "ConversationTextualApp", FakeApp)
    monkeypatch.setattr(conversation_app.ask, "ask_error", errors.append)

    result = run_conversation_app(make_state(ChatInputMode.TEXT))

    assert result.status is ConversationAppStatus.FAILED
    assert result.message == "RuntimeError: boom"
    assert errors == [result.message]
    assert closed == [True]


def test_runner_classifies_stylesheet_failure(monkeypatch) -> None:
    _install_runner_prerequisites(monkeypatch)
    errors: list[str] = []

    class FakeApp:
        _exception = conversation_app.StylesheetError("bad css")

        def __init__(self, state, *, blocker_messages):
            self.runtime = SimpleNamespace(close=lambda: None)

        def run(self, *, inline):
            return None

    monkeypatch.setattr(conversation_app, "ConversationTextualApp", FakeApp)
    monkeypatch.setattr(conversation_app.ask, "ask_error", errors.append)

    result = run_conversation_app(make_state(ChatInputMode.TEXT))

    assert result.status is ConversationAppStatus.FAILED
    assert result.message == "Couldn't load textual css"
    assert errors == [result.message]


def test_runner_classifies_missing_result(monkeypatch) -> None:
    _install_runner_prerequisites(monkeypatch)

    class FakeApp:
        _exception = None

        def __init__(self, state, *, blocker_messages):
            self.runtime = SimpleNamespace(close=lambda: None)

        def run(self, *, inline):
            return None

    monkeypatch.setattr(conversation_app, "ConversationTextualApp", FakeApp)
    monkeypatch.setattr(conversation_app.ask, "ask_error", lambda message: None)

    result = run_conversation_app(make_state(ChatInputMode.TEXT))

    assert result.status is ConversationAppStatus.FAILED
    assert "without returning a result" in result.message


def test_runner_forces_refresh_before_blockers_hints_and_app(monkeypatch) -> None:
    state = make_state(ChatInputMode.TEXT)
    events = []
    expected = ConversationAppResult(ConversationAppStatus.COMPLETED)

    def bind(project, *, refresh=False):
        assert project is state.project
        assert refresh is True
        events.append("refresh")

    class FakeApp:
        _exception = None

        def __init__(self, current, *, blocker_messages):
            assert current is state
            assert blocker_messages == ()
            events.append("app")
            self.runtime = SimpleNamespace(close=lambda: None)

        def run(self, *, inline):
            events.append("run")
            return expected

    monkeypatch.setattr(conversation_app, "can_textual", lambda: events.append("textual") or True)
    monkeypatch.setattr(Tts, "bind_project", bind)
    monkeypatch.setattr(readiness, "get_chat_blockers", lambda current: events.append("blockers") or [])
    monkeypatch.setattr(conversation_app.app_hint_util, "show_pre_inference_hints",
                        lambda prefs, project: events.append("hints") or True)
    monkeypatch.setattr(conversation_app, "ConversationTextualApp", FakeApp)

    assert run_conversation_app(state) == expected
    assert events == ["textual", "refresh", "blockers", "hints", "app", "run"]


def test_chat_start_recovers_cached_offline_remote_binding(monkeypatch) -> None:
    state = make_state(ChatInputMode.TEXT)
    state.project.tts_model_type = "chatterbox_audiocpp"
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "http://remote.test")
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    # Freeze time so recovery necessarily bypasses the failure backoff.
    monkeypatch.setattr(RemoteTtsDiscovery, "_now", staticmethod(lambda: 1000.0))
    online = [False]
    requests = []

    def respond(request):
        requests.append(request.url.path)
        if not online[0]:
            raise httpx.ConnectError("offline", request=request)
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "status": "ok", "backend": "cpu", "models": 1,
                "ui": False, "ui_management": False,
            })
        return httpx.Response(200, json={"data": [{
            "id": "recovered-model", "family": "chatterbox", "task": "clon", "mode": "offline",
            "session_options": {"chatterbox.multilingual_t3": "v3"},
        }]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: True)
    monkeypatch.setattr(Tts, "clear_tts_model", lambda: None)
    monkeypatch.setattr(Tts, "set_model_params_using_project", lambda project: None)
    monkeypatch.setattr(Tts, "get_model_support",
                        lambda project: SimpleNamespace(get_blocking_issues=lambda project, voice: []))

    assert Tts.bind_project(state.project) is not None
    cached_failure = RemoteTtsDiscovery.get_snapshot()
    assert cached_failure.issue.code == "unavailable"
    online[0] = True
    assert "offline" in readiness.get_chat_blocker_text(state, verbose=True)
    assert requests == ["/health"], "Menu blocker checks must not force discovery"
    assert RemoteTtsDiscovery.refresh() is cached_failure

    expected = ConversationAppResult(ConversationAppStatus.COMPLETED)
    blockers = []

    class FakeApp:
        _exception = None

        def __init__(self, current, *, blocker_messages):
            blockers.append(blocker_messages)
            self.runtime = SimpleNamespace(close=lambda: None)

        def run(self, *, inline):
            return expected

    monkeypatch.setattr(conversation_app, "can_textual", lambda: True)
    monkeypatch.setattr(conversation_app.app_hint_util, "show_pre_inference_hints", lambda prefs, project: True)
    monkeypatch.setattr(conversation_app, "ConversationTextualApp", FakeApp)

    assert run_conversation_app(state) == expected
    assert requests == ["/health", "/health", "/v1/models"]
    assert blockers == [()]
    assert Tts._binding_issue is None
    assert Tts._selected_server_model_id == "recovered-model"
    assert Tts.get_active_type().id == state.project.tts_model_type
