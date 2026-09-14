from types import SimpleNamespace

from tts_audiobook_tool.app_types import ReadinessIssue, SttVariant
from tts_audiobook_tool.conversation.conversation_types import ChatInputMode
from tts_audiobook_tool.textual import conversation_app
from tts_audiobook_tool.textual.conversation_app import (
    ConversationAppResult,
    ConversationAppStatus,
    run_conversation_app,
)
from tts_audiobook_tool import readiness


def make_state(mode: ChatInputMode):
    return SimpleNamespace(
        prefs=SimpleNamespace(
            chat_input_mode=mode,
            llm_url="http://llm",
            stt_variant=SttVariant.DISABLED,
        ),
        project=SimpleNamespace(),
    )


def test_text_readiness_does_not_probe_microphone_or_require_stt(monkeypatch) -> None:
    state = make_state(ChatInputMode.TEXT)
    monkeypatch.setattr(
        readiness.SoundInputDeviceInfo,
        "get_check_error",
        lambda: (_ for _ in ()).throw(AssertionError("microphone probed")),
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts.Tts.get_class",
        lambda: SimpleNamespace(get_blocking_issues=lambda project, voice: []),
    )

    assert readiness.get_chat_blockers(state) == []


def test_echo_readiness_does_not_require_llm_endpoint(monkeypatch) -> None:
    state = make_state(ChatInputMode.TEXT)
    state.prefs.chat_echo_override = True
    state.prefs.llm_url = ""
    monkeypatch.setattr(
        "tts_audiobook_tool.tts.Tts.get_class",
        lambda: SimpleNamespace(get_blocking_issues=lambda project, voice: []),
    )

    assert readiness.get_chat_blockers(state) == []


def test_microphone_readiness_retains_input_requirements(monkeypatch) -> None:
    state = make_state(ChatInputMode.MIC_ENTER)
    monkeypatch.setattr(
        readiness.SoundInputDeviceInfo, "get_check_error", lambda: "no mic"
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts.Tts.get_class",
        lambda: SimpleNamespace(get_blocking_issues=lambda project, voice: []),
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
