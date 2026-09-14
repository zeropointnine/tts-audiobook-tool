import builtins
from types import SimpleNamespace

import pytest
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import DummyHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tts_audiobook_tool import ask, ask_advanced
from tts_audiobook_tool.ask_advanced import AskAdvanced
from tts_audiobook_tool.project import Project


class _TTY:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def _run_prompt(data: bytes, prefill: str = "prefilled") -> str:
    with create_pipe_input() as pipe_input:
        pipe_input.send_bytes(data)
        return ask_advanced._ask_with_prompt_toolkit(
            "",
            prefill,
            prompt_input=pipe_input,
            prompt_output=DummyOutput(),
        )


def test_ask_advanced_accepts_selected_prefill_unchanged():
    assert _run_prompt(b"\r") == "prefilled"


def test_ask_advanced_content_replaces_selected_prefill():
    assert _run_prompt(b"X\r") == "X"


@pytest.mark.parametrize("key", [b"\x7f", b"\x1b[3~"])
def test_ask_advanced_delete_key_removes_selected_prefill(key):
    assert _run_prompt(key + b"\r") == ""


@pytest.mark.parametrize(
    ("key", "expected"),
    [(b"\x1b[H", "Xprefilled"), (b"\x1b[F", "prefilledX")],
)
def test_ask_advanced_home_and_end_collapse_selection(key, expected):
    assert _run_prompt(key + b"X\r") == expected


def test_ask_advanced_bracketed_paste_replaces_selected_prefill():
    assert _run_prompt(b"\x1b[200~PASTE\x1b[201~\r") == "PASTE"


def test_ask_advanced_bracketed_paste_inserts_after_collapsed_selection():
    assert (
        _run_prompt(b"\x1b[F\x1b[200~PASTE\x1b[201~\r")
        == "prefilledPASTE"
    )


@pytest.mark.parametrize(
    "key",
    [
        b"\x1b[A",
        b"\x1b[B",
        b"\x1b[5~",
        b"\x1b[6~",
        b"\x10",
        b"\x0e",
        b"\x12",
        b"\x13",
    ],
)
def test_ask_advanced_history_keys_are_inert(key):
    assert _run_prompt(key + b"X\r") == "X"


def test_ask_advanced_escape_cancels():
    assert _run_prompt(b"\x1b") == ""


def test_ask_advanced_ctrl_c_cancels():
    assert _run_prompt(b"\x03") == ""


def test_ask_advanced_ctrl_d_preserves_eof_behavior_for_empty_input():
    with pytest.raises(EOFError):
        _run_prompt(b"\x04", prefill="")


def test_ask_advanced_accepts_content_with_empty_prefill():
    assert _run_prompt(b"value\r", prefill="") == "value"


def test_ask_advanced_configures_plain_ansi_prompt_without_extra_features(
    monkeypatch,
):
    captured = {}

    class FakePromptSession:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, message, **kwargs):
            captured["message"] = message
            captured["kwargs"] = kwargs
            self.app = SimpleNamespace(ttimeoutlen=None)

        def prompt(self, **kwargs):
            captured["prompt_kwargs"] = kwargs
            return "result"

    monkeypatch.setattr(ask_advanced, "PromptSession", FakePromptSession)

    assert (
        ask_advanced._ask_with_prompt_toolkit("\x1b[31mPrompt: ", "prefilled")
        == "result"
    )

    message = captured["message"]
    kwargs = captured["kwargs"]
    assert isinstance(message, ANSI)
    assert message.value == "\x1b[31mPrompt: "
    assert isinstance(kwargs["history"], DummyHistory)
    assert kwargs["complete_while_typing"] is False
    assert kwargs["validate_while_typing"] is False
    assert kwargs["enable_history_search"] is False
    assert kwargs["enable_system_prompt"] is False
    assert kwargs["enable_suspend"] is False
    assert kwargs["enable_open_in_editor"] is False
    assert kwargs["mouse_support"] is False
    assert kwargs["multiline"] is False
    assert kwargs["reserve_space_for_menu"] == 0
    assert kwargs["style"].get_attrs_for_style_str("class:selected").reverse
    assert captured["prompt_kwargs"]["default"] == "prefilled"
    assert captured["prompt_kwargs"]["set_exception_handler"] is False


def test_ask_advanced_falls_back_to_input_without_tty(monkeypatch):
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return "plain input"

    monkeypatch.setattr(ask_advanced.sys, "stdin", _TTY(False))
    monkeypatch.setattr(ask_advanced.sys, "stdout", _TTY(False))
    monkeypatch.setattr(builtins, "input", fake_input)

    assert AskAdvanced.ask("prompt: ") == "plain input"
    assert prompts == ["prompt: "]


def test_ask_advanced_falls_back_for_unsafe_control_characters(monkeypatch):
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return "plain input"

    monkeypatch.setattr(builtins, "input", fake_input)

    assert AskAdvanced.ask("multiple\nlines") == "plain input"
    assert AskAdvanced.ask(prefill="escape\x1bsequence") == "plain input"
    assert prompts == ["multiple\nlines", ""]


def test_ask_advanced_rejects_non_string_arguments():
    with pytest.raises(TypeError):
        AskAdvanced.ask(message=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AskAdvanced.ask(prefill=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("is_minus_one_default", "expected_prefill"),
    [(True, "0.5"), (False, "-1.0")],
)
def test_ask_number_and_save_prefills_effective_default_for_minus_one_sentinel(
    monkeypatch, is_minus_one_default, expected_prefill
):
    project = Project()
    project.chatterbox_exaggeration = -1.0
    prefills = []

    def fake_ask_input(*, prefill):
        prefills.append(prefill)
        return ""

    monkeypatch.setattr(ask, "ask_input", fake_ask_input)
    monkeypatch.setattr(ask, "printt", lambda *args, **kwargs: None)

    ask.ask_number_and_save(
        project,
        attr="chatterbox_exaggeration",
        prompt="Enter exaggeration:",
        min_value=0.25,
        max_value=2.0,
        default_value=0.5,
        success_prefix="Value set:",
        is_minus_one_default=is_minus_one_default,
    )

    assert prefills == [expected_prefill]


def test_ask_number_and_save_unchanged_effective_default_preserves_sentinel(
    monkeypatch,
):
    project = Project()
    project.chatterbox_exaggeration = -1.0
    saves = []

    monkeypatch.setattr(ask, "ask_input", lambda *, prefill: prefill)
    monkeypatch.setattr(ask, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))

    ask.ask_number_and_save(
        project,
        attr="chatterbox_exaggeration",
        prompt="Enter exaggeration:",
        min_value=0.25,
        max_value=2.0,
        default_value=0.5,
        success_prefix="Value set:",
        is_minus_one_default=True,
    )

    assert project.chatterbox_exaggeration == -1.0
    assert saves == []


@pytest.mark.parametrize("submitted", ["0.5", "0.50"])
def test_ask_number_and_save_does_not_save_unchanged_value(monkeypatch, submitted):
    project = Project()
    project.chatterbox_exaggeration = 0.5
    saves = []

    monkeypatch.setattr(
        ask, "ask_input", lambda *, prefill: submitted
    )
    monkeypatch.setattr(ask, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))

    ask.ask_number_and_save(
        project,
        attr="chatterbox_exaggeration",
        prompt="Enter exaggeration:",
        min_value=0.25,
        max_value=2.0,
        default_value=0.5,
        success_prefix="Value set:",
        is_minus_one_default=True,
    )

    assert project.chatterbox_exaggeration == 0.5
    assert saves == []


def test_ask_number_and_save_accepts_minus_one_default_sentinel(monkeypatch):
    project = Project()
    project.chatterbox_exaggeration = 0.5
    saves = []

    monkeypatch.setattr(ask, "ask_input", lambda *, prefill: "-1")
    monkeypatch.setattr(ask, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))

    ask.ask_number_and_save(
        project,
        attr="chatterbox_exaggeration",
        prompt="Enter exaggeration:",
        min_value=0.25,
        max_value=2.0,
        default_value=0.5,
        success_prefix="Value set:",
        is_minus_one_default=True,
    )

    assert project.chatterbox_exaggeration == -1
    assert saves == [project]


@pytest.mark.parametrize(
    ("submitted", "normalizer"),
    [("hello", None), ("HELLO", str.lower)],
)
def test_ask_string_and_save_does_not_save_unchanged_value(
    monkeypatch, submitted, normalizer
):
    project = Project()
    project.qwen3_instructions = "hello"
    saves = []

    monkeypatch.setattr(
        ask,
        "ask_input",
        lambda *, prefill, lower: submitted,
    )
    monkeypatch.setattr(ask, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))

    saved = ask.ask_string_and_save(
        project,
        prompt_line="Enter instructions:",
        attr="qwen3_instructions",
        success_prefix="Instructions set:",
        normalizer=normalizer,
    )

    assert saved is False
    assert project.qwen3_instructions == "hello"
    assert saves == []


def test_ask_advanced_cancel_clears_only_current_input(monkeypatch):
    writes = []
    stdout = SimpleNamespace(
        isatty=lambda: True,
        write=writes.append,
        flush=lambda: writes.append("<flush>"),
    )

    def interrupt_input(prompt, prefill):
        raise KeyboardInterrupt

    monkeypatch.setattr(ask_advanced.sys, "stdin", _TTY(True))
    monkeypatch.setattr(ask_advanced.sys, "stdout", stdout)
    monkeypatch.setattr(
        ask_advanced, "_ask_with_prompt_toolkit", interrupt_input
    )

    assert AskAdvanced.ask("Prompt: ", "prefilled") == ""
    assert writes == ["\n", "<flush>"]
