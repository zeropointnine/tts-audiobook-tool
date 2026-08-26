import builtins
import sys
from types import SimpleNamespace

import pytest

from tts_audiobook_tool import ask, ask_advanced
from tts_audiobook_tool.ask_advanced import AskAdvanced
from tts_audiobook_tool.project import Project


class _TTY:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_ask_advanced_prefills_readline_and_places_cursor_at_end(
    monkeypatch, platform
):
    inserted = []
    hooks = []
    prompts = []
    readline = SimpleNamespace()

    def set_startup_hook(hook):
        readline.hook = hook
        hooks.append(hook)

    def fake_input(prompt=""):
        prompts.append(prompt)
        readline.hook()
        return "edited value"

    readline.set_startup_hook = set_startup_hook
    readline.insert_text = inserted.append
    readline.hook = None

    monkeypatch.setattr(ask_advanced.sys, "platform", platform)
    monkeypatch.setattr(ask_advanced.sys, "stdin", _TTY(True))
    monkeypatch.setattr(ask_advanced.sys, "stdout", _TTY(True))
    monkeypatch.setitem(sys.modules, "readline", readline)
    monkeypatch.setattr(builtins, "input", fake_input)

    assert (
        AskAdvanced.ask("Prompt: ", prefill="prefilled")
        == "edited value"
    )
    assert inserted == ["prefilled"]
    assert prompts == ["Prompt: "]
    assert callable(hooks[0])
    assert hooks[-1] is None


def test_readline_hook_is_cleared_when_input_raises(monkeypatch):
    hooks = []
    readline = SimpleNamespace(insert_text=lambda value: None)

    def set_startup_hook(hook):
        hooks.append(hook)

    def fake_input(prompt=""):
        raise EOFError

    readline.set_startup_hook = set_startup_hook
    monkeypatch.setattr(ask_advanced.sys, "stdin", _TTY(True))
    monkeypatch.setattr(ask_advanced.sys, "stdout", _TTY(True))
    monkeypatch.setitem(sys.modules, "readline", readline)
    monkeypatch.setattr(builtins, "input", fake_input)

    with pytest.raises(EOFError):
        ask_advanced._ask_with_readline("", "prefilled")

    assert callable(hooks[0])
    assert hooks[-1] is None


def test_ask_advanced_falls_back_to_input_without_tty(monkeypatch):
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return "plain input"

    monkeypatch.setattr(ask_advanced.sys, "platform", "linux")
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


def test_windows_initial_line_displays_prefill_and_flushes(monkeypatch):
    writes = []
    stdout = SimpleNamespace(
        write=writes.append,
        flush=lambda: writes.append("<flush>"),
    )
    monkeypatch.setattr(ask_advanced.sys, "stdout", stdout)

    ask_advanced._write_windows_initial_line("Prompt: ", "prefilled")

    assert writes == ["Prompt: prefilled", "<flush>"]


def test_ask_advanced_uses_windows_console_reader(monkeypatch):
    calls = []

    def fake_windows_reader(prompt, prefill):
        calls.append((prompt, prefill))
        return "edited value"

    monkeypatch.setattr(ask_advanced.sys, "platform", "win32")
    monkeypatch.setattr(
        ask_advanced, "_ask_with_windows_console", fake_windows_reader
    )

    assert AskAdvanced.ask("Prompt: ", "prefilled") == "edited value"
    assert calls == [("Prompt: ", "prefilled")]


def test_posix_input_temporarily_overrides_and_restores_sigint_handler(
    monkeypatch,
):
    def swallowed_sigint(signum, frame):
        return None

    installed_handler = swallowed_sigint
    installed = []

    def fake_getsignal(signum):
        assert signum == ask_advanced.signal.SIGINT
        return installed_handler

    def fake_signal(signum, handler):
        nonlocal installed_handler
        assert signum == ask_advanced.signal.SIGINT
        installed_handler = handler
        installed.append(handler)

    def interrupt_from_readline(prompt, prefill):
        installed_handler(ask_advanced.signal.SIGINT, None)
        raise AssertionError("SIGINT handler did not interrupt input")

    monkeypatch.setattr(ask_advanced.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(ask_advanced.signal, "signal", fake_signal)
    monkeypatch.setattr(ask_advanced, "_ask_with_readline", interrupt_from_readline)

    with pytest.raises(KeyboardInterrupt):
        ask_advanced._ask_with_posix_interruptible_input("Prompt: ", "prefilled")

    assert installed == [
        ask_advanced.signal.default_int_handler,
        swallowed_sigint,
    ]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_ask_advanced_posix_ctrl_c_cancels_only_current_input(
    monkeypatch, platform
):
    writes = []
    stdout = SimpleNamespace(
        write=writes.append,
        flush=lambda: writes.append("<flush>"),
    )

    def interrupt_input(prompt, prefill):
        raise KeyboardInterrupt

    monkeypatch.setattr(ask_advanced.sys, "platform", platform)
    monkeypatch.setattr(ask_advanced.sys, "stdout", stdout)
    monkeypatch.setattr(ask_advanced, "_ask_with_readline", interrupt_input)

    assert AskAdvanced.ask("Prompt: ", "prefilled") == ""
    assert writes == ["\n", "<flush>"]


def test_ask_advanced_windows_ctrl_c_cancels_only_current_input(monkeypatch):
    writes = []
    stdout = SimpleNamespace(
        write=writes.append,
        flush=lambda: writes.append("<flush>"),
    )

    def interrupt_input(prompt, prefill):
        raise KeyboardInterrupt

    monkeypatch.setattr(ask_advanced.sys, "platform", "win32")
    monkeypatch.setattr(ask_advanced.sys, "stdout", stdout)
    monkeypatch.setattr(
        ask_advanced, "_ask_with_windows_console", interrupt_input
    )

    assert AskAdvanced.ask("Prompt: ", "prefilled") == ""
    assert writes == ["\n", "<flush>"]
