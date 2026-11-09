from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.prefs import Prefs

HINT_A = Hint("alpha", "Alpha heading", "alpha body")
HINT_B = Hint("beta", "Beta heading", "beta body")


@pytest.fixture
def prefs() -> Prefs:
    return Prefs(hints={})


@pytest.fixture
def prompts(monkeypatch) -> list[bool]:
    calls: list[bool] = []
    monkeypatch.setattr(ask, "ask_enter_to_continue", lambda *_args, **_kwargs: calls.append(True))
    return calls


@pytest.fixture
def saves(prefs: Prefs, monkeypatch) -> list[bool]:
    calls: list[bool] = []
    real_save = prefs.save

    def counting_save() -> str:
        calls.append(True)
        return real_save()

    monkeypatch.setattr(prefs, "save", counting_save)
    return calls


def test_pending_hints_print_together_with_a_single_prompt(
    prefs: Prefs, prompts: list[bool], saves: list[bool], capsys, prefs_path: Path
) -> None:
    hints.show_hints_if_necessary(prefs, [HINT_A, HINT_B], and_prompt=True)

    printed = capsys.readouterr().out
    assert "Alpha heading" in printed
    assert "alpha body" in printed
    assert "Beta heading" in printed
    # One prompt for the whole batch, not one per hint.
    assert prompts == [True]
    assert prefs.get_hint("alpha") and prefs.get_hint("beta")
    assert saves == [True]

    saved_payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert saved_payload["hints"] == {"alpha": True, "beta": True}


def test_already_shown_hints_are_not_printed_again(
    prefs: Prefs, prompts: list[bool], saves: list[bool], capsys
) -> None:
    prefs.set_hint_true("alpha")

    hints.show_hints_if_necessary(prefs, [HINT_A, HINT_B], and_prompt=True)

    printed = capsys.readouterr().out
    assert "Alpha heading" not in printed
    assert "Beta heading" in printed
    assert prompts == [True]
    assert saves == [True]


def test_nothing_prints_prompts_or_saves_when_all_hints_were_shown(
    prefs: Prefs, prompts: list[bool], saves: list[bool], capsys
) -> None:
    prefs.set_hint_true("alpha")
    prefs.set_hint_true("beta")

    hints.show_hints_if_necessary(prefs, [HINT_A, HINT_B], and_prompt=True)

    assert capsys.readouterr().out == ""
    assert prompts == []
    assert saves == []


def test_hints_are_still_marked_when_no_prompt_is_requested(
    prefs: Prefs, prompts: list[bool], saves: list[bool], capsys
) -> None:
    hints.show_hints_if_necessary(prefs, [HINT_A, HINT_B])

    printed = capsys.readouterr().out
    assert "Alpha heading" in printed
    assert "Beta heading" in printed
    assert prompts == []
    assert prefs.get_hint("alpha") and prefs.get_hint("beta")
    assert saves == [True]


class _MossProject:
    def get_tts_model_type(self):
        return type("_Type", (), {"id": "moss_local"})()

    def get_model_setting(self, _model_id: str, _key: str):
        return "some/repo"


@pytest.fixture
def confirms(monkeypatch) -> list[str]:
    from tts_audiobook_tool.app_support import app_hint_util
    monkeypatch.setattr(app_hint_util.platform, "system", lambda: "Windows")
    calls: list[str] = []

    def fake_confirm(message: str = "") -> bool:
        calls.append(message)
        return True

    monkeypatch.setattr(ask, "ask_confirm", fake_confirm)
    return calls


@pytest.mark.parametrize("start_prompt", ["", "Start? "])
def test_confirm_hint_and_start_prompt_share_one_confirmation(
    prefs: Prefs, confirms: list[str], prefs_path: Path, capsys, start_prompt: str
) -> None:
    # The MOSS remote-code warning's confirmation doubles as the caller's start
    # prompt, so the user is asked only once (with the start prompt, if given).
    from tts_audiobook_tool.app_support import app_hint_util

    result = app_hint_util.show_pre_inference_hints(prefs, _MossProject(), start_prompt=start_prompt)  # type: ignore[arg-type]

    assert result is True
    assert confirms == [start_prompt]
    assert "some/repo" in capsys.readouterr().out
    assert prefs.get_hint("moss_remote_code")


def test_start_prompt_still_asked_when_hint_already_shown(
    prefs: Prefs, confirms: list[str], prefs_path: Path, capsys
) -> None:
    from tts_audiobook_tool.app_support import app_hint_util
    prefs.set_hint_true("moss_remote_code")

    app_hint_util.show_pre_inference_hints(prefs, _MossProject(), start_prompt="Start? ")  # type: ignore[arg-type]

    assert confirms == ["Start? "]
    assert "some/repo" not in capsys.readouterr().out


def test_declined_confirmation_does_not_mark_hint(
    prefs: Prefs, confirms: list[str], prefs_path: Path, monkeypatch
) -> None:
    from tts_audiobook_tool.app_support import app_hint_util
    monkeypatch.setattr(ask, "ask_confirm", lambda _message="": False)

    result = app_hint_util.show_pre_inference_hints(prefs, _MossProject(), start_prompt="Start? ")  # type: ignore[arg-type]

    assert result is False
    assert not prefs.get_hint("moss_remote_code")
