from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs

HINT_A = Hint("alpha", "Alpha heading", "alpha body")
HINT_B = Hint("beta", "Beta heading", "beta body")


@pytest.fixture
def prefs(tmp_path: Path, monkeypatch) -> Prefs:
    monkeypatch.setattr(
        Prefs, "get_file_path", staticmethod(lambda: str(tmp_path / PREFS_FILE_NAME))
    )
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
    prefs: Prefs, prompts: list[bool], saves: list[bool], capsys, tmp_path: Path
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

    saved_payload = json.loads((tmp_path / PREFS_FILE_NAME).read_text(encoding="utf-8"))
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
