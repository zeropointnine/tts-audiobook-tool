from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.voice import voice_menu_shared as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State


def make_state(count=3):
    return cast(State, SimpleNamespace(project=Project(
        tts_model_type="omnivoice_local",
        voice_references=[{"file_name": f"{i}.flac", "transcript": f"text {i}"} for i in range(count)],
    )))


def capture(monkeypatch, inputs):
    errors, prompts, feedback = [], [], []
    answers = iter(inputs)

    def next_answer(**_kwargs):
        return next(answers)

    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: next_answer())
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(menu, "printt", lambda text="": prompts.append(text))
    monkeypatch.setattr(
        menu, "print_feedback",
        lambda message, **kwargs: feedback.append((message, kwargs) if kwargs else message),
    )
    return errors, prompts, feedback


@pytest.mark.parametrize("inputs, order", [(["1", "3"], [1, 2, 0]), (["3", "1"], [2, 0, 1]), (["2", "1"], [1, 0, 2])])
def test_move_reorders_pairs_and_saves(monkeypatch, inputs, order):
    state = make_state()
    state.project.book = Book([BookSection([
        PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=i) for i in [-1, 0, 1, 2]
    ])])
    original = list(state.project.voice_references)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, _ = capture(monkeypatch, inputs)
    menu.VoiceMenuShared.move_voice_sample_from_menu(state)
    assert state.project.voice_references == [original[i] for i in order]
    assert [group.voice_index for group in state.project.book.phrase_groups] == [-1, 0, 1, 2]
    assert prompts == ["Enter voice sample number to move:", "Enter new position:"]
    assert not errors
    saved.assert_called_once()


@pytest.mark.parametrize("inputs, error", [
    ([""], None), (["x"], "Bad value"), (["0"], "Out of range"),
    (["4"], "Out of range"), (["1", ""], None), (["1", "x"], "Bad value"),
    (["1", "0"], "Out of range"), (["1", "4"], "Out of range"), (["2", "2"], None),
])
def test_move_invalid_cancel_and_no_change(monkeypatch, inputs, error):
    state = make_state()
    original = list(state.project.voice_references)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, _, feedback = capture(monkeypatch, inputs)
    menu.VoiceMenuShared.move_voice_sample_from_menu(state)
    assert state.project.voice_references == original
    assert not errors
    if error:
        assert feedback == [(error, {"is_error": True})]
    saved.assert_not_called()


@pytest.mark.parametrize("action", ["trim", "edit", "remove"])
@pytest.mark.parametrize("value, error", [
    ("x", "Bad value"), ("1.5", "Bad value"), ("0", "Out of range"),
    ("-1", "Out of range"), ("4", "Out of range"),
])
def test_invalid_sample_index_uses_nonblocking_error_feedback(monkeypatch, action, value, error):
    state = make_state()
    original = list(state.project.voice_references)
    saved = Mock()
    monkeypatch.setattr(Project, "save", saved)
    errors, _, feedback = capture(monkeypatch, [value])
    if action == "trim":
        menu.VoiceMenuShared.crop_voice_sample_from_menu(state)
    elif action == "edit":
        menu.VoiceMenuShared.edit_voice_sample_transcript(state)
    else:
        assert menu.VoiceMenuShared.remove_voice_sample_from_menu(
            state, state.project.get_tts_model_type(),
        ) is False
    assert not errors
    assert feedback == [(error, {"is_error": True})]
    assert state.project.voice_references == original
    saved.assert_not_called()


def test_move_save_failure_restores_order(monkeypatch):
    state = make_state()
    original = list(state.project.voice_references)
    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    errors, _, feedback = capture(monkeypatch, ["1", "3"])
    menu.VoiceMenuShared.move_voice_sample_from_menu(state)
    assert state.project.voice_references == original
    assert errors == ["disk full"]
    assert not feedback


@pytest.mark.parametrize("action", ["move", "edit"])
def test_empty_sample_actions_do_not_prompt(monkeypatch, action):
    state = make_state(0)
    errors, prompts, feedback = capture(monkeypatch, [])
    if action == "move":
        menu.VoiceMenuShared.move_voice_sample_from_menu(state)
    else:
        menu.VoiceMenuShared.edit_voice_sample_transcript(state)
    assert feedback == ["No voice samples"]
    assert not errors and not prompts


@pytest.mark.parametrize("text, expected", [("updated", "updated"), ("/clear", "")])
def test_transcription_edits_selected_pair(monkeypatch, text, expected):
    state = make_state()
    capture(monkeypatch, ["2", text])
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    items = menu.VoiceMenuShared.make_voice_sample_items(state, state.project.get_tts_model_type())
    items[4].handler(state, items[4])
    assert [v["transcript"] for v in state.project.voice_references] == ["text 0", expected, "text 2"]
    assert state.project.voice_references[1]["file_name"] == "1.flac"
    saved.assert_called_once()


@pytest.mark.parametrize("inputs", [[""], ["0"], ["4"], ["x"], ["2", ""], ["2", "text 1"]])
def test_transcription_canceled_or_invalid(monkeypatch, inputs):
    state = make_state()
    original = list(state.project.voice_references)
    capture(monkeypatch, inputs)
    saved = Mock()
    monkeypatch.setattr(Project, "save", saved)
    menu.VoiceMenuShared.edit_voice_sample_transcript(state)
    assert state.project.voice_references == original
    saved.assert_not_called()


def test_transcription_save_error(monkeypatch):
    state = make_state(1)
    errors, _, feedback = capture(monkeypatch, ["updated"])
    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    menu.VoiceMenuShared.edit_voice_sample_transcript(state)
    assert errors == ["disk full"]
    assert not feedback
