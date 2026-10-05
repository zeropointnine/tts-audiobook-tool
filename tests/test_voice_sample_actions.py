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
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: next(answers))
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(menu, "printt", lambda text="": prompts.append(text))
    monkeypatch.setattr(menu, "print_feedback", feedback.append)
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
    assert prompts == ["Enter voice sample number to move", "Enter new position"]
    assert not errors
    saved.assert_called_once()


@pytest.mark.parametrize("inputs, error", [([""], False), (["x"], True), (["0"], True), (["4"], True), (["1", ""], False), (["1", "x"], True), (["1", "0"], True), (["1", "4"], True), (["2", "2"], False)])
def test_move_invalid_cancel_and_no_change(monkeypatch, inputs, error):
    state = make_state()
    original = list(state.project.voice_references)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, _, _ = capture(monkeypatch, inputs)
    menu.VoiceMenuShared.move_voice_sample_from_menu(state)
    assert state.project.voice_references == original
    assert bool(errors) == error
    if error:
        assert errors == ["Enter a number between 1 and 3"]
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


@pytest.mark.parametrize("count, inputs, selected", [(1, [], 0), (3, ["2"], 1)])
def test_play_uses_saved_sample_and_async_player(monkeypatch, count, inputs, selected):
    state = make_state(count)
    capture(monkeypatch, inputs)
    monkeypatch.setattr(menu.ProjectVoiceUtil, "resolve_voice_file_path", lambda _, name: f"/voices/{name}")
    sound = SimpleNamespace(data=[0] * 10, sr=10)
    load = Mock(return_value=sound)
    play = Mock()
    monkeypatch.setattr(menu.SoundFileUtil, "load", load)
    monkeypatch.setattr(menu.PlaySoundUtil, "play_sound_async", play)
    menu.VoiceMenuShared.play_voice_sample_from_menu(state, state.project.get_tts_model_type())
    load.assert_called_once_with(f"/voices/{selected}.flac")
    play.assert_called_once_with(sound)


@pytest.mark.parametrize("inputs", [[""], ["x"], ["0"], ["4"]])
def test_play_invalid_or_canceled_does_not_load(monkeypatch, inputs):
    state = make_state()
    capture(monkeypatch, inputs)
    load = Mock()
    monkeypatch.setattr(menu.SoundFileUtil, "load", load)
    menu.VoiceMenuShared.play_voice_sample_from_menu(state, state.project.get_tts_model_type())
    load.assert_not_called()


def test_play_load_error(monkeypatch):
    state = make_state(1)
    errors, _, _ = capture(monkeypatch, [])
    monkeypatch.setattr(menu.SoundFileUtil, "load", lambda _: "missing audio")
    play = Mock()
    monkeypatch.setattr(menu.PlaySoundUtil, "play_sound_async", play)
    menu.VoiceMenuShared.play_voice_sample_from_menu(state, state.project.get_tts_model_type())
    assert errors == ["missing audio"]
    play.assert_not_called()


@pytest.mark.parametrize("action", ["move", "play", "edit"])
def test_empty_sample_actions_do_not_prompt(monkeypatch, action):
    state = make_state(0)
    errors, prompts, feedback = capture(monkeypatch, [])
    if action == "play":
        menu.VoiceMenuShared.play_voice_sample_from_menu(state, state.project.get_tts_model_type())
    elif action == "move":
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
