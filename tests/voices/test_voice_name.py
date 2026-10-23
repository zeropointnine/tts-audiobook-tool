"""Per-voice user-facing name (`Voice.name`): storage, validation and the voice menus."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared as menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import (
    VOICE_NAME_ONE_WORD_ERROR,
    VOICE_NAME_UNIQUE_ERROR,
    ProjectVoiceUtil,
)
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import make_currently_string


def test_empty_name_is_not_written_and_set_name_round_trips():
    assert "name" not in Voice("a.flac").to_dict()
    data = Voice("a.flac", name="Narrator").to_dict()
    assert data["name"] == "Narrator"
    assert Voice.from_dict(data) == Voice("a.flac", name="Narrator")


@pytest.mark.parametrize("stored", [None, 5, ["x"], {"a": "b"}])
def test_malformed_name_loads_empty(stored):
    voice = Voice.from_dict({"file_name": "a.flac", "name": stored})
    assert voice is not None and voice.name == ""


def test_name_survives_project_round_trip():
    # The name persists through the project's voice_references list, not just Voice.
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "name": "Alice"}, {"file_name": "b.flac"},
    ])
    assert [v.name for v in project.voice_references] == ["Alice", ""]


def test_set_name_strips_edits_only_that_voice_and_restores_on_failed_save(monkeypatch):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac"}, {"file_name": "b.flac", "pan": 0.5},
    ])
    monkeypatch.setattr(Project, "save", lambda _: "")
    assert ProjectVoiceUtil.set_voice_name_at_index_and_save(project, 1, "  Bob  ") == ""
    assert [v.name for v in project.voice_references] == ["", "Bob"]
    assert project.voice_references[1].pan == 0.5

    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    assert ProjectVoiceUtil.set_voice_name_at_index_and_save(project, 1, "Carol") == "disk full"
    assert project.voice_references[1].name == "Bob"
    with pytest.raises(IndexError):
        ProjectVoiceUtil.set_voice_name_at_index_and_save(project, 2, "x")


def test_submenu_name_item_shows_optional_then_value_and_offers_clear(monkeypatch):
    # Unset: dim "(optional)" and no Clear item. Set: "(currently: ...)" plus Clear name.
    state = cast(State, SimpleNamespace(project=Project(
        tts_model_type="omnivoice_local", voice_references=[{"file_name": "a.flac", "transcript": "t"}],
    )))
    captured = {}
    monkeypatch.setattr(menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(menu.AudioMetaUtil, "get_audio_duration", lambda _: 4.0)
    monkeypatch.setattr(Project, "save", lambda _: "")
    monkeypatch.setattr(menu, "printt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(menu, "print_feedback", lambda *_args, **_kwargs: None)
    menu.VoiceMenuShared.voice_sample_submenu(state, state.project.get_tts_model_type(), 0)

    def labels():
        return [get_string_from(state, item.label) for item in captured["items"](state)]

    assert labels()[0] == menu.make_menu_label_optional(menu.LABEL_VOICE_NAME)
    assert menu.LABEL_CLEAR_VOICE_NAME not in labels()

    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: "Narrator")
    item = captured["items"](state)[0]
    item.handler(state, item)
    assert state.project.voice_references[0].name == "Narrator"
    assert labels()[0] == f"{menu.LABEL_VOICE_NAME} {make_currently_string('Narrator')}"

    # Blank input (or cancel) leaves the name unchanged; Clear removes it.
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: "")
    item.handler(state, item)
    assert state.project.voice_references[0].name == "Narrator"
    clear = next(i for i in captured["items"](state)
                 if get_string_from(state, i.label) == menu.LABEL_CLEAR_VOICE_NAME)
    clear.handler(state, clear)
    assert state.project.voice_references[0].name == ""


def test_name_taken_ignores_own_voice_case_and_whitespace():
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "name": "Narrator"}, {"file_name": "b.flac"},
    ])
    assert ProjectVoiceUtil.is_voice_name_taken(project, 1, "  narrator ")
    assert not ProjectVoiceUtil.is_voice_name_taken(project, 0, "NARRATOR")  # its own name
    assert not ProjectVoiceUtil.is_voice_name_taken(project, 1, "Alice")
    assert not ProjectVoiceUtil.is_voice_name_taken(project, 1, "")


@pytest.mark.parametrize("name, error", [
    ("Bob", ""),
    ("  Bob  ", ""),  # surrounding whitespace is stripped, not a second word
    ("", ""),  # empty means unset
    ("Big Bob", VOICE_NAME_ONE_WORD_ERROR),
    ("Big\tBob", VOICE_NAME_ONE_WORD_ERROR),
    ("narrator", VOICE_NAME_UNIQUE_ERROR),
])
def test_voice_name_error(name, error):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "name": "Narrator"}, {"file_name": "b.flac"},
    ])
    assert ProjectVoiceUtil.get_voice_name_error(project, 1, name) == error


def test_setter_refuses_invalid_name_without_saving(monkeypatch):
    # The setter enforces the same rules as the menu, so no caller can store an invalid name.
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "name": "Narrator"}, {"file_name": "b.flac"},
    ])
    monkeypatch.setattr(Project, "save", lambda _: pytest.fail("must not save"))
    assert ProjectVoiceUtil.set_voice_name_at_index_and_save(project, 1, "Big Bob") == VOICE_NAME_ONE_WORD_ERROR
    assert ProjectVoiceUtil.set_voice_name_at_index_and_save(project, 1, "NARRATOR") == VOICE_NAME_UNIQUE_ERROR
    assert [v.name for v in project.voice_references] == ["Narrator", ""]


@pytest.mark.parametrize("entered, message", [
    ("narrator", "Name must be unique"),
    ("Big Bob", "Name must be one word (no spaces)"),
])
def test_invalid_name_is_rejected_from_menu_without_saving(monkeypatch, entered, message):
    # An invalid entry shows error feedback and leaves both voices unchanged.
    state = cast(State, SimpleNamespace(project=Project(
        tts_model_type="omnivoice_local", voice_references=[
            {"file_name": "a.flac", "name": "Narrator"}, {"file_name": "b.flac", "name": "Bob"},
        ],
    )))
    feedback = []
    monkeypatch.setattr(Project, "save", lambda _: pytest.fail("must not save"))
    monkeypatch.setattr(menu, "printt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(menu, "print_feedback", lambda message, **kwargs: feedback.append((message, kwargs)))
    monkeypatch.setattr(menu.ask, "ask_input", lambda **_: entered)
    menu.VoiceMenuShared.ask_voice_name(state, 1)
    assert [v.name for v in state.project.voice_references] == ["Narrator", "Bob"]
    assert feedback == [(message, {"is_error": True})]


def test_submenu_heading_includes_name_when_set(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(
        tts_model_type="omnivoice_local", voice_references=[{"file_name": "a.flac"}, {"file_name": "b.flac"}],
    )))
    captured = {}
    monkeypatch.setattr(menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    menu.VoiceMenuShared.voice_sample_submenu(state, state.project.get_tts_model_type(), 1)
    assert get_string_from(state, captured["heading"]) == "Voice 2"
    monkeypatch.setattr(Project, "save", lambda _: "")
    ProjectVoiceUtil.set_voice_name_at_index_and_save(state.project, 1, "MyName")
    assert get_string_from(state, captured["heading"]) == "Voice 2 MyName"


def test_voice_list_item_includes_name_when_set(monkeypatch):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "name": "MyName"}, {"file_name": "b.flac"},
    ])
    monkeypatch.setattr(menu.AudioMetaUtil, "get_audio_duration", lambda _: None)
    tts_type = project.get_tts_model_type()
    first, _ = menu.VoiceMenuShared.make_voice_sample_item_label(project, tts_type, 0)
    second, _ = menu.VoiceMenuShared.make_voice_sample_item_label(project, tts_type, 1)
    assert first.startswith("Voice 1 MyName: ")
    assert second.startswith("Voice 2: ")
