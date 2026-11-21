"""Pocket voice-cloning access validation is remembered only after success."""
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.voice import voice_pocket_menu as menu
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State


def make_project(model_code: str = "english_2026-10", **kwargs) -> Project:
    project = Project(tts_model_type="pocket_local", **kwargs)
    project.set_model_setting("pocket_local", "model_code", model_code)
    return project


def make_state(model_code: str = "english_2026-10"):
    state = State.for_worker(Prefs(project_dir="", stt_variant=SttVariant.DISABLED))
    state._project = make_project(model_code, voice_references=[
        {"file_name": "voice.flac", "transcript": "Reference."},
    ])
    return state


def test_successful_access_check_is_silent_afterward_even_across_projects(monkeypatch):
    state = make_state()
    calls, output, feedback = [], [], []
    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking",
                        lambda current: (calls.append(current) or SimpleNamespace(blocking_issues=[]), ""))
    monkeypatch.setattr(menu, "printt", lambda *args: output.append(args))
    monkeypatch.setattr(menu, "print_feedback", feedback.append)
    menu.validate_voice_file(state)
    assert state.pocket_voice_clone_access_validated
    assert calls == [state]
    assert "Validating Pocket voice cloning access" in output[0][0]
    assert feedback == ["Validated"]
    output.clear()
    feedback.clear()
    state._project = make_project(voice_references=[
        {"file_name": "another.flac", "transcript": "Another reference."},
    ])
    menu.validate_voice_file(state)
    assert calls == [state]
    assert output == feedback == []


@pytest.mark.parametrize("first_result", [
    (None, "Model worker is busy"),
    (SimpleNamespace(blocking_issues=["Access denied"]), ""),
    (None, ""),
])
def test_unsuccessful_check_retries_until_success(monkeypatch, first_result):
    state = make_state()
    results = iter([first_result, (SimpleNamespace(blocking_issues=[]), "")])
    calls, errors = [], []

    def inspect(current):
        calls.append(current)
        return next(results)

    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking", inspect)
    monkeypatch.setattr(menu.ask, "ask_error", errors.append)
    menu.validate_voice_file(state)
    assert not state.pocket_voice_clone_access_validated
    menu.validate_voice_file(state)
    assert state.pocket_voice_clone_access_validated
    menu.validate_voice_file(state)
    assert calls == [state, state]
    expected_error = first_result[1] or (first_result[0].blocking_issues[0] if first_result[0] else "")
    assert errors == ([expected_error + "\n"] if expected_error else [])


@pytest.mark.parametrize("has_clone,preset", [(False, ""), (True, "alba")])
def test_unused_clone_does_not_mark_access_validated(monkeypatch, has_clone, preset):
    state = make_state()
    if not has_clone:
        state.project.voice_references = []
    elif preset:
        # The voice's predefined voice takes precedence over its sample
        state.project.voice_references = [Voice("voice.flac", "Reference.", presets={"pocket_local": preset})]
    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("No clone access check needed"))
    menu.validate_voice_file(state)
    assert not state.pocket_voice_clone_access_validated


def test_each_new_interactive_state_starts_unvalidated(monkeypatch):
    prefs = Prefs(project_dir="", stt_variant=SttVariant.DISABLED)
    monkeypatch.setattr(Prefs, "load", lambda: prefs)
    # Avoid binding projects/starting their auxiliary resources in this constructor test.
    monkeypatch.setattr(State, "project", property(
        State.project.fget, lambda self, project: setattr(self, "_project", project)))
    first = State()
    assert not first.pocket_voice_clone_access_validated
    first.pocket_voice_clone_access_validated = True
    second = State()
    assert not second.pocket_voice_clone_access_validated
    assert first.pocket_voice_clone_access_validated


def test_access_check_waits_for_a_model_selection(monkeypatch):
    # The access check loads the model, which can't happen without a selected
    # model code; the missing selection is reported as a blocker instead.
    state = make_state(model_code="")
    monkeypatch.setattr(menu.ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("Must not load Pocket without a model selection"))
    menu.validate_voice_file(state)
    assert not state.pocket_voice_clone_access_validated


def test_missing_model_selection_blocks_without_an_instance():
    # A project without a Pocket model code is blocked up front (menus, run
    # starts, server), with no implicit default model.
    from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel

    issues = PocketBaseModel.get_blocking_issues(make_project(""), None)
    assert [issue.short for issue in issues] == ["Pocket model selection"]
    assert issues[0].verbose == PocketBaseModel.MODEL_NOT_SELECTED_MESSAGE
    assert PocketBaseModel.get_blocking_issues(make_project("dutch"), None) == []


@pytest.mark.parametrize("language_code, expected", [
    ("en", "english_2026-10"), ("en-GB", "english_2026-10"), ("nl", "dutch_24l"),
    ("fr", "french_24l"), ("ja", ""), ("", ""),
])
def test_recommended_model_follows_project_language(language_code, expected):
    # The picker hint follows the project language; unsupported languages get none.
    from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel

    project = make_project("", language_code=language_code)
    assert PocketBaseModel.get_recommended_language(project) == expected
