from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.ask as ask_module
import tts_audiobook_tool.menus.project_menu as project_menu_module
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.constants_hints import HINT_TOLERANCE_FIRST_CLASS
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.project_menu import on_language
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class SaveableProject(SimpleNamespace):
    save_count: int = 0

    def get_tts_model_type(self) -> TtsModelType:
        return TtsModelType.require_by_id("none")

    def save(self) -> None:
        self.save_count += 1


@pytest.mark.parametrize(("entered_code", "expected_code"), [(" EN ", "en"), ("Es", "es")])
def test_language_change_normalizes_and_shows_tolerance_hint(
    monkeypatch, entered_code: str, expected_code: str
) -> None:
    project = SaveableProject(language_code="fr", strictness=Strictness.LOW)
    prefs = object()
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    hint_calls: list[tuple[object, object, bool]] = []

    stub_language_prompt(monkeypatch, entered_code, hint_calls)

    on_language(state, MenuItem("Language", lambda *_: None))

    assert project.language_code == expected_code
    assert project.save_count == 1
    assert hint_calls == [(prefs, HINT_TOLERANCE_FIRST_CLASS, True)]


@pytest.mark.parametrize("entered_code", ["en-US", "EN", "eng", "es-MX", "spa"])
def test_language_change_shows_tolerance_hint_for_first_class_aliases(
    monkeypatch, entered_code: str
) -> None:
    project = SaveableProject(language_code="fr", strictness=Strictness.LOW)
    prefs = object()
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    hint_calls: list[tuple[object, object, bool]] = []

    stub_language_prompt(monkeypatch, entered_code, hint_calls)

    on_language(state, MenuItem("Language", lambda *_: None))

    assert hint_calls == [(prefs, HINT_TOLERANCE_FIRST_CLASS, True)]


@pytest.mark.parametrize("entered_code", ["", "123", "fr"])
def test_language_change_does_not_show_tolerance_hint_when_not_saved_to_en_or_es(
    monkeypatch, entered_code: str
) -> None:
    project = SaveableProject(language_code="de", strictness=Strictness.LOW)
    state = cast(State, SimpleNamespace(project=project, prefs=object()))
    hint_calls: list[tuple[object, object, bool]] = []

    stub_language_prompt(monkeypatch, entered_code, hint_calls)

    on_language(state, MenuItem("Language", lambda *_: None))

    assert hint_calls == []
    if entered_code == "fr":
        assert project.language_code == "fr"
        assert project.save_count == 1
    else:
        assert project.language_code == "de"
        assert project.save_count == 0


def stub_language_prompt(
    monkeypatch,
    entered_code: str,
    hint_calls: list[tuple[object, object, bool]],
) -> None:
    monkeypatch.setattr(project_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None)
    monkeypatch.setattr(project_menu_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_module, "print_feedback", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_module, "ask_input", lambda *args, **kwargs: entered_code)
    monkeypatch.setattr(
        project_menu_module.Whitelist,
        "set_language_code",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        project_menu_module.hints,
        "show_hint_if_necessary",
        lambda prefs, hint, and_prompt=False: hint_calls.append(
            (prefs, hint, and_prompt)
        ),
    )


@pytest.mark.parametrize("did_open", [False, True])
def test_open_existing_project_keeps_project_menu_open(monkeypatch, did_open):
    state = cast(State, SimpleNamespace(project=SimpleNamespace(dir_path="")))
    captured = []
    feedback = []

    def open_project(current):
        assert current is state
        if did_open:
            current.project.dir_path = "opened-project"
        return did_open

    monkeypatch.setattr(
        project_menu_module.ProjectMenu, "ask_and_set_existing_project", open_project,
    )
    monkeypatch.setattr(
        project_menu_module.MenuUtil, "menu",
        lambda current, heading, items, **kwargs: captured.extend(items(current)),
    )
    monkeypatch.setattr(
        project_menu_module, "print_feedback", lambda *args: feedback.append(args),
    )

    project_menu_module.ProjectMenu.menu(state)
    item = next(item for item in captured if item.label == "Open existing project")

    # MenuUtil exits only when a handler returns True.
    assert item.handler(state, item) is None
    assert feedback == ([("Project directory set:", "opened-project")] if did_open else [])
