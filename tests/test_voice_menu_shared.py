from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppStylesheetFailed,
    EditorSaveFailed,
)
from textual_editor_stubs import StubPhraseGroup, StubProject


@pytest.mark.parametrize(
    ("run_result", "expected_message"),
    [
        (
            ContentAppCompleted(EditorSaveFailed("Save failed: disk full")),
            "Save failed: disk full",
        ),
        (
            ContentAppStylesheetFailed("Couldn't load textual css"),
            "Couldn't load textual css",
        ),
    ],
)
def test_voice_sample_assignment_reports_editor_failures(
    run_result, expected_message: str, monkeypatch
) -> None:
    state = cast(
        State,
        SimpleNamespace(project=StubProject([StubPhraseGroup("Line 1")])),
    )
    feedback_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        voice_menu_shared,
        "VoiceLineEditorTextualApp",
        lambda _: object(),
    )
    monkeypatch.setattr(
        voice_menu_shared,
        "run_content_textual_app",
        lambda _: run_result,
    )
    monkeypatch.setattr(
        voice_menu_shared,
        "print_feedback",
        lambda message, **kwargs: feedback_calls.append(
            (message, kwargs.get("is_error", False))
        ),
    )

    voice_menu_shared.VoiceMenuShared.assign_voice_samples_to_text_lines(state)

    assert feedback_calls == [(expected_message, True)]

# ---------------------------------------------------------------------------
# Voice sample selection mode menu items (moved from test_voice_selection.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("voice_count, expected_item_count", [(0, 2), (1, 2), (2, 3)])
def test_voice_sample_selection_mode_item_requires_multiple_samples(
        voice_count: int,
        expected_item_count: int,
) -> None:
    project = Project.model_validate({
        "mira_voice_file_name": [f"voice-{index}.flac" for index in range(voice_count)],
    })
    state = cast(State, SimpleNamespace(project=project))

    items = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.MIRA)

    assert len(items) == expected_item_count
    assert get_string_from(state, items[0].label).startswith("Add/remove voice samples") or voice_count == 0
    if voice_count > 1:
        assert "Voice selection mode" in get_string_from(state, items[1].label)
        assert VoiceSelectMode.AUTO_ADVANCE.current_label in get_string_from(state, items[1].label)


def test_voice_sample_selection_mode_item_label_tracks_project_value() -> None:
    project = Project.model_validate({
        "mira_voice_file_name": ["voice-a.flac", "voice-b.flac"],
    })
    state = cast(State, SimpleNamespace(project=project))
    item = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.MIRA)[1]

    assert VoiceSelectMode.AUTO_ADVANCE.current_label in get_string_from(state, item.label)

    project.voice_select_mode = VoiceSelectMode.USER_DEFINED

    assert VoiceSelectMode.USER_DEFINED.current_label in get_string_from(state, item.label)


def test_voice_sample_selection_mode_submenu_uses_options_menu_and_saves_selection() -> None:
    project = Project.model_validate({
        "mira_voice_file_name": ["voice-a.flac", "voice-b.flac"],
    })
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(Project, "save") as save, \
            patch("tts_audiobook_tool.menus.voice.voice_menu_shared.MenuUtil.options_menu") as options_menu:
        VoiceMenuShared.voice_sample_selection_mode_submenu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](VoiceSelectMode.USER_DEFINED)

    assert kwargs["heading_text"] == "Voice selection mode"
    assert kwargs["labels"] == [mode.label for mode in VoiceSelectMode]
    assert kwargs["values"] == list(VoiceSelectMode)
    assert kwargs["current_value"] == VoiceSelectMode.AUTO_ADVANCE
    assert kwargs["default_value"] == VoiceSelectMode.get_default()
    assert kwargs["sublabels"] == [mode.description for mode in VoiceSelectMode]
    assert project.voice_select_mode == VoiceSelectMode.USER_DEFINED
    save.assert_called_once_with()
