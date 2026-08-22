from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.state import State
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