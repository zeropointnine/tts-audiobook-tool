import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppStylesheetFailed,
    EditorSaveFailed,
)
from textual_editor_stubs import StubPhraseGroup, StubProject


@pytest.mark.parametrize("model_id", ["none", "unknown-model"])
def test_remote_voice_menu_removes_controls_when_selection_is_unresolved(monkeypatch, model_id):
    state = cast(State, SimpleNamespace(project=Project(
        tts_model_type="omnivoice_audiocpp",
        voice_references=[{"file_name": "existing.flac", "transcript": "text"}],
    )))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu(state)
    assert captured["items"](state)
    state.project.tts_model_type = model_id
    assert captured["items"](state) == []
    assert get_string_from(state, captured["subheading"]) == ""


def test_remote_voice_menu_tracks_selection_on_redraw_and_edit(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="omnivoice_audiocpp")))
    state.project.set_model_setting("omnivoice_audiocpp", "file_name", ["old.flac"])
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu(state)
    assert "old.flac" in get_string_from(state, captured["subheading"])

    state.project.tts_model_type = "breeze_tts_2_audiocpp"
    state.project.set_model_setting("breeze_tts_2_audiocpp", "file_name", ["new.flac"])
    subheading = get_string_from(state, captured["subheading"])
    assert "new.flac" in subheading and "old.flac" not in subheading
    calls = []
    monkeypatch.setattr(VoiceMenuShared, "ask_and_set_voice_file",
                        lambda current, model, **kwargs: calls.append((model.id, kwargs)))
    monkeypatch.setattr(VoiceMenuShared, "remove_voice_sample_from_menu",
                        lambda current, model: calls.append((model.id, "remove")) or False)
    items = captured["items"](state)
    items[0].handler(state, items[0])
    items[1].handler(state, items[1])
    assert calls == [("breeze_tts_2_audiocpp", {"append": True}), ("breeze_tts_2_audiocpp", "remove")]
    state.project.tts_model_type = "none"
    assert captured["items"](state) == []
    assert get_string_from(state, captured["subheading"]) == ""


@pytest.mark.parametrize("model_id", [
    "mira_local", "dots_local", "chatterbox_audiocpp",
    "omnivoice_audiocpp", "zonos2_sglomni", "higgs_v3_sglomni",
])
@pytest.mark.parametrize("voice_count", [0, 1, 2])
def test_transcript_editing_is_available_without_changing_transcripts(model_id, voice_count):
    project = Project(tts_model_type=model_id, voice_references=[
        {"file_name": f"voice-{i}.flac", "transcript": "Retained reference text"}
        for i in range(voice_count)
    ])
    state = cast(State, SimpleNamespace(project=project))
    items = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())
    labels = [get_string_from(state, item.label) for item in items]
    assert voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION in labels
    assert all(entry["transcript"] == "Retained reference text" for entry in project.voice_references)


def test_remote_transcript_editing_is_available_on_redraw(monkeypatch):
    project = Project(tts_model_type="omnivoice_audiocpp", voice_references=[
        {"file_name": "voice.flac", "transcript": "Retained reference text"},
    ])
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu(state)
    for model_id in [
        "omnivoice_audiocpp", "chatterbox_audiocpp", "zonos2_sglomni", "higgs_v3_sglomni",
    ]:
        project.tts_model_type = model_id
        labels = [get_string_from(state, item.label) for item in captured["items"](state)]
        assert voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION in labels
        assert project.voice_references[0]["transcript"] == "Retained reference text"


def test_voice_menu_direct_actions_refresh_samples_and_preserve_callbacks(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="dots_local")))
    model = state.project.get_tts_model_type()
    captured = {}
    events = []
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 1.25)

    def add_sample(current, selected, **kwargs):
        assert selected == model and kwargs == {"append": True}
        events.append("add")
        current.project.voice_references.append({"file_name": "new.flac", "transcript": "text"})

    def remove_sample(current, selected):
        assert selected == model
        events.append("remove")
        current.project.voice_references.clear()
        return True

    monkeypatch.setattr(VoiceMenuShared, "ask_and_set_voice_file", add_sample)
    monkeypatch.setattr(VoiceMenuShared, "remove_voice_sample_from_menu", remove_sample)

    def make_items(current):
        return VoiceMenuShared.make_voice_sample_items(
            current, model,
            on_before_set_callback=lambda: events.append("before"),
            on_set_callback=lambda: events.append("after"),
            on_clear_callback=lambda: events.append("clear"),
        )

    VoiceMenuShared.menu_wrapper(state, make_items, subheading=lambda _: "Sample guidance")
    assert get_string_from(state, captured["subheading"]) == "Sample guidance"
    items = captured["items"](state)
    assert items[0].handler(state, items[0]) is None
    subheading = get_string_from(state, captured["subheading"])
    assert "Voice sample 1:" in subheading and "new.flac" in subheading
    assert "(1.2s)" in subheading
    assert subheading.endswith("\n\nSample guidance")
    items = captured["items"](state)
    assert get_string_from(state, items[0].label) == voice_menu_shared.LABEL_ADD_VOICE_SAMPLE
    assert items[1].handler(state, items[1]) is None  # clearing must not exit Voice clone
    assert get_string_from(state, captured["subheading"]) == "Sample guidance"
    # Zero samples: no move item.
    assert len(captured["items"](state)) == 5
    assert events == ["before", "add", "after", "remove", "clear"]
    # Removing from an already-empty list must not unload a model again.
    items[1].handler(state, items[1])
    assert events[-1] == "remove"
    assert events.count("clear") == 1


@pytest.mark.parametrize("voice_count", [0, 1])
def test_voice_menu_canceled_actions_keep_samples_and_clear_callback(monkeypatch, voice_count):
    project = Project(tts_model_type="mira_local", voice_references=[
        {"file_name": "existing.flac", "transcript": "text"} for _ in range(voice_count)
    ])
    state = cast(State, SimpleNamespace(project=project))
    events = []
    monkeypatch.setattr(VoiceMenuShared, "ask_and_set_voice_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(VoiceMenuShared, "remove_voice_sample_from_menu", lambda *_: False)
    items = VoiceMenuShared.make_voice_sample_items(
        state, project.get_tts_model_type(),
        no_samples_label="Custom empty label",
        on_before_set_callback=lambda: events.append("before"),
        on_set_callback=lambda: events.append("after"),
        on_clear_callback=lambda: events.append("clear"),
    )
    assert get_string_from(state, items[0].label) == (
        voice_menu_shared.LABEL_ADD_VOICE_SAMPLE if voice_count else "Custom empty label"
    )
    assert items[0].handler(state, items[0]) is None
    assert items[1].handler(state, items[1]) is None
    assert len(project.voice_references) == voice_count
    assert events == ["before", "after"]  # retain the existing post-prompt callback behavior


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
    feedback_calls: list[str] = []
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
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", feedback_calls.append)

    voice_menu_shared.VoiceMenuShared.assign_voice_samples_to_text_lines(state)

    assert feedback_calls == [expected_message]

# ---------------------------------------------------------------------------
# Voice sample selection mode menu items (moved from test_voice_selection.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("voice_count", [0, 1, 2])
def test_voice_samples_subheading_has_no_trailing_newline(monkeypatch, voice_count):
    project = Project(tts_model_type="mira_local")
    voices = [f"voice-{index}.flac" for index in range(voice_count)]
    monkeypatch.setattr(voice_menu_shared.ProjectVoiceUtil, "get_voice_values", lambda *_: voices)
    monkeypatch.setattr(voice_menu_shared.ProjectVoiceUtil, "make_voice_sample_display_label", lambda _, name, __: name)
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)

    text = VoiceMenuShared.make_voice_samples_subheading(project, TtsModelType.require_by_id("mira_local"))

    assert not text.endswith("\n")
    assert text.count("\n") == max(0, voice_count - 1)
    for index, voice in enumerate(voices, start=1):
        assert f"Voice sample {index}:" in text and voice in text


@pytest.mark.parametrize("model_id", ["omnivoice_local", "echo_tts_audiocpp", "omnivoice_audiocpp"])
def test_voice_samples_subheading_marks_only_long_samples_with_one_footnote(monkeypatch, model_id):
    project = Project(tts_model_type=model_id, voice_references=[
        {"file_name": "long.flac", "transcript": ""},
        {"file_name": "short.flac", "transcript": ""},
        {"file_name": "also-long.flac", "transcript": ""},
    ])
    durations = iter([15.1, 14.6, 16.0])
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: next(durations))

    text = VoiceMenuShared.make_voice_samples_subheading(project, project.get_tts_model_type())

    marker = f"{voice_menu_shared.COL_ERROR}*{voice_menu_shared.COL_DEFAULT}"
    lines = text.splitlines()
    assert lines[0].endswith(f"(15.1s){voice_menu_shared.COL_DEFAULT}{marker}")
    assert lines[1].endswith(f"(14.6s){voice_menu_shared.COL_DEFAULT}")
    assert lines[2].endswith(f"(16.0s){voice_menu_shared.COL_DEFAULT}{marker}")
    assert lines[3] == (
        f"  {voice_menu_shared.COL_ERROR}* {voice_menu_shared.COL_DEFAULT}"
        f"{voice_menu_shared.COL_DIM_ITALICS}Sample exceeds recommended duration "
        f"for the current model (15s){voice_menu_shared.COL_DEFAULT}"
    )
    assert text.count("Sample exceeds") == 1
    assert not text.endswith("\n")


@pytest.mark.parametrize("duration, warned", [(14.6, False), (15.0, False), (15.1, True), (15.01, True), (None, False)])
def test_voice_sample_duration_warning_uses_unrounded_duration(monkeypatch, duration, warned):
    project = Project(tts_model_type="echo_tts_audiocpp", voice_references=[
        {"file_name": "voice.flac", "transcript": ""},
    ])
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: duration)

    text = VoiceMenuShared.make_voice_samples_subheading(project, project.get_tts_model_type())

    assert ("Sample exceeds" in text) == warned
    assert (f"{voice_menu_shared.COL_ERROR}*" in text) == warned
    assert ("(" in text) == (duration is not None)


def make_crop_project(tmp_path) -> Project:
    """A dots_local project (10s recommended) with a cropped 11.2s sample."""
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    (voice_dir / "long.flac").write_bytes(b"x")
    project = Project(tts_model_type="dots_local", voice_references=[
        {
            "file_name": "long.flac",
            "transcript": "",
            "crop_file_name": "a.flac",
            "crop_start": "1.1",
            "crop_end": "11.1572",
            "crop_transcript": "cut",
        },
    ])
    project.dir_path = str(tmp_path)
    return project


def test_voice_samples_subheading_reports_cropped_duration(tmp_path, monkeypatch):
    """The saved crop drives the shown duration and the warning."""
    project = make_crop_project(tmp_path)
    entry = project.voice_references[0]
    crop_path = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry))
    crop_path.parent.mkdir(parents=True)
    crop_path.write_bytes(b"x")
    durations = {"long.flac": 11.1572, "a.flac": 9.9}
    monkeypatch.setattr(
        voice_menu_shared.AudioMetaUtil,
        "get_audio_duration",
        lambda path: durations[Path(path).name],
    )

    text = VoiceMenuShared.make_voice_samples_subheading(project, project.get_tts_model_type())

    assert "(trimmed, 9.9s)" in text
    assert "11.1572" not in text
    assert "*" not in text
    assert "Sample exceeds" not in text


def test_voice_samples_subheading_falls_back_when_cropped_file_missing(tmp_path, monkeypatch):
    """No cropped file on disk means generation uses the original, so the warning applies."""
    project = make_crop_project(tmp_path)
    durations = {"long.flac": 11.1572}
    monkeypatch.setattr(
        voice_menu_shared.AudioMetaUtil,
        "get_audio_duration",
        lambda path: durations[Path(path).name],
    )

    text = VoiceMenuShared.make_voice_samples_subheading(project, project.get_tts_model_type())

    assert "(trimmed, 11.2s)" in text
    assert f"{voice_menu_shared.COL_ERROR}*" in text
    assert "Sample exceeds recommended duration for the current model (10s)" in text


@pytest.mark.parametrize("model_id", ["chatterbox_local", "chatterbox_audiocpp", "higgs_v3_sglomni"])
def test_voice_samples_without_duration_recommendation_have_no_warning(monkeypatch, model_id):
    project = Project(tts_model_type=model_id, voice_references=[
        {"file_name": "voice.flac", "transcript": ""},
    ])
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 120.0)

    text = VoiceMenuShared.make_voice_samples_subheading(project, project.get_tts_model_type())

    assert "Voice sample 1:" in text
    assert "*" not in text
    assert "Sample exceeds" not in text


def test_voice_menu_duration_warning_tracks_current_model_on_redraw(monkeypatch):
    project = Project(tts_model_type="echo_tts_audiocpp", voice_references=[
        {"file_name": "voice.flac", "transcript": ""},
    ])
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 15.1)
    VoiceMenuShared.menu(state)

    for model_id, warned in [("echo_tts_audiocpp", True), ("fish_s2_sglomni", False),
                              ("auk_sglomni", True), ("chatterbox_audiocpp", False)]:
        project.tts_model_type = model_id
        text = get_string_from(state, captured["subheading"])
        assert ("Sample exceeds" in text) == warned
        if warned:
            max_duration = project.get_tts_model_type().value.ui["voice_sample_max_duration_s"]
            assert text.endswith(f"({max_duration:g}s){voice_menu_shared.COL_DEFAULT}")


@pytest.mark.parametrize("voice_count, expected_item_count", [(0, 5), (1, 5), (2, 7), (9, 6)])
def test_voice_sample_selection_mode_item_requires_multiple_samples(
        voice_count: int,
        expected_item_count: int,
) -> None:
    project = Project.model_validate({
        "mira_voice_file_name": [f"voice-{index}.flac" for index in range(voice_count)],
    })
    state = cast(State, SimpleNamespace(project=project))

    items = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id("mira_local"))

    assert len(items) == expected_item_count
    labels = [get_string_from(state, item.label) for item in items]
    expected = []
    if voice_count < 9:
        expected.append(voice_menu_shared.LABEL_ADD_VOICE_SAMPLE)
    expected.append(voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE)
    if voice_count > 1:
        expected.append(voice_menu_shared.LABEL_MOVE_VOICE_SAMPLE)
    expected.extend([
        voice_menu_shared.LABEL_CROP_VOICE_SAMPLE,
        voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION,
    ])
    assert items[-2 if voice_count > 1 else -1].blank_line_before
    if voice_count > 1:
        expected.append(voice_menu_shared.LABEL_VOICE_SELECTION_MODE)
        assert VoiceSelectMode.DISABLED.current_label in labels[-2]
    expected.append(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    assert all(label.startswith(prefix) for label, prefix in zip(labels, expected))
    assert not any(label.startswith("Add/remove") for label in labels)


def test_voice_sample_selection_mode_item_label_tracks_project_value() -> None:
    project = Project.model_validate({
        "mira_voice_file_name": ["voice-a.flac", "voice-b.flac"],
    })
    state = cast(State, SimpleNamespace(project=project))
    item = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id("mira_local"))[-2]

    assert VoiceSelectMode.DISABLED.current_label in get_string_from(state, item.label)

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

    assert kwargs["heading_text"] == voice_menu_shared.LABEL_VOICE_SELECTION_MODE
    assert kwargs["labels"] == [mode.label for mode in VoiceSelectMode]
    assert kwargs["values"] == list(VoiceSelectMode)
    assert kwargs["current_value"] == VoiceSelectMode.DISABLED
    assert kwargs["default_value"] == VoiceSelectMode.get_default()
    assert kwargs["sublabels"] == [mode.description for mode in VoiceSelectMode]
    assert project.voice_select_mode == VoiceSelectMode.USER_DEFINED
    save.assert_called_once_with()


@pytest.mark.parametrize("transcript", ["Transcribed sample text.", ""])
def test_transcribed_text_uses_long_pause_feedback(monkeypatch, transcript):
    from tts_audiobook_tool.app_types import Sound
    import numpy as np

    state = cast(State, SimpleNamespace(
        project=Project(tts_model_type="glm_local", language_code="en"),
        prefs=SimpleNamespace(stt_variant=voice_menu_shared.SttVariant.LARGE_V3, stt_config={}),
    ))
    sound = Sound(np.ones(24000, dtype=np.float32), 24000)
    monkeypatch.setattr(voice_menu_shared.Transcriber, "transcribe_to_words", lambda *_: ["words"])
    monkeypatch.setattr(
        voice_menu_shared.Transcriber,
        "get_flat_text_filtered_by_probability",
        lambda *_: transcript,
    )
    with patch.object(voice_menu_shared, "print_feedback") as feedback:
        assert VoiceMenuShared.transcribe_voice_sample_to_text(state, sound) == (transcript, "")
    feedback.assert_called_once_with(transcript, long_pause=True)


# --- validate_voices pre-flight ---

class _ValidateCapture:
    def __init__(self, monkeypatch):
        self.lines = []
        self.errors = []
        monkeypatch.setattr(voice_menu_shared, "printt", lambda message="": self.lines.append(message))
        monkeypatch.setattr(voice_menu_shared.ask, "ask_error", lambda message: self.errors.append(message))


def _validate_state(tmp_path, model_id, entries, *, preset=False):
    project = Project(dir_path=str(tmp_path), tts_model_type=model_id, voice_references=entries)
    if preset:
        project.set_model_setting("pocket_local", "predefined_voice", "alba")
    prefs = SimpleNamespace(stt_variant=None, stt_config={})
    return cast(State, SimpleNamespace(project=project, prefs=prefs))


@pytest.mark.parametrize("model_id, entries, preset", [
    ("none", [{"file_name": "a.flac", "transcript": ""}], False),  # no voice binding
    ("dots_local", [], False),  # empty shared list, voice not required
    ("pocket_local", [{"file_name": "a.flac", "transcript": ""}], True),  # preset takes precedence
])
def test_validate_voices_no_op_cases(tmp_path, monkeypatch, model_id, entries, preset):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, model_id, entries, preset=preset)
    assert VoiceMenuShared.validate_voices(state) is True
    assert not capture.lines and not capture.errors


@pytest.mark.parametrize("model_type", ["custom_voice", "voice_design"])
def test_validate_voices_qwen3_non_base_checkpoint_needs_no_samples(tmp_path, monkeypatch, model_type):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "qwen3tts_local", [])
    state.project.set_model_setting("qwen3tts_local", "model_type", model_type)
    assert VoiceMenuShared.validate_voices(state) is True
    assert not capture.lines and not capture.errors


def test_validate_voices_qwen3_base_checkpoint_requires_samples(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "qwen3tts_local", [])
    assert state.project.get_model_setting("qwen3tts_local", "model_type") in ("", "base")
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.errors == ["A voice clone sample is required"]


def test_validate_voices_pocket_empty_message_mentions_predefined_voice(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "pocket_local", [], preset=False)
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.errors == ["A voice clone sample or predefined voice is required"]


def test_validate_voices_blocks_required_voice_with_empty_list(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "glm_local", [])
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.errors == ["A voice clone sample is required"]


def test_validate_voices_reports_missing_and_invalid_files(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    (tmp_path / "bad.flac").write_bytes(b"junk")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda path: "decode failed" if path.endswith("bad.flac") else object())
    state = _validate_state(tmp_path, "glm_local", [
        {"file_name": "gone.flac", "transcript": "kept"},
        {"file_name": "bad.flac", "transcript": "kept"},
    ])
    assert VoiceMenuShared.validate_voices(state) is False
    assert any("Voice file gone.flac not found" in line for line in capture.lines)
    assert any("Voice file bad.flac is invalid" in line for line in capture.lines)
    assert capture.errors == ["Replace problem voice clone file"]
    assert state.project.voice_references[0]["transcript"] == "kept"


def test_validate_voices_transcribes_empty_transcript_and_saves(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    (tmp_path / "a.flac").write_bytes(b"valid")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    monkeypatch.setattr(voice_menu_shared.Transcriber, "transcribe_to_words", lambda *args, **kwargs: ["words"])
    monkeypatch.setattr(
        voice_menu_shared.Transcriber,
        "get_flat_text_filtered_by_probability",
        lambda words, min_probability: "transcribed text",
    )
    state = _validate_state(tmp_path, "glm_local", [{"file_name": "a.flac", "transcript": ""}])
    assert VoiceMenuShared.validate_voices(state) is True
    assert state.project.voice_references == [{"file_name": "a.flac", "transcript": "transcribed text"}]
    saved = json.loads((tmp_path / "project.json").read_text())
    assert saved["voice_references"] == [{"file_name": "a.flac", "transcript": "transcribed text"}]
    assert not capture.errors


def test_validate_voices_transcription_failure_blocks_launch_and_does_not_save(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    (tmp_path / "a.flac").write_bytes(b"valid")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    monkeypatch.setattr(voice_menu_shared.Transcriber, "transcribe_to_words", lambda *args, **kwargs: "stt broke")
    state = _validate_state(tmp_path, "glm_local", [{"file_name": "a.flac", "transcript": ""}])
    assert VoiceMenuShared.validate_voices(state) is False
    assert any("Voice file a.flac could not be transcribed" in line for line in capture.lines)
    assert capture.errors == ["Replace problem voice clone file"]
    assert state.project.voice_references == [{"file_name": "a.flac", "transcript": ""}]
    assert not (tmp_path / "project.json").exists()


def test_validate_voices_skips_stt_for_model_without_transcripts(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    (tmp_path / "a.flac").write_bytes(b"valid")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    monkeypatch.setattr(voice_menu_shared.Transcriber, "transcribe_to_words", lambda *_args, **_kwargs: pytest.fail("must not transcribe"))
    state = _validate_state(tmp_path, "pocket_local", [{"file_name": "a.flac", "transcript": ""}])
    assert VoiceMenuShared.validate_voices(state) is True
    assert not capture.lines and not capture.errors


class TestCropVoiceSampleMenu:

    def _make_state(self, tmp_path, entries):
        voice_dir = tmp_path / "voice"
        voice_dir.mkdir(parents=True, exist_ok=True)
        from tts_audiobook_tool.app_types import Sound
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
        import numpy as np
        for entry in entries:
            if "crop_file_name" in entry:
                (voice_dir / ProjectVoiceUtil.get_cropped_voice_relative_path(entry)).parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(0)
            data = (rng.uniform(0.01, 0.02, 24000 * 8)).astype(np.float32)
            assert SoundFileUtil.save_flac(Sound(data, 24000), str(voice_dir / entry["file_name"])) == ""
        project = Project(tts_model_type="glm_local", voice_references=list(entries))
        project.dir_path = str(tmp_path)
        return cast(State, SimpleNamespace(project=project)), voice_dir

    def _patch_asks(self, monkeypatch, answers, errors=None):
        prompts = []
        monkeypatch.setattr(
            voice_menu_shared.ask, "ask_input",
            lambda message="", prefill="", **_: (prompts.append((message, prefill)) or answers.pop(0)),
        )
        captured_errors = []
        monkeypatch.setattr(
            voice_menu_shared.ask, "ask_error", lambda msg: captured_errors.append(msg)
        )
        return prompts, captured_errors

    def test_item_present_before_transcript_item(self):
        project = Project(tts_model_type="glm_local", voice_references=[
            {"file_name": "a.flac", "transcript": "t"},
        ])
        state = cast(State, SimpleNamespace(project=project))
        items = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())
        labels = [get_string_from(state, item.label) for item in items]
        assert voice_menu_shared.LABEL_CROP_VOICE_SAMPLE == "Trim or play voice sample"
        assert voice_menu_shared.LABEL_CROP_VOICE_SAMPLE in labels
        assert labels.index(voice_menu_shared.LABEL_CROP_VOICE_SAMPLE) < labels.index(
            voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION
        )

    def _patch_crop_app(self, monkeypatch, result):
        from tts_audiobook_tool.textual import voice_crop_app
        launched = []
        monkeypatch.setattr(
            voice_crop_app, "run_voice_crop_app",
            lambda sound, title, initial: launched.append((sound, title, initial)) or result,
        )
        return launched

    def _make_state_with_duration(self, tmp_path, seconds):
        voice_dir = tmp_path / "voice"
        voice_dir.mkdir(parents=True, exist_ok=True)
        from tts_audiobook_tool.app_types import Sound
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
        import numpy as np
        data = (np.random.default_rng(0).uniform(0.01, 0.02, int(24000 * seconds))).astype(np.float32)
        assert SoundFileUtil.save_flac(Sound(data, 24000), str(voice_dir / "a.flac")) == ""
        project = Project(tts_model_type="glm_local", voice_references=[{"file_name": "a.flac", "transcript": "t"}])
        project.dir_path = str(tmp_path)
        return cast(State, SimpleNamespace(project=project))

    @pytest.mark.parametrize("seconds", [0.5, 1.95, 2.0, 8.0])
    def test_samples_of_any_duration_launch_editor(self, tmp_path, monkeypatch, seconds):
        state = self._make_state_with_duration(tmp_path, seconds=seconds)
        launched = self._patch_crop_app(monkeypatch, SimpleNamespace(saved=False, cleared=False))
        errors = []
        monkeypatch.setattr(voice_menu_shared.ask, "ask_error", errors.append)
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        assert len(launched) == 1
        assert launched[0][0].duration == pytest.approx(seconds)
        assert not errors

    def test_save_flow_applies_crop_and_transcribes(self, tmp_path, monkeypatch):
        feedback = []
        monkeypatch.setattr(voice_menu_shared, "print_feedback", feedback.append)
        state, voice_dir = self._make_state(tmp_path, [{"file_name": "a.flac", "transcript": "full"}])
        launched = self._patch_crop_app(monkeypatch, SimpleNamespace(saved=True, cleared=False, start_s=1.0, end_s=4.5))
        transcribed = []
        monkeypatch.setattr(
            VoiceMenuShared, "transcribe_voice_sample_to_text",
            lambda state, sound: transcribed.append(sound) or ("auto transcript", ""),
        )
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        assert len(launched) == 1
        sound, title, initial = launched[0]
        assert title == "a"
        assert initial is None  # no existing crop
        entry = state.project.voice_references[0]
        assert entry["crop_file_name"].endswith(".flac")
        assert "/" not in entry["crop_file_name"] and "\\" not in entry["crop_file_name"]
        assert Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, entry)).is_file()
        assert len(transcribed) == 1  # the cropped span, not the original
        assert entry["crop_start"] == "1.0"
        assert entry["crop_end"] == "4.5"
        assert entry["crop_transcript"] == "auto transcript"
        assert feedback == ["Trim saved: 1s - 4.5s (used for generation)"]

    def test_existing_crop_is_passed_as_initial(self, tmp_path, monkeypatch):
        state, _ = self._make_state(tmp_path, [{
            "file_name": "a.flac", "transcript": "t",
            "crop_file_name": "a.flac",
            "crop_start": "0.25", "crop_end": "4.0", "crop_transcript": "cut",
        }])
        launched = self._patch_crop_app(monkeypatch, SimpleNamespace(saved=False, cleared=False))
        monkeypatch.setattr(
            VoiceMenuShared, "transcribe_voice_sample_to_text", lambda state, sound: ("x", "")
        )
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        assert launched[0][2] == (0.25, 4.0)

    def test_cleared_result_reverts_existing_crop(self, tmp_path, monkeypatch):
        feedback = []
        monkeypatch.setattr(voice_menu_shared, "print_feedback", feedback.append)
        state, voice_dir = self._make_state(tmp_path, [{
            "file_name": "a.flac", "transcript": "t",
            "crop_file_name": "a.flac",
            "crop_start": "0.25", "crop_end": "4.0", "crop_transcript": "cut",
        }])
        crop_path = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(
            state.project, state.project.voice_references[0],
        ))
        crop_path.write_bytes(b"x")
        self._patch_crop_app(monkeypatch, SimpleNamespace(saved=False, cleared=True))
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        assert not crop_path.exists()
        assert state.project.voice_references == [{"file_name": "a.flac", "transcript": "t"}]
        assert feedback == ["Trim reset; using original sample a.flac"]

    def test_cleared_without_crop_is_a_noop(self, tmp_path, monkeypatch):
        feedback = []
        monkeypatch.setattr(voice_menu_shared, "print_feedback", feedback.append)
        state, _ = self._make_state(tmp_path, [{"file_name": "a.flac", "transcript": "t"}])
        self._patch_crop_app(monkeypatch, SimpleNamespace(saved=False, cleared=True))
        before = list(state.project.voice_references)
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        assert state.project.voice_references == before
        assert feedback == ["No trim to reset"]

    def test_unchanged_save_is_a_noop(self, tmp_path, monkeypatch):
        state, voice_dir = self._make_state(tmp_path, [{
            "file_name": "a.flac", "transcript": "t",
            "crop_file_name": "a.flac",
            "crop_start": "1.0", "crop_end": "4.0", "crop_transcript": "cut",
        }])
        # The app reports saved=False when values did not change.
        self._patch_crop_app(monkeypatch, SimpleNamespace(saved=False, cleared=False))
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        entry = state.project.voice_references[0]
        assert not Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, entry)).exists()
        assert entry["crop_start"] == "1.0"

    def test_reversed_adjustment_preserves_edited_crop_transcript(self, tmp_path, monkeypatch):
        import asyncio
        from tts_audiobook_tool.textual import voice_crop_app

        state, voice_dir = self._make_state(tmp_path, [{"file_name": "a.flac", "transcript": "full"}])
        assert voice_menu_shared.ProjectVoiceUtil.apply_voice_crop_and_save(
            state.project, 0, 0.1, 5.0, "Manually corrected crop transcript",
        ) == ""
        previous_entry = dict(state.project.voice_references[0])
        crop_path = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, previous_entry))
        previous_audio = crop_path.read_bytes()
        previous_settings = (tmp_path / "project.json").read_bytes()

        def run_editor(sound, title, initial):
            app = voice_crop_app.VoiceCropApp(sound, title, initial)

            async def exercise():
                async with app.run_test(size=(80, 24)) as pilot:
                    # Reversing a coarse step crosses a truncation boundary:
                    # 0.1 -> 0.6 -> 0.09999999999999998.
                    await pilot.press("S", "A", "enter")
            asyncio.run(exercise())
            assert app.return_value is not None
            return app.return_value

        monkeypatch.setattr(voice_crop_app, "run_voice_crop_app", run_editor)
        monkeypatch.setattr(
            VoiceMenuShared, "transcribe_voice_sample_to_text",
            lambda *_: pytest.fail("Unchanged audio must not be retranscribed"),
        )
        with patch.object(voice_menu_shared.ProjectVoiceUtil, "apply_voice_crop_and_save") as apply:
            VoiceMenuShared.crop_voice_sample_from_menu(state)
        apply.assert_not_called()
        assert state.project.voice_references[0] == previous_entry
        assert crop_path.read_bytes() == previous_audio
        assert (tmp_path / "project.json").read_bytes() == previous_settings

    def test_multi_sample_selection(self, tmp_path, monkeypatch):
        state, voice_dir = self._make_state(tmp_path, [
            {"file_name": "a.flac", "transcript": "A"},
            {"file_name": "b.flac", "transcript": "B"},
        ])
        positions = []
        monkeypatch.setattr(
            VoiceMenuShared, "ask_voice_sample_position",
            lambda prompt, count: positions.append(prompt) or 1,
        )
        launched = self._patch_crop_app(monkeypatch, SimpleNamespace(saved=True, cleared=False, start_s=0.25, end_s=4.0))
        monkeypatch.setattr(
            VoiceMenuShared, "transcribe_voice_sample_to_text", lambda state, sound: ("auto transcript", "")
        )
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        assert positions and "trim" in positions[0]
        assert launched[0][1] == "b"
        entry = state.project.voice_references[1]
        assert entry["crop_file_name"].endswith(".flac")
        assert "/" not in entry["crop_file_name"] and "\\" not in entry["crop_file_name"]
        assert Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, entry)).is_file()
        assert state.project.voice_references[0] == {"file_name": "a.flac", "transcript": "A"}
        assert entry["crop_start"] == "0.25"

    def test_app_error_is_reported(self, tmp_path, monkeypatch):
        state, _ = self._make_state(tmp_path, [{"file_name": "a.flac", "transcript": "t"}])
        self._patch_crop_app(monkeypatch, "terminal does not support")
        errors = []
        monkeypatch.setattr(voice_menu_shared.ask, "ask_error", errors.append)
        VoiceMenuShared.crop_voice_sample_from_menu(state)
        assert errors == ["terminal does not support"]

    def test_transcript_edit_targets_cropped_span_when_crop_active(self, tmp_path, monkeypatch):
        state, voice_dir = self._make_state(tmp_path, [{
            "file_name": "a.flac", "transcript": "full text",
            "crop_file_name": "a.flac",
            "crop_start": "1.0", "crop_end": "4.0", "crop_transcript": "cut text",
        }])
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
        import numpy as np
        from tts_audiobook_tool.app_types import Sound
        crop_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, state.project.voice_references[0])
        assert SoundFileUtil.save_flac(Sound(np.zeros(24000 * 3, dtype=np.float32), 24000), crop_path) == ""
        prompts = []
        monkeypatch.setattr(
            voice_menu_shared.ask, "ask_input",
            lambda message="", prefill="", **_: (prompts.append((message, prefill)) or "cut text 2"),
        )
        lines = []
        monkeypatch.setattr(voice_menu_shared, "printt", lambda s="": lines.append(s))
        VoiceMenuShared.edit_voice_sample_transcript(state)
        entry = state.project.voice_references[0]
        assert entry["crop_transcript"] == "cut text 2"
        assert entry["transcript"] == "full text"  # original untouched
        assert prompts[0][1] == "cut text"  # prefilled with the crop transcript
        assert "trimmed span" in prompts[0][0]

    def test_transcript_edit_falls_back_to_original_when_cropped_file_missing(self, tmp_path, monkeypatch):
        state, _ = self._make_state(tmp_path, [{
            "file_name": "a.flac", "transcript": "full text",
            "crop_file_name": "a.flac",
            "crop_start": "1.0", "crop_end": "4.0", "crop_transcript": "cut text",
        }])
        # No crops/a.flac written: the funnel falls back to the original pair.
        prompts = []
        monkeypatch.setattr(
            voice_menu_shared.ask, "ask_input",
            lambda message="", prefill="", **_: (prompts.append((message, prefill)) or "edited full"),
        )
        VoiceMenuShared.edit_voice_sample_transcript(state)
        entry = state.project.voice_references[0]
        assert entry["transcript"] == "edited full"
        assert entry["crop_transcript"] == "cut text"  # crop transcript untouched
        assert prompts[0][1] == "full text"
