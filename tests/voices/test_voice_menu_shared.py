import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types.voice import Voice, VoiceCrop
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppStylesheetFailed,
    EditorSaveFailed,
)
from textual_editor_stubs import StubPhraseGroup, StubProject


def capture_menus(monkeypatch) -> list[dict]:
    """Record the kwargs of every MenuUtil.menu call (main menu first, submenus after)."""
    menus: list[dict] = []
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: menus.append(kwargs))
    return menus


def find_item(state, items, label) -> MenuItem:
    return next(item for item in items if get_string_from(state, item.label) == label)


@pytest.mark.parametrize("model_id", ["none", "unknown-model"])
def test_remote_voice_menu_removes_controls_when_selection_is_unresolved(monkeypatch, model_id):
    state = cast(State, SimpleNamespace(project=Project(
        tts_model_type="omnivoice_audiocpp",
        voice_references=[{"file_name": "existing.flac", "transcript": "text"}],
    )))
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    VoiceMenuShared.menu(state)
    assert captured["items"](state)
    state.project.tts_model_type = model_id
    assert captured["items"](state) == []
    assert captured["subheading"] is None


def test_remote_voice_menu_tracks_selection_on_redraw_and_edit(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="omnivoice_audiocpp")))
    state.project.set_model_setting("omnivoice_audiocpp", "file_name", ["old.flac"])
    captured = {}
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    VoiceMenuShared.menu(state)
    assert captured["subheading"] is None
    assert "old" in get_string_from(state, captured["items"](state)[0].label)

    state.project.tts_model_type = "breeze_tts_2_audiocpp"
    state.project.set_model_setting("breeze_tts_2_audiocpp", "file_name", ["new.flac"])
    items = captured["items"](state)
    label = get_string_from(state, items[0].label)
    assert "new" in label and "old" not in label
    calls = []
    monkeypatch.setattr(voice_menu_shared.Project, "save", lambda _: "")
    monkeypatch.setattr(VoiceMenuShared, "voice_sample_submenu",
                        lambda current, model, index, _on_clear, **_kwargs: calls.append((model.id, index)))
    items[1].handler(state, items[1])  # Add: a default voice, then its submenu
    items[0].handler(state, items[0])
    assert calls == [("breeze_tts_2_audiocpp", 1), ("breeze_tts_2_audiocpp", 0)]
    assert state.project.voice_references[1] == Voice()
    state.project.tts_model_type = "none"
    assert captured["items"](state) == []


@pytest.mark.parametrize("model_id", [
    "mira_local", "dots_local", "chatterbox_audiocpp",
    "omnivoice_audiocpp", "zonos2_sglomni", "higgs_v3_sglomni",
])
@pytest.mark.parametrize("voice_count", [1, 2])
def test_transcript_editing_is_available_without_changing_transcripts(monkeypatch, model_id, voice_count):
    project = Project(tts_model_type=model_id, voice_references=[
        {"file_name": f"voice-{i}.flac", "transcript": "Retained reference text"}
        for i in range(voice_count)
    ])
    state = cast(State, SimpleNamespace(project=project))
    menus = capture_menus(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    items = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())
    for item in items[:voice_count]:
        item.handler(state, item)  # opens that sample's submenu
        labels = [get_string_from(state, sub.label) for sub in menus[-1]["items"](state)]
        assert any(label.startswith(voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION) for label in labels)
    assert len(menus) == voice_count
    assert all(entry.transcript == "Retained reference text" for entry in project.voice_references)


def test_remote_transcript_editing_is_available_on_redraw(monkeypatch):
    project = Project(tts_model_type="omnivoice_audiocpp", voice_references=[
        {"file_name": "voice.flac", "transcript": "Retained reference text"},
    ])
    state = cast(State, SimpleNamespace(project=project))
    menus = capture_menus(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    VoiceMenuShared.menu(state)
    make_main_items = menus[0]["items"]
    for model_id in [
        "omnivoice_audiocpp", "chatterbox_audiocpp", "zonos2_sglomni", "higgs_v3_sglomni",
    ]:
        project.tts_model_type = model_id
        voice_item = make_main_items(state)[0]
        voice_item.handler(state, voice_item)
        labels = [get_string_from(state, item.label) for item in menus[-1]["items"](state)]
        assert any(label.startswith(voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION) for label in labels)
        assert project.voice_references[0].transcript == "Retained reference text"


def test_voice_menu_direct_actions_refresh_samples_and_preserve_callbacks(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="dots_local")))
    model = state.project.get_tts_model_type()
    menus = capture_menus(monkeypatch)
    events = []
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 1.25)
    monkeypatch.setattr(voice_menu_shared.Project, "save", lambda _: "")

    def set_sample(current, selected, index):
        assert selected == model
        events.append(f"set {index}")
        current.project.voice_references[index] = Voice("new.flac", "text")

    def remove_sample(current, selected, index):
        assert selected == model and index == 0
        events.append("remove")
        ProjectVoiceUtil.remove_voice_at_index_and_save(current.project, selected, index)
        return True

    monkeypatch.setattr(VoiceMenuShared, "ask_and_set_voice_sample", set_sample)
    monkeypatch.setattr(VoiceMenuShared, "remove_voice_sample_from_menu", remove_sample)

    def make_items(current):
        return VoiceMenuShared.make_voice_sample_items(
            current, model,
            on_before_set_callback=lambda: events.append("before"),
            on_set_callback=lambda: events.append("after"),
            on_clear_callback=lambda: events.append("clear"),
        )

    VoiceMenuShared.menu_wrapper(state, make_items, subheading=lambda _: "Sample guidance")
    main = menus[0]
    items = main["items"](state)
    # A new project has one sample-less voice, then Add.
    assert get_string_from(state, items[0].label).startswith("Voice 1: ")
    assert get_string_from(state, items[1].label) == voice_menu_shared.LABEL_ADD_VOICE_SAMPLE
    assert get_string_from(state, main["subheading"]) == "Sample guidance"
    assert items[0].handler(state, items[0]) is False  # opens the voice's submenu; the menu stays
    sub_items = menus[-1]["items"](state)
    set_item = next(
        item for item in sub_items
        if get_string_from(state, item.label).startswith(voice_menu_shared.LABEL_SET_VOICE_SAMPLE)
    )
    assert get_string_from(state, set_item.label).startswith(voice_menu_shared.LABEL_SET_VOICE_SAMPLE)
    labels = [get_string_from(state, item.label) for item in sub_items]
    assert not any(label.startswith(voice_menu_shared.LABEL_CROP_VOICE_SAMPLE) for label in labels)
    assert not any(label.startswith(voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION) for label in labels)
    set_item.handler(state, set_item)
    assert events == ["before", "set 0", "after"]
    items = main["items"](state)
    label = get_string_from(state, items[0].label)
    assert "Voice 1:" in label and "new" in label and "(1.2s)" in label
    assert items[0].data == 0
    # The trim and transcript items appear once a sample is set.
    labels = [get_string_from(state, item.label) for item in menus[-1]["items"](state)]
    assert any(label.startswith(voice_menu_shared.LABEL_CROP_VOICE_SAMPLE) for label in labels)
    remove = find_item(state, menus[-1]["items"](state), voice_menu_shared.LABEL_RESET_VOICE_SAMPLE)
    assert remove.handler(state, remove) is True  # exits the voice submenu, not Voice clone
    # The only voice was reset: one default voice, Add, voice/line selections
    # and auto-assign.
    assert state.project.voice_references == [Voice()]
    assert len(main["items"](state)) == 4
    assert events == ["before", "set 0", "after", "remove", "clear"]
    # Removing a voice that has no sample must not unload a model again.
    assert remove.handler(state, remove) is True
    assert events.count("clear") == 1


def test_add_appends_default_voice_and_opens_its_submenu(monkeypatch):
    project = Project(tts_model_type="mira_local", voice_references=[
        {"file_name": "existing.flac", "transcript": "text", "pan": 0.5},
    ])
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    monkeypatch.setattr(voice_menu_shared.Project, "save", lambda _: "")
    monkeypatch.setattr(VoiceMenuShared, "ask_voice_file", lambda *_: pytest.fail("Add must not ask for a file"))
    opened = []
    monkeypatch.setattr(VoiceMenuShared, "voice_sample_submenu",
                        lambda _state, _model, index, *_args, **_kwargs: opened.append(index))
    items = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())
    add = find_item(state, items, voice_menu_shared.LABEL_ADD_VOICE_SAMPLE)
    add.handler(state, add)
    assert project.voice_references[1] == Voice()
    assert opened == [1]


def test_voice_list_pages_through_more_voices_with_add_on_last_page(monkeypatch):
    # Past a page's worth of voices, "More voices..." replaces Add and opens
    # the next page, which lists only voices (no model items or selection
    # controls); Add follows the last page's voices.
    project = Project(tts_model_type="mira_local", voice_references=[{"file_name": ""}] * 5)
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared, "VOICES_PER_PAGE", 2)
    menus = capture_menus(monkeypatch)

    def labels(items) -> list[str]:
        return [strip_ansi_codes(get_string_from(state, item.label)) for item in items]

    items = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())
    assert [label.split(":")[0] for label in labels(items)] == ["Voice 1", "Voice 2", "More voices... (3)"]
    items[-1].handler(state, items[-1])
    page_2 = menus[-1]["items"](state)
    assert [label.split(":")[0] for label in labels(page_2)] == ["Voice 3", "Voice 4", "More voices... (1)"]
    assert strip_ansi_codes(get_string_from(state, menus[-1]["heading"])) == "Voices (3-4 of 5)"
    assert get_string_from(state, menus[-1]["breadcrumb"]) == "Voices 3-4"
    page_2[-1].handler(state, page_2[-1])
    page_3 = menus[-1]["items"](state)
    assert [label.split(":")[0] for label in labels(page_3)] == [
        "Voice 5", voice_menu_shared.LABEL_ADD_VOICE_SAMPLE, voice_menu_shared.LABEL_CLEAR_VOICES]


def make_clear_state(monkeypatch, voice_references):
    project = Project(tts_model_type="mira_local", voice_references=voice_references)
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    monkeypatch.setattr(voice_menu_shared, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(voice_menu_shared, "print_feedback", lambda *args, **kwargs: None)
    return state


def test_clear_resets_voice_list_after_confirmation(monkeypatch):
    # Clear follows Add; once confirmed, the list is one default voice (so
    # Clear is no longer offered) and the clear callback runs since a voice
    # had a sample. Declining changes nothing.
    state = make_clear_state(monkeypatch, [
        {"file_name": "a.flac", "transcript": "text"}, {"file_name": "", "name": "bob"}])
    monkeypatch.setattr(voice_menu_shared.Project, "save", lambda _: "")
    events = []
    make_items = lambda: VoiceMenuShared.make_voice_sample_items(
        state, state.project.get_tts_model_type(), on_clear_callback=lambda: events.append("clear"))
    clear = make_items()[-1]
    assert get_string_from(state, clear.label) == voice_menu_shared.LABEL_CLEAR_VOICES

    monkeypatch.setattr(voice_menu_shared.ask, "ask_confirm", lambda *_: False)
    assert clear.handler(state, clear) is False
    assert len(state.project.voice_references) == 2

    monkeypatch.setattr(voice_menu_shared.ask, "ask_confirm", lambda *_: True)
    assert clear.handler(state, clear) is False  # The first page stays open
    assert state.project.voice_references == [Voice()]
    assert events == ["clear"]
    assert get_string_from(state, make_items()[-1].label) == voice_menu_shared.LABEL_ADD_VOICE_SAMPLE


def test_clear_restores_voice_list_when_saving_fails(monkeypatch):
    # In-memory voices must match the project file after a failed save.
    state = make_clear_state(monkeypatch, [{"file_name": "a.flac", "transcript": "text"}])
    original = list(state.project.voice_references)
    errors = []
    monkeypatch.setattr(voice_menu_shared.Project, "save", lambda _: "disk full")
    monkeypatch.setattr(voice_menu_shared.ask, "ask_confirm", lambda *_: True)
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", errors.append)

    assert VoiceMenuShared.clear_voices_from_menu(state) is False
    assert state.project.voice_references == original
    assert errors == ["disk full"]


def test_clear_on_a_later_page_closes_it_and_the_pages_before(monkeypatch):
    # After Clear on page 3, pages 2 and 3 are empty, so both close; page 1 stays.
    state = make_clear_state(monkeypatch, [{"file_name": "", "name": f"v{i}"} for i in range(5)])
    monkeypatch.setattr(voice_menu_shared, "VOICES_PER_PAGE", 2)
    monkeypatch.setattr(voice_menu_shared.Project, "save", lambda _: "")
    monkeypatch.setattr(voice_menu_shared.ask, "ask_confirm", lambda *_: True)
    results = []

    def run_page(**kwargs):
        # Page 2 opens page 3 via "More voices..."; page 3 selects Clear
        s = kwargs["state"]
        items = kwargs["items"](s)
        results.append(items[-1].handler(s, items[-1]))
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", run_page)

    items = VoiceMenuShared.make_voice_sample_items(state, state.project.get_tts_model_type())
    more = items[-1]
    assert more.handler(state, more) is False
    assert results == [True, True]  # Page 3's Clear, then page 2's "More voices..."
    assert state.project.voice_references == [Voice()]


def test_later_voice_page_closes_when_removals_empty_it(monkeypatch):
    # Pages are recomputed on redraw; a page whose voices were all removed
    # (from a voice's submenu) closes instead of showing an empty list.
    project = Project(tts_model_type="mira_local", voice_references=[{"file_name": ""}] * 3)
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared, "VOICES_PER_PAGE", 2)
    menus = capture_menus(monkeypatch)
    monkeypatch.setattr(VoiceMenuShared, "voice_sample_submenu",
                        lambda s, *_args, **_kwargs: setattr(s.project, "voice_references",
                                                             s.project.voice_references[:-1]))
    items = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())
    items[-1].handler(state, items[-1])
    voice_3 = menus[-1]["items"](state)[0]
    assert voice_3.handler(state, voice_3) is True  # Voice 3 removed: page 2 is empty
    assert items[0].handler(state, items[0]) is False  # The first page never closes this way


@pytest.mark.parametrize("voice_count", [0, 1])
def test_voice_menu_canceled_actions_keep_samples_and_clear_callback(monkeypatch, voice_count):
    project = Project(tts_model_type="mira_local", voice_references=[
        {"file_name": "existing.flac", "transcript": "text"} for _ in range(voice_count)
    ])
    state = cast(State, SimpleNamespace(project=project))
    original = list(project.voice_references)
    events = []
    menus = capture_menus(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    monkeypatch.setattr(VoiceMenuShared, "ask_voice_file", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(voice_menu_shared.hints, "show_hint_if_necessary", lambda *_, **__: None)
    monkeypatch.setattr(VoiceMenuShared, "remove_voice_sample_from_menu", lambda *_: False)
    monkeypatch.setattr(voice_menu_shared.Project, "save", lambda _: pytest.fail("must not save"))
    state.prefs = SimpleNamespace(last_voice_dir="", save=lambda: None)
    items = VoiceMenuShared.make_voice_sample_items(
        state, project.get_tts_model_type(),
        on_before_set_callback=lambda: events.append("before"),
        on_set_callback=lambda: events.append("after"),
        on_clear_callback=lambda: events.append("clear"),
    )
    items[0].handler(state, items[0])
    sub_items = menus[-1]["items"](state)
    set_item = next(
        item for item in sub_items
        if get_string_from(state, item.label).startswith(voice_menu_shared.LABEL_SET_VOICE_SAMPLE)
    )
    set_item.handler(state, set_item)  # Set voice sample, canceled at the file prompt
    remove_label = voice_menu_shared.LABEL_RESET_VOICE_SAMPLE
    remove = find_item(state, sub_items, remove_label)
    assert remove.handler(state, remove) is False  # canceled: stay in the submenu
    assert project.voice_references == original
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
        SimpleNamespace(
            project=StubProject([StubPhraseGroup("Line 1")]),
            editor_view_anchors={},
        ),
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
        lambda _app, _anchors: run_result,
    )
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", feedback_calls.append)

    voice_menu_shared.VoiceMenuShared.assign_voice_samples_to_text_lines(state)

    assert feedback_calls == [expected_message]

# ---------------------------------------------------------------------------
# Voice sample selection mode menu items (moved from test_voice_selection.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("voice_count", [1, 2])
def test_voice_sample_items_list_each_sample(monkeypatch, voice_count):
    project = Project(tts_model_type="mira_local", voice_references=[
        {"file_name": f"voice-{index}.flac", "transcript": ""} for index in range(voice_count)
    ])
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared.ProjectVoiceUtil, "make_voice_sample_display_label", lambda _, name, __: name)
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)

    items = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id("mira_local"))

    for index, item in enumerate(items[:voice_count]):
        label = get_string_from(state, item.label)
        assert f"Voice {index + 1}:" in label and f"voice-{index}.flac" in label
        assert "\n" not in label
        assert item.data == index
        assert item.superlabel == (voice_menu_shared.LABEL_VOICE_SAMPLES_SUPERLABEL if index == 0 else "")
        assert item.sublabel is None
    assert get_string_from(state, items[voice_count].label).startswith(voice_menu_shared.LABEL_ADD_VOICE_SAMPLE)


@pytest.mark.parametrize("model_id", ["omnivoice_local", "echo_tts_audiocpp", "omnivoice_audiocpp"])
def test_voice_sample_items_mark_only_long_samples_with_one_footnote(monkeypatch, model_id):
    project = Project(tts_model_type=model_id, voice_references=[
        {"file_name": "long.flac", "transcript": "text"},
        {"file_name": "short.flac", "transcript": "text"},
        {"file_name": "also-long.flac", "transcript": "text"},
    ])
    state = cast(State, SimpleNamespace(project=project))
    durations = iter([15.1, 14.6, 16.0])
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: next(durations))

    items = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())

    m = voice_menu_shared
    marker = f"{m.COL_ERROR}*{m.COL_DIM}"
    labels = [get_string_from(state, item.label) for item in items[:3]]
    assert labels[0].endswith(f"(15.1s{marker}){m.COL_DEFAULT}")
    assert labels[1].endswith(f"(14.6s){m.COL_DEFAULT}")
    assert labels[2].endswith(f"(16.0s{marker}){m.COL_DEFAULT}")
    assert all(item.sublabel is None for item in items)


@pytest.mark.parametrize("duration, warned", [(14.6, False), (15.0, False), (15.1, True), (15.01, True), (None, False)])
def test_voice_sample_duration_warning_uses_unrounded_duration(monkeypatch, duration, warned):
    project = Project(tts_model_type="echo_tts_audiocpp", voice_references=[
        {"file_name": "voice.flac", "transcript": ""},
    ])
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: duration)

    label, is_long = VoiceMenuShared.make_voice_sample_item_label(project, project.get_tts_model_type(), 0)
    item = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())[0]

    assert is_long == warned
    assert item.sublabel is None
    assert (f"{voice_menu_shared.COL_ERROR}*" in label) == warned
    assert ("(" in label) == (duration is not None)


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


def test_voice_sample_item_reports_cropped_duration(tmp_path, monkeypatch):
    """The saved crop drives the shown duration and the warning."""
    project = make_crop_project(tmp_path)
    state = cast(State, SimpleNamespace(project=project))
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

    item = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())[0]
    label = get_string_from(state, item.label)

    assert "(trimmed, 9.9s)" in label
    assert "11.1572" not in label
    assert "*" not in label
    assert item.sublabel is None


def test_voice_sample_item_falls_back_when_cropped_file_missing(tmp_path, monkeypatch):
    """No cropped file on disk means generation uses the original, so the warning applies."""
    project = make_crop_project(tmp_path)
    state = cast(State, SimpleNamespace(project=project))
    durations = {"long.flac": 11.1572}
    monkeypatch.setattr(
        voice_menu_shared.AudioMetaUtil,
        "get_audio_duration",
        lambda path: durations[Path(path).name],
    )

    item = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())[0]
    label = get_string_from(state, item.label)

    assert f"(trimmed, 11.2s{voice_menu_shared.COL_ERROR}*{voice_menu_shared.COL_DIM})" in label
    assert item.sublabel is None


@pytest.mark.parametrize("model_id", ["chatterbox_local", "chatterbox_audiocpp", "higgs_v3_sglomni"])
def test_voice_samples_without_duration_recommendation_have_no_warning(monkeypatch, model_id):
    project = Project(tts_model_type=model_id, voice_references=[
        {"file_name": "voice.flac", "transcript": ""},
    ])
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 120.0)

    item = VoiceMenuShared.make_voice_sample_items(state, project.get_tts_model_type())[0]
    label = get_string_from(state, item.label)

    assert "Voice 1:" in label
    assert "*" not in label
    assert item.sublabel is None


def test_voice_item_label_includes_current_model_instructions(monkeypatch):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "", "instructions": {"omnivoice_local": "only instructions"}},
    ])
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 15.3)
    tts_type = project.get_tts_model_type()

    def plain(index: int) -> str:
        label, _ = VoiceMenuShared.make_voice_sample_item_label(project, tts_type, index)
        return re.sub(r"\x1b\[[0-9;]*m|\x1b\]8;[^\x1b]*\x1b\\", "", label)

    # Instructions-only voice (no sample file), and no voice has a sample
    assert plain(0) == 'Voice 1: "only instructions"'
    project.voice_references.insert(0, Voice(
        "voice.flac", "text",
        instructions={"omnivoice_local": "some  voice\ninstruction", "other_model": "ignored"},
    ))
    assert plain(0) == 'Voice 1: voice (15.3s*); "some voice instruction"'
    # OmniVoice does not require a voice clone sample, so a voice without
    # one is fine even when another voice has one
    assert plain(1) == 'Voice 2: "only instructions"'
    # Neither a sample file nor instructions for the current model
    project.voice_references.append(Voice("", instructions={"other_model": "ignored"}))
    assert plain(2) == "Voice 3: no sample set"
    label, _ = VoiceMenuShared.make_voice_sample_item_label(project, tts_type, 2)
    assert label == "Voice 3: no sample set"  # Inherit the menu item's normal colour


def test_voice_item_label_hides_instructions_when_mode_does_not_use_them(monkeypatch):
    model_id = "cosyvoice3_audiocpp"
    project = Project(tts_model_type=model_id, voice_references=[
        Voice("voice.flac", "text", instructions={model_id: "Speak warmly"}),
    ])
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 5.0)
    tts_type = project.get_tts_model_type()

    def plain() -> str:
        label, _ = VoiceMenuShared.make_voice_sample_item_label(project, tts_type, 0)
        return re.sub(r"\x1b\[[0-9;]*m|\x1b\]8;[^\x1b]*\x1b\\", "", label)

    assert plain() == "Voice 1: voice (5.0s)"  # zero_shot default
    project.set_model_setting(model_id, "template_name", "instruct")
    assert plain() == 'Voice 1: voice (5.0s); "Speak warmly"'


@pytest.mark.parametrize("model_type, hinted", [
    ("voice_design", True), ("custom_voice", False), ("base", False), ("", False),
])
def test_voice_item_label_hints_missing_instructions_only_for_qwen_voice_design(model_type, hinted):
    model_id = "qwen3tts_local"
    project = Project(tts_model_type=model_id, voice_references=[
        Voice(instructions={"other_model": "ignored"}),
        Voice(instructions={model_id: "A calm voice"}),
    ])
    project.set_model_setting(model_id, "model_type", model_type)
    tts_type = project.get_tts_model_type()

    def label(index: int) -> str:
        return VoiceMenuShared.make_voice_sample_item_label(project, tts_type, index)[0]

    assert ("no instructions set" in label(0)) is hinted
    if hinted:
        assert re.sub(r"\x1b\[[0-9;]*m", "", label(0)) == "Voice 1: no instructions set"
        # Non-blocking, so not error-colored
        assert voice_menu_shared.COL_ERROR not in label(0)
    assert "no instructions" not in label(1)  # has instructions (previewed when supported)


@pytest.mark.parametrize("model_id, requirement", [
    ("glm_local", "required"),
    ("pocket_local", "voice sample or predefined voice required"),
])
def test_voice_item_no_sample_shows_requirement_when_required(model_id, requirement):
    project = Project(tts_model_type=model_id, voice_references=[Voice()])
    label, is_long = VoiceMenuShared.make_voice_sample_item_label(project, project.get_tts_model_type(), 0)

    assert label == (
        f"Voice 1: {voice_menu_shared.COL_ERROR}{requirement}{voice_menu_shared.COL_DEFAULT}"
    )
    assert not is_long


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
        label = get_string_from(state, captured["items"](state)[0].label)
        assert (f"{voice_menu_shared.COL_ERROR}*" in label) == warned


@pytest.mark.parametrize("voice_count, expected_item_count", [(0, 4), (1, 5), (2, 6), (9, 13)])
def test_voice_items_end_with_selection_items(
        monkeypatch,
        voice_count: int,
        expected_item_count: int,
) -> None:
    project = Project.model_validate({
        "mira_voice_file_name": [f"voice-{index}.flac" for index in range(voice_count)],
    })
    state = cast(State, SimpleNamespace(project=project))
    monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: None)

    mira = TtsModelType.require_by_id("mira_local")
    items = [*VoiceMenuShared.make_voice_sample_items(state, mira),
             *VoiceMenuShared.make_voice_selection_items(state, mira)]

    assert len(items) == expected_item_count
    labels = [get_string_from(state, item.label) for item in items]
    # No samples still means one (sample-less) voice, which has nothing to clear
    expected = [voice_menu_shared.LABEL_ADD_VOICE_SAMPLE]
    if voice_count:
        expected.append(voice_menu_shared.LABEL_CLEAR_VOICES)
    voice_count = max(voice_count, 1)
    for index, label in enumerate(labels[:voice_count]):
        assert f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} {index + 1}: " in label
    assert items[-2].blank_line_before
    expected.append(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    expected.append(voice_menu_shared.LABEL_AUTO_ASSIGN_CHARACTER_VOICES)
    assert len(labels[voice_count:]) == len(expected)
    assert all(label.startswith(prefix) for label, prefix in zip(labels[voice_count:], expected))
    assert not any(label.startswith("Add/remove") for label in labels)


@pytest.mark.parametrize("file_name", ["roster.json", "attributions.json"])
def test_auto_assign_item_is_optional_until_speaker_assign_data_exists(tmp_path, file_name) -> None:
    # The "(uses LLM; optional)" hint shows only while the project has no saved
    # roster or attributions; either file removes it
    state = cast(State, SimpleNamespace(project=SimpleNamespace(dir_path=str(tmp_path))))
    item = VoiceMenuShared.make_auto_assign_character_voices_item()

    label = strip_ansi_codes(get_string_from(state, item.label))
    assert label == f"{voice_menu_shared.LABEL_AUTO_ASSIGN_CHARACTER_VOICES} (uses LLM; optional)"

    (tmp_path / "speaker_assign").mkdir()
    (tmp_path / "speaker_assign" / file_name).write_text("{}", encoding="utf-8")
    assert get_string_from(state, item.label) == voice_menu_shared.LABEL_AUTO_ASSIGN_CHARACTER_VOICES


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
        self.pauses = []
        monkeypatch.setattr(voice_menu_shared, "printt", lambda message="": self.lines.append(message))
        monkeypatch.setattr(voice_menu_shared.ask, "ask_error", lambda message: self.errors.append(message))
        monkeypatch.setattr(voice_menu_shared.ask, "ask_enter_to_continue", lambda: self.pauses.append(None))


def _validate_state(tmp_path, model_id, entries, *, preset=False):
    project = Project(dir_path=str(tmp_path), tts_model_type=model_id, voice_references=entries)
    if preset:
        _set_presets(project, "pocket_local", "alba")
    prefs = SimpleNamespace(stt_variant=None, stt_config={})
    return cast(State, SimpleNamespace(project=project, prefs=prefs))


def _set_presets(project, model_id, preset):
    project.voice_references = [replace(voice, presets={model_id: preset}) for voice in project.voice_references]


def _validate_entry(file_name, transcript, *, cropped):
    if cropped:
        return Voice(file_name, "original text", VoiceCrop(f"trim-{file_name}", 0.0, 3.0, transcript))
    return Voice(file_name, transcript)


def _write_active_sample(tmp_path, entry):
    sample_path = tmp_path / entry.file_name
    if entry.crop is not None:
        sample_path = tmp_path / "voice" / "crops" / entry.crop.file_name
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(b"valid")
    return sample_path


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
@pytest.mark.parametrize("voice_count", [0, 1])
def test_validate_voices_qwen3_non_base_checkpoint_needs_no_samples(tmp_path, monkeypatch, model_type, voice_count):
    # Non-cloning checkpoints ignore even stale shared references in both modes.
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "qwen3tts_local", [
        {"file_name": "gone.flac", "transcript": ""} for _ in range(voice_count)
    ])
    state.project.set_model_setting("qwen3tts_local", "model_type", model_type)
    if model_type == "custom_voice":
        # CustomVoice needs a speaker per voice instead
        _set_presets(state.project, "qwen3tts_local", "Ryan")
    assert VoiceMenuShared.get_voice_problems(state) == []
    assert VoiceMenuShared.validate_voices(state) is True
    assert not capture.lines and not capture.errors


def test_validate_voices_qwen3_custom_voice_requires_a_speaker_per_voice(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "qwen3tts_local", [
        Voice("gone.flac", presets={"qwen3tts_local": "Ryan"}), Voice("gone.flac"),
    ])
    state.project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    assert VoiceMenuShared.get_voice_problems(state) == ["Voice 2 has no speaker (select a speaker)"]
    assert VoiceMenuShared.get_voice_problems(state, voice_indices=[0]) == []
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.pauses == [None]


def test_validate_voices_pocket_checks_only_voices_without_predefined_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "pocket_local", [
        Voice("gone.flac", presets={"pocket_local": "alba"}), Voice("gone.flac"), Voice(),
    ])
    assert VoiceMenuShared.get_voice_problems(state) == [
        "Voice file gone.flac not found",
        "Voice 3 has no voice sample (set a sample or select a predefined voice)",
    ]


@pytest.mark.parametrize("model_type", ["", "base"])
def test_validate_voices_qwen3_base_checkpoint_requires_samples(tmp_path, monkeypatch, model_type):
    # An unset checkpoint retains Base's required-clone rule.
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "qwen3tts_local", [])
    state.project.set_model_setting("qwen3tts_local", "model_type", model_type)
    assert VoiceMenuShared.get_voice_problems(state) == [
        "Voice 1 has no voice sample (set a sample or remove the voice)",
    ]
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.lines == [
        f"{voice_menu_shared.COL_ERROR}Voice 1 has no voice sample (set a sample or remove the voice){voice_menu_shared.COL_DEFAULT}",
        "",
    ]
    assert not capture.errors
    assert capture.pauses == [None]


def test_validate_voices_pocket_empty_message_mentions_predefined_voice(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "pocket_local", [], preset=False)
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.lines == [
        f"{voice_menu_shared.COL_ERROR}Voice 1 has no voice sample (set a sample or select a predefined voice){voice_menu_shared.COL_DEFAULT}",
        "",
    ]
    assert not capture.errors
    assert capture.pauses == [None]


def test_validate_voices_blocks_required_voice_with_empty_list(tmp_path, monkeypatch):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, "glm_local", [])
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.lines == [
        f"{voice_menu_shared.COL_ERROR}Voice 1 has no voice sample (set a sample or remove the voice){voice_menu_shared.COL_DEFAULT}",
        "",
    ]
    assert not capture.errors
    assert capture.pauses == [None]


@pytest.mark.parametrize("model_id, blocked", [("glm_local", True), ("dots_local", False)])
def test_validate_voices_reports_each_required_sample_less_voice(tmp_path, monkeypatch, model_id, blocked):
    capture = _ValidateCapture(monkeypatch)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, model_id, [Voice(pan=0.5), Voice()])
    expected = [
        f"Voice {index + 1} has no voice sample (set a sample or remove the voice)"
        for index in range(2)
    ] if blocked else []
    assert VoiceMenuShared.get_voice_problems(state) == expected
    assert VoiceMenuShared.validate_voices(state) is not blocked
    assert not capture.errors
    assert capture.lines == ([
        *(f"{voice_menu_shared.COL_ERROR}{problem}{voice_menu_shared.COL_DEFAULT}" for problem in expected),
        "",
    ] if blocked else [])
    assert capture.pauses == ([None] if blocked else [])


@pytest.mark.parametrize("model_id, blocked", [("glm_local", True), ("dots_local", False)])
def test_validate_voices_mixed_sample_less_voices_block_only_voice_requiring_models(
        tmp_path, monkeypatch, model_id, blocked):
    # A sample-less voice among sampled ones uses the model's built-in voice,
    # unless the model requires a voice clone sample.
    capture = _ValidateCapture(monkeypatch)
    (tmp_path / "a.flac").write_bytes(b"valid")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    state = _validate_state(tmp_path, model_id, [Voice("a.flac", "text"), Voice()])
    expected = ["Voice 2 has no voice sample (set a sample or remove the voice)"] if blocked else []
    assert VoiceMenuShared.get_voice_problems(state) == expected
    assert VoiceMenuShared.validate_voices(state) is not blocked
    if blocked:
        assert capture.lines[0] == f"{voice_menu_shared.COL_ERROR}{expected[0]}{voice_menu_shared.COL_DEFAULT}"
    else:
        assert not capture.lines and not capture.errors


@pytest.mark.parametrize("entries, model_id, expected", [
    # No sample anywhere: required only when the model requires a voice
    ([Voice()], "glm_local", ["required"]),
    ([Voice()], "dots_local", [""]),
    ([Voice()], "pocket_local", ["voice sample or predefined voice required"]),
    # Per voice: another voice's sample doesn't change this voice's requirement
    ([Voice("a.flac", "t"), Voice()], "pocket_local", ["", "voice sample or predefined voice required"]),
    # Mixed: sample-less voices are required only when the model requires a voice
    ([Voice("a.flac", "t"), Voice()], "glm_local", ["", "required"]),
    ([Voice("a.flac", "t"), Voice()], "dots_local", ["", ""]),
    ([Voice("a.flac", "t"), Voice()], "breeze_tts_2_audiocpp", ["", ""]),
])
def test_voice_sample_required_label(entries, model_id, expected):
    project = Project(tts_model_type=model_id, voice_references=entries)
    labels = [VoiceMenuShared.get_voice_sample_required_label(project, i) for i in range(len(entries))]
    assert labels == expected


def test_voice_sample_not_required_with_pocket_predefined_voice():
    project = Project(tts_model_type="pocket_local", voice_references=[
        Voice(presets={"pocket_local": "alba"}), Voice(),
    ])
    assert VoiceMenuShared.get_voice_sample_required_label(project, 0) == ""
    assert VoiceMenuShared.get_voice_sample_required_label(project, 1) != ""


@pytest.mark.parametrize("cropped", [False, True])
def test_validate_voices_reports_missing_and_invalid_files(tmp_path, monkeypatch, cropped):
    # Both modes reject active-file problems before attempting transcript repair.
    capture = _ValidateCapture(monkeypatch)
    entries = [_validate_entry(name, "", cropped=cropped) for name in ("gone.flac", "bad.flac")]
    sample_dir = tmp_path / "voice"
    if cropped:
        sample_dir.mkdir()
        for entry in entries:
            (sample_dir / entry.file_name).write_bytes(b"original")
        sample_dir /= "crops"
    sample_dir.mkdir(parents=True, exist_ok=True)
    bad_name = "trim-bad.flac" if cropped else "bad.flac"
    bad_path = sample_dir / bad_name
    bad_path.write_bytes(b"junk")
    loaded = []
    monkeypatch.setattr(
        voice_menu_shared.SoundFileUtil, "load",
        lambda path: loaded.append(path) or "decode failed",
    )
    monkeypatch.setattr(
        VoiceMenuShared, "transcribe_voice_sample_to_text",
        lambda *_: pytest.fail("must not transcribe"),
    )
    state = _validate_state(tmp_path, "glm_local", entries)
    expected = [
        "Trimmed voice file for gone.flac not found (reset the trim or restore trim-gone.flac)",
        "Trimmed voice file for bad.flac is invalid",
    ] if cropped else ["Voice file gone.flac not found", "Voice file bad.flac is invalid"]
    assert VoiceMenuShared.get_voice_problems(state) == expected
    assert not capture.lines and not capture.errors
    assert VoiceMenuShared.validate_voices(state) is False
    assert capture.lines == [
        *(f"{voice_menu_shared.COL_ERROR}{problem}{voice_menu_shared.COL_DEFAULT}" for problem in expected),
        "",
    ]
    assert not capture.errors
    assert capture.pauses == [None]
    assert loaded == [str(bad_path), str(bad_path)]
    assert state.project.voice_references == entries
    assert not (tmp_path / "project.json").exists()


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("transcript", ["", " \n "])
def test_validate_voices_transcribes_empty_transcript_and_saves(tmp_path, monkeypatch, cropped, transcript):
    # Only interactive validation repairs the active transcript, not the original crop source.
    capture = _ValidateCapture(monkeypatch)
    entry = _validate_entry("a.flac", transcript, cropped=cropped)
    sample_path = _write_active_sample(tmp_path, entry)
    sound = object()
    loaded = []
    transcribed = []
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda path: loaded.append(path) or sound)
    monkeypatch.setattr(
        voice_menu_shared.Transcriber, "transcribe_to_words",
        lambda sample, *_: transcribed.append(sample) or ["words"],
    )
    monkeypatch.setattr(
        voice_menu_shared.Transcriber,
        "get_flat_text_filtered_by_probability",
        lambda words, min_probability: "transcribed text",
    )
    state = _validate_state(tmp_path, "glm_local", [entry])
    label = "Trimmed voice file for a.flac" if cropped else "Voice file a.flac"
    assert VoiceMenuShared.get_voice_problems(state) == [f"{label} has no transcript"]
    assert not transcribed and not capture.lines and not capture.errors
    assert state.project.voice_references == [entry]
    assert not (tmp_path / "project.json").exists()

    assert VoiceMenuShared.validate_voices(state) is True
    expected = (
        replace(entry, crop=replace(entry.crop, transcript="transcribed text"))
        if entry.crop is not None else replace(entry, transcript="transcribed text")
    )
    assert state.project.voice_references == [expected]
    saved = json.loads((tmp_path / "project.json").read_text())
    assert saved["voice_references"] == [expected.to_dict()]
    assert loaded == [str(sample_path), str(sample_path)]
    assert transcribed == [sound]
    assert not capture.errors


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("result", [("", "stt broke"), ("", "")])
def test_validate_voices_transcription_failure_blocks_launch_and_does_not_save(tmp_path, monkeypatch, cropped, result):
    # STT errors and empty successful results both leave the active transcript unchanged.
    capture = _ValidateCapture(monkeypatch)
    entry = _validate_entry("a.flac", "", cropped=cropped)
    _write_active_sample(tmp_path, entry)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", lambda *_: result)
    state = _validate_state(tmp_path, "glm_local", [entry])
    assert VoiceMenuShared.validate_voices(state) is False
    label = "Trimmed voice file for a.flac" if cropped else "Voice file a.flac"
    assert capture.lines == [f"{voice_menu_shared.COL_ERROR}{label} could not be transcribed{voice_menu_shared.COL_DEFAULT}", ""]
    assert not capture.errors
    assert capture.pauses == [None]
    assert state.project.voice_references == [entry]
    assert not (tmp_path / "project.json").exists()


@pytest.mark.parametrize("cropped", [False, True])
def test_validate_voices_reports_transcript_save_failure(tmp_path, monkeypatch, cropped):
    # Crop/noncrop save failures keep their existing messages and use the matching updater.
    capture = _ValidateCapture(monkeypatch)
    entry = _validate_entry("a.flac", "", cropped=cropped)
    _write_active_sample(tmp_path, entry)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", lambda *_: ("fixed", ""))
    updater = "set_voice_crop_transcript_at_index_and_save" if cropped else "set_voice_transcript_at_index_and_save"
    other_updater = "set_voice_transcript_at_index_and_save" if cropped else "set_voice_crop_transcript_at_index_and_save"
    saved = []
    monkeypatch.setattr(ProjectVoiceUtil, updater, lambda *args: saved.append(args) or "disk full")
    monkeypatch.setattr(ProjectVoiceUtil, other_updater, lambda *_: pytest.fail("wrong transcript updater"))
    state = _validate_state(tmp_path, "glm_local", [entry])
    assert VoiceMenuShared.validate_voices(state) is False
    transcript_label = "trim transcript" if cropped else "transcript"
    assert capture.lines == [
        f"{voice_menu_shared.COL_ERROR}Could not save {transcript_label} for voice file a.flac: disk full{voice_menu_shared.COL_DEFAULT}",
        "",
    ]
    assert not capture.errors
    assert capture.pauses == [None]
    assert saved == [(state.project, 0, "fixed")]


@pytest.mark.parametrize("cropped", [False, True])
def test_validate_voices_skips_stt_for_model_without_transcripts(tmp_path, monkeypatch, cropped):
    # Transcript-free models still validate the active audio in both modes.
    capture = _ValidateCapture(monkeypatch)
    entry = _validate_entry("a.flac", "", cropped=cropped)
    sample_path = _write_active_sample(tmp_path, entry)
    loaded = []
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda path: loaded.append(path) or object())
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", lambda *_: pytest.fail("must not transcribe"))
    state = _validate_state(tmp_path, "pocket_local", [entry])
    assert VoiceMenuShared.get_voice_problems(state) == []
    assert VoiceMenuShared.validate_voices(state) is True
    assert loaded == [str(sample_path), str(sample_path)]
    assert not capture.lines and not capture.errors
    assert state.project.voice_references == [entry]
    assert not (tmp_path / "project.json").exists()


@pytest.mark.parametrize("cropped", [False, True])
def test_voice_validation_preserves_existing_active_transcript(tmp_path, monkeypatch, cropped):
    # An existing crop transcript is sufficient even when the original transcript is empty.
    capture = _ValidateCapture(monkeypatch)
    entry = _validate_entry("a.flac", " kept ", cropped=cropped)
    if cropped:
        entry = replace(entry, transcript="")
    _write_active_sample(tmp_path, entry)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", lambda *_: pytest.fail("must not transcribe"))
    state = _validate_state(tmp_path, "glm_local", [entry])
    assert VoiceMenuShared.get_voice_problems(state) == []
    assert VoiceMenuShared.validate_voices(state) is True
    assert state.project.voice_references == [entry]
    assert not capture.lines and not capture.errors
    assert not (tmp_path / "project.json").exists()


# --- get_voice_problems (console-free pre-flight used by the quick-gen modal) ---

@pytest.mark.parametrize("model_id, entries, preset", [
    ("none", [{"file_name": "a.flac", "transcript": ""}], False),
    ("dots_local", [], False),
    ("pocket_local", [{"file_name": "a.flac", "transcript": ""}], True),
])
def test_get_voice_problems_is_empty_where_validate_is_a_no_op(tmp_path, monkeypatch, model_id, entries, preset):
    # Same "nothing to check" scope as validate_voices.
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: pytest.fail("must not load"))
    state = _validate_state(tmp_path, model_id, entries, preset=preset)
    assert VoiceMenuShared.get_voice_problems(state) == []


def test_get_voice_problems_reports_required_voice_missing(tmp_path):
    state = _validate_state(tmp_path, "qwen3tts_local", [])
    assert VoiceMenuShared.get_voice_problems(state) == [
        "Voice 1 has no voice sample (set a sample or remove the voice)",
    ]


@pytest.mark.parametrize("cropped", [False, True])
def test_get_voice_problems_lists_every_problem_without_side_effects(tmp_path, monkeypatch, cropped, capsys):
    # List all active-file blockers without console output, transcription, or saving.
    capture = _ValidateCapture(monkeypatch)
    entries = [
        _validate_entry("gone.flac", "kept", cropped=cropped),
        _validate_entry("bad.flac", "kept", cropped=cropped),
        _validate_entry("ok.flac", "", cropped=cropped),
    ]
    sample_dir = tmp_path / "voice" / "crops" if cropped else tmp_path
    sample_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries[1:]:
        name = entry.crop.file_name if entry.crop is not None else entry.file_name
        (sample_dir / name).write_bytes(b"sample")
    monkeypatch.setattr(
        voice_menu_shared.SoundFileUtil, "load",
        lambda path: "decode failed" if path.endswith("bad.flac") else object(),
    )
    monkeypatch.setattr(
        VoiceMenuShared, "transcribe_voice_sample_to_text",
        lambda *_: pytest.fail("must not transcribe"),
    )
    state = _validate_state(tmp_path, "glm_local", entries)
    expected = [
        "Trimmed voice file for gone.flac not found (reset the trim or restore trim-gone.flac)",
        "Trimmed voice file for bad.flac is invalid",
        "Trimmed voice file for ok.flac has no transcript",
    ] if cropped else [
        "Voice file gone.flac not found",
        "Voice file bad.flac is invalid",
        "Voice file ok.flac has no transcript",
    ]
    assert VoiceMenuShared.get_voice_problems(state) == expected
    assert not capture.lines and not capture.errors
    assert capsys.readouterr().out == ""
    assert state.project.voice_references == entries
    assert not (tmp_path / "project.json").exists()


def test_voice_preflight_skips_transcript_for_model_that_stores_but_never_reads_it(tmp_path, monkeypatch):
    # MOSS local keeps shared transcripts in storage but clones from audio
    # alone, so an empty transcript neither blocks nor triggers STT.
    (tmp_path / "a.flac").write_bytes(b"valid")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", lambda *_: pytest.fail("must not transcribe"))
    state = _validate_state(tmp_path, "moss_local", [{"file_name": "a.flac", "transcript": ""}])
    assert VoiceMenuShared.get_voice_problems(state) == []
    assert VoiceMenuShared.validate_voices(state) is True


def test_get_voice_problems_ignores_empty_transcript_for_model_without_transcripts(tmp_path, monkeypatch):
    (tmp_path / "a.flac").write_bytes(b"valid")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda *_: object())
    state = _validate_state(tmp_path, "pocket_local", [{"file_name": "a.flac", "transcript": ""}])
    assert VoiceMenuShared.get_voice_problems(state) == []


class TestCropVoiceSampleMenu:

    def _make_state(self, tmp_path, entries):
        voice_dir = tmp_path / "voice"
        voice_dir.mkdir(parents=True, exist_ok=True)
        from tts_audiobook_tool.app_types import Sound
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
        import numpy as np
        project = Project(tts_model_type="glm_local", voice_references=list(entries))
        for entry in project.voice_references:
            if entry.crop is not None:
                (voice_dir / ProjectVoiceUtil.get_cropped_voice_relative_path(entry)).parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(0)
            data = (rng.uniform(0.01, 0.02, 24000 * 8)).astype(np.float32)
            assert SoundFileUtil.save_flac(Sound(data, 24000), str(voice_dir / entry.file_name)) == ""
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

    def test_item_present_before_transcript_item(self, monkeypatch):
        project = Project(tts_model_type="glm_local", voice_references=[
            {"file_name": "a.flac", "transcript": "t"},
        ])
        state = cast(State, SimpleNamespace(project=project))
        menus = capture_menus(monkeypatch)
        monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: 3.0)
        VoiceMenuShared.voice_sample_submenu(state, project.get_tts_model_type(), 0)
        labels = [get_string_from(state, item.label) for item in menus[-1]["items"](state)]
        assert voice_menu_shared.LABEL_CROP_VOICE_SAMPLE == "Trim or play"
        crop_label = voice_menu_shared.make_menu_label(voice_menu_shared.LABEL_CROP_VOICE_SAMPLE, "3.0s")
        assert crop_label in labels
        edit_label = voice_menu_shared.make_menu_label(voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION, '"t"')
        assert labels.index(crop_label) < labels.index(edit_label)

    @pytest.mark.parametrize("duration, warned", [(15.0, False), (15.3, True)])
    def test_crop_item_notes_long_sample(self, monkeypatch, duration, warned):
        project = Project(tts_model_type="echo_tts_audiocpp", voice_references=[
            {"file_name": "a.flac", "transcript": "t"},
        ])
        state = cast(State, SimpleNamespace(project=project))
        menus = capture_menus(monkeypatch)
        monkeypatch.setattr(voice_menu_shared.AudioMetaUtil, "get_audio_duration", lambda _: duration)
        VoiceMenuShared.voice_sample_submenu(state, project.get_tts_model_type(), 0)
        crop_label = get_string_from(state, menus[-1]["items"](state)[3].label)
        m = voice_menu_shared
        note = (
            f" {m.COL_ERROR}*{m.COL_DIM_ITALICS}exceeds current model's "
            f"recommended limit of 15s{m.COL_DEFAULT}"
        )
        assert crop_label.endswith(note) == warned

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
        VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
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
        VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
        assert len(launched) == 1
        sound, title, initial = launched[0]
        assert title == "a"
        assert initial is None  # no existing crop
        entry = state.project.voice_references[0]
        assert entry.crop is not None
        assert entry.crop.file_name.endswith(".flac")
        assert "/" not in entry.crop.file_name and "\\" not in entry.crop.file_name
        assert Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, entry)).is_file()
        assert len(transcribed) == 1  # the cropped span, not the original
        assert entry.crop_range == (1.0, 4.5)
        assert entry.crop.transcript == "auto transcript"
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
        VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
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
        VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
        assert not crop_path.exists()
        assert state.project.voice_references == [Voice("a.flac", "t")]
        assert feedback == ["Trim reset; using original sample a.flac"]

    def test_cleared_without_crop_is_a_noop(self, tmp_path, monkeypatch):
        feedback = []
        monkeypatch.setattr(voice_menu_shared, "print_feedback", feedback.append)
        state, _ = self._make_state(tmp_path, [{"file_name": "a.flac", "transcript": "t"}])
        self._patch_crop_app(monkeypatch, SimpleNamespace(saved=False, cleared=True))
        before = list(state.project.voice_references)
        VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
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
        VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
        entry = state.project.voice_references[0]
        assert not Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, entry)).exists()
        assert entry.crop_range == (1.0, 4.0)

    def test_reversed_adjustment_preserves_edited_crop_transcript(self, tmp_path, monkeypatch):
        import asyncio
        from tts_audiobook_tool.textual import voice_crop_app

        state, voice_dir = self._make_state(tmp_path, [{"file_name": "a.flac", "transcript": "full"}])
        assert voice_menu_shared.ProjectVoiceUtil.apply_voice_crop_and_save(
            state.project, 0, 0.1, 5.0, "Manually corrected crop transcript",
        ) == ""
        previous_entry = state.project.voice_references[0]
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
            VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
        apply.assert_not_called()
        assert state.project.voice_references[0] == previous_entry
        assert crop_path.read_bytes() == previous_audio
        assert (tmp_path / "project.json").read_bytes() == previous_settings

    def test_multi_sample_index_targets_that_sample(self, tmp_path, monkeypatch):
        state, voice_dir = self._make_state(tmp_path, [
            {"file_name": "a.flac", "transcript": "A"},
            {"file_name": "b.flac", "transcript": "B"},
        ])
        monkeypatch.setattr(
            VoiceMenuShared, "ask_voice_sample_position",
            lambda *_: pytest.fail("the sample index is given; must not prompt"),
        )
        launched = self._patch_crop_app(monkeypatch, SimpleNamespace(saved=True, cleared=False, start_s=0.25, end_s=4.0))
        monkeypatch.setattr(
            VoiceMenuShared, "transcribe_voice_sample_to_text", lambda state, sound: ("auto transcript", "")
        )
        VoiceMenuShared.crop_voice_sample_from_menu(state, 1)
        assert launched[0][1] == "b"
        entry = state.project.voice_references[1]
        assert entry.crop is not None
        assert entry.crop.file_name.endswith(".flac")
        assert "/" not in entry.crop.file_name and "\\" not in entry.crop.file_name
        assert Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, entry)).is_file()
        assert state.project.voice_references[0] == Voice("a.flac", "A")
        assert entry.crop_range == (0.25, 4.0)

    def test_app_error_is_reported(self, tmp_path, monkeypatch):
        state, _ = self._make_state(tmp_path, [{"file_name": "a.flac", "transcript": "t"}])
        self._patch_crop_app(monkeypatch, "terminal does not support")
        errors = []
        monkeypatch.setattr(voice_menu_shared.ask, "ask_error", errors.append)
        VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
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
        VoiceMenuShared.edit_voice_sample_transcript(state, 0)
        entry = state.project.voice_references[0]
        assert entry.crop is not None and entry.crop.transcript == "cut text 2"
        assert entry.transcript == "full text"  # original untouched
        assert prompts[0][1] == "cut text"  # prefilled with the crop transcript
        assert "trimmed span" in lines[0]  # the prompt is printed above the input line

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
        VoiceMenuShared.edit_voice_sample_transcript(state, 0)
        entry = state.project.voice_references[0]
        assert entry.transcript == "edited full"
        assert entry.crop is not None and entry.crop.transcript == "cut text"  # crop transcript untouched
        assert prompts[0][1] == "full text"
