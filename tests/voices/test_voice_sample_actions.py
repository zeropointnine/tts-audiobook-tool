from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from dataclasses import replace

from tts_audiobook_tool.app_types import Book, BookSection, RoomReverb
from tts_audiobook_tool.app_types.voice import Voice, VoiceCrop, stereo_position_label
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.menu_util import get_string_from
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


def open_submenu(monkeypatch, state, index, on_clear_callback=None):
    """Open the per-sample submenu; returns the captured MenuUtil.menu kwargs."""
    captured = {}
    monkeypatch.setattr(menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(menu.AudioMetaUtil, "get_audio_duration", lambda _: 4.0)
    menu.VoiceMenuShared.voice_sample_submenu(
        state, state.project.get_tts_model_type(), index, on_clear_callback,
    )
    return captured


def find_item(state, items, label):
    return next(item for item in items if get_string_from(state, item.label) == label)


@pytest.mark.parametrize("index, position, order", [(0, "3", [1, 2, 0]), (2, "1", [2, 0, 1]), (1, "1", [1, 0, 2])])
def test_move_reorders_pairs_and_saves(monkeypatch, index, position, order):
    state = make_state()
    state.project.book = Book([BookSection([
        PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=i) for i in [-1, 0, 1, 2]
    ])])
    original = list(state.project.voice_references)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, feedback = capture(monkeypatch, [position])
    assert menu.VoiceMenuShared.move_voice_sample_from_menu(state, index) == int(position) - 1
    assert state.project.voice_references == [original[i] for i in order]
    assert [group.voice_index for group in state.project.book.phrase_groups] == [-1, 0, 1, 2]
    assert prompts == [f"Enter new position {menu.COL_DIM}(1-3):"]
    assert feedback == [f"Moved voice sample {index + 1} to position {position}"]
    assert not errors
    saved.assert_called_once()


@pytest.mark.parametrize("inputs, expected_feedback", [
    ([""], []),
    (["x"], [("Bad value", {"is_error": True})]),
    (["0"], [("Out of range", {"is_error": True})]),
    (["4"], [("Out of range", {"is_error": True})]),
    (["2"], ["Voice sample position unchanged"]),
])
def test_move_invalid_cancel_and_no_change(monkeypatch, inputs, expected_feedback):
    state = make_state()
    original = list(state.project.voice_references)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, _, feedback = capture(monkeypatch, inputs)
    assert menu.VoiceMenuShared.move_voice_sample_from_menu(state, 1) == 1
    assert state.project.voice_references == original
    assert not errors
    assert feedback == expected_feedback
    saved.assert_not_called()


@pytest.mark.parametrize("action", ["move", "trim", "edit", "remove"])
@pytest.mark.parametrize("index", [-1, 3])
def test_invalid_sample_index_does_not_prompt_or_save(monkeypatch, action, index):
    state = make_state()
    original = list(state.project.voice_references)
    saved = Mock()
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, feedback = capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_confirm", lambda *_: pytest.fail("must not confirm"))
    expected_feedback = []
    if action == "move":
        assert menu.VoiceMenuShared.move_voice_sample_from_menu(state, index) == index
    elif action == "trim":
        menu.VoiceMenuShared.crop_voice_sample_from_menu(state, index)
        expected_feedback = ["No such voice sample"]
    elif action == "edit":
        menu.VoiceMenuShared.edit_voice_sample_transcript(state, index)
        expected_feedback = ["No such voice sample"]
    else:
        assert menu.VoiceMenuShared.remove_voice_sample_from_menu(
            state, state.project.get_tts_model_type(), index,
        ) is False
    assert not errors and not prompts
    assert feedback == expected_feedback
    assert state.project.voice_references == original
    saved.assert_not_called()


def test_move_save_failure_restores_order(monkeypatch):
    state = make_state()
    original = list(state.project.voice_references)
    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    errors, _, feedback = capture(monkeypatch, ["3"])
    assert menu.VoiceMenuShared.move_voice_sample_from_menu(state, 0) == 0
    assert state.project.voice_references == original
    assert errors == ["disk full"]
    assert not feedback


@pytest.mark.parametrize("action", ["move", "trim", "edit"])
def test_sample_less_voice_actions_do_not_prompt(monkeypatch, action):
    # A project with no samples still has one (default) voice.
    state = make_state(0)
    assert state.project.voice_references == [Voice()]
    errors, prompts, feedback = capture(monkeypatch, [])
    if action == "move":
        assert menu.VoiceMenuShared.move_voice_sample_from_menu(state, 0) == 0
        assert not feedback
    elif action == "trim":
        menu.VoiceMenuShared.crop_voice_sample_from_menu(state, 0)
        assert feedback == ["No voice sample"]
    else:
        menu.VoiceMenuShared.edit_voice_sample_transcript(state, 0)
        assert feedback == ["No voice sample"]
    assert not errors and not prompts


@pytest.mark.parametrize("text, expected", [("updated", "updated"), ("/clear", "")])
def test_transcription_edits_selected_pair(monkeypatch, text, expected):
    state = make_state()
    capture(monkeypatch, [text])
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    items = open_submenu(monkeypatch, state, 1)["items"](state)
    edit = find_item(state, items, menu.make_menu_label(menu.LABEL_EDIT_VOICE_TRANSCRIPTION, '"text 1"'))
    edit.handler(state, edit)
    assert [v.transcript for v in state.project.voice_references] == ["text 0", expected, "text 2"]
    assert state.project.voice_references[1].file_name == "1.flac"
    saved.assert_called_once()


@pytest.mark.parametrize("index, inputs", [(1, [""]), (1, ["text 1"]), (-1, []), (3, [])])
def test_transcription_canceled_or_invalid(monkeypatch, index, inputs):
    state = make_state()
    original = list(state.project.voice_references)
    capture(monkeypatch, inputs)
    saved = Mock()
    monkeypatch.setattr(Project, "save", saved)
    menu.VoiceMenuShared.edit_voice_sample_transcript(state, index)
    assert state.project.voice_references == original
    saved.assert_not_called()


@pytest.mark.parametrize("uses_transcript", [True, False])
def test_transcription_prompt_is_on_its_own_line_with_unused_note(monkeypatch, uses_transcript):
    state = make_state(1)
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    _, prompts, _ = capture(monkeypatch, ["updated"])
    monkeypatch.setattr(menu.Tts, "requires_reference_transcript", lambda _: uses_transcript)
    menu.VoiceMenuShared.edit_voice_sample_transcript(state, 0)
    expected = ['Enter voice sample transcript (or "/clear" to clear):']
    if not uses_transcript:
        expected.append(f"{menu.COL_DIM}(Note that the current TTS model does not utilize the transcript value)")
    assert prompts == expected


def test_transcription_save_error(monkeypatch):
    state = make_state(1)
    errors, _, feedback = capture(monkeypatch, ["updated"])
    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    menu.VoiceMenuShared.edit_voice_sample_transcript(state, 0)
    assert errors == ["disk full"]
    assert not feedback


@pytest.mark.parametrize("confirmed", [True, False])
def test_remove_requires_confirmation(monkeypatch, confirmed):
    state = make_state()
    original = list(state.project.voice_references)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, feedback = capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_confirm", lambda *_: confirmed)
    assert menu.VoiceMenuShared.remove_voice_sample_from_menu(
        state, state.project.get_tts_model_type(), 1,
    ) is confirmed
    assert prompts == ["Remove Voice 2 (1)?"]
    assert not errors
    if confirmed:
        assert state.project.voice_references == [original[0], original[2]]
        assert feedback == ["Removed Voice 2 (1)"]
        saved.assert_called_once()
    else:
        assert state.project.voice_references == original
        assert not feedback
        saved.assert_not_called()


# --- Per-sample submenu ---

@pytest.mark.parametrize("count", [1, 2])
def test_submenu_items_for_one_and_multiple_samples(monkeypatch, count):
    state = make_state(count)
    captured = open_submenu(monkeypatch, state, 0)
    assert get_string_from(state, captured["heading"]) == "Voice 1"
    assert get_string_from(state, captured["breadcrumb"]) == "Voice 1"
    items = captured["items"](state)
    expected = [
        menu.make_menu_label_optional(menu.LABEL_VOICE_NAME),
        menu.make_menu_label(menu.LABEL_SET_VOICE_SAMPLE, "0"),
        menu.LABEL_CLEAR_VOICE_SAMPLE,
        menu.make_menu_label(menu.LABEL_CROP_VOICE_SAMPLE, "4.0s"),
        menu.make_menu_label(menu.LABEL_EDIT_VOICE_TRANSCRIPTION, '"text 0"'),
        menu.make_menu_label("Voice design/OmniVoice", "none"),
    ]
    if count > 1:
        expected.extend([menu.LABEL_MOVE_VOICE_SAMPLE, menu.LABEL_REMOVE_VOICE_SAMPLE])
    else:
        expected.append(menu.LABEL_RESET_VOICE_SAMPLE)
    expected.extend([
        menu.make_menu_label(menu.LABEL_STEREO_POSITION, "Center", "Center"),
        menu.make_menu_label(menu.LABEL_VOLUME, "100%", "100%"),
        menu.make_menu_label(menu.LABEL_ROOM_REVERB, RoomReverb.OFF.label, RoomReverb.OFF.label),
    ])
    assert [get_string_from(state, item.label) for item in items] == expected
    assert items[3].sublabel is None
    # The Effects group starts at Stereo position
    assert items[-3].superlabel == menu.LABEL_EFFECTS_SUPERLABEL
    assert not items[-2].superlabel and not items[-1].superlabel
    assert items[-3].sublabel is None


@pytest.mark.parametrize("trimmed, duration, value", [
    (False, 13.1, "13.1s"),
    (True, 13.1, "trimmed, 13.1s"),
    (True, None, "trimmed"),
    (False, None, None),
])
def test_trim_item_uses_currently_treatment(monkeypatch, trimmed, duration, value):
    state = make_state(1)
    if trimmed:
        state.project.voice_references[0] = replace(
            state.project.voice_references[0], crop=VoiceCrop("trim.flac", 0, 13.1),
        )
    captured = open_submenu(monkeypatch, state, 0)
    monkeypatch.setattr(menu.AudioMetaUtil, "get_audio_duration", lambda _: duration)
    items = captured["items"](state)
    expected = (
        menu.make_menu_label(menu.LABEL_CROP_VOICE_SAMPLE, value)
        if value else menu.LABEL_CROP_VOICE_SAMPLE
    )
    assert get_string_from(state, items[3].label) == expected


@pytest.mark.parametrize("count, confirmed, cleared", [(1, True, True), (2, True, False), (1, False, False)])
def test_submenu_remove_exits_and_clears_when_no_sample_remains(monkeypatch, count, confirmed, cleared):
    state = make_state(count)
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_confirm", lambda *_: confirmed)
    events = []
    captured = open_submenu(monkeypatch, state, 0, lambda: events.append("clear"))
    label = menu.LABEL_REMOVE_VOICE_SAMPLE if count > 1 else menu.LABEL_RESET_VOICE_SAMPLE
    remove = find_item(state, captured["items"](state), label)
    assert remove.handler(state, remove) is confirmed  # True exits the submenu
    if count == 1 and confirmed:
        # Removing the only voice leaves a fresh default voice.
        assert state.project.voice_references == [Voice()]
    else:
        assert len(state.project.voice_references) == count - confirmed
    assert events == (["clear"] if cleared else [])


def test_resetting_sole_default_voice_needs_no_confirmation(monkeypatch):
    state = make_state(0)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    _, prompts, feedback = capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_confirm", lambda *_: pytest.fail("must not confirm"))
    assert menu.VoiceMenuShared.remove_voice_sample_from_menu(state, state.project.get_tts_model_type(), 0)
    assert state.project.voice_references == [Voice()]
    assert not prompts
    assert feedback == ["Reset Voice 1 to default"]


def test_resetting_sole_voice_with_values_confirms(monkeypatch):
    state = make_state(0)
    state.project.voice_references = [Voice(pan=0.5)]
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    _, prompts, _ = capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_confirm", lambda *_: True)
    assert menu.VoiceMenuShared.remove_voice_sample_from_menu(state, state.project.get_tts_model_type(), 0)
    assert prompts == ["Reset Voice 1 to default?"]
    assert state.project.voice_references == [Voice()]


def test_submenu_for_sample_less_voice_hides_sample_items(monkeypatch):
    state = make_state(0)
    items = open_submenu(monkeypatch, state, 0)["items"](state)
    labels = [get_string_from(state, item.label) for item in items]
    # OmniVoice does not require a voice clone sample
    assert labels[0] == menu.make_menu_label_optional(menu.LABEL_VOICE_NAME)
    assert labels[1] == menu.make_menu_label(menu.LABEL_SET_VOICE_SAMPLE, "none")
    assert labels[2:] == [
        menu.make_menu_label("Voice design/OmniVoice", "none"),
        menu.LABEL_RESET_VOICE_SAMPLE,
        menu.make_menu_label(menu.LABEL_STEREO_POSITION, "Center", "Center"),
        menu.make_menu_label(menu.LABEL_VOLUME, "100%", "100%"),
        menu.make_menu_label(menu.LABEL_ROOM_REVERB, RoomReverb.OFF.label, RoomReverb.OFF.label),
    ]


@pytest.mark.parametrize("model_id", ["breeze_tts_2_audiocpp", "glm_local"])
@pytest.mark.parametrize("count", [1, 2])
def test_clear_sample_keeps_voice_and_submenu_open(monkeypatch, model_id, count):
    state = make_state(count)
    state.project.tts_model_type = model_id
    state.project.voice_references[0] = replace(
        state.project.voice_references[0], pan=-0.3, room_reverb=RoomReverb.SMALL_DRR17,
        presets={"pocket_local": "alba"}, instructions={model_id: "warm"},
    )
    original = list(state.project.voice_references)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, feedback = capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_confirm", lambda *_: pytest.fail("must not confirm"))
    stopped = Mock()
    monkeypatch.setattr(menu.PlaySoundUtil, "stop_sound_async", stopped)
    cleared = Mock()
    captured = open_submenu(monkeypatch, state, 0, cleared)
    clear = find_item(state, captured["items"](state), menu.LABEL_CLEAR_VOICE_SAMPLE)
    assert clear.handler(state, clear) is None  # Does not exit the submenu
    assert state.project.voice_references == [
        replace(original[0], file_name="", transcript="", crop=None), *original[1:],
    ]
    assert get_string_from(state, captured["heading"]) == "Voice 1"
    labels = [get_string_from(state, item.label) for item in captured["items"](state)]
    assert "currently:" in labels[1] and "none" in labels[1]
    assert not any(label.startswith(menu.LABEL_CROP_VOICE_SAMPLE) for label in labels)
    assert not any(label.startswith(menu.LABEL_EDIT_VOICE_TRANSCRIPTION) for label in labels)
    assert menu.LABEL_CLEAR_VOICE_SAMPLE not in labels
    assert menu.VoiceMenuShared.get_voice_sample_required_label(state.project, 0) == (
        "required" if model_id == "glm_local" else ""
    )
    assert not errors and not prompts
    assert feedback == ["Voice sample cleared"]
    assert cleared.call_count == (count == 1)
    saved.assert_called_once()
    stopped.assert_called_once()


def test_clear_sample_save_failure_preserves_voice_and_does_not_call_hook(monkeypatch):
    state = make_state(1)
    original = list(state.project.voice_references)
    monkeypatch.setattr(Project, "save", lambda _: "disk full")
    errors, _, feedback = capture(monkeypatch, [])
    cleared = Mock()
    captured = open_submenu(monkeypatch, state, 0, cleared)
    clear = find_item(state, captured["items"](state), menu.LABEL_CLEAR_VOICE_SAMPLE)
    assert clear.handler(state, clear) is None
    assert state.project.voice_references == original
    assert errors == ["disk full"]
    assert not feedback
    cleared.assert_not_called()
    assert find_item(state, captured["items"](state), menu.LABEL_CLEAR_VOICE_SAMPLE)


def test_submenu_follows_moved_sample(monkeypatch):
    state = make_state()
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    capture(monkeypatch, ["3"])
    captured = open_submenu(monkeypatch, state, 0)
    move = find_item(state, captured["items"](state), menu.LABEL_MOVE_VOICE_SAMPLE)
    move.handler(state, move)
    assert [v.file_name for v in state.project.voice_references] == ["1.flac", "2.flac", "0.flac"]
    assert get_string_from(state, captured["heading"]) == "Voice 3"
    assert get_string_from(state, captured["breadcrumb"]) == "Voice 3"
    edit_label = get_string_from(state, captured["items"](state)[4].label)
    assert edit_label == menu.make_menu_label(menu.LABEL_EDIT_VOICE_TRANSCRIPTION, '"text 0"')
    clear = find_item(state, captured["items"](state), menu.LABEL_CLEAR_VOICE_SAMPLE)
    assert clear.handler(state, clear) is None
    assert [v.file_name for v in state.project.voice_references] == ["1.flac", "2.flac", ""]
    assert [v.transcript for v in state.project.voice_references] == ["text 1", "text 2", ""]


@pytest.mark.parametrize("transcript, expected", [
    ("Two  words\n here", '"Two words here"'),
    ("x" * 40, f'"{"x" * 40}"'),
    ("x" * 41, f'"{"x" * 37}..."'),
])
def test_transcript_preview_is_flattened_and_ellipsized(transcript, expected):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "transcript": transcript},
    ])
    assert menu.VoiceMenuShared.make_transcript_preview(project, project.get_tts_model_type(), 0) == expected


@pytest.mark.parametrize("uses_transcript", [True, False])
def test_transcript_preview_marks_missing_transcript(monkeypatch, uses_transcript):
    monkeypatch.setattr(menu.Tts, "requires_reference_transcript", lambda _: uses_transcript)
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "transcript": " \n "},
    ])
    color = menu.COL_ERROR if uses_transcript else menu.COL_DIM
    assert menu.VoiceMenuShared.make_transcript_preview(
        project, project.get_tts_model_type(), 0,
    ) == "missing"
    state = cast(State, SimpleNamespace(project=project))
    items = open_submenu(monkeypatch, state, 0)["items"](state)
    expected = menu.make_menu_label(menu.LABEL_EDIT_VOICE_TRANSCRIPTION, "missing", color_code=color)
    assert find_item(state, items, expected)


@pytest.mark.parametrize("save_error", ["", "disk full"])
def test_room_reverb_submenu_saves_selected_preset(monkeypatch, save_error):
    state = make_state(2)
    saved = Mock(return_value=save_error)
    monkeypatch.setattr(Project, "save", saved)
    errors, _, feedback = capture(monkeypatch, [])
    options = {}
    monkeypatch.setattr(menu.MenuUtil, "options_menu", lambda **kwargs: options.update(kwargs))
    make_items = open_submenu(monkeypatch, state, 1)["items"]
    reverb = make_items(state)[-1]
    reverb.handler(state, reverb)
    assert options["heading_text"] == menu.LABEL_ROOM_REVERB
    assert options["values"] == list(RoomReverb)
    assert options["current_value"] == RoomReverb.OFF
    options["on_select"](RoomReverb.SMALL_DRR17)
    assert [v.room_reverb for v in state.project.voice_references] == [RoomReverb.OFF, RoomReverb.SMALL_DRR17]
    saved.assert_called_once()
    if save_error:
        assert errors == [save_error] and not feedback
    else:
        assert not errors
        assert feedback == [f"{menu.LABEL_ROOM_REVERB} set to: {RoomReverb.SMALL_DRR17.label}"]
        assert get_string_from(state, make_items(state)[-1].label) == menu.make_menu_label(
            menu.LABEL_ROOM_REVERB, RoomReverb.SMALL_DRR17.label, RoomReverb.OFF.label,
        )


@pytest.mark.parametrize("pan, label", [(0.0, "Center"), (-0.3, "Left 30"), (1.0, "Right 100"), (0.004, "Center")])
def test_stereo_position_label(pan, label):
    assert stereo_position_label(pan) == label


def test_stereo_position_item_has_no_sublabel(monkeypatch):
    state = make_state(2)
    item = open_submenu(monkeypatch, state, 0)["items"](state)[-3]
    assert item.sublabel is None


@pytest.mark.parametrize("save_error", ["", "disk full"])
def test_stereo_position_prompt_saves_value(monkeypatch, save_error):
    state = make_state(2)
    saved = Mock(return_value=save_error)
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, feedback = capture(monkeypatch, ["-30"])
    make_items = open_submenu(monkeypatch, state, 1)["items"]
    item = make_items(state)[-3]
    item.handler(state, item)
    assert prompts == [
        "Enter stereo position, from -100 (full left) to 100 (full right), or 0 for center:",
        f"{menu.COL_DIM}(Subtle values like -30 or 30 work well for separating voices.)",
    ]
    assert [v.pan for v in state.project.voice_references] == [0.0, -0.3]
    saved.assert_called_once()
    if save_error:
        assert errors == [save_error] and not feedback
    else:
        assert not errors
        assert feedback == [f"{menu.LABEL_STEREO_POSITION} set to: Left 30"]
        assert get_string_from(state, make_items(state)[-3].label) == menu.make_menu_label(
            menu.LABEL_STEREO_POSITION, "Left 30", "Center",
        )


@pytest.mark.parametrize("answer, error", [
    ("", None), ("0", None), ("x", "Bad value"), ("101", "Out of range"), ("-100.6", "Out of range"),
    ("nan", "Bad value"), ("inf", "Bad value"),
])
def test_stereo_position_prompt_cancel_unchanged_or_invalid(monkeypatch, answer, error):
    state = make_state(1)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, _, feedback = capture(monkeypatch, [answer])
    menu.VoiceMenuShared.ask_stereo_position(state, 0)
    assert state.project.voice_references[0].pan == 0.0
    assert errors == ([error] if error else [])
    assert not feedback
    saved.assert_not_called()


def test_stereo_position_prompt_prefills_current_and_rounds(monkeypatch):
    state = make_state(1)
    state.project.voice_references = [replace(state.project.voice_references[0], pan=0.25)]
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    prefills = []
    capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_input", lambda **kwargs: prefills.append(kwargs["prefill"]) or "-12.4")
    menu.VoiceMenuShared.ask_stereo_position(state, 0)
    assert prefills == ["25"]
    assert state.project.voice_references[0].pan == -0.12


@pytest.mark.parametrize("save_error", ["", "disk full"])
def test_volume_prompt_saves_value(monkeypatch, save_error):
    # Volume sits between Stereo position and Room reverb
    state = make_state(2)
    saved = Mock(return_value=save_error)
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, feedback = capture(monkeypatch, ["70"])
    make_items = open_submenu(monkeypatch, state, 1)["items"]
    item = make_items(state)[-2]
    item.handler(state, item)
    assert prompts == ["Enter volume, from 30 to 100 (percent):"]
    assert [v.volume for v in state.project.voice_references] == [100, 70]
    saved.assert_called_once()
    if save_error:
        assert errors == [save_error] and not feedback
    else:
        assert not errors
        assert feedback == [f"{menu.LABEL_VOLUME} set to: 70%"]
        assert get_string_from(state, make_items(state)[-2].label) == menu.make_menu_label(
            menu.LABEL_VOLUME, "70%", "100%",
        )


@pytest.mark.parametrize("answer, error", [
    ("", None), ("100", None), ("x", "Bad value"), ("29", "Out of range"), ("101", "Out of range"),
    ("nan", "Bad value"), ("inf", "Bad value"),
])
def test_volume_prompt_cancel_unchanged_or_invalid(monkeypatch, answer, error):
    state = make_state(1)
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    errors, _, feedback = capture(monkeypatch, [answer])
    menu.VoiceMenuShared.ask_volume(state, 0)
    assert state.project.voice_references[0].volume == 100
    assert errors == ([error] if error else [])
    assert not feedback
    saved.assert_not_called()


def test_volume_prompt_accepts_percent_sign_and_rounds(monkeypatch):
    state = make_state(1)
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    capture(monkeypatch, ["64.6%"])
    menu.VoiceMenuShared.ask_volume(state, 0)
    assert state.project.voice_references[0].volume == 65
