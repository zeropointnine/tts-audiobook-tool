from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from dataclasses import replace

from tts_audiobook_tool.app_types import Book, BookSection, RoomReverb, VoiceSelectMode
from tts_audiobook_tool.app_types.voice import stereo_position_label
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


@pytest.mark.parametrize("action", ["move", "edit"])
def test_empty_sample_actions_do_not_prompt(monkeypatch, action):
    state = make_state(0)
    errors, prompts, feedback = capture(monkeypatch, [])
    if action == "move":
        assert menu.VoiceMenuShared.move_voice_sample_from_menu(state, 0) == 0
        assert not feedback
    else:
        menu.VoiceMenuShared.edit_voice_sample_transcript(state, 0)
        assert feedback == ["No such voice sample"]
    assert not errors and not prompts


@pytest.mark.parametrize("text, expected", [("updated", "updated"), ("/clear", "")])
def test_transcription_edits_selected_pair(monkeypatch, text, expected):
    state = make_state()
    capture(monkeypatch, [text])
    saved = Mock(return_value="")
    monkeypatch.setattr(Project, "save", saved)
    items = open_submenu(monkeypatch, state, 1)["items"](state)
    edit = find_item(state, items, f'{menu.LABEL_EDIT_VOICE_TRANSCRIPTION} {menu.COL_DIM}("text 1")')
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
    assert prompts == ["Remove voice sample 2: 1.flac?"]
    assert not errors
    if confirmed:
        assert state.project.voice_references == [original[0], original[2]]
        assert feedback == ["Removed 1.flac"]
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
    assert get_string_from(state, captured["heading"]) == f"Options for 0.flac {menu.COL_DIM}(voice index 1)"
    assert get_string_from(state, captured["breadcrumb"]) == "Voice sample 1"
    items = captured["items"](state)
    expected = [
        f"{menu.LABEL_CROP_VOICE_SAMPLE} {menu.COL_DIM}(4.0s)",
        f'{menu.LABEL_EDIT_VOICE_TRANSCRIPTION} {menu.COL_DIM}("text 0")',
    ]
    if count > 1:
        expected.append(menu.LABEL_MOVE_VOICE_SAMPLE)
    expected.extend([
        menu.LABEL_REMOVE_VOICE_SAMPLE,
        menu.make_menu_label(menu.LABEL_STEREO_POSITION, "Center"),
        menu.make_menu_label(menu.LABEL_ROOM_REVERB, RoomReverb.OFF.label),
    ])
    assert [get_string_from(state, item.label) for item in items] == expected
    assert items[1].sublabel is None
    # The Effects group starts at Stereo position
    assert items[-2].superlabel == menu.LABEL_EFFECTS_SUPERLABEL
    assert not items[-1].superlabel
    assert items[-2].sublabel is None


@pytest.mark.parametrize("count, confirmed, cleared", [(1, True, True), (2, True, False), (1, False, False)])
def test_submenu_remove_exits_and_clears_when_list_empties(monkeypatch, count, confirmed, cleared):
    state = make_state(count)
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    capture(monkeypatch, [])
    monkeypatch.setattr(menu.ask, "ask_confirm", lambda *_: confirmed)
    events = []
    captured = open_submenu(monkeypatch, state, 0, lambda: events.append("clear"))
    remove = find_item(state, captured["items"](state), menu.LABEL_REMOVE_VOICE_SAMPLE)
    assert remove.handler(state, remove) is confirmed  # True exits the submenu
    assert len(state.project.voice_references) == count - confirmed
    assert events == (["clear"] if cleared else [])


def test_submenu_follows_moved_sample(monkeypatch):
    state = make_state()
    monkeypatch.setattr(Project, "save", Mock(return_value=""))
    capture(monkeypatch, ["3"])
    captured = open_submenu(monkeypatch, state, 0)
    move = find_item(state, captured["items"](state), menu.LABEL_MOVE_VOICE_SAMPLE)
    move.handler(state, move)
    assert [v.file_name for v in state.project.voice_references] == ["1.flac", "2.flac", "0.flac"]
    assert get_string_from(state, captured["heading"]) == f"Options for 0.flac {menu.COL_DIM}(voice index 3)"
    assert get_string_from(state, captured["breadcrumb"]) == "Voice sample 3"
    edit_label = get_string_from(state, captured["items"](state)[1].label)
    assert edit_label == f'{menu.LABEL_EDIT_VOICE_TRANSCRIPTION} {menu.COL_DIM}("text 0")'


@pytest.mark.parametrize("transcript, expected", [
    ("Two  words\n here", '("Two words here")'),
    ("x" * 40, f'("{"x" * 40}")'),
    ("x" * 41, f'("{"x" * 37}...")'),
    (" \n ", "(none)"),
])
def test_transcript_preview_is_flattened_and_ellipsized(transcript, expected):
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        {"file_name": "a.flac", "transcript": transcript},
    ])
    assert menu.VoiceMenuShared.make_transcript_preview(project, project.get_tts_model_type(), 0) == expected


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
            menu.LABEL_ROOM_REVERB, RoomReverb.SMALL_DRR17.label,
        )


@pytest.mark.parametrize("pan, label", [(0.0, "Center"), (-0.3, "Left 30"), (1.0, "Right 100"), (0.004, "Center")])
def test_stereo_position_label(pan, label):
    assert stereo_position_label(pan) == label


@pytest.mark.parametrize("mode, noted", [
    (VoiceSelectMode.USER_DEFINED, False), (VoiceSelectMode.DISABLED, False), (VoiceSelectMode.AUTO_ADVANCE, True),
])
def test_stereo_position_item_notes_auto_advance(monkeypatch, mode, noted):
    state = make_state(2)
    state.project.voice_select_mode = mode
    item = open_submenu(monkeypatch, state, 0)["items"](state)[-2]
    assert (item.sublabel == "No effect in auto-advance voice selection mode") is noted
    assert noted or item.sublabel is None


@pytest.mark.parametrize("save_error", ["", "disk full"])
def test_stereo_position_prompt_saves_value(monkeypatch, save_error):
    state = make_state(2)
    saved = Mock(return_value=save_error)
    monkeypatch.setattr(Project, "save", saved)
    errors, prompts, feedback = capture(monkeypatch, ["-30"])
    make_items = open_submenu(monkeypatch, state, 1)["items"]
    item = make_items(state)[-2]
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
        assert get_string_from(state, make_items(state)[-2].label) == menu.make_menu_label(
            menu.LABEL_STEREO_POSITION, "Left 30",
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
