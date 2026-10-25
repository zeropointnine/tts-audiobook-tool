from collections import Counter
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.menu_util import MenuUtil, get_string_from
from tts_audiobook_tool.menus.speaker_assign import voices_menu
from tts_audiobook_tool.menus.speaker_assign.voices_menu import (
    make_apply_preview,
    make_character_label,
    run_apply,
    sort_characters,
    voice_submenu,
)
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.speaker_assign.roster import RosterCharacter
from tts_audiobook_tool.speaker_assign.voice_apply import VoiceApplyPlan
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes

from tests.speaker_assign.test_voice_apply import ROSTER_IDS, make_fixture


def make_state(*voice_names: str) -> State:
    project = SimpleNamespace(voice_references=[Voice(name=name) for name in voice_names])
    return cast(State, SimpleNamespace(project=project))


def test_characters_sort_by_dialog_count_with_ties_in_roster_order() -> None:
    characters = tuple(RosterCharacter(id=id, name=id.title()) for id in ("ALICE", "BOB", "CAROL", "DAVE"))
    counts = Counter({"BOB": 50, "CAROL": 2, "DAVE": 50})
    assert [c.id for c in sort_characters(characters, counts)] == ["BOB", "DAVE", "CAROL", "ALICE"]


def test_characters_without_dialog_can_be_omitted() -> None:
    # Once dialog has been assigned, characters with no passages aren't listed
    characters = tuple(RosterCharacter(id=id, name=id.title()) for id in ("ALICE", "BOB", "CAROL"))
    counts = Counter({"BOB": 3, "CAROL": 0})
    assert [c.id for c in sort_characters(characters, counts, omit_unused=True)] == ["BOB"]


def test_character_label_shows_count_and_chosen_voice() -> None:
    # The chosen voice is shown by its title (number plus name, if any); a
    # voice index past the project's voices is flagged as missing.
    state = make_state("", "bob")
    assert strip_ansi_codes(make_character_label(state, "ALICE", 1234, None)) == "ALICE (1,234)"
    assert strip_ansi_codes(make_character_label(state, "BOB", 50, 1)) == "BOB (50) -> Voice 2 bob"
    assert strip_ansi_codes(make_character_label(state, "CAROL", 2, 0)) == "CAROL (2) -> Voice 1"
    assert strip_ansi_codes(make_character_label(state, "DAVE", 0, 5)) == "DAVE (0) -> Voice 6 (missing)"


def test_voice_map_set_clear_and_set_all() -> None:
    # Copies are returned; "set all" covers exactly the given characters and
    # clearing all leaves no entries (stale ids included).
    voice_map = VoiceMapFile({"ALICE": 0, "GONE": 2})
    assert voice_map.with_voice("BOB", 1).voices == {"ALICE": 0, "GONE": 2, "BOB": 1}
    assert voice_map.with_voice("ALICE", None).voices == {"GONE": 2}
    assert voice_map.voices == {"ALICE": 0, "GONE": 2}
    assert VoiceMapFile.all_set_to(["ALICE", "BOB"], 1).voices == {"ALICE": 1, "BOB": 1}
    assert VoiceMapFile.all_set_to(["ALICE", "BOB"], None) == VoiceMapFile()


def test_voice_submenu_lists_voices_then_clear_and_reports_choice(monkeypatch) -> None:
    # One item per project voice (current one marked), then the clear item;
    # selecting passes the voice index, or None for clear.
    captured = {}
    monkeypatch.setattr(MenuUtil, "menu", lambda state, **kwargs: captured.update(kwargs))
    state = make_state("general", "narrator")
    chosen = []
    voice_submenu(state, heading="Assign ALICE to...", current=1, clear_label="Clear", on_select=chosen.append)

    items = captured["items"]
    assert captured["heading"] == "Assign ALICE to..." and captured["one_shot"]
    assert [strip_ansi_codes(get_string_from(state, item.label)) for item in items] == [
        "Voice 1 general", "Voice 2 narrator (selected)", "Clear"]
    for item in items:
        item.handler(state, item)
    assert chosen == [0, 1, None]


def make_apply_state():
    book, attributions = make_fixture()
    saved = []
    project = SimpleNamespace(
        book=book,
        voice_references=[Voice(name="general"), Voice(name="narrator"), Voice(name="bob")],
        sound_segments=None,
        save=lambda: saved.append("project") or "",
    )
    return cast(State, SimpleNamespace(project=project)), attributions, saved


def characters(*ids: str) -> list[RosterCharacter]:
    return [RosterCharacter(id=id, name=id.title()) for id in ids]


@pytest.fixture
def quiet_apply(monkeypatch):
    """Stubs the console interactions of run_apply; returns printed feedback."""
    feedback = []
    monkeypatch.setattr(voices_menu, "print_feedback", lambda message, **_: feedback.append(message))
    monkeypatch.setattr(voices_menu, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(MenuUtil, "print_screen_heading", lambda *args, **kwargs: None)
    monkeypatch.setattr(voices_menu.ask, "ask_confirm", lambda *args: True)
    return feedback


def test_apply_preview_groups_by_voice_and_warns() -> None:
    # Characters sharing a voice share a row; unchosen speakers are listed as
    # unchanged; stale results and existing audio each add a note.
    state, attributions, _ = make_apply_state()
    voice_map = VoiceMapFile({"ALICE": 2, "CAROL": 2, "BOB": 0})
    plan = VoiceApplyPlan.make(state.project.book, ROSTER_IDS, attributions, voice_map)
    changed = plan.changed_lines(state.project.book)
    lines = [strip_ansi_codes(line) for line in make_apply_preview(
        state, characters("ALICE", "BOB", "CAROL"), voice_map, plan, changed, generated={1})]

    assert "  Voice 1 general  BOB (1)" in lines
    assert "  Voice 3 bob      ALICE (1), CAROL (1)" in lines
    assert any(line.startswith("  Unchanged") and "UNKNOWN (1)" in line and "NOT_SPEECH (1)" in line
               for line in lines)
    assert "3 lines change voice. Voices already set on these lines are replaced." in lines
    text = "\n".join(lines)
    assert "1 passage skipped" in text
    assert "1 of these lines already have generated audio" in text


def test_run_apply_sets_voices_and_saves_text(monkeypatch, quiet_apply) -> None:
    state, attributions, saved = make_apply_state()
    monkeypatch.setattr(ProjectTextIOUtil, "save_book", lambda project: saved.append("text") or "")

    run_apply(state, characters("ALICE", "BOB", "CAROL"), attributions, VoiceMapFile({"ALICE": 2}))

    groups = state.project.book.flat_phrase_groups
    assert [groups[line].voice_index for line in (1, 2, 3)] == [2, 2, 1]
    assert saved == ["text"]
    assert quiet_apply == ["Updated 2 lines"]


def test_run_apply_rolls_back_when_saving_fails(monkeypatch, quiet_apply) -> None:
    # In-memory voices must match the file on disk after a failed save.
    state, attributions, saved = make_apply_state()
    errors = []
    monkeypatch.setattr(ProjectTextIOUtil, "save_book", lambda project: "disk full")
    monkeypatch.setattr(voices_menu.ask, "ask_error", errors.append)

    run_apply(state, characters("ALICE", "BOB", "CAROL"), attributions, VoiceMapFile({"ALICE": 2}))

    groups = state.project.book.flat_phrase_groups
    assert [groups[line].voice_index for line in (1, 2)] == [1, 1]
    assert errors and "disk full" in errors[0]
    assert saved == []


def test_run_apply_refuses_missing_voices_and_reports_no_ops(quiet_apply, monkeypatch) -> None:
    state, attributions, _ = make_apply_state()
    errors = []
    monkeypatch.setattr(voices_menu.ask, "ask_error", errors.append)
    roster = characters("ALICE", "BOB", "CAROL")

    run_apply(state, roster, attributions, VoiceMapFile({"ALICE": 7}))
    assert errors and "ALICE" in errors[0]

    run_apply(state, roster, attributions, VoiceMapFile())
    run_apply(state, roster, attributions, VoiceMapFile({"CAROL": 2}))  # Line 6 already has voice 2
    assert quiet_apply == ["No voices chosen", "Nothing to change: the dialog lines already have these voices"]
