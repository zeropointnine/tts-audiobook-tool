from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.menus.speaker_assign import seed_voices_menu
from tts_audiobook_tool.menus.speaker_assign.seed_voices_menu import SeedPlan, make_numbered_columns
from tts_audiobook_tool.project_support.project_voice_util import VOICE_NAME_MAX_LENGTH
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes

from tests.speaker_assign.test_voice_apply import make_fixture


def test_plan_skips_existing_names_case_insensitively_and_truncates_long_ids() -> None:
    # Names are compared case-insensitively (as voice names are); ids past the
    # voice-name limit are truncated, and a truncated name can collide too.
    voices = [Voice(), Voice(name="Bob")]
    long_a = "A" * VOICE_NAME_MAX_LENGTH + "_1"
    long_b = "A" * VOICE_NAME_MAX_LENGTH + "_2"
    plan = SeedPlan.make(voices, ["ALICE", "BOB", long_a, long_b])
    assert plan.added == (("ALICE", "ALICE"), (long_a, "A" * VOICE_NAME_MAX_LENGTH))
    assert plan.skipped == 2


def test_plan_maps_added_characters_to_their_new_voices() -> None:
    # The added voices follow the existing ones; other mappings are kept.
    plan = SeedPlan(added=(("ALICE", "ALICE"), ("CAROL", "CAROL")), skipped=0)
    voice_map = plan.make_voice_map(VoiceMapFile({"ALICE": 0, "BOB": 1}), first_index=3)
    assert voice_map.voices == {"ALICE": 3, "BOB": 1, "CAROL": 4}


@pytest.mark.parametrize("added, skipped, expected", [
    (1, 0, "Added 1 item, mapped to its character in step 3."),
    (2, 1, "Added 2 items, mapped to their characters in step 3. Skipped 1 item which already exists."),
    (0, 2, "Added 0 items. Skipped 2 items which already exist."),
])
def test_plan_message(added: int, skipped: int, expected: str) -> None:
    plan = SeedPlan(added=tuple((f"C{i}", f"C{i}") for i in range(added)), skipped=skipped)
    assert plan.make_message() == expected


def test_numbered_list_shows_counts_and_uses_two_columns_past_ten_items() -> None:
    # Each id is followed by its (dimmed) dialog count. Numbered down the first
    # column, then the second; the first gets the extra row. Columns align by
    # visible width, color codes aside.
    def plain(lines: list[str]) -> list[str]:
        return [strip_ansi_codes(line) for line in lines]

    assert plain(make_numbered_columns(["A", "B"], {"A": 1234, "B": 5})) == ["[1] A (1,234)", "[2] B (5)"]
    ids = [f"ID{i}" for i in range(1, 12)]
    counts = {"ID1": 100, "ID2": 5}
    lines = plain(make_numbered_columns(ids, counts))
    assert len(lines) == 6
    assert lines[0] == "[1] ID1 (100)    [7] ID7 (0)"
    assert lines[1] == "[2] ID2 (5)      [8] ID8 (0)"
    assert lines[5] == "[6] ID6 (0)"


@pytest.fixture
def seed_state(monkeypatch):
    """A project whose step 2 results give ALICE, BOB and CAROL a passage each
    (DAVE none), with the console interactions stubbed. Returns (state, recorded)."""
    book, attributions = make_fixture()
    recorded = {"saved_map": None, "messages": [], "feedback": [], "errors": [], "project_saves": 0}

    def save_project() -> str:
        recorded["project_saves"] += 1
        return recorded.get("project_error", "")

    project = SimpleNamespace(
        dir_path="project", book=book, voice_references=[Voice(), Voice(name="carol")], save=save_project)
    roster = Roster(tuple(RosterCharacter(id=id, name=id.title()) for id in ("ALICE", "BOB", "CAROL", "DAVE")))
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_roster_file", lambda _: SimpleNamespace(roster=roster))
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_attributions_file", lambda _: attributions)
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_voices_file", lambda _: VoiceMapFile({"BOB": 0}))
    monkeypatch.setattr(SpeakerAssignFileUtil, "save_voices_file",
                        lambda _, voice_map: recorded.update(saved_map=voice_map) or "")
    monkeypatch.setattr(MenuUtil, "print_screen_heading",
                        lambda state, heading, subheading="", **_: recorded.update(subheading=subheading))
    monkeypatch.setattr(seed_voices_menu, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(seed_voices_menu, "print_feedback", lambda message, **_: recorded["feedback"].append(message))
    monkeypatch.setattr(seed_voices_menu.ask, "ask_enter_to_continue", recorded["messages"].append)
    monkeypatch.setattr(seed_voices_menu.ask, "ask_error", recorded["errors"].append)
    return cast(State, SimpleNamespace(project=project)), recorded


def run_seed(monkeypatch, state: State, inp: str) -> None:
    monkeypatch.setattr(seed_voices_menu.ask, "ask_input", lambda *args, **kwargs: inp)
    seed_voices_menu.seed_voices_menu(state)


def test_seed_appends_in_list_order_and_maps_characters(monkeypatch, seed_state) -> None:
    # The list is most dialog first, characters without dialog (DAVE) left out.
    # Input order doesn't matter, out-of-range numbers are ignored, and CAROL
    # is skipped since a voice already has that name.
    state, recorded = seed_state
    run_seed(monkeypatch, state, "3, 1-2, 9")

    assert strip_ansi_codes(recorded["subheading"]).endswith("[1] ALICE (1)\n[2] BOB (1)\n[3] CAROL (1)")
    assert [voice.name for voice in state.project.voice_references] == ["", "carol", "ALICE", "BOB"]
    assert recorded["saved_map"].voices == {"ALICE": 2, "BOB": 3}
    assert recorded["messages"] == [
        "Added 2 items, mapped to their characters in step 3. Skipped 1 item which already exists."]


def test_seed_rejects_bad_syntax_without_saving(monkeypatch, seed_state) -> None:
    state, recorded = seed_state
    run_seed(monkeypatch, state, "1, x")

    assert recorded["feedback"] == ["Bad value: x"]
    assert recorded["project_saves"] == 0 and recorded["saved_map"] is None
    assert recorded["messages"] == []


def test_seed_restores_voices_when_the_project_save_fails(monkeypatch, seed_state) -> None:
    # In-memory voices must match the file on disk; the voice map isn't saved.
    state, recorded = seed_state
    recorded["project_error"] = "disk full"
    run_seed(monkeypatch, state, "1")

    assert [voice.name for voice in state.project.voice_references] == ["", "carol"]
    assert recorded["errors"] == ["disk full"]
    assert recorded["saved_map"] is None
