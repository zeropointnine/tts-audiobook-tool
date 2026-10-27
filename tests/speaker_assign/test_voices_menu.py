from collections import Counter
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil, get_string_from
from tts_audiobook_tool.menus.speaker_assign import voices_menu
from tts_audiobook_tool.menus.speaker_assign.voices_menu import (
    make_character_label,
    make_character_subheading,
    run_apply,
    sort_characters,
    voice_submenu,
)
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.speaker_assign.roster import RosterCharacter
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes

from tests.speaker_assign.test_voice_apply import make_fixture


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


def test_character_labels_line_up_their_arrows() -> None:
    # Padded to the widest "ID (count)" of the whole list (every page), so the
    # arrows form a column; a label without a voice isn't padded.
    state = make_state("", "bob")
    roster = characters("SUNDAY_AHZMUNDIN", "CHIMP", "ALICE")
    counts = Counter({"SUNDAY_AHZMUNDIN": 473, "CHIMP": 232})
    width = voices_menu.make_character_label_width(roster, counts)
    assert width == len("SUNDAY_AHZMUNDIN (473)")
    labels = [strip_ansi_codes(make_character_label(state, c.id, counts[c.id], 1, width)) for c in roster]
    assert labels == [
        "SUNDAY_AHZMUNDIN (473) -> Voice 2 bob",
        "CHIMP (232)            -> Voice 2 bob",
        "ALICE (0)              -> Voice 2 bob",
    ]
    assert strip_ansi_codes(make_character_label(state, "CHIMP", 232, None, width)) == "CHIMP (232)"


def test_character_subheading_shows_name_details_and_voice_hint() -> None:
    # The voice picker's subheading: the name, known gender and narrator flag,
    # then the voice hint on its own line (omitted when there is none).
    baron = RosterCharacter(id="BARON", name="Baron de Verdira", gender="male", voice_hint="elderly; deep")
    assert strip_ansi_codes(make_character_subheading(baron)) == "Baron de Verdira (male)\nvoice hint: elderly; deep"
    narrator = RosterCharacter(id="NARRATOR", name="The physician", is_narrator=True)
    assert strip_ansi_codes(make_character_subheading(narrator)) == "The physician (narrator)"


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

    items = captured["items"](state)
    assert captured["heading"] == "Assign ALICE to..."
    assert [strip_ansi_codes(get_string_from(state, item.label)) for item in items] == [
        "Voice 1 general", "Voice 2 narrator (selected)", "Clear"]
    for item in items:
        assert item.handler(state, item) is True  # A choice closes the picker
    assert chosen == [0, 1, None]


def test_voice_picker_pages_chain_through_more_and_repeat_clear(monkeypatch) -> None:
    # Past a page's worth of voices, "More voices..." opens the next page;
    # every page ends with the clear item. A choice on a later page closes the
    # earlier pages too, while backing out of it returns to the previous page.
    menus = []
    monkeypatch.setattr(MenuUtil, "menu", lambda state, **kwargs: menus.append(kwargs))
    monkeypatch.setattr(voices_menu, "VOICES_PER_PAGE", 2)
    state = make_state("a", "b", "c", "d", "e")
    chosen = []
    voice_submenu(state, heading="Assign ALICE to...", current=4, clear_label="Clear", on_select=chosen.append,
                  subheading="Alice")

    def labels(menu: dict) -> list[str]:
        return [strip_ansi_codes(get_string_from(state, item.label)) for item in menu["items"](state)]

    assert labels(menus[0]) == ["Voice 1 a", "Voice 2 b", "More voices... (3)", "Clear"]
    more = menus[0]["items"](state)[2]
    assert more.handler(state, more) is False  # Backed out of page 2 (stubbed menu)
    assert labels(menus[1]) == ["Voice 3 c", "Voice 4 d", "More voices... (1)", "Clear"]
    assert strip_ansi_codes(menus[1]["heading"]) == "Assign ALICE to... (3-4 of 5)"
    assert menus[1]["breadcrumb"] == "Voices 3-4"
    assert [menu["subheading"] for menu in menus] == ["Alice", "Alice"]

    # Choosing on page 3 reports True up through page 2's "More voices..."
    def choose_on_page_3(state, **kwargs):
        menus.append(kwargs)
        last = kwargs["items"](state)[0]
        last.handler(state, last)
    monkeypatch.setattr(MenuUtil, "menu", choose_on_page_3)
    more = menus[1]["items"](state)[2]
    assert more.handler(state, more) is True
    assert labels(menus[2]) == ["Voice 5 e (selected)", "Clear"]
    assert chosen == [4]


def test_character_pages_chain_through_more_and_repeat_actions(monkeypatch) -> None:
    # Each page lists a page's worth of characters, then "More characters..."
    # while any are left (it opens the next page), then the same action items.
    menus = []
    monkeypatch.setattr(MenuUtil, "menu", lambda state, **kwargs: menus.append(kwargs))
    monkeypatch.setattr(voices_menu, "CHARACTERS_PER_PAGE", 2)
    state = make_state()
    roster = characters("A", "B", "C", "D", "E")

    def labels(menu: dict) -> list[str]:
        return [strip_ansi_codes(get_string_from(state, item.label)) for item in menu["items"](state)]

    def make_item(character: RosterCharacter) -> MenuItem:
        return MenuItem(character.id, lambda _, __: None)

    def make_action_items() -> list[MenuItem]:
        return [MenuItem("Set all to...", lambda _, __: None), MenuItem("Apply", lambda _, __: None)]

    voices_menu.characters_page_menu(state, roster, 0, make_item, make_action_items, "desc")
    while labels(menus[-1])[-3].startswith("More characters"):
        more = menus[-1]["items"](state)[-3]
        more.handler(state, more)

    assert [labels(menu) for menu in menus] == [
        ["A", "B", "More characters... (3)", "Set all to...", "Apply"],
        ["C", "D", "More characters... (1)", "Set all to...", "Apply"],
        ["E", "Set all to...", "Apply"],
    ]
    assert [menu["subheading"] for menu in menus] == ["desc", "", ""]
    assert strip_ansi_codes(menus[2]["heading"]) == "Map to voices (5-5 of 5)"


def test_apply_on_a_later_character_page_closes_every_page(monkeypatch) -> None:
    # Apply closes its page once done; each earlier page's "More characters..."
    # then reports done too, so the whole menu exits. Before that, returning
    # from a later page keeps the earlier one open.
    monkeypatch.setattr(voices_menu, "CHARACTERS_PER_PAGE", 2)
    state = make_state()
    roster = characters("A", "B", "C", "D", "E")
    done = False
    results = []

    def apply(_, __) -> bool:
        nonlocal done
        done = True
        return done

    def make_action_items() -> list[MenuItem]:
        return [MenuItem("Apply", apply)]

    def run_page(state, **kwargs):
        # Each page selects its last-but-one item ("More characters...") if
        # it has one, else Apply
        items = kwargs["items"](state)
        item = items[-2] if len(items) > 1 and "More" in str(items[-2].label) else items[-1]
        results.append(item.handler(state, item))
    monkeypatch.setattr(MenuUtil, "menu", run_page)

    voices_menu.characters_page_menu(
        state, roster, 0, lambda c: MenuItem(c.id, lambda _, __: None), make_action_items, "", lambda: done)
    assert results == [True, True, True]  # Page 3's Apply, then pages 2 and 1


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
    """Stubs the console interactions of run_apply (confirming); returns
    printed feedback and, after it, the confirmation prompts."""
    feedback, prompts = [], []
    monkeypatch.setattr(voices_menu, "print_feedback", lambda message, **_: feedback.append(message))
    monkeypatch.setattr(voices_menu.ask, "ask_confirm", lambda message="": prompts.append(message) or True)
    return feedback, prompts


def test_run_apply_confirms_then_sets_voices_and_saves_text(monkeypatch, quiet_apply) -> None:
    # The standard confirmation prompt (the Apply label gives the line count);
    # ALICE's passage spans 2 lines.
    feedback, prompts = quiet_apply
    state, attributions, saved = make_apply_state()
    monkeypatch.setattr(ProjectTextIOUtil, "save_book", lambda project: saved.append("text") or "")

    assert run_apply(state, characters("ALICE", "BOB", "CAROL"), attributions, VoiceMapFile({"ALICE": 2}))

    assert prompts == [""]  # ie, ask_confirm's default message
    groups = state.project.book.flat_phrase_groups
    assert [groups[line].voice_index for line in (1, 2, 3)] == [2, 2, 1]
    assert saved == ["text"]
    assert feedback == ["Updated 2 lines"]


def test_apply_label_counts_changing_lines_of_those_covered() -> None:
    # Of the lines in the chosen characters' passages, those whose voice would
    # change (line 6, CAROL's, already has voice 2); no count without a choice.
    state, attributions, _ = make_apply_state()
    book = state.project.book
    roster = characters("ALICE", "BOB", "CAROL")

    def label(voice_map: VoiceMapFile) -> str:
        return strip_ansi_codes(voices_menu.make_apply_label(book, roster, attributions, voice_map))

    assert label(VoiceMapFile()) == "Apply"
    assert label(VoiceMapFile({"ALICE": 2, "CAROL": 2})) == "Apply (changes voice for 2 of 3 lines)"
    assert label(VoiceMapFile({"CAROL": 2})) == "Apply (changes voice for 0 of 1 line)"


def test_run_apply_declined_changes_nothing(monkeypatch, quiet_apply) -> None:
    state, attributions, saved = make_apply_state()
    monkeypatch.setattr(voices_menu.ask, "ask_confirm", lambda *args: False)
    monkeypatch.setattr(ProjectTextIOUtil, "save_book", lambda project: pytest.fail("must not save"))

    assert not run_apply(state, characters("ALICE", "BOB", "CAROL"), attributions, VoiceMapFile({"ALICE": 2}))

    groups = state.project.book.flat_phrase_groups
    assert [groups[line].voice_index for line in (1, 2)] == [1, 1]
    assert quiet_apply[0] == []


def test_run_apply_rolls_back_when_saving_fails(monkeypatch, quiet_apply) -> None:
    # In-memory voices must match the file on disk after a failed save.
    state, attributions, saved = make_apply_state()
    errors = []
    monkeypatch.setattr(ProjectTextIOUtil, "save_book", lambda project: "disk full")
    monkeypatch.setattr(voices_menu.ask, "ask_error", errors.append)

    assert not run_apply(state, characters("ALICE", "BOB", "CAROL"), attributions, VoiceMapFile({"ALICE": 2}))

    groups = state.project.book.flat_phrase_groups
    assert [groups[line].voice_index for line in (1, 2)] == [1, 1]
    assert errors and "disk full" in errors[0]
    assert saved == []


def test_run_apply_refuses_missing_voices_and_reports_no_ops(quiet_apply, monkeypatch) -> None:
    state, attributions, _ = make_apply_state()
    errors = []
    monkeypatch.setattr(voices_menu.ask, "ask_error", errors.append)
    roster = characters("ALICE", "BOB", "CAROL")

    assert not run_apply(state, roster, attributions, VoiceMapFile({"ALICE": 7}))
    assert errors and "ALICE" in errors[0]

    assert not run_apply(state, roster, attributions, VoiceMapFile())
    # Line 6 already has voice 2
    assert not run_apply(state, roster, attributions, VoiceMapFile({"CAROL": 2}))
    assert quiet_apply[0] == ["No voices chosen", "Nothing to change: the dialog lines already have these voices"]
