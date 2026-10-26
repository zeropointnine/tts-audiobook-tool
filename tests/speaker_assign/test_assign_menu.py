import os
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.menus.speaker_assign import assign_menu
from tts_audiobook_tool.menus.speaker_assign.assign_menu import (
    AssignData,
    ROSTER_TEXT_NOTICE,
    make_subheading,
    play_outcome_sound,
)
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.attribution_runner import RunOutcome
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter, RosterFile
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book
from tests.speaker_assign.test_voice_apply import ROSTER_IDS, make_fixture


@pytest.mark.parametrize("stale", [False, True])
def test_roster_notice_explains_reuse_without_implying_assignments_were_deleted(stale):
    # Roster provenance is advisory, separate from invalidation of text-bound assignments.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    hashes = ("old text",) if stale else UtteranceUtil.section_text_hashes(book)
    roster = RosterFile(Roster(()), section_text_hashes=hashes)
    saved = AttributionsFile()
    data = AssignData(roster, saved, AssignPlan.make(book, saved))
    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book)))
    heading = make_subheading(state, data)
    assert (ROSTER_TEXT_NOTICE in heading) == stale
    if stale:
        assert f"{COL_DIM}{ROSTER_TEXT_NOTICE}" in heading
        assert f"{COL_ERROR}{ROSTER_TEXT_NOTICE}" not in heading


@pytest.mark.parametrize("outcome, has_failures, expected", [
    (RunOutcome.FINISHED, False, "done"),
    (RunOutcome.FINISHED, True, None),
    (RunOutcome.FAILED, False, None),
    (RunOutcome.STOPPED, False, None),
    (RunOutcome.CANCELLED, False, None),
])
def test_outcome_sound(monkeypatch, outcome, has_failures, expected):
    # Only a fully successful run plays a sound (done); errors and user stops play nothing.
    played: list[str] = []
    monkeypatch.setattr(app_support, "play_done_sound", lambda: played.append("done"))
    monkeypatch.setattr(app_support, "play_fatal_gen_sound", lambda: played.append("error"))
    play_outcome_sound(outcome, has_failures)
    assert played == ([expected] if expected else [])


def test_clear_deletes_saved_assignments_and_drops_the_item(monkeypatch, tmp_path):
    # "Clear" is offered while any results are saved; once confirmed, it deletes
    # them and the menu falls back to a plain "Start". Step 3's voice mappings
    # stay: the roster's character ids they map are unchanged.
    book, attributions = make_fixture()
    roster_file = RosterFile(Roster(tuple(RosterCharacter(id=id, name=id) for id in sorted(ROSTER_IDS))))
    path = SpeakerAssignFileUtil.attributions_path(str(tmp_path))
    os.makedirs(os.path.dirname(path))
    voices_path = SpeakerAssignFileUtil.voices_path(str(tmp_path))
    for file_path in (path, voices_path):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("{}")

    def load(_) -> AssignData:
        saved = attributions if os.path.exists(path) else AttributionsFile()
        return AssignData(roster_file, saved, AssignPlan.make(book, saved))

    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book, dir_path=str(tmp_path))))
    monkeypatch.setattr(SpeakerAssignFileUtil, "has_roster", lambda _: True)
    monkeypatch.setattr(assign_menu, "load_assign_data", load)
    monkeypatch.setattr(assign_menu, "printt", lambda *_: None)
    monkeypatch.setattr(assign_menu.ask, "ask_confirm", lambda *_: True)
    feedback: list[str] = []
    monkeypatch.setattr(assign_menu, "print_feedback", lambda s, **_: feedback.append(s))
    monkeypatch.setattr(MenuUtil, "print_heading", lambda *_, **__: None)
    monkeypatch.setattr(MenuUtil, "make_breadcrumb_text", lambda *_: "")
    captured = []
    monkeypatch.setattr(MenuUtil, "menu", lambda *_, items, **__: captured.append(items))

    assign_menu.assign_menu(state)
    items_maker = captured[0]
    clear = items_maker(state)[-1]
    assert clear.label == "Clear"

    clear.handler(state, clear)

    assert not os.path.exists(path)
    assert os.path.exists(voices_path)
    assert feedback == ["Dialog assignments cleared"]
    assert [item.label for item in items_maker(state)] == ["Start"]
