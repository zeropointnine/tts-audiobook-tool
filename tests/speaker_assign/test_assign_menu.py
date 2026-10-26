from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.speaker_assign import assign_menu
from tts_audiobook_tool.menus.speaker_assign.assign_menu import (
    AssignData,
    ROSTER_TEXT_NOTICE,
    make_subheading,
    play_outcome_sound,
)
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.attribution_runner import RunOutcome
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter, RosterFile
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
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


def print_assign_results(monkeypatch, book, attributions) -> list[str]:
    roster = Roster(tuple(RosterCharacter(id=id, name=id.title()) for id in sorted(ROSTER_IDS)))
    data = AssignData(RosterFile(roster), attributions, AssignPlan.make(book, attributions))
    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book)))
    monkeypatch.setattr(assign_menu, "load_assign_data", lambda _: data)
    lines: list[str] = []
    monkeypatch.setattr(assign_menu, "printt", lambda s="": lines.append(strip_ansi_codes(s)))
    assert assign_menu.print_assign_results(state)
    return lines


def test_assign_results_list_not_speech_as_a_row(monkeypatch):
    # NOT_SPEECH is a table row like any other id, so the rows add up to the
    # progress total and no extra note is printed for it.
    lines = print_assign_results(monkeypatch, *make_fixture())

    assert any(line.split()[1:2] == ["UNKNOWN"] for line in lines)
    assert any(line.split()[:2] == ["1", "NOT_SPEECH"] for line in lines)
    i = next(i for i, line in enumerate(lines) if "dialog passages assigned" in line)
    assert lines[i + 1:] == [""]


@pytest.mark.parametrize("one_offs, expected", [
    ([], "     7  OTHER  Not found in roster: bot (4), gearhead (3)"),
    (["OTHER:guard"], "     8  OTHER  Not found in roster: bot (4), gearhead (3), ..."),
    (["OTHER"], "     8  OTHER  Not found in roster: bot (4), gearhead (3), ..."),
])
def test_assign_results_name_recurring_others_in_the_other_row(monkeypatch, one_offs, expected):
    # The OTHER row counts every non-roster speaker but names only the
    # recurring ones, most first; "..." marks that some (one-off or unnamed)
    # aren't named.
    speakers = ["OTHER:gearhead"] * 3 + ["OTHER:bot"] * 4 + one_offs
    book = make_book([(f'"Line {i}." ', Reason.PARAGRAPH, DIALOG) for i in range(len(speakers))])
    attributions = AttributionsFile((RangeAttribution(
        0, len(speakers), UtteranceUtil.range_text_hash(book, 0, len(speakers)),
        tuple(UtteranceAttribution((i,), speaker) for i, speaker in enumerate(speakers))),))

    lines = print_assign_results(monkeypatch, book, attributions)

    assert expected in lines
    assert not any(line.startswith("-") for line in lines)
