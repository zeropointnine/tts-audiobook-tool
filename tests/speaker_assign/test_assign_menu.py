from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.speaker_assign.assign_menu import AssignData, ROSTER_TEXT_NOTICE, make_subheading
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterFile
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book


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
