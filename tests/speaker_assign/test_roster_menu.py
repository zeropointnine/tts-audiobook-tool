from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.speaker_assign.roster_menu import roster_menu
from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu import SpeakerAssignMenu
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterFile
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book


@pytest.mark.parametrize("status", ["current", "stale", "unreadable", "missing"])
def test_roster_menu_labels_mark_only_known_stale_data(monkeypatch, status):
    # Both navigation levels expose stale provenance without marking missing/unreadable data.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    hashes = ("old text",) if status == "stale" else UtteranceUtil.section_text_hashes(book)
    roster = RosterFile(Roster(()), section_text_hashes=hashes)
    state = cast(State, SimpleNamespace(project=SimpleNamespace(
        book=book, dir_path="unused", phrase_groups=book.flat_phrase_groups,
    )))
    monkeypatch.setattr(SpeakerAssignFileUtil, "has_roster", lambda _: status != "missing")
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_roster_file", lambda _: (
        "Cannot load roster" if status in ("unreadable", "missing") else roster
    ))
    captured = []

    def capture_menu(state, *, items, **kwargs):
        captured.append([
            item.label(state) if callable(item.label) else item.label
            for item in (items(state)[:1] if kwargs["heading"] == "Assign character voices using LLM" else items(state))
        ])

    monkeypatch.setattr(MenuUtil, "menu", capture_menu)
    SpeakerAssignMenu.menu(state)
    roster_menu(state)

    marker = f"{COL_ERROR}*" if status == "stale" else ""
    expected_step = (
        "Step 1: Create character roster" if status == "missing" else
        f"Step 1: Create/view character roster {COL_DIM}(completed)" + marker
    )
    assert captured[0][0] == expected_step
    assert captured[1] == (["Start"] if status == "missing" else ["Start", "View roster data" + marker])


@pytest.mark.parametrize("has_assignments", [False, True])
def test_roster_menu_omits_discard_notice(monkeypatch, tmp_path, has_assignments):
    # The description omits the discard warning regardless of saved step-2 data.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book, dir_path=str(tmp_path))))
    if has_assignments:
        path = tmp_path / "attributions.json"
        path.write_text("{}", encoding="utf-8")
    else:
        path = tmp_path / "missing.json"
    monkeypatch.setattr(SpeakerAssignFileUtil, "attributions_path", lambda _: str(path))
    captured = []
    monkeypatch.setattr(MenuUtil, "menu", lambda *args, **kwargs: captured.append(kwargs["subheading"]))

    roster_menu(state)

    notice = "Re-creating the roster discards any dialog assignments from step 2."
    assert notice not in captured[0]
