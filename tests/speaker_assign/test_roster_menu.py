from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.speaker_assign import speaker_assign_menu
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
    book.segmentation_settings = book.segmentation_settings._replace(dialog_segmentation=True)
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


@pytest.mark.parametrize("dialog_segmentation", [False, True])
@pytest.mark.parametrize("has_dialog", [False, True])
def test_step_one_checks_dialog_segmentation_on_selection(monkeypatch, dialog_segmentation, has_dialog):
    # The import setting controls the gate; the warning appears only after selection.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, has_dialog)])
    book.segmentation_settings = book.segmentation_settings._replace(dialog_segmentation=dialog_segmentation)
    state = cast(State, SimpleNamespace(project=SimpleNamespace(
        book=book, dir_path="unused", phrase_groups=book.flat_phrase_groups,
    )))
    monkeypatch.setattr(SpeakerAssignFileUtil, "has_roster", lambda _: False)
    captured = []
    warnings = []
    opened = []
    monkeypatch.setattr(MenuUtil, "menu", lambda *args, **kwargs: captured.append(kwargs))
    monkeypatch.setattr(speaker_assign_menu.ask, "ask_enter_to_continue", warnings.append)
    monkeypatch.setattr(speaker_assign_menu, "roster_menu", opened.append)

    SpeakerAssignMenu.menu(state)

    menu = captured[0]
    assert "imported without dialog segmentation" not in menu["subheading"]
    assert "Re-import it" not in menu["subheading"]
    assert "Requires text imported with dialog segmentation enabled." not in menu["subheading"]
    assert not warnings
    step_one = menu["items"](state)[0]
    expected_label = "Step 1: Create character roster"
    if not dialog_segmentation:
        expected_label += f" {COL_DIM}(requires dialog segmentation)"
    assert step_one.label(state) == expected_label

    step_one.handler(state, step_one)

    if dialog_segmentation:
        assert opened == [state]
        assert not warnings
        if not has_dialog:
            assert f"{COL_ERROR}No dialog found in the project text." in menu["subheading"]
    else:
        assert not opened
        assert warnings == [
            f"{COL_ERROR}The text was imported without dialog segmentation.\n"
            "Re-import it with dialog segmentation enabled (Text > Segmentation settings).\n"
        ]
        if not has_dialog:
            assert menu["subheading"] == speaker_assign_menu.SPEAKER_ASSIGN_DESC


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
