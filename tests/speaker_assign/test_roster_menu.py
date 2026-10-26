import os
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.speaker_assign import speaker_assign_menu
from tts_audiobook_tool.menus.speaker_assign import roster_menu as roster_menu_module
from tts_audiobook_tool.menus.speaker_assign.roster_menu import create_roster, roster_menu
from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu import SpeakerAssignMenu
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult, LlmRequestUtil
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterFile
from tts_audiobook_tool.speaker_assign.roster_util import NormalizedRoster, RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book


@pytest.mark.parametrize("status", ["current", "stale", "unreadable", "missing"])
def test_roster_menu_labels_mark_only_known_stale_data(monkeypatch, status):
    # Step 1 and the "View character/dialog results" item expose stale provenance without
    # marking missing/unreadable data; the view item needs a saved roster, and
    # step 1 offers "Start over" and "Clear" once one exists.
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
            for item in (items(state) if callable(items) else items)
        ])

    monkeypatch.setattr(MenuUtil, "menu", capture_menu)
    SpeakerAssignMenu.menu(state)
    roster_menu(state)

    marker = f"{COL_ERROR}*" if status == "stale" else ""
    expected_step = (
        "Step 1: Generate character roster" if status == "missing" else
        f"Step 1: Generate character roster {COL_DIM}(completed{marker}{COL_DIM})"
    )
    assert captured[0][0] == expected_step
    assert len(captured[0]) == (3 if status == "missing" else 4)
    if status != "missing":
        assert captured[0][-1] == "View character/dialog results" + marker
    # An existing roster (even an unreadable one) can be replaced or cleared
    assert captured[1] == (["Start"] if status == "missing" else ["Start over", "Clear"])


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
    expected_label = "Step 1: Generate character roster"
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


@pytest.mark.parametrize("outcome, expected", [
    ("success", ["done"]),
    ("error", []),
    ("bad_reply", []),
    ("cancelled", []),
])
def test_create_roster_outcome_sound(monkeypatch, tmp_path, outcome, expected):
    # Only a saved roster plays a sound (done); request/reply errors and a user cancel play nothing.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    state = cast(State, SimpleNamespace(
        prefs=None, project=SimpleNamespace(book=book, dir_path=str(tmp_path)),
    ))
    played: list[str] = []
    monkeypatch.setattr(app_support, "play_done_sound", lambda: played.append("done"))
    monkeypatch.setattr(app_support, "play_fatal_gen_sound", lambda: played.append("error"))
    monkeypatch.setattr(roster_menu_module, "print_llm_info", lambda _: None)
    monkeypatch.setattr(roster_menu_module, "print_characters_info", lambda _: False)
    monkeypatch.setattr(roster_menu_module.ask, "ask_error", lambda *_: None)
    monkeypatch.setattr(roster_menu_module.ask, "ask_enter_to_continue", lambda *_: None)
    monkeypatch.setattr(LlmRequestUtil, "make_session", lambda *_: None)
    monkeypatch.setattr(LlmRequestUtil, "describe_llm", lambda _: {})
    monkeypatch.setattr(LlmRequestUtil, "describe_error", lambda *_: "error")
    monkeypatch.setattr(SpeakerAssignFileUtil, "save_roster_file", lambda *_: "")
    monkeypatch.setattr(RosterUtil, "normalize_roster", lambda _: NormalizedRoster(Roster(())))
    reply = "not json" if outcome == "bad_reply" else '{"characters": []}'
    result = LlmRequestResult(
        reply=reply, reasoning="", elapsed=1.0,
        cancelled=outcome == "cancelled",
        error=RuntimeError("boom") if outcome == "error" else None,
        finish_reason="stop",
    )
    monkeypatch.setattr(LlmRequestUtil, "send_cancellable", lambda *_, **__: result)

    create_roster(state)

    assert played == expected


@pytest.mark.parametrize("confirmed", [False, True])
def test_clear_roster_deletes_roster_and_dependents_after_confirmation(monkeypatch, tmp_path, confirmed):
    # Clearing step 1 also discards step 2's results and step 3's voice
    # mappings (both keyed by the roster's character ids), only once confirmed.
    state = cast(State, SimpleNamespace(project=SimpleNamespace(dir_path=str(tmp_path))))
    paths = [SpeakerAssignFileUtil.roster_path(str(tmp_path)), *write_dependents(tmp_path)]
    with open(paths[0], "w", encoding="utf-8") as file:
        file.write("{}")
    printed: list[str] = []
    feedback: list[str] = []
    monkeypatch.setattr(roster_menu_module, "printt", lambda s="": printed.append(s))
    monkeypatch.setattr(roster_menu_module.ask, "ask_confirm", lambda *_: confirmed)
    monkeypatch.setattr(roster_menu_module, "print_feedback", lambda s, **_: feedback.append(s))

    roster_menu_module.clear_roster(state)

    assert any("Dialog assignments from step 2" in line for line in printed)
    assert any("Voice mappings from step 3" in line for line in printed)
    assert [os.path.exists(path) for path in paths] == [not confirmed] * 3
    assert feedback == (["Roster cleared"] if confirmed else [])


def write_dependents(tmp_path) -> list[str]:
    """Writes placeholder step 2 and 3 files; returns their paths."""
    paths = [SpeakerAssignFileUtil.attributions_path(str(tmp_path)), SpeakerAssignFileUtil.voices_path(str(tmp_path))]
    os.makedirs(SpeakerAssignFileUtil.dir_path(str(tmp_path)), exist_ok=True)
    for path in paths:
        with open(path, "w", encoding="utf-8") as file:
            file.write("{}")
    return paths


def test_regenerated_roster_discards_dialog_assignments_and_voice_mappings(monkeypatch, tmp_path):
    # A new roster may use different character ids, so step 2 and step 3 data
    # are both discarded once it's saved, and the run says so.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    state = cast(State, SimpleNamespace(prefs=None, project=SimpleNamespace(book=book, dir_path=str(tmp_path))))
    paths = write_dependents(tmp_path)
    printed: list[str] = []
    monkeypatch.setattr(roster_menu_module, "printt", lambda s="": printed.append(s))
    monkeypatch.setattr(roster_menu_module, "print_llm_info", lambda _: None)
    monkeypatch.setattr(roster_menu_module, "print_characters_info", lambda _: False)
    monkeypatch.setattr(app_support, "play_done_sound", lambda: None)
    monkeypatch.setattr(LlmRequestUtil, "make_session", lambda *_: None)
    monkeypatch.setattr(LlmRequestUtil, "describe_llm", lambda _: {})
    monkeypatch.setattr(SpeakerAssignFileUtil, "save_roster_file", lambda *_: "")
    monkeypatch.setattr(RosterUtil, "normalize_roster", lambda _: NormalizedRoster(Roster(())))
    result = LlmRequestResult(reply='{"characters": []}', reasoning="", elapsed=1.0, finish_reason="stop")
    monkeypatch.setattr(LlmRequestUtil, "send_cancellable", lambda *_, **__: result)

    create_roster(state)

    assert not any(os.path.exists(path) for path in paths)
    assert f"{COL_DIM}Discarded dialog assignments and voice mappings made with the previous roster" in printed


@pytest.mark.parametrize("existing", ["nothing", "roster", "dependents"])
def test_roster_replacement_is_confirmed_only_when_data_would_be_lost(monkeypatch, tmp_path, existing):
    # Starting step 1 asks first (in place, under the menu) only when it would
    # replace a roster or discard step 2/3 data, listing what would be lost.
    state = cast(State, SimpleNamespace(project=SimpleNamespace(dir_path=str(tmp_path))))
    if existing == "dependents":
        write_dependents(tmp_path)
    monkeypatch.setattr(SpeakerAssignFileUtil, "has_roster", lambda _: existing == "roster")
    printed: list[str] = []
    asked: list[str] = []
    monkeypatch.setattr(roster_menu_module, "printt", lambda s="": printed.append(s))
    monkeypatch.setattr(roster_menu_module.ask, "ask_confirm", lambda *_: asked.append("asked") or False)

    confirmed = roster_menu_module.confirm_roster_replacement(state)

    assert confirmed == (existing == "nothing")
    assert asked == ([] if existing == "nothing" else ["asked"])
    assert any("already exists" in line for line in printed) == (existing == "roster")
    assert any("Voice mappings from step 3" in line for line in printed) == (existing == "dependents")
