from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_DIM_ITALICS, COL_ERROR
from tts_audiobook_tool.menus.speaker_assign import speaker_assign_menu
from tts_audiobook_tool.menus.speaker_assign import roster_menu as roster_menu_module
from tts_audiobook_tool.menus.speaker_assign.roster_menu import create_roster, roster_menu
from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu import SpeakerAssignMenu
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult, LlmRequestUtil
from tts_audiobook_tool.speaker_assign.roster import Prominence, Roster, RosterCharacter, RosterFile
from tts_audiobook_tool.speaker_assign.roster_util import NormalizedRoster, RosterUtil
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
            for item in (items(state)[:1] if kwargs["heading"] == "Auto-assign character voices" else items(state))
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
    assert captured[1] == (["Start"] if status == "missing" else [
        "Start",
        "View roster data" + marker,
        f"View roster data, expanded{marker} {COL_DIM}(possible spoilers)",
    ])


def test_expanded_roster_view_prints_character_blocks(monkeypatch):
    # The expanded view shows each character as a block with its aliases and description.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    roster = Roster((
        RosterCharacter(
            id="HARU_KOYAMA", name="Haru Koyama", aliases=("Haru", "Miss Haru", "Koyama"),
            gender="female", description="A teenage girl from Tokyo.", prominence=Prominence.MAJOR,
        ),
    ))
    roster_file = RosterFile(roster, section_text_hashes=UtteranceUtil.section_text_hashes(book))
    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book, dir_path="unused")))
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_roster_file", lambda _: roster_file)
    lines: list[str] = []
    monkeypatch.setattr(roster_menu_module, "printt", lambda s="": lines.append(s))

    assert roster_menu_module.print_roster_info(state, expanded=True)

    i = lines.index(
        f"{COL_DEFAULT}[1] {COL_DIM}id: {COL_ACCENT}HARU_KOYAMA  {COL_DIM}tier: {COL_DEFAULT}major      "
        f"{COL_DIM}gender: {COL_DEFAULT}female"
    )
    assert lines[i + 1:i + 6] == [
        f"    {COL_DIM}name: {COL_DEFAULT}Haru Koyama",
        f"    {COL_DIM}aliases: {COL_DEFAULT}Haru, Miss Haru, Koyama",
        "",
        f"{COL_DIM_ITALICS}A teenage girl from Tokyo.",
        "",
    ]


@pytest.mark.parametrize("expanded", [False, True])
def test_roster_view_puts_stale_warning_first_and_provenance_last(monkeypatch, expanded):
    # The stale warning opens the printout; the created/model line closes it after a blank line.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    roster = Roster((RosterCharacter(id="HARU", name="Haru"),))
    roster_file = RosterFile(
        roster, created="2026-10-24T18:43:10-04:00", llm={"model": "some/model"},
        section_text_hashes=("old text",),
    )
    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book, dir_path="unused")))
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_roster_file", lambda _: roster_file)
    lines: list[str] = []
    monkeypatch.setattr(roster_menu_module, "printt", lambda s="": lines.append(s))

    assert roster_menu_module.print_roster_info(state, expanded=expanded)

    assert lines[:2] == [
        f"{COL_ERROR}* {COL_DIM_ITALICS}The project text has changed since this roster was created", "",
    ]
    assert lines[-3:] == [
        "",
        f"{COL_DEFAULT}Created: {COL_DIM}2026-10-24 18:43, {COL_DEFAULT}model: {COL_DIM}some/model",
        "",
    ]
    assert lines[-4] != ""


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
    monkeypatch.setattr(roster_menu_module, "print_roster_info", lambda _: False)
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
