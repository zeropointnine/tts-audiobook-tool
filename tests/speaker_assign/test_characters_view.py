from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_DIM_ITALICS, COL_ERROR
from tts_audiobook_tool.menus.speaker_assign import characters_view
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.roster import Prominence, Roster, RosterCharacter, RosterFile
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book
from tests.speaker_assign.test_voice_apply import ROSTER_IDS, make_fixture


def print_characters(
        monkeypatch, book, roster_file, attributions=None, expanded=False, strip=True) -> list[str]:
    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book, dir_path="unused")))
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_roster_file", lambda _: roster_file)
    monkeypatch.setattr(SpeakerAssignFileUtil, "load_attributions_file", lambda _: attributions or AttributionsFile())
    lines: list[str] = []
    monkeypatch.setattr(characters_view, "printt", lambda s="": lines.append(strip_ansi_codes(s) if strip else s))
    assert characters_view.print_characters_info(state, expanded=expanded)
    return lines


def make_roster_file(book, ids) -> RosterFile:
    return RosterFile(
        Roster(tuple(RosterCharacter(id=id, name=id.title()) for id in ids)),
        section_text_hashes=UtteranceUtil.section_text_hashes(book),
    )


def test_expanded_view_prints_character_blocks(monkeypatch):
    # The expanded view shows each character as a block with its aliases and description.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    roster = Roster((
        RosterCharacter(
            id="HARU_KOYAMA", name="Haru Koyama", aliases=("Haru", "Miss Haru", "Koyama"),
            gender="female", description="A teenage girl from Tokyo.", prominence=Prominence.MAJOR,
        ),
    ))
    roster_file = RosterFile(roster, section_text_hashes=UtteranceUtil.section_text_hashes(book))

    lines = print_characters(monkeypatch, book, roster_file, expanded=True, strip=False)

    i = lines.index(
        f"{COL_DIM}[1] id: {COL_ACCENT}HARU_KOYAMA  {COL_DIM}tier: {COL_DEFAULT}major      "
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
def test_view_puts_stale_warning_first_and_provenance_last(monkeypatch, expanded):
    # The stale warning opens the printout; the created/model line closes it after a blank line.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    roster_file = RosterFile(
        Roster((RosterCharacter(id="HARU", name="Haru"),)),
        created="2026-10-24T18:43:10-04:00", llm={"model": "some/model"},
        section_text_hashes=("old text",),
    )

    lines = print_characters(monkeypatch, book, roster_file, expanded=expanded, strip=False)

    assert lines[:2] == [
        f"{COL_ERROR}* {COL_DIM_ITALICS}The project text has changed since this roster was created", "",
    ]
    assert lines[-3:] == [
        "",
        f"{COL_DEFAULT}Created: {COL_DIM}2026-10-24 18:43, {COL_DEFAULT}model: {COL_DIM}some/model",
        "",
    ]
    assert lines[-4] != ""


def test_before_attribution_lists_roster_order_with_index(monkeypatch):
    # Without step 2 results, the first column is the roster index, in roster
    # order, and there's no progress line.
    book, _ = make_fixture()
    lines = print_characters(monkeypatch, book, make_roster_file(book, ["CAROL", "ALICE", "BOB"]))

    assert lines[0].split() == ["#", "id", "tier", "gender", "name"]
    assert [line.split()[:2] for line in lines[1:4]] == [["1", "CAROL"], ["2", "ALICE"], ["3", "BOB"]]
    assert not any("dialog passages assigned" in line for line in lines)


def test_after_attribution_lists_characters_by_count_then_reserved_rows(monkeypatch):
    # With step 2 results, the first column is the dialog count: characters
    # most first (unused ones included), then the reserved answers, then the
    # progress line. Stale results (chapter 2) aren't counted.
    book, attributions = make_fixture()
    lines = print_characters(monkeypatch, book, make_roster_file(book, ["DAVE", "CAROL", "ALICE", "BOB"]), attributions)

    assert lines[0].split() == ["count", "id", "tier", "gender", "name"]
    assert [line.split()[:2] for line in lines[1:7]] == [
        ["1", "CAROL"], ["1", "ALICE"], ["1", "BOB"], ["0", "DAVE"], ["1", "UNKNOWN"], ["1", "NOT_SPEECH"],
    ]
    assert any("5 of 6 dialog passages assigned" in line for line in lines)


def test_expanded_view_shows_counts_and_reserved_rows(monkeypatch):
    # The expanded blocks gain a count once step 2 has results; the reserved
    # answers follow as a small table.
    book, attributions = make_fixture()
    lines = print_characters(monkeypatch, book, make_roster_file(book, sorted(ROSTER_IDS)), attributions, expanded=True)

    assert any(line.startswith("[1] id: ALICE") and line.endswith("count: 1") for line in lines)
    assert any(line.split()[:2] == ["1", "NOT_SPEECH"] for line in lines)
    assert not any(line.startswith("Full roster data") for line in lines)


def test_not_speech_is_listed_as_a_row(monkeypatch):
    # NOT_SPEECH is a table row like any other id, so the rows add up to the
    # progress total and no extra note is printed for it.
    book, attributions = make_fixture()
    lines = print_characters(monkeypatch, book, make_roster_file(book, sorted(ROSTER_IDS)), attributions)

    assert any(line.split()[1:2] == ["UNKNOWN"] for line in lines)
    assert any(line.split()[:2] == ["1", "NOT_SPEECH"] for line in lines)
    i = next(i for i, line in enumerate(lines) if "dialog passages assigned" in line)
    assert lines[i + 1] == ""
    assert lines[i + 2].startswith("Full roster data")


@pytest.mark.parametrize("one_offs, expected", [
    ([], "Not found in roster: bot (4), gearhead (3)"),
    (["OTHER:guard"], "Not found in roster: bot (4), gearhead (3), ..."),
    (["OTHER"], "Not found in roster: bot (4), gearhead (3), ..."),
])
def test_other_row_names_recurring_others(monkeypatch, one_offs, expected):
    # The OTHER row counts every non-roster speaker but names only the
    # recurring ones, most first; "..." marks that some (one-off or unnamed)
    # aren't named.
    speakers = ["OTHER:gearhead"] * 3 + ["OTHER:bot"] * 4 + one_offs
    book = make_book([(f'"Line {i}." ', Reason.PARAGRAPH, DIALOG) for i in range(len(speakers))])
    attributions = AttributionsFile((RangeAttribution(
        0, len(speakers), UtteranceUtil.range_text_hash(book, 0, len(speakers)),
        tuple(UtteranceAttribution((i,), speaker) for i, speaker in enumerate(speakers))),))

    lines = print_characters(monkeypatch, book, make_roster_file(book, sorted(ROSTER_IDS)), attributions)

    row = next(line for line in lines if line.split()[1:2] == ["OTHER"])
    assert row.split()[0] == str(len(speakers))
    assert row.endswith(expected)
