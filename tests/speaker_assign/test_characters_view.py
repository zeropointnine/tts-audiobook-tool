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


@pytest.mark.parametrize("index, view", [(0, "Summary"), (1, "Expanded")])
def test_heading_folds_results_into_title(monkeypatch, index, view):
    # Both screen titles include the results view instead of repeating it in the breadcrumb.
    from tts_audiobook_tool.menus.menu_util import MenuFrame, MenuUtil

    state = cast(State, SimpleNamespace())
    frames = [MenuFrame(heading=heading) for heading in (
        "Main", "Voices", "Auto-assign character voices", characters_view.RESULTS_HEADING,
    )]
    monkeypatch.setattr(MenuUtil, "menu_frames", frames)
    headings = []
    monkeypatch.setattr(MenuUtil, "print_heading", lambda state, title, **kwargs: headings.append((title, kwargs)))
    monkeypatch.setattr(characters_view, "print_characters_info", lambda state, **kwargs: False)

    def choose_view(state, *, items, **kwargs):
        items[index].handler(state, items[index])

    monkeypatch.setattr(MenuUtil, "menu", choose_view)
    characters_view.characters_menu(state)

    assert headings == [(f"Character/dialog results - {view}", {
        "breadcrumb_text": "Main > Voices > Auto-assign character voices >",
    })]
    assert MenuUtil.menu_frames == frames


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
        ["1", "CAROL"], ["1", "ALICE"], ["1", "BOB"], ["0", "DAVE"], ["1", "OTHER"], ["1", "UNKNOWN"],
    ]
    assert any("5 of 6 dialog passages assigned" in line for line in lines)


def test_expanded_view_shows_counts_and_reserved_rows(monkeypatch):
    # The expanded blocks gain a count once step 2 has results; the reserved
    # answers follow as a small table.
    book, attributions = make_fixture()
    lines = print_characters(monkeypatch, book, make_roster_file(book, sorted(ROSTER_IDS)), attributions, expanded=True)

    assert any(line.startswith("[1] id: ALICE") and line.endswith("count: 1") for line in lines)
    assert any(line.split()[:2] == ["1", "OTHER"] for line in lines)
    assert not any("NOT_SPEECH" in line for line in lines)
    assert not any(line.startswith("Full roster data") for line in lines)


def test_not_speech_is_displayed_under_other(monkeypatch):
    # Non-speech-only results still produce an OTHER row, keeping the progress total intact.
    book, attributions = make_fixture()
    lines = print_characters(monkeypatch, book, make_roster_file(book, sorted(ROSTER_IDS)), attributions)

    assert any(line.split()[1:2] == ["UNKNOWN"] for line in lines)
    assert any(line.split()[:2] == ["1", "OTHER"] for line in lines)
    assert not any("NOT_SPEECH" in line for line in lines)
    i = next(i for i, line in enumerate(lines) if "dialog passages assigned" in line)
    assert lines[i + 1] == ""
    assert lines[i + 2].startswith("Full roster data")


@pytest.mark.parametrize("label", ["OTHER", "UNKNOWN"])
@pytest.mark.parametrize("summary", [False, True])
def test_reserved_rows_have_white_dashes_only_in_summary(monkeypatch, label, summary):
    # Summary placeholders fill both columns in the default color; expanded rows stay compact.
    lines = []
    monkeypatch.setattr(characters_view, "printt", lines.append)
    description = characters_view.RESERVED_LABEL_DESCRIPTIONS[label]

    characters_view.print_extra_rows([(label, 1, description)], 5, 10, tier_gender_gap=summary)

    columns = f"{'-':<10}  {'-':<7}  " if summary else ""
    assert lines == [f"    1  {COL_ACCENT}{label:<10}{COL_DEFAULT}  {columns}{COL_DIM}{description}"]


@pytest.mark.parametrize("one_offs, expected", [
    ([], "Non-roster speakers or non-dialogue text: bot (4), gearhead (3)"),
    (["OTHER:guard"], "Non-roster speakers or non-dialogue text: bot (4), gearhead (3), ..."),
    (["OTHER"], "Non-roster speakers or non-dialogue text: bot (4), gearhead (3), ..."),
    (["NOT_SPEECH"], "Non-roster speakers or non-dialogue text: bot (4), gearhead (3)"),
])
@pytest.mark.parametrize("expanded", [False, True])
def test_other_row_names_recurring_others(monkeypatch, one_offs, expected, expanded):
    # The OTHER row counts every non-roster speaker but names only the
    # recurring ones, most first; "..." marks that some (one-off or unnamed)
    # aren't named.
    speakers = ["OTHER:gearhead"] * 3 + ["OTHER:bot"] * 4 + one_offs
    book = make_book([(f'"Line {i}." ', Reason.PARAGRAPH, DIALOG) for i in range(len(speakers))])
    attributions = AttributionsFile((RangeAttribution(
        0, len(speakers), UtteranceUtil.range_text_hash(book, 0, len(speakers)),
        tuple(UtteranceAttribution((i,), speaker) for i, speaker in enumerate(speakers))),))

    original_counts = characters_view.AttributionSummary.make(attributions, book).counts.copy()
    lines = print_characters(
        monkeypatch, book, make_roster_file(book, sorted(ROSTER_IDS)), attributions, expanded=expanded,
    )

    assert not any("NOT_SPEECH" in line for line in lines)
    assert characters_view.AttributionSummary.make(attributions, book).counts == original_counts
    assert [u.speaker for u in attributions.ranges[0].utterances] == speakers
    row = next(line for line in lines if line.split()[1:2] == ["OTHER"])
    assert row.split()[0] == str(len(speakers))
    assert row.endswith(expected)
