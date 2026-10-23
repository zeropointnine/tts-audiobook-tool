import json
import os

from tts_audiobook_tool.speaker_assign.roster import (
    NarrativePerson,
    Prominence,
    Roster,
    RosterCharacter,
    RosterFile,
    RosterNarration,
)
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil


def make_roster_file() -> RosterFile:
    roster = Roster(
        characters=(
            RosterCharacter(id="NARRATOR", name="The Physician", aliases=("Simone",),
                            is_narrator=True, prominence=Prominence.MAJOR),
            RosterCharacter(id="HELENE", name="Hélène", aliases=("madame",), gender="female"),
        ),
        narration=RosterNarration(NarrativePerson.FIRST, "A physician at a château."),
    )
    return RosterFile(
        roster=roster,
        created="2026-10-22T12:55:07-04:00",
        llm={"url": "https://example.test/v1/chat/completions", "model": "m", "extra_params": {}},
        book_title="Leech",
        section_text_hashes=["aaaa", "bbbb"],  # type: ignore[arg-type]
        raw_reply='{"characters": []}',
    )


def test_save_then_load_round_trips(tmp_path) -> None:
    # The roster file is written under [project]/speaker_assign/ (created on
    # demand) and loads back equal, non-ASCII names included.
    project_dir = str(tmp_path)
    roster_file = make_roster_file()
    assert not SpeakerAssignFileUtil.has_roster(project_dir)

    assert SpeakerAssignFileUtil.save_roster_file(project_dir, roster_file) == ""

    assert SpeakerAssignFileUtil.roster_path(project_dir) == os.path.join(project_dir, "speaker_assign", "roster.json")
    assert SpeakerAssignFileUtil.has_roster(project_dir)
    assert SpeakerAssignFileUtil.load_roster_file(project_dir) == roster_file


def test_stored_form_is_versioned_and_nests_the_roster(tmp_path) -> None:
    # The on-disk payload carries a format tag and keeps the roster as its own object.
    project_dir = str(tmp_path)
    SpeakerAssignFileUtil.save_roster_file(project_dir, make_roster_file())
    with open(SpeakerAssignFileUtil.roster_path(project_dir), encoding="utf-8") as file:
        stored = json.load(file)
    assert stored["format"] == "speaker_roster.v1"
    assert [c["id"] for c in stored["roster"]["characters"]] == ["NARRATOR", "HELENE"]
    assert stored["section_text_hashes"] == ["aaaa", "bbbb"]


def test_missing_file_is_an_error_string(tmp_path) -> None:
    result = SpeakerAssignFileUtil.load_roster_file(str(tmp_path))
    assert isinstance(result, str) and "No speaker roster" in result


def test_other_format_or_invalid_roster_is_an_error_string(tmp_path) -> None:
    # No migration: files without the current format tag (eg the prototype's
    # older cache shape) and rosters violating the id rules don't load.
    project_dir = str(tmp_path)
    os.makedirs(SpeakerAssignFileUtil.dir_path(project_dir))
    path = SpeakerAssignFileUtil.roster_path(project_dir)

    for payload in (
        {"version": 1, "characters": [{"id": "BARON", "name": "Baron"}]},
        {"format": "speaker_roster.v1", "roster": {"characters": [{"id": "OLD DOC", "name": "x"}]}},
    ):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        assert isinstance(SpeakerAssignFileUtil.load_roster_file(project_dir), str)


def test_malformed_provenance_loads_as_defaults() -> None:
    # Provenance is diagnostic, so malformed values fall back instead of failing.
    stored = make_roster_file().to_dict()
    stored.update(llm="not a dict", section_text_hashes="nope", created=5, raw_reply=None)
    roster_file = RosterFile.from_dict(stored)
    assert roster_file.llm == {} and roster_file.section_text_hashes == ()
    assert roster_file.created == "" and roster_file.raw_reply == ""
    assert roster_file.roster == make_roster_file().roster
