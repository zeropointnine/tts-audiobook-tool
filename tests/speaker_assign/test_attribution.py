import json

import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.attribution import (
    AttributionsFile,
    RangeAttribution,
    UtteranceAttribution,
)
from tts_audiobook_tool.speaker_assign.line_speakers import load_line_speakers
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book


BOOK = make_book([
    ("“Hello.” ", Reason.PHRASE_QUOTE_END, DIALOG),     # 0
    ("he said.\n\n", Reason.PARAGRAPH, NARRATION),     # 1
    ("“Hi,” ", Reason.PHRASE_QUOTE_END, DIALOG),       # 2
    ("she said.\n\n", Reason.PARAGRAPH, NARRATION),    # 3
])


def attribution(start: int, end: int, speakers: dict[int, str], created: str = "") -> RangeAttribution:
    return RangeAttribution(
        start=start,
        end=end,
        text_hash=UtteranceUtil.range_text_hash(BOOK, start, end),
        utterances=tuple(UtteranceAttribution((line,), speaker) for line, speaker in speakers.items()),
        created=created,
        llm={"model": "m"},
    )


def test_round_trip() -> None:
    file = AttributionsFile((attribution(0, 2, {0: "BARON"}), attribution(2, 4, {2: "OTHER:maid"})))
    stored = json.loads(json.dumps(file.to_dict()))
    assert stored["format"] == "speaker_attributions.v1"
    assert AttributionsFile.from_dict(stored) == file


def test_with_range_replaces_and_makes_newest() -> None:
    # Saving a range again replaces its entry and moves it to the newest
    # position, so it takes precedence over overlapping older ranges.
    whole = attribution(0, 4, {0: "BARON", 2: "DIDIER"})
    part = attribution(0, 2, {0: "NARRATOR"})
    file = AttributionsFile().with_range(part).with_range(whole)
    assert file.speakers_by_line(BOOK) == {0: "BARON", 2: "DIDIER"}
    file = file.with_range(attribution(0, 2, {0: "HELENE"}))
    assert [r.key for r in file.ranges] == [(0, 4), (0, 2)]
    assert file.speakers_by_line(BOOK) == {0: "HELENE", 2: "DIDIER"}


def test_ranges_whose_text_changed_are_ignored() -> None:
    file = AttributionsFile((attribution(0, 2, {0: "BARON"}), attribution(2, 4, {2: "DIDIER"})))
    edited = make_book([
        ("“Hello.” ", Reason.PHRASE_QUOTE_END, DIALOG),
        ("he said.\n\n", Reason.PARAGRAPH, NARRATION),
        ("“Hello there,” ", Reason.PHRASE_QUOTE_END, DIALOG),
        ("she said.\n\n", Reason.PARAGRAPH, NARRATION),
    ])
    assert [r.key for r in file.valid_ranges(edited)] == [(0, 2)]
    assert file.speakers_by_line(edited) == {0: "BARON"}


def test_invalid_data_is_rejected() -> None:
    with pytest.raises(ValueError):
        RangeAttribution(start=2, end=2, text_hash="", utterances=())
    with pytest.raises(ValueError, match="outside"):
        RangeAttribution(start=0, end=2, text_hash="", utterances=(UtteranceAttribution((3,), "BARON"),))
    with pytest.raises(ValueError):
        UtteranceAttribution((), "BARON")
    with pytest.raises(ValueError):
        AttributionsFile((attribution(0, 2, {0: "A"}), attribution(0, 2, {0: "B"})))
    with pytest.raises(ValueError):
        AttributionsFile.from_dict({"format": "speaker_attributions.prototype", "attributions": {}})


def test_file_util_save_load_delete(tmp_path) -> None:
    # No file yet loads as an empty set of attributions; saving writes
    # [project]/speaker_assign/attributions.json; deleting is idempotent.
    project_dir = str(tmp_path)
    assert SpeakerAssignFileUtil.load_attributions_file(project_dir) == AttributionsFile()

    file = AttributionsFile((attribution(0, 4, {0: "BARON", 2: "DIDIER"}),))
    assert SpeakerAssignFileUtil.save_attributions_file(project_dir, file) == ""
    assert SpeakerAssignFileUtil.load_attributions_file(project_dir) == file

    assert SpeakerAssignFileUtil.delete_attributions_file(project_dir) == ""
    assert SpeakerAssignFileUtil.delete_attributions_file(project_dir) == ""
    assert SpeakerAssignFileUtil.load_attributions_file(project_dir) == AttributionsFile()


def test_file_util_reports_an_invalid_file(tmp_path) -> None:
    project_dir = str(tmp_path)
    SpeakerAssignFileUtil.save_attributions_file(project_dir, AttributionsFile())
    with open(SpeakerAssignFileUtil.attributions_path(project_dir), "w", encoding="utf-8") as file:
        file.write("{not json")
    assert isinstance(SpeakerAssignFileUtil.load_attributions_file(project_dir), str)


def test_speaker_of_each_line_covers_continuation_lines() -> None:
    # A quotation split across lines is one utterance; every one of its lines
    # gets the speaker, whereas speakers_by_line only keys the first line.
    utterance = UtteranceAttribution((0, 1), "BARON")
    file = AttributionsFile((RangeAttribution(0, 4, UtteranceUtil.range_text_hash(BOOK, 0, 4), (utterance,)),))
    assert file.speakers_by_line(BOOK) == {0: "BARON"}
    assert file.speaker_of_each_line(BOOK) == {0: "BARON", 1: "BARON"}


def test_load_line_speakers(tmp_path) -> None:
    # Missing and unreadable files both load as no data (the data is optional
    # to the views that display it).
    project_dir = str(tmp_path)
    assert load_line_speakers(project_dir, BOOK) == {}
    assert load_line_speakers("", BOOK) == {}

    file = AttributionsFile((attribution(0, 4, {0: "BARON", 2: "OTHER:maid"}),))
    SpeakerAssignFileUtil.save_attributions_file(project_dir, file)
    assert load_line_speakers(project_dir, BOOK) == {0: "BARON", 2: "OTHER:maid"}

    with open(SpeakerAssignFileUtil.attributions_path(project_dir), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert load_line_speakers(project_dir, BOOK) == {}
