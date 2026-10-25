import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.attribution_reply_util import AttributionReplyUtil, ParsedReply
from tts_audiobook_tool.speaker_assign.speaker_label_util import SpeakerKind, SpeakerLabelUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book


ROSTER_IDS = frozenset({"BARON", "DIDIER"})


def test_extract_json_tolerates_fences_and_prose() -> None:
    assert AttributionReplyUtil.extract_json('Sure!\n```json\n{"speakers": {}}\n```') == {"speakers": {}}
    assert AttributionReplyUtil.extract_json('Here: {"a": 1} done') == {"a": 1}
    with pytest.raises(ValueError):
        AttributionReplyUtil.extract_json("no json here")


def test_parse_reply_normalizes_answers() -> None:
    # Roster ids and reserved answers match case-insensitively; other answers
    # are kept for review as OTHER:<answer>; ids that weren't requested are
    # ignored; missing ids are reported; uncertain ids are limited to requested ones.
    value = {
        "speakers": {"1": "baron", "2": "not speech", "3": "Other:guard", "4": "Sylvie", "5": "unknown", "9": "BARON"},
        "uncertain": [2, "4", 9, "x"],
    }
    parsed = AttributionReplyUtil.parse_reply(value, requested_ids={1, 2, 3, 4, 5, 6}, roster_ids=ROSTER_IDS)
    assert parsed.speakers == {1: "BARON", 2: "NOT_SPEECH", 3: "Other:guard", 4: "OTHER:Sylvie", 5: "UNKNOWN"}
    assert parsed.uncertain == {2, 4}
    assert parsed.missing == (6,)
    with pytest.raises(ValueError):
        AttributionReplyUtil.parse_reply({"answers": {}}, {1}, ROSTER_IDS)


def test_merge_fills_missing_ids_from_a_follow_up() -> None:
    first = ParsedReply({1: "BARON"}, frozenset(), (2, 3))
    follow_up = ParsedReply({2: "DIDIER"}, frozenset({2}), ())
    merged = AttributionReplyUtil.merge(first, follow_up)
    assert merged == ParsedReply({1: "BARON", 2: "DIDIER"}, frozenset({2}), (3,))


def test_make_range_attribution_marks_unanswered_utterances_unknown() -> None:
    book = make_book([
        ("“One.” ", Reason.PHRASE_QUOTE_END, DIALOG),
        ("he said. ", Reason.SENTENCE, NARRATION),
        ("“Two.”\n\n", Reason.PARAGRAPH, DIALOG),
    ])
    utterances = UtteranceUtil.find_utterances(book, 0, 3)
    result = AttributionReplyUtil.make_range_attribution(
        book, 0, 3, utterances, ParsedReply({1: "BARON"}, frozenset({1}), (2,)), created="t", llm={"model": "m"})
    assert [(u.lines, u.speaker, u.uncertain, u.text) for u in result.utterances] == [
        ((0,), "BARON", True, "“One.”"), ((2,), "UNKNOWN", False, "“Two.”")]
    assert result.text_hash == UtteranceUtil.range_text_hash(book, 0, 3) and result.is_valid_for(book)


@pytest.mark.parametrize("label, kind, counting_id", [
    ("BARON", SpeakerKind.CHARACTER, "BARON"),
    ("OTHER:guard", SpeakerKind.OTHER, "OTHER"),
    ("OTHER", SpeakerKind.OTHER, "OTHER"),
    ("UNKNOWN", SpeakerKind.UNKNOWN, "UNKNOWN"),
    ("NOT_SPEECH", SpeakerKind.NOT_SPEECH, "NOT_SPEECH"),
])
def test_speaker_label_kinds(label: str, kind: SpeakerKind, counting_id: str) -> None:
    assert SpeakerLabelUtil.kind(label, ROSTER_IDS) is kind
    assert SpeakerLabelUtil.counting_id(label) == counting_id


def test_other_name() -> None:
    assert SpeakerLabelUtil.other_name("OTHER: Anis del Mono man") == "anis del mono man"
    assert SpeakerLabelUtil.other_name("OTHER") == "" and SpeakerLabelUtil.other_name("BARON") == ""
