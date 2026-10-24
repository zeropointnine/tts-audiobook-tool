from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.speaker_assign_types import NOT_SPEECH, UNKNOWN
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.speaker_assign.voice_apply import VoiceApplyPlan
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book

ROSTER_IDS = {"ALICE", "BOB", "CAROL"}

# The voice given to the fixture's dialog lines (as by the import's opt-in
# "assign dialog to voice sample 2"); line 6 has voice 2 instead
DIALOG_VOICE = 1


def make_fixture():
    # Chapter 1: lines 0-6; ALICE's passage spans lines 1-2.
    # Chapter 2: lines 7-8; its saved results go stale below.
    book = make_book(
        [
            ("She looked up. ", Reason.SENTENCE, NARRATION),
            ("“Hello,” ", Reason.PHRASE_QUOTE_END, DIALOG),
            ("how are you?” ", Reason.PARAGRAPH, DIALOG),
            ("“Hi.” ", Reason.PARAGRAPH, DIALOG),
            ("“Who?” ", Reason.PARAGRAPH, DIALOG),
            ("“Exit,” ", Reason.PARAGRAPH, DIALOG),
            ("“Quiet.” ", Reason.PARAGRAPH, DIALOG),
        ],
        [
            ("Later. ", Reason.SENTENCE, NARRATION),
            ("“Bye.” ", Reason.PARAGRAPH, DIALOG),
        ],
    )
    for line in book.dialog_line_indices():
        book.flat_phrase_groups[line].voice_index = DIALOG_VOICE
    book.flat_phrase_groups[6].voice_index = 2
    chapter_1 = RangeAttribution(0, 7, UtteranceUtil.range_text_hash(book, 0, 7), (
        UtteranceAttribution((1, 2), "ALICE"),
        UtteranceAttribution((3,), "BOB"),
        UtteranceAttribution((4,), UNKNOWN),
        UtteranceAttribution((5,), NOT_SPEECH),
        UtteranceAttribution((6,), "CAROL"),
    ))
    chapter_2 = RangeAttribution(7, 9, "stale-hash", (UtteranceAttribution((8,), "ALICE"),))
    return book, AttributionsFile((chapter_1, chapter_2))


def test_plan_covers_only_characters_with_a_chosen_voice() -> None:
    # Characters without a voice (BOB), UNKNOWN and NOT_SPEECH are left as is;
    # passages from results whose text changed are skipped, not applied.
    book, attributions = make_fixture()
    voice_map = VoiceMapFile({"ALICE": 2, "CAROL": 2})
    plan = VoiceApplyPlan.make(book, ROSTER_IDS, attributions, voice_map)

    assert plan.line_voices == {1: 2, 2: 2, 6: 2}
    assert plan.applied == {"ALICE": 1, "CAROL": 1}
    assert plan.unchanged == {"BOB": 1, UNKNOWN: 1, NOT_SPEECH: 1}
    assert plan.skipped == 1


def test_voice_map_entries_for_ids_not_in_the_roster_are_ignored() -> None:
    # A stale entry (eg from a previous roster) matching a speaker label must
    # not apply: only current roster ids count.
    book, attributions = make_fixture()
    plan = VoiceApplyPlan.make(book, {"ALICE"}, attributions, VoiceMapFile({"BOB": 3}))
    assert plan.line_voices == {} and not plan.applied
    assert plan.unchanged["BOB"] == 1


def test_apply_changes_only_differing_lines_and_restore_reverts() -> None:
    # Line 6 already has voice 2, so only ALICE's two lines change.
    book, attributions = make_fixture()
    plan = VoiceApplyPlan.make(book, ROSTER_IDS, attributions, VoiceMapFile({"ALICE": 2, "CAROL": 2}))
    assert plan.changed_lines(book) == [1, 2]

    previous = plan.apply(book)
    groups = book.flat_phrase_groups
    assert previous == {1: DIALOG_VOICE, 2: DIALOG_VOICE}
    assert [groups[line].voice_index for line in (1, 2, 3, 6)] == [2, 2, DIALOG_VOICE, 2]
    assert plan.changed_lines(book) == []

    VoiceApplyPlan.restore(book, previous)
    assert [groups[line].voice_index for line in (1, 2)] == [DIALOG_VOICE, DIALOG_VOICE]


def test_dialog_given_voice_1_is_still_dialog() -> None:
    # Dialog status is independent of the voice: dialog mapped to the first
    # voice (index 0, the narrator's) is still found as dialog afterwards.
    book, attributions = make_fixture()
    plan = VoiceApplyPlan.make(book, ROSTER_IDS, attributions, VoiceMapFile({"ALICE": 0}))
    plan.apply(book)
    assert book.flat_phrase_groups[1].voice_index == 0
    assert UtteranceUtil.find_utterances(book, 0, 7)[0].lines == (1, 2)
