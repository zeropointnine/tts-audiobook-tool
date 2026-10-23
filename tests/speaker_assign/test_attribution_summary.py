from collections import Counter

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.attribution import (
    AttributionsFile,
    RangeAttribution,
    UtteranceAttribution,
)
from tts_audiobook_tool.speaker_assign.attribution_summary import AttributionSummary
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book

BOOK = make_book([
    ("“A.” ", Reason.PHRASE, DIALOG),       # 0
    ("“B.” ", Reason.PHRASE, DIALOG),       # 1
    ("“C.” ", Reason.PHRASE, DIALOG),       # 2
    ("“D.” ", Reason.PHRASE, DIALOG),       # 3
    ("“E.”\n\n", Reason.PARAGRAPH, DIALOG),  # 4
    ("Narration.\n\n", Reason.PARAGRAPH, NARRATION),
    ("“F.”\n\n", Reason.PARAGRAPH, DIALOG),  # 6  not attributed
])


def test_summary_counts_others_as_one_and_finds_recurring_ones() -> None:
    # OTHER:<name> labels count as OTHER; one named 3+ times is a roster
    # candidate. Utterances without saved results are counted separately.
    speakers = ["BARON", "OTHER:guard", "OTHER:Guard", "OTHER:guard", "OTHER:maid"]
    attributions = AttributionsFile((RangeAttribution(
        start=0, end=5, text_hash=UtteranceUtil.range_text_hash(BOOK, 0, 5),
        utterances=tuple(UtteranceAttribution((line,), speaker, uncertain=line == 0)
                         for line, speaker in enumerate(speakers)),
    ),))
    summary = AttributionSummary.make(attributions, BOOK)
    assert summary.counts == Counter({"OTHER": 4, "BARON": 1})
    assert summary.recurring_others == Counter({"guard": 3})
    assert (summary.attributed, summary.unattributed, summary.uncertain) == (5, 1, 1)
