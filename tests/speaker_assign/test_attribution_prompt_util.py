from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.attribution_prompt_util import AttributionPromptUtil
from tts_audiobook_tool.speaker_assign.roster import (
    NarrativePerson,
    Roster,
    RosterCharacter,
    RosterNarration,
)

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book


BOOK = make_book([
    ("“Who goes there?” ", Reason.PHRASE_QUOTE_END, DIALOG),  # 0
    ("the guard asked.\n\n", Reason.PARAGRAPH, NARRATION),    # 1
    ("“A friend,” ", Reason.PHRASE_QUOTE_END, DIALOG),        # 2
    ("I said.\n\n", Reason.PARAGRAPH, NARRATION),             # 3
    ("“Pass.”\n\n", Reason.PARAGRAPH, DIALOG),                # 4
])
ROSTER = Roster(
    (RosterCharacter(id="NARRATOR", name="The Physician", aliases=("Simone",), is_narrator=True),
     RosterCharacter(id="BARON", name="Baron", gender="male", description="The old lord.")),
    RosterNarration(NarrativePerson.FIRST, "spoiler-free notes"),
)


def test_message_has_roster_then_tagged_passage() -> None:
    # The passage's utterances are numbered from 1 within the range; the
    # roster includes LLM-facing fields; there's no context at the book start.
    message = AttributionPromptUtil.build_message(BOOK, 0, 5, ROSTER, {})
    assert "- NARRATOR: The Physician (the first-person narrator); unknown; aliases: Simone;" in message
    assert "- BARON: Baron; male; aliases: -; The old lord." in message
    assert "narrated in the first person" in message
    assert "CONTEXT" not in message
    assert '<q id="1">“Who goes there?”</q>' in message and '<q id="3">“Pass.”</q>' in message


def test_context_carries_already_attributed_speakers() -> None:
    # Attributing from line 4 on: earlier attributed utterances appear in the
    # context tagged with their speaker; unattributed ones stay untagged.
    speakers = {2: "NARRATOR"}
    message = AttributionPromptUtil.build_message(BOOK, 4, 5, ROSTER, speakers)
    context = message.split("===== CONTEXT: PRECEDING TEXT (do not attribute) =====")[1].split("===== TEXT TO")[0]
    assert '<q speaker="NARRATOR">“A friend,”</q>' in context
    assert "“Who goes there?”" in context and "Who goes there?”</q>" not in context
    assert '<q id="1">“Pass.”</q>' in message


def test_missing_ids_message() -> None:
    assert AttributionPromptUtil.missing_ids_message([3, 7]).startswith("Your reply omitted these ids: 3, 7.")
