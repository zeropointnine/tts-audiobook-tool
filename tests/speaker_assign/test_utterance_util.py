from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book


# A conversation as the app's dialog segmentation leaves it: quoted pieces are
# their own (dialog) lines, attributions are narration lines.
BOOK = make_book(
    [
        ("He said, ", Reason.PHRASE, NARRATION),            # 0
        ("“Hello, ", Reason.PHRASE, DIALOG),                # 1  opens a quotation...
        ("my friend.”\n\n", Reason.PARAGRAPH, DIALOG),      # 2  ...continued (no opening quote)
        ("“Bye.” ", Reason.PHRASE_QUOTE_END, DIALOG),      # 3  a new quotation right after
        ("she said.\n\n", Reason.PARAGRAPH, NARRATION),     # 4
    ],
    [
        ("and the rest of it.”\n\n", Reason.PARAGRAPH, DIALOG),  # 5  dialog at a section start
        ("Narration.\n\n", Reason.PARAGRAPH, NARRATION),         # 6
    ],
)


def test_utterances_group_continuations_and_split_at_opening_quotes() -> None:
    # A dialog line without an opening quote continues the previous utterance;
    # an opening quote starts a new one, even directly after dialog.
    utterances = UtteranceUtil.find_utterances(BOOK, 0, 5)
    assert [u.lines for u in utterances] == [(1, 2), (3,)]
    assert utterances[0].text == "“Hello, my friend.”\n\n"


def test_utterances_never_continue_across_a_section_boundary() -> None:
    utterances = UtteranceUtil.find_utterances(BOOK, 0, 7)
    assert [u.lines for u in utterances] == [(1, 2), (3,), (5,)]


def test_ranges_are_half_open() -> None:
    # Line 2 is outside [0, 2), so the first utterance is cut short there.
    assert [u.lines for u in UtteranceUtil.find_utterances(BOOK, 0, 2)] == [(1,)]
    assert UtteranceUtil.range_text(BOOK, 3, 5) == "“Bye.” she said.\n\n"


def test_annotate_range_tags_utterances_by_position() -> None:
    # Tags get the utterance's 1-based position in the range; trailing
    # whitespace stays outside the tag; a None tag leaves an utterance untagged;
    # a header marks where a new section begins.
    text = UtteranceUtil.annotate_range(
        BOOK, 0, 7, lambda number, _: None if number == 2 else f'<q id="{number}">', section_headers=True)
    assert text == (
        'He said, <q id="1">“Hello, my friend.”</q>\n\n“Bye.” she said.\n\n'
        '\n===== Section 2: Chapter 2 =====\n'
        '<q id="3">and the rest of it.”</q>\n\nNarration.\n\n'
    )


def test_range_labels_show_inclusive_line_numbers() -> None:
    assert UtteranceUtil.range_label(BOOK, 0, 5) == "Section 1: Chapter 1 (lines 0-4)"
    assert UtteranceUtil.range_label(BOOK, 1, 3) == "Lines 1-2 (in Section 1: Chapter 1)"
    assert UtteranceUtil.range_label(BOOK, 3, 7) == "Lines 3-6 (Section 1: Chapter 1 to Section 2: Chapter 2)"


def test_text_hashes_detect_text_changes() -> None:
    before = UtteranceUtil.section_text_hashes(BOOK)
    changed = make_book(
        [("Changed.\n\n", Reason.PARAGRAPH, NARRATION)],
        [("and the rest of it.”\n\n", Reason.PARAGRAPH, DIALOG), ("Narration.\n\n", Reason.PARAGRAPH, NARRATION)],
    )
    after = UtteranceUtil.section_text_hashes(changed)
    assert before[0] != after[0] and before[1] == after[1]
