from __future__ import annotations

import pytest

from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import ConcreteWord
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.enhance import enhance_alignment
from tts_audiobook_tool.enhance import enhance_anchored_alignment as anchored


def phrases(*texts: str) -> list[Phrase]:
    return [Phrase(text, Reason.SENTENCE) for text in texts]


def transcript(text: str, start: float = 0.0, step: float = 0.5) -> list[ConcreteWord]:
    """One Whisper-like word per whitespace token, `step` apart, each 0.4 s long."""
    return [
        ConcreteWord(start + index * step, start + index * step + 0.4, f" {value}", 1.0)
        for index, value in enumerate(text.split())
    ]


def spans(result: anchored.AnchoredAlignmentResult) -> list[tuple[float, float]]:
    return [(round(item.time_start, 2), round(item.time_end, 2)) for item in result.timed_phrases]


def test_recovers_after_audio_only_passage_longer_than_greedy_lookahead() -> None:
    # Narration absent from the text (eg a "characters so far" segment) is
    # just a gap; the greedy aligner cannot skip more than 250 words and
    # orphans everything after it.
    source = phrases("The witch left the tower.", "Afterword", "Thank you for reading this book.")
    words = (
        transcript("the witch left the tower")
        + transcript(" ".join(f"extra{k}" for k in range(600)), start=10.0)
        + transcript("afterword thank you for reading this book", start=400.0)
    )

    result = anchored.align_phrases_detailed(source, words, "en")

    assert spans(result) == [(0.0, 10.0), (400.0, 400.5), (400.5, 403.4)]
    assert result.unmatched_count == 0
    greedy, _, _ = enhance_alignment.align_phrases_with_state(source, words, print_info=False)
    assert [(item.time_start, item.time_end) for item in greedy[1:]] == [(0.0, 0.0), (0.0, 0.0)]


def test_repeated_short_lines_are_placed_in_order() -> None:
    # Short repeated lines are ambiguous locally; global alignment keeps each
    # one on its own occurrence instead of jumping ahead to a later one.
    source = phrases("Yes.", "She nodded slowly.", "Yes.", "Then she left.", "Yes.")
    words = transcript("yes she nodded slowly yes then she left yes")

    result = anchored.align_phrases_detailed(source, words, "en")

    assert [start for start, _ in spans(result)] == [0.0, 0.5, 2.0, 2.5, 4.0]


def test_spelled_out_number_matches_digits() -> None:
    source = phrases("There were one hundred and fifty of them.")
    words = transcript("there were 150 of them")

    result = anchored.align_phrases_detailed(source, words, "en")

    alignment = result.phrase_alignments[0]
    assert alignment.status == anchored.STATUS_TIMED
    assert alignment.matched_count == alignment.token_count == 8


def test_untranscribed_interjection_is_placed_in_the_pause_before_the_next_phrase() -> None:
    # Whisper drops the "Um"; it is attached in the silence between its
    # neighbors, and the previous phrase now ends at its last word.
    source = phrases("She paused.", "“Um,", "I think so.”")
    words = transcript("she paused") + transcript("i think so", start=2.0)

    result = anchored.align_phrases_detailed(source, words, "en")

    assert spans(result) == [(0.0, 0.9), (0.9, 2.0), (2.0, 3.4)]
    assert [item.status for item in result.phrase_alignments] == ["timed", "attached", "timed"]
    assert result.unmatched_count == 0


def test_interjection_without_a_pause_gets_a_zero_length_span_at_the_boundary() -> None:
    source = phrases("She paused.", "Um,", "I think so.")
    words = transcript("she paused i think so", step=0.42)

    result = anchored.align_phrases_detailed(source, words, "en")

    start, end = spans(result)[1]
    assert start == end == 0.84
    assert result.phrase_alignments[1].status == anchored.STATUS_ATTACHED


def test_interjection_inside_an_unmatched_passage_stays_unmatched_with_it() -> None:
    source = phrases("She paused.", "Um,", "The narrator skipped this whole line.", "Then she left.")
    words = transcript("she paused then she left")

    result = anchored.align_phrases_detailed(source, words, "en")

    assert spans(result)[1:3] == [(0.0, 0.0), (0.0, 0.0)]
    assert result.unmatched_count == 2


def test_unnarrated_text_stays_unmatched_between_timed_neighbors() -> None:
    source = phrases("Chapter one begins.", "Thank you for buying this ebook.", "The story continued.")
    words = transcript("chapter one begins the story continued")

    result = anchored.align_phrases_detailed(source, words, "en")

    assert spans(result) == [(0.0, 1.5), (0.0, 0.0), (1.5, 2.9)]


def test_duplicated_transcript_word_does_not_start_the_phrase() -> None:
    # Old transcripts can repeat a word at a window seam ("the the"); the
    # phrase starts at the later copy rather than at the duplicate.
    source = phrases("Nero chuckled confidently.", "The sound of a meow was audible.")
    words = transcript("nero chuckled confidently the the sound of a meow was audible")

    result = anchored.align_phrases_detailed(source, words, "en")

    assert spans(result)[1][0] == 2.0


def test_spans_stay_ordered_when_phrases_share_a_whisper_word() -> None:
    # "twenty-plus" is one Whisper word but its tokens fall in two phrases.
    source = phrases("She had slain twenty", "plus dragons that year.")
    words = transcript("she had slain twenty-plus dragons that year")

    result = anchored.align_phrases_detailed(source, words, "en")

    (first_start, first_end), (second_start, second_end) = spans(result)
    assert first_start < first_end <= second_start < second_end


def test_unrelated_text_is_left_unaligned_without_gap_filling(monkeypatch) -> None:
    # The wrong book for the audio: the only shared unique words are
    # coincidences, so their anchor chain is far too sparse to trust, and the
    # costly edit-distance pass must not run over the unrelated text.
    source_words = [f"alpha{k}" for k in range(200)]
    transcript_words = [f"omega{k}" for k in range(200)]
    for position, word in ((50, "zebra"), (150, "quokka")):
        source_words[position] = transcript_words[position + 3] = word
    monkeypatch.setattr(
        anchored, "_align_window_dp", lambda *_args: pytest.fail("gap-filled unrelated text")
    )

    result = anchored.align_phrases_detailed(
        phrases(" ".join(source_words)), transcript(" ".join(transcript_words)), "en"
    )

    assert spans(result) == [(0.0, 0.0)]


def test_exhausted_dp_budget_leaves_remaining_gaps_unaligned(monkeypatch, caplog) -> None:
    # Safety net: once the per-run DP budget is spent, gaps are left
    # unaligned (and logged) instead of growing the runtime without bound.
    monkeypatch.setattr(anchored, "DP_CELL_BUDGET_MIN", 0)
    monkeypatch.setattr(anchored, "DP_CELL_BUDGET_PER_TOKEN", 0)

    with caplog.at_level("WARNING", logger="tts-audiobook-tool"):
        result = anchored.align_phrases_detailed(phrases("Hello there."), transcript("hello there"), "en")

    assert spans(result) == [(0.0, 0.0)]
    assert "DP budget exhausted" in caplog.text


def test_empty_transcript_leaves_everything_unmatched() -> None:
    result = anchored.align_phrases_detailed(phrases("Hello.", "Um,"), [], "en")
    assert spans(result) == [(0.0, 0.0), (0.0, 0.0)]
    assert anchored.align_phrases_detailed([], transcript("hello"), "en").timed_phrases == []


def test_interrupt_returns_no_phrases() -> None:
    Interrupts().set("thinking")
    Interrupts()._flag = True
    try:
        timed, interrupted = anchored.align_phrases(phrases("Hello."), transcript("hello"), "en")
    finally:
        Interrupts().clear()
    assert interrupted
    assert timed == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wh-wh-wh-what do I do?!", ["what", "do", "i", "do"]),
        ("N-n-no,", ["no"]),
        ("Ha-ha-ha", ["ha", "ha", "ha"]),  # repetition of a whole word is kept
        ("“Ahhhhh!”", ["ah"]),
        ("I don’t know.", ["i", "dont", "know"]),
        ("150", ["one", "hundred", "and", "fifty"]),
        ("the 3rd time", ["the", "third", "time"]),
        ("well-known", ["well", "known"]),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert anchored.tokenize(text, "en") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("um,", True),
        ("“Ahhh…", True),
        ("Heh-heh-heh.", True),
        ("U-um,", True),
        ("“Nooooo!”", True),  # elongated words are vocalizations
        ("Hahaha!", True),
        ("I…", False),
        ("he asked.", False),
        ("Yes.", False),
        ("Banana.", False),
        ("…", False),  # no words: handled as non-verbal, not as an interjection
    ],
)
def test_is_interjection_only(text: str, expected: bool) -> None:
    assert anchored.is_interjection_only(text) is expected
