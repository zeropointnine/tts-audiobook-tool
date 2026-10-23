from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.chunk_plan_util import ChunkPlanUtil

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book


TEN_WORDS = "one two three four five six seven eight nine ten "


def narration(count: int) -> list:
    return [(TEN_WORDS + "\n\n", Reason.PARAGRAPH, NARRATION) for _ in range(count)]


def test_chunks_cover_the_book_contiguously_within_the_limit() -> None:
    book = make_book(narration(30), narration(30), narration(30))
    chunks = ChunkPlanUtil.plan_chunks(book, chunk_words=200)
    assert chunks[0][0] == 0 and chunks[-1][1] == 90
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    assert all(sum(len(g.text.split()) for g in book.flat_phrase_groups[s:e]) <= 200 for s, e in chunks)


def test_small_sections_merge_and_headings_join_the_following_text() -> None:
    # A one-word chapter heading (ending with a space break) is too small to
    # stand alone; it joins the text after it, so no chunk ends on a heading.
    heading = ("II\n\n\n", Reason.SPACE_BREAK, NARRATION)
    book = make_book(narration(8), [heading, *narration(8)])
    assert ChunkPlanUtil.plan_chunks(book, chunk_words=100) == [(0, 8), (8, 17)]
    # With room for both, everything merges into one chunk.
    assert ChunkPlanUtil.plan_chunks(book, chunk_words=1000) == [(0, 17)]


def test_oversized_pieces_split_away_from_conversations() -> None:
    # A 40-line piece over the limit: its middle is a conversation (dialog
    # lines 15-24), so the split lands in narration, not between dialog lines.
    lines = narration(40)
    for i in range(15, 25):
        lines[i] = ("“Line of dialog here, with ten words in it.”\n\n", Reason.PARAGRAPH, DIALOG)
    book = make_book(lines)
    chunks = ChunkPlanUtil.plan_chunks(book, chunk_words=300)
    groups = book.flat_phrase_groups
    for _, end in chunks[:-1]:
        assert not (groups[end - 1].voice_index >= 1 and groups[end].voice_index >= 1)


def test_empty_book_has_no_chunks() -> None:
    assert ChunkPlanUtil.plan_chunks(make_book([]), chunk_words=100) == []
