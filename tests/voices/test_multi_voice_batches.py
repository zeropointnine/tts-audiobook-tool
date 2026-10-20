from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import (
    BATCH_ITERATIONS_PER_GROUP,
    SubBatch,
    bucket_items,
    effective_voice_indices,
    make_multi_voice_rounds,
    make_retry_round,
    make_single_voice_rounds,
)


def make_phrase_group(num_words: int, voice_index: int = -1) -> PhraseGroup:
    text = " ".join(["word"] * num_words)
    return PhraseGroup([Phrase(text, Reason.SENTENCE)], voice_index=voice_index)


def flatten(rounds: list[list[SubBatch]]) -> list[SubBatch]:
    return [sub for round in rounds for sub in round]


def test_rounds_are_voice_homogeneous_and_cover_every_item() -> None:
    items = [(i, 0) for i in range(10)]
    groups = [make_phrase_group(5 + (i % 4)) for i in range(10)]
    voice_of_index = {i: i % 2 for i in range(10)}

    subs = flatten(make_multi_voice_rounds(items, groups, voice_of_index, batch_size=3, sort_by_words=True))

    assert subs
    for sub in subs:
        assert 1 <= len(sub.items) <= 3
        for index, _ in sub.items:
            assert voice_of_index[index] == sub.voice_selection_index
    covered = sorted(index for sub in subs for index, _ in sub.items)
    assert covered == list(range(10))


def test_word_count_sorting_alternates_per_window() -> None:
    batch_size = 2
    items = [(i, 0) for i in range(24)]
    groups = [make_phrase_group((i * 3) % 7 + 1) for i in range(24)]
    voice_of_index = {i: i % 2 for i in range(24)}

    rounds = make_multi_voice_rounds(items, groups, voice_of_index, batch_size=batch_size, sort_by_words=True)

    window_size = batch_size * BATCH_ITERATIONS_PER_GROUP * 2
    assert window_size == 20
    assert len(rounds) == 2

    def voice_word_counts(round_idx: int, voice: int) -> list[int]:
        counts: list[int] = []
        for sub in rounds[round_idx]:
            if sub.voice_selection_index != voice:
                continue
            for index, _ in sub.items:
                counts.append(groups[index].num_words)
        return counts

    for voice in (0, 1):
        first = voice_word_counts(0, voice)
        assert first, f"voice {voice} should have items in the first window"
        assert first == sorted(first, reverse=True), "first window should sort descending"
        second = voice_word_counts(1, voice)
        assert second == sorted(second), "second window should sort ascending"


def test_no_word_sort_keeps_index_order_within_voice() -> None:
    items = [(i, 0) for i in range(12)]
    groups = [make_phrase_group((i * 3) % 7 + 1) for i in range(12)]
    voice_of_index = {i: i % 3 for i in range(12)}

    subs = flatten(make_multi_voice_rounds(items, groups, voice_of_index, batch_size=2, sort_by_words=False))

    for sub in subs:
        index_list = [index for index, _ in sub.items]
        assert index_list == sorted(index_list)


def test_single_distinct_voice_matches_single_voice_path() -> None:
    items = [(i, 0) for i in range(24)]
    groups = [make_phrase_group((i * 3) % 7 + 1) for i in range(24)]
    voice_of_index = {i: 0 for i in range(24)}

    multi = flatten(make_multi_voice_rounds(items, groups, voice_of_index, batch_size=2, sort_by_words=True))
    flat = [index for sub in multi for index, _ in sub.items]
    expected = [index for index, _ in bucket_items(list(items), groups, 2)]

    assert flat == expected
    assert all(sub.voice_selection_index == 0 for sub in multi)


def test_multi_voice_rounds_empty_items() -> None:
    assert make_multi_voice_rounds([], [], {}, 2, True) == []


def test_single_voice_rounds_chunk_flat_queue() -> None:
    items = [(i, 0) for i in range(5)]

    rounds = make_single_voice_rounds(items, 2)

    assert rounds == [
        [SubBatch(voice_selection_index=None, items=((0, 0), (1, 0)))],
        [SubBatch(voice_selection_index=None, items=((2, 0), (3, 0)))],
        [SubBatch(voice_selection_index=None, items=((4, 0),))],
    ]


def test_retry_round_groups_by_voice_and_chunks() -> None:
    items = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    voice_of_index = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}

    retry_round = make_retry_round(items, voice_of_index, batch_size=2)

    assert retry_round == [
        SubBatch(voice_selection_index=0, items=((0, 1), (2, 1))),
        SubBatch(voice_selection_index=0, items=((4, 1),)),
        SubBatch(voice_selection_index=1, items=((1, 1), (3, 1))),
    ]


def test_retry_round_without_voice_map_is_single_voiceless_group() -> None:
    items = [(0, 1), (1, 1), (2, 1)]

    retry_round = make_retry_round(items, None, batch_size=2)

    assert retry_round == [
        SubBatch(voice_selection_index=None, items=((0, 1), (1, 1))),
        SubBatch(voice_selection_index=None, items=((2, 1),)),
    ]


def test_effective_voice_indices_clamps_stale_and_unassigned() -> None:
    groups = [
        make_phrase_group(1, -1),  # unassigned -> voice sample 1
        make_phrase_group(1, 0),
        make_phrase_group(1, 1),
        make_phrase_group(1, 9),  # stale -> clamped to last
    ]

    assert effective_voice_indices(groups, [0, 1, 2, 3], 2) == {0: 0, 1: 0, 2: 1, 3: 1}