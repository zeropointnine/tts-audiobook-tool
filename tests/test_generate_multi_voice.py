from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from generate_files_test_support import StubValidationResult, generate_files_mock_stack


def make_phrase_group(num_words: int, voice_index: int) -> PhraseGroup:
    text = " ".join(["word"] * num_words)
    return PhraseGroup([Phrase(text, Reason.SENTENCE)], voice_index=voice_index)


def run_generate_files(
        voice_select_mode: VoiceSelectMode,
        voice_indices: list[int],
        num_words: int,
        batch_size: int,
        max_retries: int,
        failing_indices: set[int],
) -> tuple[bool, list[tuple[list[int], int]]]:
    """
    Runs generate_files with a mocked model layer.
    Returns (did_interrupt, per-call (indices, voice_selection_index)).
    All lines have equal word counts, so word-count ordering keeps index order.
    """
    phrase_groups = [make_phrase_group(num_words, voice) for voice in voice_indices]
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        max_retries=max_retries,
        phrase_groups=phrase_groups,
        sound_segments=sound_segments,
        generate_range_string="all",
        save=MagicMock(return_value=""),
        voice_select_mode=voice_select_mode,
        mira_voice_file_name=["voice-a.flac", "voice-b.flac"],
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(stt_variant=None, stt_config=None, save_debug_files=False),
        ),
    )
    calls: list[tuple[list[int], int]] = []
    failing = set(failing_indices)

    def generate_and_validate_batch(**kwargs: object) -> list[StubValidationResult]:
        indices = list(kwargs["indices"])  # type: ignore[index]
        voice = kwargs["voice_selection_index"]
        calls.append((indices, voice))
        results = []
        for index in indices:
            is_fail = index in failing
            if is_fail:
                failing.discard(index)
            results.append(StubValidationResult(is_fail))
        return results

    with generate_files_mock_stack(
        generate_and_validate_batch, model_type=TtsModelType.MIRA
    ):
        did_interrupt = GenerateUtil.generate_files(
            state, set(range(len(voice_indices))), batch_size=batch_size, is_regen=False,
        )

    return did_interrupt, calls


def test_multi_voice_batches_are_voice_homogeneous_and_cover_all_lines() -> None:
    voice_indices = [0, 1, 0, 1, 0, 1, 0, 1]
    did_interrupt, calls = run_generate_files(
        VoiceSelectMode.USER_DEFINED, voice_indices, num_words=5,
        batch_size=2, max_retries=1, failing_indices=set(),
    )

    assert not did_interrupt
    assert calls
    covered: list[int] = []
    for indices, voice in calls:
        assert 1 <= len(indices) <= 2
        for index in indices:
            assert voice_indices[index] == voice
        covered.extend(indices)
    assert sorted(covered) == list(range(8))
    assert len(covered) == 8  # no line generated twice


def test_multi_voice_retries_reuse_line_voice_and_run_at_head() -> None:
    voice_indices = [0, 1, 0, 1]
    did_interrupt, calls = run_generate_files(
        VoiceSelectMode.USER_DEFINED, voice_indices, num_words=5,
        batch_size=2, max_retries=1, failing_indices={1},
    )

    assert not did_interrupt
    assert calls == [
        ([0, 2], 0),
        ([1, 3], 1),
        ([1], 1),
    ]


def test_single_voice_projects_pass_no_explicit_voice() -> None:
    voice_indices = [0, 0, 0, 0]
    did_interrupt, calls = run_generate_files(
        VoiceSelectMode.AUTO_ADVANCE, voice_indices, num_words=5,
        batch_size=2, max_retries=1, failing_indices=set(),
    )

    assert not did_interrupt
    for indices, voice in calls:
        assert voice is None
    assert sorted(index for indices, _ in calls for index in indices) == [0, 1, 2, 3]