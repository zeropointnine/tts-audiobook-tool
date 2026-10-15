from unittest.mock import patch

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.sound_pipeline import BreakEffectTracker, SoundPipeline
from tts_audiobook_tool.sound.sound_util import SoundUtil


def run_tracker(
    reasons: list[Reason],
    section_start_indices: list[int],
    start_group_index: int = 0,
    effects_enabled: bool = True,
) -> list[Reason | None]:
    tracker = BreakEffectTracker(section_start_indices, start_group_index=start_group_index)
    return [
        tracker.next_break_effect(i, reason, effects_enabled=effects_enabled, is_final=False)
        for i, reason in enumerate(reasons, start=start_group_index)
    ]


class TestBreakEffectTrackerRule1:
    """The first sound segment of a section never gets a SPACE_BREAK effect."""

    def test_first_segment_of_book_suppresses_space_break(self) -> None:
        assert run_tracker([Reason.SPACE_BREAK], [0]) == [None]

    def test_first_segment_of_each_section_suppresses_space_break(self) -> None:
        # Section starts at 0 and 3 (marker- or book-section-style)
        reasons = [
            Reason.SPACE_BREAK,  # 0: first of section 0 -> suppressed
            Reason.SPACE_BREAK,  # 1: continues run -> suppressed
            Reason.SENTENCE,    # 2: resets run
            Reason.SPACE_BREAK,  # 3: first of section 1 -> suppressed
            Reason.SPACE_BREAK,  # 4: continues run -> suppressed
        ]
        assert run_tracker(reasons, [0, 3]) == [None, None, None, None, None]

    def test_non_initial_space_break_keeps_effect(self) -> None:
        reasons = [Reason.SENTENCE, Reason.SPACE_BREAK]
        assert run_tracker(reasons, [0]) == [None, Reason.SPACE_BREAK]


class TestBreakEffectTrackerRule2:
    """Only the first of consecutive would-be SPACE_BREAK segments keeps the effect."""

    def test_run_collapses_to_first(self) -> None:
        reasons = [Reason.SENTENCE, Reason.SPACE_BREAK, Reason.SPACE_BREAK, Reason.SPACE_BREAK]
        assert run_tracker(reasons, [0]) == [
            None,
            Reason.SPACE_BREAK,
            None,
            None,
        ]

    def test_non_space_break_segment_resets_run(self) -> None:
        reasons = [Reason.SENTENCE, Reason.SPACE_BREAK, Reason.PARAGRAPH, Reason.SPACE_BREAK]
        assert run_tracker(reasons, [0]) == [None, Reason.SPACE_BREAK, None, Reason.SPACE_BREAK]

    def test_rule1_suppressed_segment_still_anchors_run(self) -> None:
        # The run is counted on the would-be assignment: when rule 1 removes
        # the section's first SPACE_BREAK effect, its followers are suppressed
        # as well rather than re-anchoring the run.
        reasons = [Reason.SPACE_BREAK, Reason.SPACE_BREAK, Reason.SENTENCE, Reason.SPACE_BREAK]
        assert run_tracker(reasons, [0]) == [None, None, None, Reason.SPACE_BREAK]

    def test_run_across_missing_segments_uses_emission_order(self) -> None:
        # Callers feed the tracker only emitted segments; simulating a missing
        # segment simply means it never appears in the call sequence.
        reasons = [Reason.SENTENCE, Reason.SPACE_BREAK, Reason.SPACE_BREAK]
        assert run_tracker(reasons, [0]) == [None, Reason.SPACE_BREAK, None]


class TestBreakEffectTrackerSectionBreak:
    def test_section_break_never_suppressed_at_section_start(self) -> None:
        assert run_tracker([Reason.SECTION_BREAK], [0]) == [Reason.SECTION_BREAK]

    def test_consecutive_section_breaks_all_kept(self) -> None:
        reasons = [Reason.SECTION_BREAK, Reason.SECTION_BREAK]
        assert run_tracker(reasons, [0, 1]) == [Reason.SECTION_BREAK, Reason.SECTION_BREAK]

    def test_section_break_resets_space_break_run(self) -> None:
        reasons = [Reason.SENTENCE, Reason.SPACE_BREAK, Reason.SECTION_BREAK, Reason.SPACE_BREAK]
        assert run_tracker(reasons, [0]) == [
            None,
            Reason.SPACE_BREAK,
            Reason.SECTION_BREAK,
            Reason.SPACE_BREAK,
        ]


class TestBreakEffectTrackerGating:
    def test_effects_disabled_returns_none(self) -> None:
        reasons = [Reason.SPACE_BREAK, Reason.SECTION_BREAK]
        assert run_tracker(reasons, [0], effects_enabled=False) == [None, None]

    def test_final_segment_returns_none(self) -> None:
        tracker = BreakEffectTracker([0])
        assert tracker.next_break_effect(
            0, Reason.SPACE_BREAK, effects_enabled=True, is_final=True
        ) is None

    def test_stream_starting_mid_section_does_not_trigger_rule1(self) -> None:
        # Sections start at 0 and 5; the stream begins mid-section at group 7.
        assert run_tracker([Reason.SPACE_BREAK], [0, 5], start_group_index=7) == [Reason.SPACE_BREAK]

    def test_stream_starting_at_section_start_triggers_rule1(self) -> None:
        assert run_tracker([Reason.SPACE_BREAK], [0, 5], start_group_index=5) == [None]

    def test_unsorted_and_unnormalized_starts_are_normalized(self) -> None:
        # Duplicates, unsorted input, and a missing 0 must all behave.
        assert run_tracker([Reason.SPACE_BREAK], [5, 0, 5]) == [None]


class TestAppendPauseOrSectionEffect:
    def test_uses_selected_pause_policy(self) -> None:
        sr = 48000
        sound = Sound(np.zeros(sr, dtype=np.float32), sr)

        normal_result = SoundPipeline.append_pause_or_section_effect(
            sound,
            reason=Reason.SPACE_BREAK,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            break_effect=None,
        )
        shorter_result = SoundPipeline.append_pause_or_section_effect(
            sound,
            reason=Reason.SPACE_BREAK,
            reason_pauses=ReasonPauseTypes.SHORTER.value,
            break_effect=None,
        )

        normal_pause = ReasonPauseTypes.NORMAL.value.get_pause_for(Reason.SPACE_BREAK)
        shorter_pause = ReasonPauseTypes.SHORTER.value.get_pause_for(Reason.SPACE_BREAK)
        assert normal_result.duration == sound.duration + normal_pause
        assert shorter_result.duration == sound.duration + shorter_pause
        assert shorter_result.duration < normal_result.duration

    def test_break_effect_is_matching_asset_centered_in_pause(self) -> None:
        from tts_audiobook_tool.constants import (
            SECTION_BREAK_SOUND_EFFECT_PATH,
            SPACE_BREAK_SOUND_EFFECT_PATH,
        )
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

        sr = 48000
        sound = Sound(np.full(sr, 0.5, dtype=np.float32), sr)
        effect_len = sr // 2
        effect = Sound(np.ones(effect_len, dtype=np.float32), sr)
        reason_pauses = ReasonPauseTypes.NORMAL.value

        with patch.object(SoundFileUtil, "load", return_value=effect) as mock:
            for break_effect, path in (
                (Reason.SPACE_BREAK, SPACE_BREAK_SOUND_EFFECT_PATH),
                (Reason.SECTION_BREAK, SECTION_BREAK_SOUND_EFFECT_PATH),
            ):
                result = SoundPipeline.append_pause_or_section_effect(
                    sound,
                    reason=Reason.SPACE_BREAK,
                    reason_pauses=reason_pauses,
                    break_effect=break_effect,
                    pause_duration_override=0.0,  # ignored for effects
                )
                assert mock.call_args.args[0] == path
                assert mock.call_args.args[1] == sr

                appended = result.data[len(sound.data):]
                assert len(appended) == int(sr * reason_pauses.get_pause_for(break_effect))
                nonzero = np.flatnonzero(appended)
                lead = nonzero[0]
                trail = len(appended) - 1 - nonzero[-1]
                assert len(nonzero) == effect_len
                assert abs(lead - trail) <= 1
                assert lead > 0

    def test_break_effect_load_failure_falls_back_to_pause(self) -> None:
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

        sr = 48000
        sound = Sound(np.zeros(sr, dtype=np.float32), sr)
        reason_pauses = ReasonPauseTypes.NORMAL.value

        with patch.object(SoundFileUtil, "load", return_value="boom"):
            result = SoundPipeline.append_pause_or_section_effect(
                sound,
                reason=Reason.SECTION_BREAK,
                reason_pauses=reason_pauses,
                break_effect=Reason.SECTION_BREAK,
                pause_duration_override=0.1,
            )

        expected = int(sr * reason_pauses.get_pause_for(Reason.SECTION_BREAK))
        assert len(result.data) == len(sound.data) + expected


class TestCenterInSilence:
    def test_odd_pad_puts_extra_sample_at_end(self) -> None:
        sound = Sound(np.ones(4, dtype=np.float32), 10)
        result = SoundUtil.center_in_silence(sound, 0.9)  # 9 samples, pad 5
        assert result.data.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
        assert result.data.dtype == np.float32

    def test_longer_sound_is_unchanged(self) -> None:
        sound = Sound(np.ones(20, dtype=np.float32), 10)
        assert SoundUtil.center_in_silence(sound, 1.0) is sound
