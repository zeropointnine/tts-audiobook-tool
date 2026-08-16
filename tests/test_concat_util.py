from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.concat_util import ConcatUtil, make_stem
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline


def make_fake_project(num_groups: int, missing_indices: frozenset[int] = frozenset()) -> MagicMock:
    project = MagicMock()
    project.markers = []
    project.sound_segments_path = "/tmp/sounds"
    project.dir_path = "/tmp/my project"
    project.phrase_groups = [
        MagicMock(as_flattened_phrase=MagicMock(return_value=MagicMock(text=f"Phrase {i}")))
        for i in range(num_groups)
    ]

    def get_best_file_for(index: int) -> str:
        return "" if index in missing_indices else f"seg{index}.flac"

    project.sound_segments.get_best_file_for.side_effect = get_best_file_for
    return project


@pytest.mark.parametrize("reason", [Reason.SPACE_BREAK, Reason.SECTION_BREAK])
def test_concatenate_does_not_append_break_effect_to_final_segment(reason: Reason) -> None:
    sound = Sound(np.zeros(48000, dtype=np.float32), 48000)
    phrase = Phrase("Final phrase", reason)
    process = MagicMock()

    with patch.object(ConcatUtil, "init_ffmpeg_stream", return_value=process), \
         patch.object(ConcatUtil, "close_ffmpeg_stream"), \
         patch.object(ConcatUtil, "add_audio_to_ffmpeg_stream"), \
         patch.object(SoundPipeline, "make_concat_rendered_sound_segment", return_value=sound), \
         patch.object(SoundPipeline, "append_pause_or_section_effect", return_value=sound) as append_mock:
        result = ConcatUtil.concatenate_sound_segments(
            dest_path="output.flac",
            phrases_and_paths=[(phrase, "segment.flac", False)],
            use_break_sound_effect=True,
            high_shelf=HighShelfEq.DISABLED,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            print_progress=False,
        )

    assert result == [sound.duration]
    append_mock.assert_called_once_with(
        sound,
        reason=reason,
        reason_pauses=ReasonPauseTypes.NORMAL.value,
        use_break_sound_effect=False,
        is_first_in_section=False,
        pause_duration_override=None,
    )


class TestMakeStemMissingCount:
    """
    Regression tests: out-of-range phrase groups must NOT be counted as
    "missing" in the output filename (only in-range groups without generated
    audio).
    """

    def test_no_missing_tag_when_all_in_range_files_exist(self) -> None:
        project = make_fake_project(num_groups=15)

        with patch.object(SoundSegmentUtil, "get_common_model_tag", return_value=""), \
             patch.object(SoundSegmentUtil, "get_common_voice_tag", return_value=""):
            stem = make_stem(
                project=project,
                index_start=0,
                index_end=2,
                file_cut_index=0,
                num_chapters=3,
            )

        assert "missing" not in stem
        assert "[1 of 3] [1-3]" in stem

    def test_missing_tag_counts_only_in_range_gaps(self) -> None:
        project = make_fake_project(num_groups=8, missing_indices=frozenset({5}))

        with patch.object(SoundSegmentUtil, "get_common_model_tag", return_value=""), \
             patch.object(SoundSegmentUtil, "get_common_voice_tag", return_value=""):
            stem = make_stem(
                project=project,
                index_start=2,
                index_end=6,
                file_cut_index=1,
                num_chapters=3,
            )

        assert "[1 missing]" in stem
        assert "[2 of 3] [3-7] [1 missing]" in stem

    def test_num_missing_in_ignores_out_of_range_groups(self) -> None:
        project = make_fake_project(num_groups=15)
        assert ConcatUtil.num_missing_in(project, 0, 2) == 0
        assert ConcatUtil.num_missing_in(project, 8, 14) == 0

        project_with_gap = make_fake_project(num_groups=8, missing_indices=frozenset({4, 6}))
        assert ConcatUtil.num_missing_in(project_with_gap, 3, 7) == 2
        assert ConcatUtil.num_missing_in(project_with_gap, 0, 2) == 0
