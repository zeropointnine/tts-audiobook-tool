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
             patch.object(SoundSegmentUtil, "get_voice_tag_summary", return_value=("", 0)):
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
             patch.object(SoundSegmentUtil, "get_voice_tag_summary", return_value=("", 0)):
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


class TestVoiceTagSummary:
    """
    The concatenated output stem shows the most common voice tag across the
    segments being concatenated; "+n" counts the other distinct voice tags
    in the set (e.g. "[belle_concerned +1]").
    """

    @staticmethod
    def _seg_filename(index: int, voice: str) -> str:
        return f"[{index+1:05d}] [0123456789abcdef] [qwen3] [{voice}] Phrase_{index}.flac"

    def test_uniform_voice_has_no_extra_count(self) -> None:
        paths = [self._seg_filename(i, "belle_concerned") for i in range(4)]
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("belle_concerned", 0)

    def test_dominant_voice_plus_one_other(self) -> None:
        paths = [self._seg_filename(i, "belle_concerned") for i in range(3)]
        paths.append(self._seg_filename(3, "other_voice"))
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("belle_concerned", 1)

    def test_dominant_voice_plus_two_others(self) -> None:
        paths = [self._seg_filename(i, "belle_concerned") for i in range(2)]
        paths.append(self._seg_filename(2, "voice_a"))
        paths.append(self._seg_filename(3, "voice_b"))
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("belle_concerned", 2)

    def test_all_ties_break_alphabetically(self) -> None:
        paths = [
            self._seg_filename(0, "zeta"),
            self._seg_filename(1, "alpha"),
            self._seg_filename(2, "belle_concerned"),
        ]
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("alpha", 2)

    def test_voiceless_and_unparsable_segments_are_ignored(self) -> None:
        paths = [self._seg_filename(i, "belle_concerned") for i in range(2)]
        paths.append("not-a-segment-file.flac")  # unparsable
        paths.append("[00003] [0123456789abcdef] [qwen3] Phrase_3.flac")  # parsable, no voice tag
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("belle_concerned", 0)

    def test_no_voice_tags_at_all(self) -> None:
        assert SoundSegmentUtil.get_voice_tag_summary([]) == ("", 0)
        paths = ["[00001] [0123456789abcdef] [qwen3] Phrase.flac", "junk.flac"]
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("", 0)

    def test_none_placeholder_does_not_compete_with_real_voices(self) -> None:
        paths = [self._seg_filename(i, "none") for i in range(3)]
        paths += [self._seg_filename(i, "belle_concerned") for i in range(3, 5)]
        paths.append(self._seg_filename(5, "other_voice"))
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("belle_concerned", 1)

    def test_real_voice_dominating_over_none(self) -> None:
        paths = [self._seg_filename(0, "none")]
        paths += [self._seg_filename(i, "belle_concerned") for i in range(1, 4)]
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("belle_concerned", 0)

    def test_only_none_reports_none(self) -> None:
        paths = [self._seg_filename(i, "none") for i in range(2)]
        paths.append("not-a-segment-file.flac")
        assert SoundSegmentUtil.get_voice_tag_summary(paths) == ("none", 0)

    def test_make_stem_formats_voice_tag_with_extra_count(self) -> None:
        voices = ["belle_concerned"] * 3 + ["other_voice"]
        project = make_fake_project(num_groups=len(voices))
        project.sound_segments.get_best_file_for.side_effect = (
            lambda i: f"[{i+1:05d}] [0123456789abcdef] [qwen3] [{voices[i]}] Phrase_{i}.flac"
        )

        with patch.object(SoundSegmentUtil, "get_common_model_tag", return_value=""):
            stem = make_stem(
                project=project,
                index_start=0,
                index_end=len(voices) - 1,
                file_cut_index=0,
                num_chapters=1,
            )

        assert "[belle_concerned +1]" in stem

    def test_make_stem_uniform_voice_has_no_extra_count(self) -> None:
        voices = ["belle_concerned"] * 4
        project = make_fake_project(num_groups=len(voices))
        project.sound_segments.get_best_file_for.side_effect = (
            lambda i: f"[{i+1:05d}] [0123456789abcdef] [qwen3] [{voices[i]}] Phrase_{i}.flac"
        )

        with patch.object(SoundSegmentUtil, "get_common_model_tag", return_value=""):
            stem = make_stem(
                project=project,
                index_start=0,
                index_end=len(voices) - 1,
                file_cut_index=0,
                num_chapters=1,
            )

        assert "[belle_concerned]" in stem

    def test_make_stem_prefers_real_voice_over_none_placeholder(self) -> None:
        voices = ["none", "none", "belle_concerned", "belle_concerned", "other_voice"]
        project = make_fake_project(num_groups=len(voices))
        project.sound_segments.get_best_file_for.side_effect = (
            lambda i: f"[{i+1:05d}] [0123456789abcdef] [qwen3] [{voices[i]}] Phrase_{i}.flac"
        )

        with patch.object(SoundSegmentUtil, "get_common_model_tag", return_value=""):
            stem = make_stem(
                project=project,
                index_start=0,
                index_end=len(voices) - 1,
                file_cut_index=0,
                num_chapters=1,
            )

        assert "[belle_concerned +1]" in stem
        assert "none" not in stem

    def test_make_stem_all_none_segments_show_none_tag(self) -> None:
        voices = ["none"] * 4
        project = make_fake_project(num_groups=len(voices))
        project.sound_segments.get_best_file_for.side_effect = (
            lambda i: f"[{i+1:05d}] [0123456789abcdef] [qwen3] [{voices[i]}] Phrase_{i}.flac"
        )

        with patch.object(SoundSegmentUtil, "get_common_model_tag", return_value=""):
            stem = make_stem(
                project=project,
                index_start=0,
                index_end=len(voices) - 1,
                file_cut_index=0,
                num_chapters=1,
            )

        assert "[none]" in stem
