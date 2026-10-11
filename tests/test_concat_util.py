import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.concat_util import (
    ConcatRenderResult,
    ConcatUtil,
    make_stem,
    make_subdivided_timed_phrases,
)
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.sentence_gap_util import SentenceGapUtil
from tts_audiobook_tool.sound.silence_util import GapTrimMap, SilenceGapTrim
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


def test_subdivided_metadata_preserves_phrases_for_missing_audio_groups() -> None:
    groups = [
        PhraseGroup([
            Phrase("First phrase, ", Reason.PHRASE),
            Phrase("second phrase. ", Reason.SENTENCE),
        ]),
        PhraseGroup([
            Phrase("Third phrase.\n\n", Reason.PARAGRAPH),
        ]),
    ]
    timed_groups = [
        TimedPhrase(group.text, 0.0, 0.0)
        for group in groups
    ]

    segments, bookmarks, group_start_indices = make_subdivided_timed_phrases(
        timed_phrases=timed_groups,
        phrase_groups=groups,
        sound_paths=["", ""],
        sound_durations=[0.0, 0.0],
        bookmark_indices=[1],
    )

    assert len(segments) == 2
    first_group = segments[0]
    second_group = segments[1]
    assert isinstance(first_group, list)
    assert isinstance(second_group, list)
    assert [segment.text for segment in first_group] == [
        "First phrase, ",
        "second phrase. ",
    ]
    assert [segment.text for segment in second_group] == ["Third phrase.\n\n"]
    assert all(
        segment.time_start == 0.0 and segment.time_end == 0.0
        for group in (first_group, second_group)
        for segment in group
    )
    assert bookmarks == [2]
    assert group_start_indices == [0, 2]


def test_subdivided_metadata_nests_valid_sidecars_and_keeps_flat_leaf_indices() -> None:
    groups = [
        PhraseGroup([
            Phrase("First phrase, ", Reason.PHRASE),
            Phrase("second phrase. ", Reason.SENTENCE),
        ]),
        PhraseGroup([
            Phrase("Missing one, ", Reason.PHRASE),
            Phrase("missing two.", Reason.SENTENCE),
        ]),
    ]
    timed_groups = [
        TimedPhrase(groups[0].text, 0.0, 3.0),
        TimedPhrase(groups[1].text, 0.0, 0.0),
    ]
    sidecar_path = MagicMock()
    sidecar_path.exists.return_value = True
    sidecar_segments = [
        TimedPhrase("First phrase, ", 0.0, 1.0),
        TimedPhrase("second phrase. ", 1.0, 2.5),
    ]

    with patch("tts_audiobook_tool.concat_util.get_segment_stt_info_path", return_value=sidecar_path), \
         patch("tts_audiobook_tool.concat_util.SegmentTranscriptUtil.load_timed_phrases", return_value=sidecar_segments):
        segments, bookmarks, group_start_indices = make_subdivided_timed_phrases(
            timed_phrases=timed_groups,
            phrase_groups=groups,
            sound_paths=["generated.flac", ""],
            sound_durations=[3.0, 0.0],
            bookmark_indices=[0, 1],
        )

    assert len(segments) == 2
    assert isinstance(segments[0], list)
    assert [segment.text for segment in segments[0]] == ["First phrase, ", "second phrase. "]
    assert segments[0][-1].time_end == 3.0
    assert isinstance(segments[1], list)
    assert [segment.text for segment in segments[1]] == ["Missing one, ", "missing two."]
    assert all(
        segment.time_start == 0.0 and segment.time_end == 0.0
        for segment in segments[1]
    )
    assert bookmarks == [0, 2]
    assert group_start_indices == [0, 2]


def test_subdivided_metadata_keeps_parse_failure_as_flat_segment() -> None:
    group = PhraseGroup([Phrase("Original group.", Reason.SENTENCE)])
    original = TimedPhrase(group.text, 0.0, 2.0)
    sidecar_path = MagicMock()
    sidecar_path.exists.return_value = True

    with patch("tts_audiobook_tool.concat_util.get_segment_stt_info_path", return_value=sidecar_path), \
         patch("tts_audiobook_tool.concat_util.SegmentTranscriptUtil.load_timed_phrases", return_value="bad sidecar"):
        segments, bookmarks, group_start_indices = make_subdivided_timed_phrases(
            timed_phrases=[original],
            phrase_groups=[group],
            sound_paths=["generated.flac"],
            sound_durations=[2.0],
            bookmark_indices=[0],
        )

    assert segments == [original]
    assert bookmarks == [0]
    assert group_start_indices == [0]


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
            phrases_and_paths=[(phrase, "segment.flac")],
            use_break_sound_effect=True,
            high_shelf=HighShelfEq.DISABLED,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            print_progress=False,
        )

    assert result.durations == [sound.duration]
    append_mock.assert_called_once_with(
        sound,
        reason=reason,
        reason_pauses=ReasonPauseTypes.NORMAL.value,
        break_effect=None,
        pause_duration_override=None,
    )


def concatenate_with_mocks(phrases: list[Phrase]) -> list[Reason | None]:
    """
    Runs the concat flow over one present segment per phrase and returns the
    `break_effect` decision recorded for each flushed segment.
    """
    sound = Sound(np.zeros(48000, dtype=np.float32), 48000)
    decisions: list[Reason | None] = []

    def fake_append(sound_in, reason, reason_pauses, break_effect, pause_duration_override):
        decisions.append(break_effect)
        return sound_in

    with patch.object(ConcatUtil, "init_ffmpeg_stream", return_value=MagicMock()), \
         patch.object(ConcatUtil, "close_ffmpeg_stream"), \
         patch.object(ConcatUtil, "add_audio_to_ffmpeg_stream"), \
         patch.object(ConcatUtil, "PSEUDO_SILENCE_COMPENSATION_ENABLED", False), \
         patch.object(
             SoundPipeline,
             "make_concat_rendered_sound_segment",
             side_effect=lambda phrase, path, high_shelf, **kwargs: sound,
         ), \
         patch.object(SoundPipeline, "append_pause_or_section_effect", side_effect=fake_append):
        ConcatUtil.concatenate_sound_segments(
            dest_path="output.flac",
            phrases_and_paths=[(phrase, f"segment{i}.flac") for i, phrase in enumerate(phrases)],
            use_break_sound_effect=True,
            high_shelf=HighShelfEq.DISABLED,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            print_progress=False,
        )

    return decisions


def test_concatenate_collapses_consecutive_space_break_runs() -> None:
    phrases = [
        Phrase("One", Reason.SENTENCE),
        Phrase("Two", Reason.SPACE_BREAK),
        Phrase("Three", Reason.SPACE_BREAK),
        Phrase("Four", Reason.SPACE_BREAK),
    ]

    decisions = concatenate_with_mocks(phrases)

    assert decisions == [None, Reason.SPACE_BREAK, None, None]


def test_concatenate_applies_rule1_per_section_and_missing_leading_audio() -> None:
    sound = Sound(np.zeros(48000, dtype=np.float32), 48000)
    phrases = [
        Phrase("Zeroth", Reason.SENTENCE),
        Phrase("First", Reason.SPACE_BREAK),   # group 1, section start, file missing
        Phrase("Second", Reason.SPACE_BREAK),  # group 2: first present of section -> rule 1
        Phrase("Third", Reason.SPACE_BREAK),   # group 3: run continues -> rule 2
    ]
    decisions: list[Reason | None] = []

    def fake_append(sound_in, reason, reason_pauses, break_effect, pause_duration_override):
        decisions.append(break_effect)
        return sound_in

    phrases_and_paths = [
        (phrases[0], "segment0.flac"),
        (phrases[1], ""),  # missing audio
        (phrases[2], "segment2.flac"),
        (phrases[3], "segment3.flac"),
    ]

    with patch.object(ConcatUtil, "init_ffmpeg_stream", return_value=MagicMock()), \
         patch.object(ConcatUtil, "close_ffmpeg_stream"), \
         patch.object(ConcatUtil, "add_audio_to_ffmpeg_stream"), \
         patch.object(ConcatUtil, "PSEUDO_SILENCE_COMPENSATION_ENABLED", False), \
         patch.object(
             SoundPipeline,
             "make_concat_rendered_sound_segment",
             side_effect=lambda phrase, path, high_shelf, **kwargs: sound,
         ), \
         patch.object(SoundPipeline, "append_pause_or_section_effect", side_effect=fake_append):
        ConcatUtil.concatenate_sound_segments(
            dest_path="output.flac",
            phrases_and_paths=phrases_and_paths,
            use_break_sound_effect=True,
            high_shelf=HighShelfEq.DISABLED,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            print_progress=False,
            section_start_indices=[0, 1],
        )

    # Only present segments get decisions, aligned to their group indices
    assert decisions == [None, None, None]


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
# ---------------------------------------------------------------------------
# Silence gap limiting (applied at concat time, not generation time)
# ---------------------------------------------------------------------------

GAP_SR = 16_000


def make_gap_sound() -> Sound:
    """ Tone, 1.0s of silence, tone """
    tone = (np.sin(2 * np.pi * 440 * np.arange(int(GAP_SR * 0.5)) / GAP_SR) * 0.5).astype(np.float32)
    silence = np.zeros(int(GAP_SR * 1.0), dtype=np.float32)
    return Sound(np.concatenate([tone, silence, tone]), GAP_SR)


def concatenate_gap_sound(
    limit_silence_gaps: bool,
    sound: Sound | None = None,
    min_sentence_silence: float = 0.0,
    segment_path: str = "segment.flac",
) -> object:
    sound = sound if sound is not None else make_gap_sound()
    phrase = Phrase("Gap phrase.", Reason.SENTENCE)

    with patch.object(ConcatUtil, "init_ffmpeg_stream", return_value=MagicMock()), \
         patch.object(ConcatUtil, "close_ffmpeg_stream"), \
         patch.object(ConcatUtil, "add_audio_to_ffmpeg_stream"), \
         patch.object(ConcatUtil, "PSEUDO_SILENCE_COMPENSATION_ENABLED", False), \
         patch.object(SoundPipeline, "make_concat_rendered_sound_segment", return_value=sound), \
         patch.object(SoundPipeline, "append_pause_or_section_effect", side_effect=lambda sound_in, **kwargs: sound_in):
        return ConcatUtil.concatenate_sound_segments(
            dest_path="output.flac",
            phrases_and_paths=[(phrase, segment_path)],
            use_break_sound_effect=False,
            high_shelf=HighShelfEq.DISABLED,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            print_progress=False,
            limit_silence_gaps=limit_silence_gaps,
            limit_silence_gaps_duration=0.2,
            min_sentence_silence=min_sentence_silence,
        )


def test_concatenate_without_gap_limiting_keeps_duration_and_no_maps() -> None:
    sound = make_gap_sound()
    result = concatenate_gap_sound(limit_silence_gaps=False)
    assert isinstance(result, ConcatRenderResult)
    assert result.durations == [sound.duration]
    assert result.gap_trim_maps == [None]
    assert result.gap_trim_count == 0


def test_concatenate_with_gap_limiting_cuts_audio_and_records_map() -> None:
    sound = make_gap_sound()
    result = concatenate_gap_sound(limit_silence_gaps=True)
    assert isinstance(result, ConcatRenderResult)

    gap_map = result.gap_trim_maps[0]
    assert gap_map is not None
    assert result.gap_trim_count == 1
    assert result.gap_removed_duration == pytest.approx(0.8, abs=0.08)

    # The written duration is the cut one, and the map agrees with it.
    assert result.durations[0] < sound.duration
    assert result.durations[0] == pytest.approx(sound.duration - 0.8, abs=0.08)
    assert gap_map.new_duration == pytest.approx(result.durations[0], abs=0.01)

    # A word timing after the cut maps earlier than it was.
    assert gap_map.map_time(sound.duration - 0.1) < sound.duration - 0.1


def test_subdivided_metadata_maps_word_timings_through_gap_cut() -> None:
    group = PhraseGroup([
        Phrase("First phrase, ", Reason.PHRASE),
        Phrase("second phrase.", Reason.SENTENCE),
    ])
    timed_groups = [TimedPhrase(group.text, 0.0, 3.0)]
    sidecar_path = MagicMock()
    sidecar_path.exists.return_value = True
    sidecar_segments = [
        TimedPhrase("First phrase, ", 0.0, 1.0),
        TimedPhrase("second phrase.", 2.0, 3.0),
    ]
    # A 1.0s gap (1.0s-2.0s) cut down to 0.2s: everything after shifts 0.8s earlier.
    gap_map = GapTrimMap(3.0, [SilenceGapTrim(1.0, 2.0, 1.0, 1.2)])

    with patch("tts_audiobook_tool.concat_util.get_segment_stt_info_path", return_value=sidecar_path), \
         patch("tts_audiobook_tool.concat_util.SegmentTranscriptUtil.load_timed_phrases", return_value=sidecar_segments):
        segments, _, _ = make_subdivided_timed_phrases(
            timed_phrases=timed_groups,
            phrase_groups=[group],
            sound_paths=["generated.flac"],
            sound_durations=[2.5],
            bookmark_indices=[0],
            gap_trim_maps=[gap_map],
        )

    assert [segment.text for segment in segments[0]] == ["First phrase, ", "second phrase."]
    assert segments[0][0].time_end == pytest.approx(1.0)
    assert segments[0][1].time_start == pytest.approx(1.0)
    # Tail is clamped to the (cut) segment duration.
    assert segments[0][-1].time_end == 2.5


@pytest.fixture
def sentence_gap_sidecar(tmp_path: Path):
    from test_sentence_gap_util import Utterance

    u = Utterance(gap_s=0.15)
    payload = {
        "version": 4, "type": "segment_stt_info", "language_code": "en",
        "index_1b": 1, "source": "Hello there friend. How are you today",
        "prompt": "", "transcript": "", "normalized_source": "",
        "normalized_transcript": "", "findings": {},
        "timed_phrases": TimedPhrase.timed_phrases_to_dicts(u.timed_phrases, include_reason=True),
        "transcript_words": u.words,
    }
    segment_path = tmp_path / "segment.flac"
    sidecar_path = segment_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    return u.sound, segment_path, sidecar_path, payload


@pytest.mark.parametrize("collection,field,value", [
    ("transcript_words", "start", "not a time"),
    ("transcript_words", "start", None),
    ("transcript_words", "start", True),
    ("transcript_words", "start", float("nan")),
    ("transcript_words", "end", float("inf")),
    ("transcript_words", "end", float("-inf")),
    ("transcript_words", "word", 42),
    ("timed_phrases", "time_start", "not a time"),
    ("timed_phrases", "time_end", float("nan")),
    ("timed_phrases", "time_end", float("inf")),
    ("timed_phrases", "text", None),
    ("timed_phrases", "reason", 4),
    ("timed_phrases", "reason", []),
])
def test_sentence_gap_extension_rejects_malformed_fields(
    sentence_gap_sidecar, collection, field, value,
) -> None:
    # Real JSON parsing can succeed while leaving unsafe values in timing objects.
    sound, segment_path, sidecar_path, payload = sentence_gap_sidecar
    payload[collection][0][field] = value
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with patch("tts_audiobook_tool.concat_util.L.w") as warning:
        result, insertions = ConcatUtil.extend_sentence_gaps_for_segment(sound, str(segment_path), 0.4)

    assert result is sound
    assert insertions == []
    warning.assert_called_once()
    assert str(sidecar_path) in warning.call_args.args[0]


@pytest.mark.parametrize("collection,field", [
    ("transcript_words", "start"),
    ("transcript_words", "end"),
    ("transcript_words", "word"),
    ("timed_phrases", "time_end"),
])
def test_sentence_gap_extension_missing_fields_do_not_abort_concat(
    sentence_gap_sidecar, collection, field,
) -> None:
    # The original missing-word-start crash must leave both the segment and concat intact.
    sound, segment_path, sidecar_path, payload = sentence_gap_sidecar
    del payload[collection][0][field]
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with patch("tts_audiobook_tool.concat_util.L.w") as warning:
        unchanged, insertions = ConcatUtil.extend_sentence_gaps_for_segment(sound, str(segment_path), 0.4)
        result = concatenate_gap_sound(False, sound, 0.4, str(segment_path))

    assert unchanged is sound
    assert insertions == []
    assert isinstance(result, ConcatRenderResult)
    assert result.durations == [sound.duration]
    assert result.gap_trim_maps == [None]
    assert warning.call_count == 2
    assert all(str(sidecar_path) in call.args[0] for call in warning.call_args_list)


@pytest.mark.parametrize("malformation", ["invalid_json", "legacy_missing_field", "non_dict_word"])
def test_sentence_gap_extension_handles_sidecar_parsing_failures(sentence_gap_sidecar, malformation) -> None:
    # In particular, from_dict's legacy branch can raise outside load's JSON guard.
    sound, segment_path, sidecar_path, payload = sentence_gap_sidecar
    if malformation == "legacy_missing_field":
        del payload["findings"]
        del payload["normalized_source"]
    elif malformation == "non_dict_word":
        payload["transcript_words"][0] = None
    sidecar_path.write_text("{" if malformation == "invalid_json" else json.dumps(payload), encoding="utf-8")

    with patch("tts_audiobook_tool.concat_util.L.w") as warning:
        result, insertions = ConcatUtil.extend_sentence_gaps_for_segment(sound, str(segment_path), 0.4)

    assert result is sound
    assert insertions == []
    warning.assert_called_once()
    assert str(sidecar_path) in warning.call_args.args[0]


@pytest.mark.parametrize("legacy", [False, True])
def test_sentence_gap_extension_accepts_valid_sidecars(sentence_gap_sidecar, legacy) -> None:
    # Existing sidecars without reasons still skip safely; modern ones still extend.
    sound, segment_path, sidecar_path, payload = sentence_gap_sidecar
    # Whisper may emit zero-length/overlapping words or times just beyond the clip.
    payload["transcript_words"][0]["end"] = 0
    payload["transcript_words"][1]["start"] = 0
    payload["transcript_words"][-1]["end"] = sound.duration + 0.01
    if legacy:
        for phrase in payload["timed_phrases"]:
            del phrase["reason"]
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with patch("tts_audiobook_tool.concat_util.L.w") as warning:
        result, insertions = ConcatUtil.extend_sentence_gaps_for_segment(sound, str(segment_path), 0.4)

    warning.assert_not_called()
    if legacy:
        assert result is sound
        assert insertions == []
    else:
        assert len(insertions) == 1
        assert result.duration > sound.duration + 0.2


def test_sentence_gap_extension_inserts_digital_silence(sentence_gap_sidecar) -> None:
    # Even a gap with an audible noise floor is extended with digital silence
    from test_sentence_gap_util import Utterance

    _, segment_path, _, _ = sentence_gap_sidecar
    sound = Utterance(gap_s=0.15, floor_level=0.0003).sound  # same timings as the sidecar
    result, insertions = ConcatUtil.extend_sentence_gaps_for_segment(sound, str(segment_path), 0.4)

    assert len(insertions) == 1
    start = round(insertions[0].position * sound.sr)
    stop = start + round(insertions[0].duration * sound.sr)
    assert not np.any(result.data[start:stop])


@pytest.mark.parametrize("error", [ValueError("pipeline bug"), RuntimeError("runtime failure")])
def test_sentence_gap_extension_does_not_hide_pipeline_errors(sentence_gap_sidecar, error) -> None:
    # Data fallback must not silently swallow unrelated failures in audio processing.
    sound, segment_path, _, _ = sentence_gap_sidecar
    with patch.object(SoundPipeline, "extend_sentence_gaps_with_insertions", side_effect=error), \
         pytest.raises(type(error), match=str(error)):
        ConcatUtil.extend_sentence_gaps_for_segment(sound, str(segment_path), 0.4)


def test_sentence_gap_extension_runs_before_limiting_and_map_composes() -> None:
    # A 0.15s sentence gap is extended to 0.4s, then the limiter (max 0.2s) cuts
    # it back down: the two steps overlap, extension first. The recorded map
    # must take a sidecar time through both so it lands on the final audio.
    from test_sentence_gap_util import Utterance

    u = Utterance(gap_s=0.15)
    info = MagicMock(transcript_words=u.words, timed_phrases=u.timed_phrases)
    sidecar_path = MagicMock()
    sidecar_path.exists.return_value = True

    with patch("tts_audiobook_tool.concat_util.get_segment_stt_info_path", return_value=sidecar_path), \
         patch("tts_audiobook_tool.concat_util.SegmentTranscriptUtil.load", return_value=info):
        extended_only = concatenate_gap_sound(False, u.sound, min_sentence_silence=0.4)
        both = concatenate_gap_sound(True, u.sound, min_sentence_silence=0.4)
        disabled = concatenate_gap_sound(False, u.sound, min_sentence_silence=0.0)

    assert isinstance(extended_only, ConcatRenderResult)
    assert isinstance(both, ConcatRenderResult)
    assert isinstance(disabled, ConcatRenderResult)

    # Disabled: untouched, no map
    assert disabled.gap_trim_maps == [None]

    # Extension alone: audio grows and the map shifts later words by the insertion
    ext_map = extended_only.gap_trim_maps[0]
    assert ext_map is not None
    assert extended_only.durations[0] > u.sound.duration + 0.2
    assert extended_only.durations[0] == pytest.approx(ext_map.new_duration, abs=0.01)
    last_word_start = u.words[3]["start"]
    assert ext_map.map_time(last_word_start + 0.01) > last_word_start + 0.2

    # Both: limiter acts on the extended audio, and the map agrees with the result
    both_map = both.gap_trim_maps[0]
    assert both_map is not None
    assert len(both_map.insertions) == 1
    assert both.durations[0] < extended_only.durations[0]
    assert both_map.new_duration == pytest.approx(both.durations[0], abs=0.01)
    assert both_map.map_time(u.sound.duration) == pytest.approx(both.durations[0], abs=0.01)
