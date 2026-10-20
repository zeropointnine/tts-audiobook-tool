from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.app_types.segment_transcript_data import SegmentTranscriptData
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.segment_transcript_util import (
    SegmentTranscriptUtil,
)


class StubSoundSegments:
    def __init__(self, sound_segment=None) -> None:
        self.sound_segment = sound_segment

    def get_best_item_for(self, _index: int):
        return self.sound_segment


def make_project(sound_segment=None) -> Project:
    return cast(
        Project,
        SimpleNamespace(
            sound_segments=StubSoundSegments(sound_segment),
            sound_segments_path="/project/segments",
            phrase_groups=[SimpleNamespace(text="Legacy source text")],
            language_code="en",
            strictness=Strictness.MODERATE,
        ),
    )


def strip_formatting(lines: list[str]) -> str:
    return text_util.strip_ansi_codes(
        text_util.combine_ansi_lines(lines)
    )


def test_make_info_text_reports_missing_segment_without_printing(capsys) -> None:
    lines = SegmentTranscriptUtil.make_info_text_lines(2, make_project())

    plain_text = strip_formatting(lines)
    assert all("\n" not in line for line in lines)
    assert "Line: 3" in plain_text
    assert "No generated sound segment found." in plain_text
    assert capsys.readouterr().out == ""


def test_make_info_text_formats_modern_sidecar_and_word_diff(capsys) -> None:
    sound_segment = SimpleNamespace(file_name="line-1.flac", num_errors=1)
    project = make_project(sound_segment)
    info = SegmentTranscriptData(
        version=SegmentTranscriptUtil.VERSION,
        type=SegmentTranscriptUtil.TYPE,
        language_code="en",
        index_1b=1,
        source="the cat",
        prompt="Read the cat",
        transcript="a dog",
        normalized_source="the cat",
        normalized_transcript="a dog",
        timed_phrases=[],
        transcript_words=[],
        findings=ValidationFindings(transcript_errors=["the/a", "cat/dog"]),
    )

    with (
        patch.object(SegmentTranscriptUtil, "load", return_value=info),
        patch.object(
            SegmentTranscriptUtil,
            "make_word_error_visualization",
            return_value="[=/=: the/a] [=/=: cat/dog]",
        ),
    ):
        lines = SegmentTranscriptUtil.make_info_text_lines(0, project)

    text = text_util.combine_ansi_lines(lines)
    plain_text = strip_formatting(lines)
    assert all("\n" not in line for line in lines)
    assert "Line: 1, Word error fail" in plain_text
    assert "Filename: line-1.flac" in plain_text
    assert "Source text              : the cat" in plain_text
    assert "TTS prompt               : Read the cat" in plain_text
    assert "STT transcript           : a dog" in plain_text
    assert "Word error visualization:" in plain_text
    assert "[=/=: the/a]" in plain_text
    assert "[=/=: cat/dog]" in plain_text
    assert "file:///project/segments/line-1.flac" in text
    assert capsys.readouterr().out == ""


def test_make_info_text_formats_sidecar_load_failure(capsys) -> None:
    sound_segment = SimpleNamespace(file_name="broken.flac", num_errors=0)
    project = make_project(sound_segment)

    with (
        patch.object(SegmentTranscriptUtil, "load", return_value="invalid JSON"),
        patch.object(
            SegmentTranscriptUtil,
            "load_timed_phrases",
            return_value="invalid timing data",
        ),
    ):
        lines = SegmentTranscriptUtil.make_info_text_lines(0, project)

    plain_text = strip_formatting(lines)
    assert all("\n" not in line for line in lines)
    assert "Could not load segment STT info: invalid JSON" in plain_text
    assert "Filename: broken.flac" in plain_text
    assert capsys.readouterr().out == ""


def test_make_info_text_formats_legacy_timing_sidecar(capsys) -> None:
    sound_segment = SimpleNamespace(file_name="legacy.flac", num_errors=2)
    project = make_project(sound_segment)

    with (
        patch.object(SegmentTranscriptUtil, "load", return_value="legacy payload"),
        patch.object(SegmentTranscriptUtil, "load_timed_phrases", return_value=[]),
    ):
        lines = SegmentTranscriptUtil.make_info_text_lines(0, project)

    plain_text = strip_formatting(lines)
    assert all("\n" not in line for line in lines)
    assert "Line: 1, word errors detected: 2" in plain_text
    assert "Filename: legacy.flac" in plain_text
    assert "Legacy timing data; detailed STT info unavailable" in plain_text
    assert capsys.readouterr().out == ""
