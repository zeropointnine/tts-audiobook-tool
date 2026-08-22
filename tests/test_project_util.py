from typing import cast

import pytest

from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project import Project
from textual_editor_stubs import (
    StubPhraseGroup,
    StubProject,
    StubSoundSegment,
    StubSoundSegments,
)


@pytest.mark.parametrize(
    ("group_count", "segments", "range_string", "expected_range", "expected_save_calls"),
    [
        # Generated items inside the requested range are dropped, and a
        # stale segment index outside the phrase groups is ignored.
        (
            5,
            {1: ["segment-1.flac"], 3: ["segment-3.flac"], 9: ["stale-segment.flac"]},
            "1-4",
            "1, 3",
            ["1, 3"],
        ),
        # A range that only covers generated items collapses to "none".
        (
            2,
            {0: ["segment-0.flac"], 1: ["segment-1.flac"]},
            "all",
            "none",
            ["none"],
        ),
        # A range with no generated items is persisted unchanged.
        (2, {1: ["segment-1.flac"]}, "1", "1", []),
    ],
)
def test_persist_range_without_generated_items_updates_and_saves_project(
    group_count: int,
    segments: dict[int, list[str]],
    range_string: str,
    expected_range: str,
    expected_save_calls: list[str],
) -> None:
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(group_count)],
        StubSoundSegments(
            {
                index: [StubSoundSegment(name) for name in names]
                for index, names in segments.items()
            }
        ),
        generate_range_string=range_string,
    )

    error = ProjectUtil.persist_range_without_generated_items(
        cast(Project, project)
    )

    assert error == ""
    assert project.generate_range_string == expected_range
    assert project.save_calls == expected_save_calls