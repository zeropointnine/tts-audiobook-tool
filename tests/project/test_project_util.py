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

def test_get_latest_concat_files_lists_every_abr_file_per_subdir(tmp_path) -> None:
    # Every .abr file in each concat subdirectory is listed (newest subdirectory first,
    # files in natural order), debug artifacts are skipped, and the result is clamped to `limit`.
    import os
    from types import SimpleNamespace

    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    for name in ("book [1 of 1].abr.m4b", "notes.txt"):
        (older / name).write_text("")
    for name in ("book [10 of 12].abr.flac", "book [2 of 12].abr.flac", "book [2 of 12] [concat].abr.flac"):
        (newer / name).write_text("")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    project = cast(Project, SimpleNamespace(concat_path=str(tmp_path)))

    paths = [path for path, _ in ProjectUtil.get_latest_concat_files(project, limit=10)]
    assert paths == [
        str(newer / "book [2 of 12].abr.flac"),
        str(newer / "book [10 of 12].abr.flac"),
        str(older / "book [1 of 1].abr.m4b"),
    ]

    assert len(ProjectUtil.get_latest_concat_files(project, limit=2)) == 2
