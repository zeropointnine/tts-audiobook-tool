"""Staging area for quick-generation output that awaits user review.

A reviewable quick generation writes its sound segment (and the parallel
STT/timing sidecar) into a per-job directory under the project's
``segments_temp/`` instead of ``segments/``. The files carry their normal
segment file names, so keeping the new sound is a same-volume move into
``segments/``; discarding it just deletes the job directory.

``segments_temp/`` lives outside ``segments/`` so the segment directory watcher
ignores it. Everything in it is disposable: the whole directory is deleted
when the Generate editor exits.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from tts_audiobook_tool.app_types import SoundSegment
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings
from tts_audiobook_tool.constants import PROJECT_SEGMENTS_TEMP_SUBDIR
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.sound_segment_util import (
    UNKNOWN_ERROR_RANK,
    SoundSegmentUtil,
    get_segment_error_rank,
    get_segment_stt_info_path,
)
from tts_audiobook_tool.util import delete_silently, make_error_string


def get_segments_temp_path(project_dir: str) -> str:
    return os.path.join(project_dir, PROJECT_SEGMENTS_TEMP_SUBDIR)


def make_staging_dir_path(project_dir: str) -> str:
    """Unique, not yet created, per-job staging directory path."""
    return os.path.join(get_segments_temp_path(project_dir), uuid.uuid4().hex)


def list_staged_segments(
    staging_dir: str, phrase_index: int | None = None
) -> list[tuple[Path, SoundSegment]]:
    """Recognizable, non-empty segment files in a staging directory, by name.

    When ``phrase_index`` is given, only that line's takes are listed.
    """
    directory = Path(staging_dir) if staging_dir else None
    if directory is None or not directory.is_dir():
        return []
    result: list[tuple[Path, SoundSegment]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".flac":
            continue
        if path.stat().st_size == 0:
            continue
        segment = SoundSegmentUtil.make_from_file_name(path.name)
        if segment is None:
            continue
        if phrase_index is not None and segment.idx != phrase_index:
            continue
        result.append((path, segment))
    return result


def get_best_staged_segment(
    staging_dir: str, phrase_index: int
) -> tuple[Path, SoundSegment] | None:
    """A line's staged take with the fewest word errors (first by name on ties)."""
    items = list_staged_segments(staging_dir, phrase_index)
    if not items:
        return None
    return min(items, key=lambda item: get_segment_error_rank(item[1], staging_dir))


def prune_staged_segments(staging_dir: str, phrase_index: int) -> None:
    """Keep only a line's best staged take (and its sidecar); delete the rest.

    The staging counterpart of ``ProjectSoundSegments.delete_redundants_for``:
    retries each save a take, and review needs exactly one candidate per line.
    """
    best = get_best_staged_segment(staging_dir, phrase_index)
    if best is None:
        return
    for path, _ in list_staged_segments(staging_dir, phrase_index):
        if path != best[0]:
            delete_silently(str(path))
            delete_silently(str(get_segment_stt_info_path(path)))


# Rename primitive; a seam for failure-injection tests.
_move = os.replace


def accept_staged_segment(project: Project, phrase_index: int, staged_path: Path) -> str:
    """Move a staged take into ``segments/``, replacing the line's old audio.

    The new take can have exactly the same name as an old one (normal segment
    names have no timestamp). Such an old pair is first moved aside into the
    staging directory, so a failure part-way through can restore it: the
    audio/sidecar pair in ``segments/`` is then either entirely old or
    entirely new. The line's other old takes are deleted only on success.
    Returns an error message, or an empty string on success.
    """
    dest_dir = Path(project.sound_segments_path)
    # The catalog lists only segments matching the line's current text, the
    # same set a delete of the line's generated sound would remove.
    old_paths = [
        dest_dir / segment.file_name
        for segment in project.sound_segments.sound_segments_map.get(phrase_index, [])
    ]
    dest_path = dest_dir / staged_path.name
    staged_json = get_segment_stt_info_path(staged_path)
    dest_json = get_segment_stt_info_path(dest_path)
    backup_dir = staged_path.parent / "replaced"

    moves: list[tuple[Path, Path]] = []

    def move(source: Path, target: Path) -> None:
        _move(source, target)
        moves.append((source, target))

    try:
        os.makedirs(dest_dir, exist_ok=True)
        # Move a same-named old pair aside; the new audio must never be paired
        # with the old sidecar.
        for path in (dest_path, dest_json):
            if path.exists():
                os.makedirs(backup_dir, exist_ok=True)
                move(path, backup_dir / path.name)
        if staged_json.exists():
            move(staged_json, dest_json)
        move(staged_path, dest_path)
    except OSError as e:
        for source, target in reversed(moves):
            try:
                _move(target, source)
            except OSError:
                pass
        project.sound_segments.force_invalidate()
        return f"Couldn't keep new sound: {make_error_string(e)}"
    stale_paths = [path for path in old_paths if path.name != dest_path.name]
    project.sound_segments.delete_path_snapshot(stale_paths)
    shutil.rmtree(backup_dir, ignore_errors=True)
    return ""


def describe_segment_word_errors(path: str | Path) -> str:
    """Short word-error description of a segment file, from its name.

    A name carries an error count only when it is non-zero, so a validated
    segment (one with an STT sidecar) without a count has no word errors.
    Returns an empty string when the segment was never validated.
    """
    segment = SoundSegmentUtil.make_from_file_name(str(path))
    if segment is None:
        return ""
    rank = get_segment_error_rank(segment, Path(path).parent)
    if rank == UNKNOWN_ERROR_RANK:
        return ""
    if rank == ValidationFindings.LEGACY_INVALID_SCORE:
        return "failed validation"
    if rank == 0:
        return "no word errors"
    noun = "word error" if rank == 1 else "word errors"
    return f"{rank} {noun}"


def delete_staging_dir(staging_dir: str) -> None:
    if staging_dir:
        shutil.rmtree(staging_dir, ignore_errors=True)


def delete_segments_temp(project_dir: str) -> None:
    """Housekeeping: delete every staged quick-generation output of a project."""
    if project_dir:
        shutil.rmtree(get_segments_temp_path(project_dir), ignore_errors=True)
