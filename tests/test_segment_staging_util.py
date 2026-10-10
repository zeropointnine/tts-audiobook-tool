from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support import segment_staging_util as staging
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil

HASH = "0123456789abcdef"


def segment_name(num_errors: int | None, text: str = "Text", line: int = 3) -> str:
    errors = f" [{num_errors}]" if num_errors is not None else ""
    return f"[{line:05d}] [{HASH}] [none] [voice]{errors} {text}.flac"


def write_segment(directory: Path, name: str, *, sidecar: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(name.encode())
    if sidecar:
        path.with_suffix(".json").write_text(name)
    return path


def test_prune_keeps_fewest_error_take_with_its_sidecar(tmp_path) -> None:
    # Retries each stage a take; review needs exactly the best one.
    worse = write_segment(tmp_path, segment_name(4, "A"))
    best = write_segment(tmp_path, segment_name(1, "B"))
    staging.prune_staged_segments(str(tmp_path), 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [best.name, best.with_suffix(".json").name]
    )
    assert not worse.exists()


def test_prune_keeps_clean_retry_over_failed_take(tmp_path) -> None:
    # A clean take's name has no error count; its sidecar shows it was
    # validated, so it ranks as zero errors and survives a failed→clean retry.
    failed = write_segment(tmp_path, segment_name(4, "A"))
    clean = write_segment(tmp_path, segment_name(None, "B"))
    staging.prune_staged_segments(str(tmp_path), 2)
    assert clean.exists() and clean.with_suffix(".json").exists()
    assert not failed.exists()


def test_prune_is_per_line(tmp_path) -> None:
    # Another line's take in the same staging directory is never pruned.
    other = write_segment(tmp_path, segment_name(5, "Other", line=4))
    write_segment(tmp_path, segment_name(4, "A"))
    best = write_segment(tmp_path, segment_name(1, "B"))
    staging.prune_staged_segments(str(tmp_path), 2)
    assert other.exists()
    assert staging.get_best_staged_segment(str(tmp_path), 2) == (
        best,
        SoundSegmentUtil.make_from_file_name(best.name),
    )


def test_best_staged_segment_ignores_unrecognized_files(tmp_path) -> None:
    write_segment(tmp_path, "debug [raw].flac")
    assert staging.get_best_staged_segment(str(tmp_path), 2) is None
    assert staging.get_best_staged_segment(str(tmp_path / "missing"), 2) is None


def make_project(tmp_path: Path) -> Project:
    segments_dir = tmp_path / "segments"

    def sound_segments_map() -> dict:
        result: dict = {}
        for path in segments_dir.glob("*.flac"):
            segment = SoundSegmentUtil.make_from_file_name(path.name)
            if segment is not None:
                result.setdefault(segment.idx, []).append(segment)
        return result

    class SoundSegments:
        @property
        def sound_segments_map(self) -> dict:
            return sound_segments_map()

        def force_invalidate(self) -> None:
            pass

        def delete_path_snapshot(self, paths) -> None:
            for path in paths:
                Path(path).unlink(missing_ok=True)
                Path(path).with_suffix(".json").unlink(missing_ok=True)

    return cast(
        Project,
        SimpleNamespace(sound_segments_path=str(segments_dir), sound_segments=SoundSegments()),
    )


def test_accept_moves_take_and_replaces_old_audio(tmp_path) -> None:
    # Keeping the new sound moves it (with its sidecar) into segments/ and
    # removes every old take of the line, even a better-scoring one.
    old = write_segment(tmp_path / "segments", segment_name(0, "Old"))
    staged = write_segment(tmp_path / "temp", segment_name(3, "New"))
    error = staging.accept_staged_segment(make_project(tmp_path), 2, staged)
    assert error == ""
    assert sorted(p.name for p in (tmp_path / "segments").iterdir()) == sorted(
        [staged.name, staged.with_suffix(".json").name]
    )
    assert not old.exists()
    assert not staged.exists()


def test_accept_same_name_as_old_take_keeps_the_new_file(tmp_path) -> None:
    # Segment names have no timestamp, so the new take can collide with the
    # old one; the replacement must not be deleted as "old audio".
    name = segment_name(None, "Same")
    write_segment(tmp_path / "segments", name)
    staged = write_segment(tmp_path / "temp", name, sidecar=False)
    staged.write_bytes(b"new audio")
    error = staging.accept_staged_segment(make_project(tmp_path), 2, staged)
    assert error == ""
    dest = tmp_path / "segments" / name
    assert dest.read_bytes() == b"new audio"
    # The old sidecar must not be paired with the new audio.
    assert not dest.with_suffix(".json").exists()


@pytest.mark.parametrize("fail_on_move", [1, 2, 3, 4])
def test_failed_accept_restores_same_named_old_pair(tmp_path, monkeypatch, fail_on_move) -> None:
    # Whichever move fails (old audio/sidecar aside, new sidecar/audio in),
    # segments/ is rolled back to the old, consistent audio/sidecar pair and
    # the line's other takes are kept.
    name = segment_name(None, "Same")
    segments = tmp_path / "segments"
    old = write_segment(segments, name)
    other = write_segment(segments, segment_name(6, "Other"))
    staged = write_segment(tmp_path / "temp", name)
    staged.write_bytes(b"new audio")
    staged.with_suffix(".json").write_text("new json")
    real_move = staging._move
    calls = {"count": 0}

    def flaky_move(source, target) -> None:
        calls["count"] += 1
        if calls["count"] == fail_on_move:
            raise OSError("disk full")
        real_move(source, target)

    monkeypatch.setattr(staging, "_move", flaky_move)
    error = staging.accept_staged_segment(make_project(tmp_path), 2, staged)
    assert "disk full" in error
    assert old.read_bytes() == name.encode()
    assert old.with_suffix(".json").read_text() == name
    assert other.exists()
    # The staged pair is back in place too.
    assert staged.read_bytes() == b"new audio"
    assert staged.with_suffix(".json").read_text() == "new json"


def test_project_best_item_prefers_validated_clean_take(tmp_path) -> None:
    # The project catalog uses the same ranking as staging: a clean take
    # (untagged, with sidecar) beats a take with word errors, while an
    # untagged take without a sidecar (unknown) ranks worst.
    from tts_audiobook_tool.project_support.project_sound_segments import ProjectSoundSegments

    segments = tmp_path / "segments"
    failed = SoundSegmentUtil.make_from_file_name(segment_name(4, "A"))
    clean = SoundSegmentUtil.make_from_file_name(
        write_segment(segments, segment_name(None, "B")).name
    )
    unknown = SoundSegmentUtil.make_from_file_name(
        write_segment(segments, segment_name(None, "C"), sidecar=False).name
    )
    catalog = cast(ProjectSoundSegments, SimpleNamespace(
        project=SimpleNamespace(sound_segments_path=str(segments)),
        sound_segments_map={2: [unknown, failed, clean]},
    ))
    assert ProjectSoundSegments.get_best_item_for(catalog, 2) == clean
    catalog.sound_segments_map[2] = [unknown, failed]
    assert ProjectSoundSegments.get_best_item_for(catalog, 2) == failed


def test_describe_word_errors(tmp_path) -> None:
    assert staging.describe_segment_word_errors(tmp_path / segment_name(1)) == "1 word error"
    assert staging.describe_segment_word_errors(tmp_path / segment_name(3)) == "3 word errors"
    assert (
        staging.describe_segment_word_errors(tmp_path / segment_name(99))
        == "failed validation"
    )
    # No count in the name: clean if validated (has a sidecar), else unknown.
    validated = write_segment(tmp_path, segment_name(None, "V"))
    unvalidated = write_segment(tmp_path, segment_name(None, "U"), sidecar=False)
    assert staging.describe_segment_word_errors(validated) == "no word errors"
    assert staging.describe_segment_word_errors(unvalidated) == ""


def test_delete_segments_temp(tmp_path) -> None:
    write_segment(Path(staging.make_staging_dir_path(str(tmp_path))), segment_name(1))
    staging.delete_segments_temp(str(tmp_path))
    assert not Path(staging.get_segments_temp_path(str(tmp_path))).exists()
