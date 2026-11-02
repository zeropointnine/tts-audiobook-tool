"""Concat export through the real output writers (no writer mocks).

Rendering is stubbed (see concat_test_support), but every rendered segment is
streamed into a real FLAC or M4B file, which is then decoded and inspected.
All of it runs under the no-FFmpeg guard.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile

from audio_fixture_support import assert_no_handles
from concat_test_support import concatenate
from testx.pyav_m4b_chapters import parse_mp4
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book, BookSection, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.concat_util import ConcatRenderResult
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound import m4b_chapter_util
from tts_audiobook_tool.sound.audio_writers import staging_path_for

pytestmark = pytest.mark.no_ffmpeg_exe

SR = APP_SAMPLE_RATE
PAUSE_S = 0.3
SEGMENT_SECONDS = [0.7, 1.13, 0.0, 0.52, 0.91]  # 0.0: a missing segment
TITLES = ["Part one", "Line two.", "Missing.", "Part two", "Line five."]


def burst(seconds: float) -> np.ndarray:
    """A tone starting at full level on the first sample (easy onset to find)."""
    t = np.arange(round(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * 1000 * t + np.pi / 2)).astype(np.float32)


def phrases_and_paths() -> list[tuple[Phrase, str]]:
    return [
        (Phrase(title, Reason.SENTENCE), f"segment{i}.flac" if seconds else "")
        for i, (title, seconds) in enumerate(zip(TITLES, SEGMENT_SECONDS))
    ]


def project_with_two_sections() -> Project:
    def group(text: str) -> PhraseGroup:
        return PhraseGroup([Phrase(text, Reason.SENTENCE)])
    return Project.model_validate({"book": Book(sections=[
        BookSection(title="Part one", phrase_groups=[group(t) for t in TITLES[:3]]),
        BookSection(title="Part two", phrase_groups=[group(t) for t in TITLES[3:]]),
    ])})


def render_bursts():
    renders = iter(Sound(burst(s), SR) for s in SEGMENT_SECONDS if s)
    return lambda *_args, **_kwargs: next(renders)


def append_pause(sound_in: Sound, **_kwargs) -> Sound:
    return Sound(np.concatenate([sound_in.data, np.zeros(round(PAUSE_S * SR), dtype=np.float32)]), sound_in.sr)


def run_export(dest: Path, **kwargs):
    return concatenate(
        phrases_and_paths(), render=kwargs.pop("render", render_bursts()), append=append_pause,
        real_writer=True, dest_path=str(dest), **kwargs,
    )


def onsets(decoded: np.ndarray, expected_starts: list[float]) -> list[float]:
    """Time of the first loud sample at or after 50 ms before each expected start."""
    found = []
    for start in expected_starts:
        begin = max(0, round((start - 0.05) * SR))
        index = int(np.argmax(np.abs(decoded[begin:]) > 0.25)) + begin
        found.append(index / SR)
    return found


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "Hörbuch 書"
    directory.mkdir()
    return directory


def test_flac_export_is_sample_exact(out_dir: Path) -> None:
    dest = out_dir / "buch [concat].flac"
    result = run_export(dest)
    assert isinstance(result, ConcatRenderResult)

    data, rate = soundfile.read(str(dest), dtype="float32")
    assert rate == SR
    assert len(data) == round(sum(result.durations) * SR)
    expected = np.concatenate([append_pause(Sound(burst(s), SR)).data for s in SEGMENT_SECONDS if s])
    np.testing.assert_allclose(data, np.round(expected * 32767) / 32768, atol=1 / 32768)
    assert_no_handles(dest, staging_path_for(str(dest)))


def test_m4b_duration_chapters_and_abr_timing_line_up(out_dir: Path) -> None:
    # One check across the whole timing chain: the durations the concat
    # reports drive both the ABR text timings and the chapters, and both must
    # match where each segment actually starts in the decoded AAC (priming
    # hidden by the edit list) and how long the file is.
    dest = out_dir / "buch [concat].m4b"
    project = project_with_two_sections()
    result = run_export(
        dest,
        make_chapters=lambda durations: m4b_chapter_util.make_chapters(project, durations),
        container_metadata={"title": "Buch", "artist": "App"},
    )
    assert isinstance(result, ConcatRenderResult)
    durations = result.durations
    assert durations[2] == 0.0

    timed = TimedPhrase.make_list_using([phrase for phrase, _ in phrases_and_paths()], durations)
    starts = [phrase.time_start for phrase, seconds in zip(timed, SEGMENT_SECONDS) if seconds]

    with av.open(str(dest)) as container:
        assert container.duration is not None
        assert container.duration / av.time_base == pytest.approx(sum(durations), abs=1e-3)
        chapters = [(c["metadata"]["title"], float(c["start"] * c["time_base"])) for c in container.chapters()]
        decoded = np.concatenate([frame.to_ndarray()[0] for frame in container.decode(audio=0)])

    assert chapters == [("Part one", 0.0), ("Part two", pytest.approx(timed[3].time_start, abs=1e-3))]
    for found, expected in zip(onsets(decoded, starts), starts):
        assert found == pytest.approx(expected, abs=0.002)
    assert sum(durations) * SR <= len(decoded) < sum(durations) * SR + 1024
    assert parse_mp4(dest).faststart
    assert_no_handles(dest, staging_path_for(str(dest)))


def test_interrupt_removes_partial_output(out_dir: Path) -> None:
    dest = out_dir / "buch [concat].m4b"
    renders = iter(Sound(burst(s), SR) for s in SEGMENT_SECONDS if s)
    count = 0

    def render(*_args, **_kwargs) -> Sound:
        nonlocal count
        count += 1
        if count == 3:
            Interrupts()._flag = True  # what Control-C does
        return next(renders)

    try:
        result = run_export(dest, render=render)
    finally:
        Interrupts().clear()
    assert result == "Interrupted by user"
    assert os.listdir(out_dir) == []
    assert_no_handles(dest, staging_path_for(str(dest)))


def test_render_error_and_exception_remove_partial_output(out_dir: Path) -> None:
    dest = out_dir / "buch [concat].flac"
    sounds = iter([Sound(burst(0.5), SR), "Couldn't load segment"])
    assert run_export(dest, render=lambda *_a, **_k: next(sounds)) == "Couldn't load segment"
    assert os.listdir(out_dir) == []

    def explode(*_args, **_kwargs) -> Sound:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_export(dest, render=explode)
    assert os.listdir(out_dir) == []
    assert_no_handles(dest, staging_path_for(str(dest)))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /dev/full")
def test_encoder_write_failure_is_reported_not_swallowed(out_dir: Path) -> None:
    # The old FFmpeg pipe ignored BrokenPipeError and exit codes; a failing
    # write must now end the export with an error and no output
    dest = out_dir / "buch [concat].m4b"
    os.symlink("/dev/full", staging_path_for(str(dest)))
    result = run_export(dest)
    assert isinstance(result, str) and result.startswith("Error ")
    assert os.listdir(out_dir) == []
    assert_no_handles(dest, "/dev/full")


def test_make_file_end_to_end_m4b(out_dir: Path) -> None:
    # The whole export flow with little stubbed: real segment rendering, the
    # AAC writer with chapters, then the ABR tag on the final file. Only the
    # final file remains.
    #
    # Segment files are read with soundfile here because SoundFileUtil.load
    # still goes through librosa, whose first load per process probes
    # `ffmpeg -version` via audioread. Phase 3 of the pip-only plan replaces
    # that loader; drop the stub then.
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import patch

    from tts_audiobook_tool.app_types import ExportType
    from tts_audiobook_tool.app_types.app_metadata import AppMetadata
    from tts_audiobook_tool.concat_util import ConcatUtil
    from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.state import State
    from no_ffmpeg_guard import no_ffmpeg_exe

    project = project_with_two_sections()
    project.export_type = ExportType.AAC
    segment_paths = []
    for i, seconds in enumerate(SEGMENT_SECONDS):
        if not seconds:
            segment_paths.append("")
            continue
        path = out_dir / f"segment {i} ü.flac"
        soundfile.write(str(path), burst(seconds), SR, subtype="PCM_16")
        segment_paths.append(str(path))
    pairs = [(phrase, path) for (phrase, _), path in zip(phrases_and_paths(), segment_paths)]
    state = SimpleNamespace(project=project, prefs=SimpleNamespace(aac_bitrate="96k", save_debug_files=False))
    stem = str(out_dir / "Buch ü")

    with no_ffmpeg_exe(), \
            patch.object(ProjectTextIOUtil, "load_raw_text", return_value="\n".join(TITLES)), \
            patch.object(ConcatUtil, "make_phrases_and_paths", return_value=pairs), \
            patch.object(SoundFileUtil, "load", side_effect=lambda path, target_sr=0: Sound(*soundfile.read(path, dtype="float32"))), \
            patch("tts_audiobook_tool.concat_util.DEV", False):
        final_path, error = ConcatUtil.make_file(cast(State, state), 0, len(TITLES) - 1, [], stem, use_upsampler=False)

    assert error == ""
    assert final_path == stem + ".abr.m4b"
    assert sorted(p.name for p in out_dir.iterdir() if not p.name.startswith("segment")) == ["Buch ü.abr.m4b"]
    structure = parse_mp4(Path(final_path))
    assert structure.faststart and structure.has_chpl and structure.chapter_track_handlers == ["text"]
    meta = AppMetadata.load_from_mp4(final_path)
    assert meta is not None
    with av.open(final_path) as container:
        assert container.duration is not None
        chapters = [(c["metadata"]["title"], float(c["start"] * c["time_base"])) for c in container.chapters()]
        assert container.metadata["title"] == "Buch ü"
        duration = container.duration / av.time_base
    starts = [segment.time_start for segment in meta.timed_phrases if isinstance(segment, TimedPhrase)]
    assert [title for title, _ in chapters] == ["Part one", "Part two"]
    assert chapters[1][1] == pytest.approx(starts[3], abs=1e-3)
    assert duration == pytest.approx(max(segment.time_end for segment in meta.timed_phrases
                                         if isinstance(segment, TimedPhrase)), abs=1e-3)
    assert_no_handles(final_path, stem + " [concat].m4b", staging_path_for(stem + " [concat].m4b"))
