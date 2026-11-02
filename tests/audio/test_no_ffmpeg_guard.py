"""The no-FFmpeg guard blocks every route to an FFmpeg executable and nothing else."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from no_ffmpeg_guard import FfmpegExecutableLaunched, NoFfmpegGuard, ffmpeg_offender, no_ffmpeg_exe


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (["ffmpeg", "-version"], {}),
        (["/usr/bin/ffprobe", "-i", "x.m4a"], {}),
        ([r"C:\tools\ffmpeg\bin\ffmpeg.exe", "-i", "x"], {}),
        (["FFMPEG.EXE"], {}),
        # Bundled/versioned binaries such as imageio-ffmpeg's.
        (["/opt/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2", "-i", "x"], {}),
        ("ffmpeg -i in.wav out.flac", {"shell": True}),
        ("cd /tmp && ffprobe in.wav | head", {"shell": True}),
        (["sh", "-c", "exec ffmpeg -version"], {}),
        (["python3"], {"executable": "/usr/local/bin/ffmpeg"}),
        (Path("/usr/bin/ffmpeg"), {}),
        (b"ffmpeg", {}),
    ],
)
def test_ffmpeg_launch_forms_are_recognized(args, kwargs) -> None:
    assert ffmpeg_offender(args, kwargs.get("executable"), kwargs.get("shell", False)) is not None


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ([sys.executable, "-c", "pass"], {}),
        (["xdg-open", "/books/ffmpeg/out.m4b"], {}),
        (["python", "ffmpeg_util.py"], {}),
        ("python ffmpeg_util.py --ffmpeg-path x", {"shell": True}),
        ([], {}),
    ],
)
def test_other_launches_are_not_flagged(args, kwargs) -> None:
    # Only the program being run matters, not FFmpeg-ish words in its arguments.
    assert ffmpeg_offender(args, kwargs.get("executable"), kwargs.get("shell", False)) is None


@pytest.mark.parametrize(
    "launch",
    [
        lambda: subprocess.run(["ffmpeg", "-version"], capture_output=True),
        lambda: subprocess.check_output("ffprobe -version", shell=True),
        lambda: os.popen("ffmpeg -version").read(),
        lambda: os.system("ffmpeg -version"),
        pytest.param(
            lambda: os.posix_spawnp("ffmpeg", ["ffmpeg", "-version"], os.environ),
            marks=pytest.mark.skipif(not hasattr(os, "posix_spawnp"), reason="POSIX only"),
        ),
    ],
)
def test_guard_blocks_launches_before_they_start(launch) -> None:
    with NoFfmpegGuard() as guard:
        with pytest.raises(FfmpegExecutableLaunched):
            launch()
    assert len(guard.violations) == 1


def test_guard_blocks_pydub_decoding(tmp_path: Path) -> None:
    # pydub decodes everything but WAV by shelling out to ffmpeg/avconv.
    pydub = pytest.importorskip("pydub")
    source = tmp_path / "in.mp3"
    source.write_bytes(b"\0" * 64)
    with NoFfmpegGuard() as guard:
        with pytest.raises(FfmpegExecutableLaunched):
            pydub.AudioSegment.from_file(str(source), format="mp3")
    assert guard.violations


def test_swallowed_launch_still_fails_at_block_exit() -> None:
    # Production code often turns errors into returned message strings; the
    # recorded violation must still fail the guarded block.
    with pytest.raises(FfmpegExecutableLaunched, match="ffmpeg"):
        with no_ffmpeg_exe():
            try:
                subprocess.run(["ffmpeg", "-version"])
            except Exception:
                pass


def test_guard_allows_other_processes_and_restores_patches() -> None:
    original_init = subprocess.Popen.__init__
    original_system = os.system
    with no_ffmpeg_exe():
        result = subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "ok"
        assert subprocess.Popen.__init__ is not original_init
    assert subprocess.Popen.__init__ is original_init
    assert os.system is original_system


def test_nested_guards_attribute_the_launch_to_the_inner_guard() -> None:
    # The suite-wide guard (--no-ffmpeg-exe) wraps tests that install their own.
    with NoFfmpegGuard() as outer:
        with NoFfmpegGuard() as inner:
            with pytest.raises(FfmpegExecutableLaunched):
                subprocess.run(["ffmpeg", "-version"])
    assert len(inner.violations) == 1
    assert outer.violations == []


@pytest.mark.no_ffmpeg_exe
def test_marker_installs_guard(no_ffmpeg_exe: NoFfmpegGuard) -> None:
    with pytest.raises(FfmpegExecutableLaunched):
        subprocess.run(["ffprobe", "-version"])
    # The fixture fails this test at teardown unless the violation is cleared.
    no_ffmpeg_exe.violations.clear()
