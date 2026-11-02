"""
Milestone A smoke test: the app's own audio I/O works with no system FFmpeg.

Runs the real app code paths for export, enhancement, preview, and both Tools
commands on synthetic audio (no model weights, no `ffmpeg` to make inputs),
and prints one PASS/FAIL line per check:

- FLAC export and chaptered M4B export (`ConcatUtil.make_file`): ABR tag, M4B
  chapters, fast-start
- Enhancement: windowed decode of MP3 and M4A input; staged output over an
  existing destination (MP3 re-encoded to AAC, M4A copied and tagged)
- Preview range decode (`SoundFileUtil.load_range`), M4A and MP3
- Tools: concatenation of mixed WAV/FLAC/MP3/M4A to FLAC and to M4A, and the
  ABR FLAC-to-AAC conversion
- App startup checks (`Start.start()` up to launching the UI), in a child process

Every check runs twice: in a plain folder and in one with non-ASCII characters.
After each check, the script verifies that:

- nothing tried to launch an FFmpeg executable (the no-FFmpeg guard from
  tests/no_ffmpeg_guard.py is active for the whole run);
- this process holds no open handle to a file in the work folder, and every
  output can be deleted (on Windows, the deletions themselves are the proof);
- every loaded FFmpeg library comes from PyAV's bundled directory (Linux and
  Windows; psutil cannot list loaded libraries on macOS).

`--ffmpeg-on-path DIR` prepends DIR (an FFmpeg *shared* build: its bin folder,
with the DLLs on Windows) to PATH, and on Linux/macOS also DIR and DIR/../lib
to the library search path, then reruns the script in a fresh process with
that environment, to prove the app does not use it.

Usage (from the repository root, in any app venv):

    python -m testx.no_ffmpeg_smoke
    python -m testx.no_ffmpeg_smoke --ffmpeg-on-path C:\\ffmpeg-shared\\bin
    python -m testx.no_ffmpeg_smoke --keep --dir /tmp/smoke

Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, cast

REPO = Path(__file__).resolve().parents[1]
CHILD_MARKER = "TTS_AUDIOBOOK_TOOL_SMOKE_CHILD"
NON_ASCII_DIR = "Hörbuch 書 ü"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ffmpeg-on-path", metavar="DIR", help="prepend an FFmpeg shared build to PATH first")
    parser.add_argument("--dir", help="parent folder for the work folder (default: system temp)")
    parser.add_argument("--keep", action="store_true", help="keep the work folder")
    parser.add_argument("--skip-startup", action="store_true", help="skip the app startup check")
    return parser.parse_args()


# --- --ffmpeg-on-path: rerun in a fresh process before importing anything ---

def rerun_with_ffmpeg_on_path(directory: str) -> int:
    folder = Path(directory).resolve()
    if not folder.is_dir():
        print(f"FAIL  --ffmpeg-on-path: not a folder: {folder}")
        return 2
    env = dict(os.environ)
    env[CHILD_MARKER] = str(folder)
    env["PATH"] = str(folder) + os.pathsep + env.get("PATH", "")
    library_var = {"linux": "LD_LIBRARY_PATH", "darwin": "DYLD_LIBRARY_PATH"}.get(sys.platform)
    if library_var:
        dirs = [str(folder)] + ([str(folder.parent / "lib")] if (folder.parent / "lib").is_dir() else [])
        env[library_var] = os.pathsep.join(dirs + ([env[library_var]] if env.get(library_var) else []))
        print(f"INFO  {library_var}={env[library_var]}")
    print(f"INFO  PATH starts with {folder}; rerunning in a fresh process")
    args = [arg for arg in sys.argv[1:]]
    index = args.index("--ffmpeg-on-path") if "--ffmpeg-on-path" in args else -1
    if index >= 0:
        del args[index:index + 2]
    else:
        args = [arg for arg in args if not arg.startswith("--ffmpeg-on-path=")]
    sys.stdout.flush()
    return subprocess.call([sys.executable, "-m", "testx.no_ffmpeg_smoke", *args], cwd=str(REPO), env=env)


def ffmpeg_shared_libs_in(folder: Path) -> list[str]:
    from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import ffmpeg_lib_family

    found = []
    for candidate in (folder, folder.parent / "lib"):
        if candidate.is_dir():
            found += [p.name for p in candidate.iterdir() if ffmpeg_lib_family(p.name)]
    return sorted(found)


# --- Synthetic audio (PyAV / soundfile only) ---

def tone_steps(sample_rate: int, seconds: float, channels: int = 1, step_s: float = 10.0):
    """A sine whose frequency is 300 Hz + 100 Hz per `step_s` band, so positions are recognizable."""
    import numpy as np

    t = np.arange(round(sample_rate * seconds)) / sample_rate
    frequency = 300 + 100 * np.floor(t / step_s)
    mono = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return mono if channels == 1 else np.stack([mono] * channels, axis=1)


def write_encoded(path: Path, signal, sample_rate: int, codec: str) -> None:
    """MP3 ("libmp3lame") or AAC ("aac", .m4a) via PyAV; (n,) mono or (n, channels)."""
    import av
    import numpy as np

    planar = signal.reshape(1, -1) if signal.ndim == 1 else signal.T
    layout = "mono" if planar.shape[0] == 1 else "stereo"
    container_format = {"aac": "ipod", "libmp3lame": "mp3"}[codec]
    with av.open(str(path), "w", format=container_format) as container:
        stream = cast(av.AudioStream, container.add_stream(codec, rate=sample_rate, layout=layout))
        for start in range(0, planar.shape[1], 1152):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(planar[:, start:start + 1152], dtype=np.float32), format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = start
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def dominant_frequency(samples, sample_rate: int) -> float:
    import numpy as np

    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return float(frequencies[np.abs(np.fft.rfft(samples)).argmax()])


def media_duration(path: Path) -> tuple[float, int, int]:
    """(seconds, sample rate, channels) of the default audio stream, by decoding it."""
    import av

    with av.open(str(path)) as container:
        stream = cast("av.AudioStream | None", container.streams.best("audio"))
        assert stream is not None
        samples = sum(frame.samples for frame in container.decode(stream) if isinstance(frame, av.AudioFrame))
        return samples / stream.rate, stream.rate, stream.channels


# --- Checks ---

class CheckFailed(Exception):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


class Smoke:

    def __init__(self, work: Path, guard) -> None:
        self.work = work
        self.guard = guard
        self.failures = 0
        self.passes = 0

    # Each check returns a short detail string; it gets its own folder

    def run(self, label: str, check: Callable[[Path], str], folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        violations_before = len(self.guard.violations) if self.guard else 0
        problems: list[str] = []
        detail = ""
        started = time.perf_counter()
        app_output = io.StringIO()  # the app's progress output, shown only on failure
        try:
            with contextlib.redirect_stdout(app_output):
                detail = check(folder)
        except CheckFailed as e:
            problems.append(str(e))
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")
            app_output.write(traceback.format_exc())
        if self.guard and len(self.guard.violations) > violations_before:
            problems.append("FFmpeg executable launch attempted:\n" + "\n".join(self.guard.violations[violations_before:]))
        problems += self.handle_problems(folder)
        problems += self.library_problems()
        try:
            shutil.rmtree(folder)
        except OSError as e:
            problems.append(f"couldn't delete outputs (file still locked?): {e}")
        elapsed = time.perf_counter() - started
        if problems:
            self.failures += 1
            print(app_output.getvalue().rstrip())
            print(f"FAIL  {label}  ({'; '.join(problems)})", flush=True)
        else:
            self.passes += 1
            print(f"PASS  {label}  ({detail}; {elapsed:.1f}s)" if detail else f"PASS  {label}", flush=True)

    def handle_problems(self, folder: Path) -> list[str]:
        import psutil

        root = os.path.normcase(os.path.realpath(folder))
        try:
            open_files = psutil.Process().open_files()
        except Exception as e:  # pragma: no cover - platform specific
            return [f"couldn't list open files: {e}"]
        held = [f.path for f in open_files if os.path.normcase(os.path.realpath(f.path)).startswith(root)]
        return [f"open handle(s) left: {held}"] if held else []

    def library_problems(self) -> list[str]:
        if sys.platform == "darwin":
            return []
        from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import ffmpeg_lib_report

        report = ffmpeg_lib_report()
        if not report.only_bundled:
            return ["FFmpeg libraries loaded from outside PyAV's bundle: "
                    + ", ".join(lib.path for lib in report.outside_bundled)]
        return []


# Export (ConcatUtil.make_file, real writers, real segment loading)

SEGMENT_SECONDS = [0.7, 1.1, 0.0, 0.5, 0.9]  # 0.0: a missing segment
TITLES = ["Part one", "Line two.", "Missing.", "Part two", "Line five."]


def export(folder: Path, aac: bool) -> Path:
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import patch

    import soundfile

    from tts_audiobook_tool.app_types import Book, BookSection, ExportType
    from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
    from tts_audiobook_tool.concat_util import ConcatUtil
    from tts_audiobook_tool.constants import APP_SAMPLE_RATE
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
    from tts_audiobook_tool.state import State

    def group(text: str) -> PhraseGroup:
        return PhraseGroup([Phrase(text, Reason.SENTENCE)])

    project = Project.model_validate({"book": Book(sections=[
        BookSection(title="Part one", phrase_groups=[group(t) for t in TITLES[:3]]),
        BookSection(title="Part two ü", phrase_groups=[group(t) for t in TITLES[3:]]),
    ])})
    project.export_type = ExportType.AAC if aac else ExportType.FLAC
    pairs = []
    for i, (title, seconds) in enumerate(zip(TITLES, SEGMENT_SECONDS)):
        path = ""
        if seconds:
            path = str(folder / f"segment {i} ü.flac")
            soundfile.write(path, tone_steps(APP_SAMPLE_RATE, seconds, step_s=0.25), APP_SAMPLE_RATE, subtype="PCM_16")
        pairs.append((Phrase(title, Reason.SENTENCE), path))
    state = SimpleNamespace(project=project, prefs=SimpleNamespace(aac_bitrate="96k", save_debug_files=False))
    stem = str(folder / "Buch ü")
    with patch.object(ProjectTextIOUtil, "load_raw_text", return_value="\n".join(TITLES)), \
            patch.object(ConcatUtil, "make_phrases_and_paths", return_value=pairs), \
            patch("tts_audiobook_tool.concat_util.DEV", False):
        final_path, error = ConcatUtil.make_file(cast(State, state), 0, len(TITLES) - 1, [], stem, use_upsampler=False)
    expect(error == "", f"export error: {error}")
    others = sorted(p.name for p in folder.iterdir() if not p.name.startswith("segment"))
    expect(others == [Path(final_path).name], f"unexpected files: {others}")
    return Path(final_path)


def check_export_flac(folder: Path) -> str:
    import soundfile

    from tts_audiobook_tool.app_types.app_metadata import AppMetadata

    path = export(folder, aac=False)
    expect(path.name == "Buch ü.abr.flac", f"output name {path.name}")
    info = soundfile.info(str(path))
    meta = AppMetadata.load_from_flac(str(path))
    expect(meta is not None and len(meta.timed_phrases) == len(TITLES), "ABR tag missing")
    return f"{info.duration:.2f}s, ABR tag read back"


def check_export_m4b(folder: Path) -> str:
    import av

    from testx.pyav_m4b_chapters import parse_mp4
    from tts_audiobook_tool.app_types.app_metadata import AppMetadata

    path = export(folder, aac=True)
    structure = parse_mp4(path)
    expect(structure.faststart, "not fast-start (moov after mdat)")
    expect(structure.has_chpl and structure.chapter_track_handlers == ["text"], "chapter atoms missing")
    with av.open(str(path)) as container:
        titles = [chapter["metadata"].get("title") for chapter in container.chapters()]
    expect(titles == ["Part one", "Part two ü"], f"chapters {titles}")
    meta = AppMetadata.load_from_mp4(str(path))
    expect(meta is not None and len(meta.timed_phrases) == len(TITLES), "ABR tag missing")
    return "2 chapters, fast-start, ABR tag read back"


def check_flac_to_aac_tool(folder: Path) -> str:
    from tts_audiobook_tool.app_types.app_metadata import AppMetadata
    from tts_audiobook_tool.sound.transcode_util import TranscodeUtil

    flac = export(folder, aac=False)
    m4a, error = TranscodeUtil.transcode_abr_flac_to_aac(str(flac))
    expect(error == "", f"transcode error: {error}")
    expect(AppMetadata.load_from_mp4(m4a) is not None, "ABR tag not carried over")
    seconds, _rate, _channels = media_duration(Path(m4a))
    return f"{Path(m4a).name}, {seconds:.2f}s, ABR tag read back"


# Enhancement

def enhance_source(folder: Path, suffix: str) -> Path:
    path = folder / f"Roman ü{suffix}"
    signal = tone_steps(44100, 40.0, channels=2)
    write_encoded(path, signal, 44100, "libmp3lame" if suffix == ".mp3" else "aac")
    return path


def check_enhance_decode(suffix: str) -> Callable[[Path], str]:
    def check(folder: Path) -> str:
        import numpy as np

        from tts_audiobook_tool.enhance.enhance_alignment import _stream_audio_with_overlap

        source = enhance_source(folder, suffix)
        windows = list(_stream_audio_with_overlap(str(source)))
        # 40 s in 30 s windows with 5 s overlap: 0-30 s and 25-40 s
        expect(len(windows) == 2 and len(windows[0]) == 30 * 16000 and abs(len(windows[1]) - 15 * 16000) < 1600,
               f"window lengths {[len(w) for w in windows]}")
        expect(all(w.dtype == np.float32 for w in windows), "windows are not float32")
        # Window 2 starts at 25 s: its first 5 s are band 2 (500 Hz)
        frequency = dominant_frequency(windows[1][:4 * 16000], 16000)
        expect(abs(frequency - 500) < 5, f"window 2 starts at the wrong place ({frequency:.0f} Hz)")
        return f"{len(windows)} windows at 16 kHz"
    return check


def check_enhance_output(suffix: str) -> Callable[[Path], str]:
    def check(folder: Path) -> str:
        from types import SimpleNamespace
        from typing import cast

        from tts_audiobook_tool.app_types.app_metadata import AppMetadata
        from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
        from tts_audiobook_tool.enhance import enhance_flow
        from tts_audiobook_tool.enhance.enhance_artifacts import EnhanceArtifacts
        from tts_audiobook_tool.state import State

        source = enhance_source(folder, suffix)
        artifacts = EnhanceArtifacts.from_audio_path(source)
        destination = artifacts.flat_output_path
        destination.write_bytes(b"previous output")
        state = cast(State, SimpleNamespace(prefs=SimpleNamespace(aac_bitrate="64k")))
        metadata = AppMetadata(
            timed_phrases=[TimedPhrase("Hello", 0.0, 1.0)], title="Roman", version=4, bookmark_indices=[],
            raw_text="", has_break_audio=False, project_snapshot={}, sections=[],
        )
        for attempt in range(2):  # the second run replaces the first one's output
            error = enhance_flow._write_staged_output(state, artifacts, destination, metadata)
            expect(error == "", f"attempt {attempt + 1}: {error}")
        expect(AppMetadata.load_from_file(str(destination)) is not None, "ABR tag missing")
        seconds, _rate, _channels = media_duration(destination)
        expect(abs(seconds - 40.0) < 0.1, f"output duration {seconds:.2f}s")
        names = sorted(p.name for p in folder.iterdir())
        expect(names == sorted([source.name, destination.name]), f"leftover files: {names}")
        return f"{'copied' if suffix == '.m4a' else 're-encoded'}, replaced existing output twice"
    return check


def check_preview(suffix: str) -> Callable[[Path], str]:
    def check(folder: Path) -> str:
        from tts_audiobook_tool.app_types import Sound
        from tts_audiobook_tool.constants import APP_SAMPLE_RATE
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

        source = enhance_source(folder, suffix)
        sound = SoundFileUtil.load_range(str(source), 32.0, 34.5)
        expect(isinstance(sound, Sound), f"load_range error: {sound}")
        assert isinstance(sound, Sound)
        expect(sound.sr == APP_SAMPLE_RATE and len(sound.data) == round(2.5 * APP_SAMPLE_RATE),
               f"{len(sound.data)} samples at {sound.sr}")
        frequency = dominant_frequency(sound.data, sound.sr)
        expect(abs(frequency - 600) < 5, f"wrong position ({frequency:.0f} Hz, expected 600)")
        return "32.0-34.5 s, 600 Hz band found"
    return check


# Tools: directory concatenation

def check_tools_concat(folder: Path) -> str:
    import soundfile

    from tts_audiobook_tool.sound.mp3_concat import SoundConcatTranscodeUtil

    soundfile.write(str(folder / "1 a.wav"), tone_steps(22050, 2.0), 22050, subtype="PCM_16")
    soundfile.write(str(folder / "2 b ü.flac"), tone_steps(44100, 2.0, channels=2), 44100, subtype="PCM_16")
    write_encoded(folder / "3 c.mp3", tone_steps(32000, 2.0), 32000, "libmp3lame")
    write_encoded(folder / "4 d.m4a", tone_steps(48000, 2.0, channels=2), 48000, "aac")
    inputs = sorted(p.name for p in folder.iterdir())
    results = []
    for name, is_flac in (("concatenated.flac", True), ("concatenated.m4a", False)):
        error = SoundConcatTranscodeUtil.concatenate_audio_files(inputs, str(folder), name, is_flac)
        expect(error == "", f"{name}: {error}")
        seconds, rate, channels = media_duration(folder / name)
        expect(abs(seconds - 8.0) < 0.15 and rate == 48000 and channels == 2,
               f"{name}: {seconds:.3f}s, {rate} Hz, {channels} ch")
        results.append(f"{name} {seconds:.2f}s")
    return ", ".join(results)


# App startup, in a child process (it imports the TTS layer, which a model venv
# may extend with packages that load system FFmpeg themselves)

STARTUP_CODE = r"""
import sys
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/tests"]
from no_ffmpeg_guard import NoFfmpegGuard
guard = NoFfmpegGuard().__enter__()
from tts_audiobook_tool.start import Start
reached = []
Start.exit_on_unsupported_terminal = lambda self: None
Start.init_logging = lambda self: None
Start.start_app_or_server = lambda self: reached.append(True)
sys.argv = ["tts_audiobook_tool"]
try:
    Start().start()
except SystemExit as e:
    print(f"STARTUP exited with {e.code}")
if guard.violations:
    print("STARTUP FFmpeg launch:", guard.violations)
print("STARTUP_OK" if reached and not guard.violations else "STARTUP_FAILED")
"""


def check_startup() -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-c", STARTUP_CODE, str(REPO)], cwd=str(REPO), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    output = (completed.stdout + completed.stderr).strip()
    ok = completed.returncode == 0 and "STARTUP_OK" in completed.stdout
    return ok, output.splitlines()[-1] if ok else output[-2000:]


# --- Main ---

def main() -> int:
    args = parse_args()
    if args.ffmpeg_on_path and not os.environ.get(CHILD_MARKER):
        return rerun_with_ffmpeg_on_path(args.ffmpeg_on_path)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(errors="backslashreplace")
    sys.path[:0] = [str(REPO), str(REPO / "tests")]

    import importlib

    # tests/no_ffmpeg_guard.py (test-only module, on sys.path from here on)
    guard = importlib.import_module("no_ffmpeg_guard").NoFfmpegGuard().__enter__()  # active for the whole run

    import av
    import soundfile

    print(f"Python {sys.version.split()[0]} on {sys.platform}; PyAV {av.__version__} "
          f"(FFmpeg {av.ffmpeg_version_info}); soundfile {soundfile.__version__}")
    print(f"INFO  ffmpeg on PATH: {shutil.which('ffmpeg')}; ffprobe on PATH: {shutil.which('ffprobe')}")
    failures = 0
    if os.environ.get(CHILD_MARKER):
        folder = Path(os.environ[CHILD_MARKER])
        libs = ffmpeg_shared_libs_in(folder)
        print(f"INFO  --ffmpeg-on-path {folder}: {len(libs)} FFmpeg shared libraries {libs}")
        if not libs:
            print("FAIL  --ffmpeg-on-path folder contains no FFmpeg shared libraries (not a shared build?)")
            failures += 1

    parent = Path(args.dir) if args.dir else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="no_ffmpeg_smoke_", dir=parent))
    smoke = Smoke(work, guard)
    checks: list[tuple[str, Callable[[Path], str]]] = [
        ("export FLAC", check_export_flac),
        ("export M4B with chapters", check_export_m4b),
        ("enhance decode MP3", check_enhance_decode(".mp3")),
        ("enhance decode M4A", check_enhance_decode(".m4a")),
        ("enhance output MP3 -> AAC, replace existing", check_enhance_output(".mp3")),
        ("enhance output M4A copy, replace existing", check_enhance_output(".m4a")),
        ("preview range M4A", check_preview(".m4a")),
        ("preview range MP3", check_preview(".mp3")),
        ("Tools concatenation (mixed formats)", check_tools_concat),
        ("Tools FLAC -> AAC", check_flac_to_aac_tool),
    ]
    try:
        for base in ("plain", NON_ASCII_DIR):
            for index, (label, check) in enumerate(checks):
                smoke.run(f"{label} [{base}]", check, work / base / f"check{index}")
    finally:
        if args.keep:
            print(f"INFO  work folder kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    if sys.platform == "darwin":
        print("SKIP  loaded FFmpeg libraries (psutil memory_maps() is unavailable on macOS)")
    else:
        from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import ffmpeg_lib_report

        report = ffmpeg_lib_report()
        print(report.describe())
        if report.libs and report.only_bundled:
            print("PASS  FFmpeg libraries only from PyAV's bundle")
        else:
            print("FAIL  FFmpeg libraries only from PyAV's bundle")
            failures += 1

    if args.skip_startup:
        print("SKIP  app startup checks")
    else:
        ok, detail = check_startup()
        print(f"{'PASS' if ok else 'FAIL'}  app startup checks  ({detail})")
        failures += 0 if ok else 1

    failures += smoke.failures
    print("RESULT", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
