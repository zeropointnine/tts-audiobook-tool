"""
Voice-clone generation check for the local TTS model of the venv it runs in,
meant to be run inside the no-system-FFmpeg runner:

    python3 -m testx.no_system_ffmpeg_run --venv venv-cb -- -m testx.no_ffmpeg_model_clone

What it does (no ffmpeg executable needed anywhere):

1. Makes non-WAV reference sources with PyAV: an M4A and an MP3 cut from the
   committed demo audiobook `browser_player/waves-chatterbox.abr.m4a` (its
   first two phrases, ~10 s of speech; the transcript comes from its ABR
   timing metadata). `--voice PATH --transcript TEXT` uses another source.
2. In a temporary project, imports the M4A and the MP3 the way the app's voice
   menu does (`SoundFileUtil.load` -> voice-clone post-processing ->
   `ProjectVoiceUtil.set_voice_and_save`, which saves a canonical FLAC), and
   adds a third, legacy voice that references a raw MP3 in the project root
   (as pre-FLAC versions of the app stored samples), which the model gets as
   a canonical copy made at generation time.
3. Loads the model the app's way (`Tts`, project defaults) and generates one
   prompt per voice; writes the results to the output dir.
4. Checks: each generation returns non-silent audio of plausible length; no
   ffmpeg/ffprobe on PATH and no FFmpeg library mapped from outside the venv
   (only enforced inside the runner, i.e. with NO_SYSTEM_FFMPEG_RUNNER=1;
   outside it they are reported). Also reports, without failing, which
   FFmpeg-capable modules (TorchCodec, pydub, audioread, gradio) got imported.

It does not judge voice similarity: listen to the outputs for that.

Options: --out DIR (default: a temp dir, printed), --cpu, --prompt TEXT,
--voices m4a,mp3,legacy (subset), --set model_id.setting=value (repeatable,
JSON value, eg --set qwen3tts_local.target='"Qwen/..."').
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SOURCE = REPO_ROOT / "browser_player" / "waves-chatterbox.abr.m4a"
DEMO_PHRASES = 2
FFMPEG_CAPABLE_MODULES = ("torchcodec", "torchcodec.decoders", "pydub", "audioread", "gradio")
PROMPT = "The lantern burned steadily beside the window while the rain moved quietly across the garden."


def demo_reference() -> tuple[str, float, str]:
    """(source path, end seconds, transcript) of the demo audiobook's first phrases."""
    from tts_audiobook_tool.app_types.app_metadata import AppMetadata

    from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase

    meta = AppMetadata.load_from_file(str(DEMO_SOURCE))
    phrases = [item for item in (meta.timed_phrases if meta else []) if isinstance(item, TimedPhrase)]
    if len(phrases) < DEMO_PHRASES:
        raise SystemExit(f"No ABR timing metadata in {DEMO_SOURCE}")
    phrases = phrases[:DEMO_PHRASES]
    return str(DEMO_SOURCE), phrases[-1].time_end, " ".join(p.text.strip() for p in phrases)


def write_sources(source: str, end_s: float | None, work_dir: Path) -> dict[str, Path]:
    """The reference cut as M4A (app AAC writer) and MP3 (PyAV libmp3lame)."""
    import av

    from tts_audiobook_tool.sound.audio_decoder import read_audio
    from tts_audiobook_tool.sound.audio_writers import open_writer

    data, sr = read_audio(source, sample_rate=44100, channels=1, start=0.0, duration=end_s)
    m4a = work_dir / "reference.m4a"
    with open_writer(str(m4a), sr, 1) as writer:
        writer.write(data)

    mp3 = work_dir / "reference.mp3"
    with av.open(str(mp3), "w", format="mp3") as container:
        stream: Any = container.add_stream("libmp3lame", rate=sr, layout="mono")
        for start in range(0, len(data), 1152):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(data[start:start + 1152], dtype=np.float32).reshape(1, -1),
                format="fltp", layout="mono")
            frame.sample_rate = sr
            frame.pts = start
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return {"m4a": m4a, "mp3": mp3}


def import_voice(project, model_type, path: Path, transcript: str) -> str:
    """The voice menu's import, minus the UI: load, post-process, save as FLAC."""
    from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline

    sound = SoundFileUtil.load(str(path))
    if isinstance(sound, str):
        raise SystemExit(f"Import of {path} failed: {sound}")
    sound = SoundPipeline.apply_voice_clone_post_processing(sound)
    err = ProjectVoiceUtil.set_voice_and_save(project, sound, path.stem + "_" + path.suffix[1:], transcript,
                                              model_type, append=True)
    if err:
        raise SystemExit(f"Import of {path} failed: {err}")
    return project.voice_references[-1].file_name


def ffmpeg_environment_problems() -> list[str]:
    problems = []
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name):
            problems.append(f"{name} on PATH: {shutil.which(name)}")
    try:
        from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import ffmpeg_lib_report
        report = ffmpeg_lib_report()
    except NotImplementedError:
        return problems
    prefix = os.path.realpath(sys.prefix) + os.sep
    print(report.describe())
    for lib in report.libs:
        if not os.path.realpath(lib.path).startswith(prefix):
            problems.append(f"FFmpeg library from outside the venv: {lib.path}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", help="reference source (any format the app imports)")
    parser.add_argument("--transcript", default="", help="transcript of --voice")
    parser.add_argument("--voices", default="m4a,mp3,legacy")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="MODEL.SETTING=JSON")
    args = parser.parse_args()

    in_runner = os.environ.get("NO_SYSTEM_FFMPEG_RUNNER") == "1"
    print(f"no-system-FFmpeg runner: {'yes' if in_runner else 'NO (checks are informational)'}")

    from tts_audiobook_tool.l import L
    L.init("no-ffmpeg-model-clone")
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
    from tts_audiobook_tool.app_types.voice import Voice
    from tts_audiobook_tool.tts import Tts

    model_type, matches = Tts.init_local_model_type()
    if model_type.id == "none" or matches != 1:
        print(f"Needs exactly one local model library in this venv (found {matches})")
        return 1
    print(f"Model: {model_type.value.ui['proper_name']} ({model_type.id})")

    if args.voice:
        source, end_s, transcript = args.voice, None, args.transcript
    else:
        source, end_s, transcript = demo_reference()
    print(f"Reference: {source} (to {end_s if end_s else 'end'} s); transcript: {transcript!r}")

    work_dir = Path(tempfile.mkdtemp(prefix="no-ffmpeg-clone-"))
    out_dir = Path(args.out) if args.out else work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir = work_dir / "project ü"
    project_dir.mkdir()
    sources = write_sources(source, end_s, work_dir)

    project = Project.model_validate({"dir_path": str(project_dir)})
    project.set_tts_selection(model_type.id)
    for item in args.set:
        key, _, value = item.partition("=")
        model_id, _, name = key.partition(".")
        project.set_model_setting(model_id, name, json.loads(value))

    wanted = [v.strip() for v in args.voices.split(",") if v.strip()]
    labels = []
    for label in ("m4a", "mp3"):
        if label in wanted:
            import_voice(project, model_type, sources[label], transcript)
            labels.append(label)
    if "legacy" in wanted:
        # A pre-FLAC project: the raw file in the project root, referenced as is
        shutil.copy(sources["mp3"], project_dir / "legacy.mp3")
        voices = [v for v in project.voice_references if v.has_sample]
        project.voice_references = voices + [Voice(file_name="legacy.mp3", transcript=transcript)]
        labels.append("legacy")
    project.save()

    for index, (label, voice) in enumerate(zip(labels, project.voice_references)):
        model_path = ProjectVoiceUtil.resolve_model_voice_file_path(project, voice.file_name)
        print(f"voice {index} [{label}]: stored {voice.file_name!r} -> model gets {model_path}")

    Tts.set_type(model_type)
    Tts.set_model_params_using_project(project)
    Tts.set_force_cpu(args.cpu)
    print("Loading model...", flush=True)
    started = time.perf_counter()
    instance = Tts.get_instance()
    print(f"Loaded in {time.perf_counter() - started:.1f}s")
    issues = Tts.get_model_support(project).get_blocking_issues(project, instance)
    if issues:
        print(f"Blocking issues: {issues}")
        return 1

    import soundfile

    failures: list[str] = []
    for index, label in enumerate(labels):
        started = time.perf_counter()
        result = Tts.generate_using_project(project, [args.prompt], voice_selection_index=index)
        elapsed = time.perf_counter() - started
        if isinstance(result, str):
            failures.append(f"[{label}] generation failed: {result}")
            print(f"[{label}] FAIL: {result}")
            continue
        sound = result[0]
        duration = len(sound.data) / sound.sr
        peak = float(np.max(np.abs(sound.data))) if len(sound.data) else 0.0
        out_path = out_dir / f"{model_type.id}_{label}.flac"
        soundfile.write(out_path, np.clip(sound.data, -1, 1), sound.sr, subtype="PCM_16")
        ok = 1.0 < duration < 30.0 and peak > 0.01
        print(f"[{label}] {'ok' if ok else 'FAIL'}: {duration:.1f}s @ {sound.sr} Hz, peak {peak:.2f}, "
              f"{elapsed:.1f}s -> {out_path}")
        if not ok:
            failures.append(f"[{label}] implausible output: {duration:.1f}s, peak {peak:.3f}")

    # Informational: an import alone launches nothing (eg librosa imports
    # audioread); a hard requirement would have failed generation in the runner.
    imported = [name for name in FFMPEG_CAPABLE_MODULES if name in sys.modules]
    print(f"FFmpeg-capable modules imported (informational): {', '.join(imported) or 'none'}")

    problems = ffmpeg_environment_problems()
    if in_runner:
        failures += problems
    else:
        for problem in problems:
            print(f"note: {problem}")

    for failure in failures:
        print(f"FAIL: {failure}")
    print(f"model clone check ({model_type.id}): {'FAIL' if failures else 'PASS'}")
    print(f"outputs: {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
