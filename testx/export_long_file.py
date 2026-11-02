"""
Long-file check for the export writers: stream a synthetic multi-hour signal
through the FLAC and AAC/M4B writers the way concat does (one rendered segment
at a time), then verify the results.

Checks, per format:
- bounded memory: peak RSS growth while writing stays under a limit;
- complete finalization: the file exists only after close, the staging file is
  gone, FLAC frame count is exact, the M4B has its chapters and is fast-start;
- duration: FLAC exact; M4B container duration within 1 ms, and decoding the
  last seconds ends within one AAC frame of the written length;
- seekability: seeking to several points (including near the end) lands on the
  right segment (each segment has its own tone frequency);
- M4B only: the Mutagen ABR tag (written after close, as in the app) keeps the
  chapters and fast-start placement.

Usage:
    ./venv-base/bin/python -m testx.export_long_file [--hours 3] [--dir DIR] [--keep]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import av
import numpy as np
import psutil
import soundfile

from testx.pyav_m4b_chapters import Checker, parse_mp4
from tts_audiobook_tool.constants import APP_META_MP4_MEAN, APP_META_MP4_TAG, APP_SAMPLE_RATE
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.audio_writers import AudioWriter, Chapter, open_writer, staging_path_for

SR = APP_SAMPLE_RATE
SEGMENT_S = 9.5  # a typical-ish rendered segment plus pause
CHAPTER_EVERY_S = 30 * 60
RSS_LIMIT_MB = 200


def segment_frequency(index: int) -> float:
    return 200.0 + (index % 40) * 25.0


def make_segment(index: int) -> np.ndarray:
    n = round(SEGMENT_S * SR)
    t = np.arange(n) / SR
    tone = 0.25 * np.sin(2 * np.pi * segment_frequency(index) * t)
    return tone.astype(np.float32)


def dominant_frequency(samples: np.ndarray) -> float:
    window = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(window))
    return float(np.argmax(spectrum)) * SR / len(samples)


def stream(writer: AudioWriter, segments: int, label: str) -> tuple[float, float]:
    """Writes every segment; returns (peak RSS growth MB, seconds)."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    started = time.time()
    for index in range(segments):
        writer.write(make_segment(index))
        if index % 50 == 0:
            peak = max(peak, process.memory_info().rss)
            done = (index + 1) / segments
            print(f"\r{label}: {done:6.1%}", end="", flush=True)
    peak = max(peak, process.memory_info().rss)
    print(f"\r{label}: writing done ({time.time() - started:.0f}s), finalizing...", flush=True)
    writer.close()
    peak = max(peak, process.memory_info().rss)
    return (peak - baseline) / 2**20, time.time() - started


def seek_points(total_s: float) -> list[float]:
    return [SEGMENT_S * 0.5, total_s * 0.37, total_s * 0.81, total_s - SEGMENT_S * 0.5]


def check_flac(checker: Checker, path: Path, segments: int, total_samples: int) -> None:
    info = soundfile.info(str(path))
    checker.check("flac: frame count exact", info.frames == total_samples, f"{info.frames} vs {total_samples}")
    with soundfile.SoundFile(str(path)) as file:
        for seconds in seek_points(total_samples / SR):
            file.seek(round(seconds * SR))
            samples = file.read(SR // 5, dtype="float32")
            expected = segment_frequency(int(seconds // SEGMENT_S))
            found = dominant_frequency(samples)
            checker.check(f"flac: seek to {seconds:.1f}s", abs(found - expected) <= 5, f"{found:.1f} vs {expected:.1f} Hz")


def check_m4b(checker: Checker, path: Path, chapters: list[Chapter], total_samples: int) -> None:
    total_s = total_samples / SR
    structure = parse_mp4(path)
    checker.check(f"{path.name}: moov before mdat", structure.faststart, f"{structure.top_level}")
    checker.check(f"{path.name}: chapter track and chpl",
                  structure.has_chpl and structure.chapter_track_handlers == ["text"])
    with av.open(str(path)) as container:
        assert container.duration is not None
        duration = container.duration / av.time_base
        checker.check(f"{path.name}: container duration", abs(duration - total_s) < 1e-3, f"{duration:.4f} vs {total_s:.4f}")
        read = [(c["metadata"]["title"], round(float(c["start"] * (c["time_base"] or 1)), 3)) for c in container.chapters()]
        checker.check(f"{path.name}: chapters", read == [(c.title, round(c.start, 3)) for c in chapters],
                      f"{len(read)} chapters")

        audio = container.streams.audio[0]
        time_base = audio.time_base
        assert time_base is not None
        for seconds in seek_points(total_s):
            container.seek(int(seconds / time_base), stream=audio)
            collected: list[np.ndarray] = []
            first_time = None
            for frame in container.decode(audio):
                assert frame.pts is not None
                frame_time = float(frame.pts * time_base)
                if first_time is None:
                    first_time = frame_time
                if frame_time + frame.samples / SR < seconds:
                    continue
                collected.append(frame.to_ndarray()[0])
                if sum(len(c) for c in collected) >= SR // 5:
                    break
            samples = np.concatenate(collected)[:SR // 5]
            expected = segment_frequency(int(seconds // SEGMENT_S))
            found = dominant_frequency(samples)
            ok = first_time is not None and first_time <= seconds and abs(found - expected) <= 10
            checker.check(f"{path.name}: seek to {seconds:.1f}s", ok,
                          f"landed {first_time}, {found:.1f} vs {expected:.1f} Hz")

        # Decode the tail to check the total decoded length
        start = max(0.0, total_s - 20)
        container.seek(int(start / time_base), stream=audio)
        last_end = 0
        for frame in container.decode(audio):
            assert frame.pts is not None
            last_end = frame.pts + frame.samples  # pts in 1/SR units for AAC in MP4
        overshoot = last_end - total_samples
        checker.check(f"{path.name}: decoded end within one AAC frame", 0 <= overshoot < 1024, f"{overshoot:+d} samples")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--dir", type=Path, help="work directory (default: a temp dir)")
    parser.add_argument("--keep", action="store_true", help="keep the output files")
    args = parser.parse_args()

    segments = int(args.hours * 3600 / SEGMENT_S)
    total_samples = segments * round(SEGMENT_S * SR)
    total_s = total_samples / SR
    chapters = [
        Chapter(start, min(start + CHAPTER_EVERY_S, total_s), f"Kapitel {i + 1} – ü")
        for i, start in enumerate(np.arange(0, total_s, CHAPTER_EVERY_S).tolist())
    ]
    print(f"PyAV {av.__version__} (FFmpeg {av.ffmpeg_version_info}), {segments} segments, {total_s / 3600:.2f} h")

    checker = Checker()
    work = Path(tempfile.mkdtemp(prefix="long export ü ", dir=args.dir))
    try:
        flac = work / "long [concat].flac"
        writer = open_writer(str(flac), SR, 1)
        checker.check("flac: nothing at destination while writing", not flac.exists())
        growth, seconds = stream(writer, segments, "flac")
        checker.check("flac: peak RSS growth", growth < RSS_LIMIT_MB, f"{growth:.1f} MB, {seconds:.0f}s, "
                      f"{flac.stat().st_size / 2**20:.0f} MB file")
        checker.check("flac: staging removed", not os.path.exists(staging_path_for(str(flac))))
        check_flac(checker, flac, segments, total_samples)
        if not args.keep:
            flac.unlink()

        m4b = work / "long [concat].m4b"
        writer = open_writer(str(m4b), SR, 2, aac_bitrate="96k", metadata={"title": "Long", "artist": "Test"})
        original_write = writer.write
        writer.write = lambda data: original_write(np.stack([data, data], axis=1))  # type: ignore[method-assign]
        checker.check("m4b: nothing at destination while writing", not m4b.exists())
        writer.set_chapters(chapters)
        growth, seconds = stream(writer, segments, "m4b")
        checker.check("m4b: peak RSS growth", growth < RSS_LIMIT_MB, f"{growth:.1f} MB, {seconds:.0f}s, "
                      f"{m4b.stat().st_size / 2**20:.0f} MB file")
        checker.check("m4b: staging removed", not os.path.exists(staging_path_for(str(m4b))))
        check_m4b(checker, m4b, chapters, total_samples)

        final = work / "long.abr.m4b"
        error = AudioMetaUtil.set_mp4_metadata_tag(str(m4b), APP_META_MP4_MEAN, APP_META_MP4_TAG, '{"x": 1}', str(final))
        checker.check("m4b: ABR tag written", error == "", error)
        check_m4b(checker, final, chapters, total_samples)
    finally:
        if args.keep:
            print(f"Kept files in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print("RESULT", "PASS" if checker.failures == 0 else f"FAIL ({checker.failures})")
    return 0 if checker.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
