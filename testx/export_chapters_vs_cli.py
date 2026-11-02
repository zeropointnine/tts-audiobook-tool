"""
Compare a chaptered M4B from the app's export writer against the legacy
FFmpeg CLI export, for the same audio, chapters, and ABR tag.

New path (production since the pip-only migration):
    AacWriter (PyAV) with title/artist and chapters set at close
    -> Mutagen ABR tag (AppMetadata.save_to_mp4's helper).

Legacy path (what the app did before; developer-only baseline, needs `ffmpeg`):
    s16le PCM piped to `ffmpeg` with the app's AAC output arguments
    -> FFMETADATA chapter remux (`-map_chapters`, `-c copy`,
       `-movflags use_metadata_tags`)
    -> the same Mutagen ABR tag.

Reports and checks, per file: `chpl` atom, chapter text track via `tref`/`chap`,
chapter titles/times, title/artist, tag atoms, fast-start placement, decoded
duration, and the ABR tag round trip. The new file must match the baseline on
chapter structure and chapter list, and must be fast-start.

Expected differences: the legacy file is not fast-start, and its tags are
FFmpeg `mdta` keys holding only the copied brand/encoder values (the legacy
remux never actually wrote title/artist); the new file has iTunes-style
`©nam`/`©ART`/`©too` atoms.

Usage:
    ./venv-base/bin/python -m testx.export_chapters_vs_cli [--keep DIR]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import numpy as np

from testx.pyav_m4b_chapters import Checker, parse_mp4, read_back
from tts_audiobook_tool.constants import (
    APP_META_MP4_MEAN, APP_META_MP4_TAG, APP_NAME, make_ffmpeg_arguments_output_aac,
)
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.audio_writers import AacWriter, Chapter

SAMPLE_RATE = 48000
BITRATE = "96k"
TITLE = "Compare Book [1-12]"
# Segment durations (seconds); chapters start at segments 0, 3 and 7
SEGMENTS = [1.25, 0.75, 2.0, 1.5, 0.5, 1.0, 2.25, 1.75, 0.8, 1.2, 0.6, 1.4]
CHAPTER_STARTS = [(0, "Opening"), (3, "Second chapter"), (7, "Third — ü 書")]
ABR_JSON = '{"version": 5, "note": "compare ✓"}'


def make_audio() -> np.ndarray:
    """A tone burst per segment (so boundaries are audible), stereo."""
    rng = np.random.default_rng(1)
    parts = []
    for index, seconds in enumerate(SEGMENTS):
        n = round(seconds * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        tone = 0.3 * np.sin(2 * np.pi * (220 + 40 * index) * t) * np.minimum(1.0, (n - np.arange(n)) / 2400)
        noise = 0.01 * rng.standard_normal(n)
        mono = (tone + noise).astype(np.float32)
        parts.append(np.stack([mono, 0.8 * mono], axis=1))
    return np.concatenate(parts)


def make_chapters() -> list[Chapter]:
    sums = np.concatenate([[0.0], np.cumsum(SEGMENTS)])
    chapters = []
    for i, (segment_index, title) in enumerate(CHAPTER_STARTS):
        end_index = CHAPTER_STARTS[i + 1][0] if i + 1 < len(CHAPTER_STARTS) else len(SEGMENTS)
        chapters.append(Chapter(float(sums[segment_index]), float(sums[end_index]), title))
    return chapters


def write_new(path: Path, audio: np.ndarray) -> None:
    with AacWriter(str(path), SAMPLE_RATE, 2, bitrate=BITRATE,
                   metadata={"title": TITLE, "artist": APP_NAME}) as writer:
        for start in range(0, len(audio), 30000):
            writer.write(audio[start:start + 30000])
        writer.set_chapters(make_chapters())


def write_legacy(path: Path, audio: np.ndarray, work: Path) -> None:
    concat = work / "legacy [concat].m4b"
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", "-",
         *make_ffmpeg_arguments_output_aac(BITRATE), str(concat)],
        input=pcm.tobytes(), check=True,
    )
    meta = f";FFMETADATA1\ntitle={TITLE}\nartist={APP_NAME}\n\n"
    for chapter in make_chapters():
        meta += (f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(chapter.start * 1000)}\n"
                 f"END={int(chapter.end * 1000)}\ntitle={chapter.title}\n\n")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(concat), "-i", "-",
         "-map", "0", "-map_metadata", "0", "-map_chapters", "1", "-c", "copy",
         "-movflags", "use_metadata_tags", str(path)],
        input=meta, text=True, encoding="utf-8", check=True,
    )


def add_abr_tag(src: Path, dest: Path) -> None:
    error = AudioMetaUtil.set_mp4_metadata_tag(str(src), APP_META_MP4_MEAN, APP_META_MP4_TAG, ABR_JSON, str(dest))
    if error:
        raise RuntimeError(error)


def decoded_samples(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(frame.samples for frame in container.decode(audio=0))


def describe(checker: Checker, label: str, path: Path, legacy: bool):
    structure = parse_mp4(path)
    chapters, metadata, _ = read_back(path)
    tag, error = AudioMetaUtil.get_mp4_metadata_tag(str(path), APP_META_MP4_MEAN, APP_META_MP4_TAG)
    samples = decoded_samples(path)
    expected = [(c.title, round(c.start, 3), round(c.end, 3)) for c in make_chapters()]
    print(f"INFO  {label}: top-level {structure.top_level}")
    print(f"INFO  {label}: meta handler {structure.meta_handler!r}, tag atoms {structure.tag_atoms}")
    print(f"INFO  {label}: decoded {samples} samples ({samples / SAMPLE_RATE:.4f}s), "
          f"written {round(sum(SEGMENTS) * SAMPLE_RATE)}")
    checker.check(f"{label}: chpl atom", structure.has_chpl)
    checker.check(f"{label}: chapter text track via tref/chap",
                  len(structure.chapter_refs) == 1 and structure.chapter_track_handlers == ["text"],
                  f"refs={structure.chapter_refs} handlers={structure.track_handlers}")
    checker.check(f"{label}: chapter titles/times", chapters == expected, f"{chapters}")
    if legacy:
        # `-map_metadata 0` takes global tags from the chapterless input, so the
        # FFMETADATA title/artist never made it in; `use_metadata_tags` stores
        # the copied brand/encoder tags as mdta keys instead.
        print(f"INFO  {label}: title/artist {metadata}")
    else:
        checker.check(f"{label}: title/artist", metadata == {"title": TITLE, "artist": APP_NAME}, f"{metadata}")
        checker.check(f"{label}: moov before mdat", structure.faststart, f"top={structure.top_level}")
    checker.check(f"{label}: ABR tag round trip", tag == ABR_JSON and not error, error)
    written = round(sum(SEGMENTS) * SAMPLE_RATE)
    checker.check(f"{label}: decoded duration within one AAC frame",
                  written <= samples < written + 1024, f"{samples - written:+d} samples")
    return structure, chapters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", type=Path, help="write the files into this directory and keep them")
    args = parser.parse_args()

    print(f"PyAV {av.__version__}, FFmpeg {av.ffmpeg_version_info}")
    checker = Checker()
    audio = make_audio()
    with tempfile.TemporaryDirectory() as temp:
        work = args.keep or Path(temp)
        work.mkdir(parents=True, exist_ok=True)

        new_concat = work / "new [concat].m4b"
        new_final = work / "new.abr.m4b"
        write_new(new_concat, audio)
        new_before_tag = parse_mp4(new_concat)
        add_abr_tag(new_concat, new_final)
        new_structure, new_chapters = describe(checker, "new", new_final, legacy=False)
        checker.check(
            "new: Mutagen tag rewrite keeps chpl, chapter track, and placement",
            (new_before_tag.has_chpl, new_before_tag.chapter_refs, new_before_tag.track_handlers,
             new_before_tag.top_level)
            == (new_structure.has_chpl, new_structure.chapter_refs, new_structure.track_handlers,
                new_structure.top_level),
        )

        if shutil.which("ffmpeg") is None:
            print("SKIP  legacy FFmpeg CLI baseline (no ffmpeg on PATH)")
        else:
            legacy_chaptered = work / "legacy [chaptermeta].m4b"
            legacy_final = work / "legacy.abr.m4b"
            write_legacy(legacy_chaptered, audio, work)
            add_abr_tag(legacy_chaptered, legacy_final)
            legacy_structure, legacy_chapters = describe(checker, "legacy", legacy_final, legacy=True)
            checker.check(
                "new matches legacy: chpl, chapter track/tref, chapter list",
                (new_structure.has_chpl, new_structure.track_handlers, new_structure.chapter_refs, new_chapters)
                == (legacy_structure.has_chpl, legacy_structure.track_handlers, legacy_structure.chapter_refs,
                    legacy_chapters),
            )
            legacy_only = sorted(set(legacy_structure.tag_atoms) - set(new_structure.tag_atoms))
            new_only = sorted(set(new_structure.tag_atoms) - set(legacy_structure.tag_atoms))
            print(f"INFO  tag atoms only in legacy: {legacy_only}; only in new: {new_only}")

    print("RESULT", "PASS" if checker.failures == 0 else f"FAIL ({checker.failures})")
    return 0 if checker.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
