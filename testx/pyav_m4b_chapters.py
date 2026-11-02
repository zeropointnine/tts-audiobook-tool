"""
Verify that the installed PyAV writes chaptered, fast-start M4B files with
the structure the app relies on.

Encodes a short AAC signal into M4B with PyAV twice — chapters set just
before close() and chapters set before the first packet — and checks each
file for:

- a `chpl` (Nero) chapter atom under moov/udta,
- a QuickTime chapter text track referenced by a `tref`/`chap` atom,
- chapter titles/times read back through PyAV,
- container title/artist metadata,
- `moov` before `mdat` (fast-start).

`parse_mp4()` is also used by the export writer tests and
`testx/export_chapters_vs_cli.py`.

If an `ffmpeg` executable is on PATH, it also builds a baseline with the
FFmpeg CLI chapter remux the app historically used and compares its chapter
structure. The baseline is developer-only; the PyAV checks do not need FFmpeg.

Rerun after any PyAV version bump.

Usage:
    ./venv-base/bin/python -m testx.pyav_m4b_chapters [--keep DIR]
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 44100
DURATION_S = 6.0
CHAPTERS = [("Opening", 0.0, 2.0), ("Middle part", 2.0, 4.5), ("Ending — ü", 4.5, DURATION_S)]
TITLE = "Probe Book"
ARTIST = "Probe Author"

CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"tref", b"edts", b"dinf"}


@dataclass
class Mp4Structure:
    top_level: list[str] = field(default_factory=list)
    has_chpl: bool = False
    track_handlers: dict[int, str] = field(default_factory=dict)
    chapter_refs: list[int] = field(default_factory=list)
    # moov/udta/meta: its hdlr type ("mdir" for iTunes-style tags, "mdta" for
    # FFmpeg's use_metadata_tags keys) and the ilst item atoms, with freeform
    # "----" items as "----:mean:name"
    meta_handler: str = ""
    tag_atoms: list[str] = field(default_factory=list)

    @property
    def faststart(self) -> bool:
        return (
            "moov" in self.top_level and "mdat" in self.top_level
            and self.top_level.index("moov") < self.top_level.index("mdat")
        )

    @property
    def chapter_track_handlers(self) -> list[str]:
        return [self.track_handlers.get(track_id, "?") for track_id in self.chapter_refs]


def _iter_boxes(data: bytes, start: int, end: int):
    position = start
    while position + 8 <= end:
        size, box_type = struct.unpack(">I4s", data[position:position + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[position + 8:position + 16])[0]
            header = 16
        elif size == 0:
            size = end - position
        if size < header:
            break
        yield box_type, position + header, position + size
        position += size


def parse_mp4(path: Path) -> Mp4Structure:
    data = path.read_bytes()
    result = Mp4Structure()

    def walk_trak(start: int, end: int) -> None:
        track_id = -1
        handler = "?"
        refs: list[int] = []

        def walk(inner_start: int, inner_end: int) -> None:
            nonlocal track_id, handler
            for box_type, body_start, body_end in _iter_boxes(data, inner_start, inner_end):
                if box_type == b"tkhd":
                    version = data[body_start]
                    offset = body_start + 4 + (16 if version == 1 else 8)
                    track_id = struct.unpack(">I", data[offset:offset + 4])[0]
                elif box_type == b"hdlr":
                    handler = data[body_start + 8:body_start + 12].decode("latin-1")
                elif box_type == b"chap":
                    count = (body_end - body_start) // 4
                    refs.extend(struct.unpack(f">{count}I", data[body_start:body_start + 4 * count]))
                elif box_type in CONTAINER_BOXES:
                    walk(body_start, body_end)

        walk(start, end)
        result.track_handlers[track_id] = handler
        result.chapter_refs.extend(refs)

    def freeform_name(start: int, end: int) -> str:
        parts = {}
        for box_type, body_start, body_end in _iter_boxes(data, start, end):
            if box_type in (b"mean", b"name"):
                parts[box_type] = data[body_start + 4:body_end].decode("utf-8", "replace")
        return f"----:{parts.get(b'mean', '')}:{parts.get(b'name', '')}"

    def walk_meta(start: int, end: int) -> None:
        # ISO full box (version/flags first), unless QuickTime-style (hdlr first)
        if data[start + 4:start + 8] != b"hdlr":
            start += 4
        for box_type, body_start, body_end in _iter_boxes(data, start, end):
            if box_type == b"hdlr":
                result.meta_handler = data[body_start + 8:body_start + 12].decode("latin-1")
            elif box_type == b"ilst":
                for item_type, item_start, item_end in _iter_boxes(data, body_start, body_end):
                    if item_type == b"----":
                        result.tag_atoms.append(freeform_name(item_start, item_end))
                    else:
                        result.tag_atoms.append(item_type.decode("latin-1"))

    def walk(start: int, end: int, parent: bytes = b"") -> None:
        for box_type, body_start, body_end in _iter_boxes(data, start, end):
            if box_type == b"trak":
                walk_trak(body_start, body_end)
            elif box_type == b"chpl":
                result.has_chpl = True
            elif box_type == b"meta" and parent == b"udta":
                walk_meta(body_start, body_end)
            elif box_type in CONTAINER_BOXES:
                walk(body_start, body_end, box_type)

    for box_type, body_start, body_end in _iter_boxes(data, 0, len(data)):
        result.top_level.append(box_type.decode("latin-1"))
        if box_type in CONTAINER_BOXES:
            walk(body_start, body_end, box_type)
    return result


def make_chapters() -> list:
    time_base = Fraction(1, 1000)
    return [
        {
            "id": index + 1,
            "start": round(start * 1000),
            "end": round(end * 1000),
            "time_base": time_base,
            "metadata": {"title": title},
        }
        for index, (title, start, end) in enumerate(CHAPTERS)
    ]


def write_pyav_m4b(path: Path, chapters_before_first_packet: bool, with_chapters: bool = True) -> None:
    t = np.arange(round(SAMPLE_RATE * DURATION_S)) / SAMPLE_RATE
    signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with av.open(str(path), "w", format="ipod", options={"movflags": "+faststart"}) as container:
        container.metadata["title"] = TITLE
        container.metadata["artist"] = ARTIST
        stream = container.add_stream("aac", rate=SAMPLE_RATE, layout="mono")
        stream.bit_rate = 64000
        if with_chapters and chapters_before_first_packet:
            container.set_chapters(make_chapters())
        frame_size = 4096
        pts = 0
        for start in range(0, len(signal), frame_size):
            chunk = signal[start:start + frame_size]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="fltp", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            frame.pts = pts
            pts += len(chunk)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
        if with_chapters and not chapters_before_first_packet:
            container.set_chapters(make_chapters())


def write_cli_baseline(source: Path, path: Path) -> None:
    """The app's historical remux (`M4bChapterUtil.make_copy_with_metadata`): a
    chapterless AAC file plus an FFMETADATA chapter list, stream-copied. Like the
    app, it does not request fast-start."""
    lines = [";FFMETADATA1"]
    for title, start, end in CHAPTERS:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(start * 1000)}", f"END={round(end * 1000)}", f"title={title}"]
    metadata_path = path.with_suffix(".ffmeta.txt")
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-i", str(metadata_path),
            "-map", "0", "-map_metadata", "0", "-map_chapters", "1",
            "-c", "copy", "-movflags", "use_metadata_tags", str(path),
        ],
        check=True,
    )


def read_back(path: Path) -> tuple[list[tuple[str, float, float]], dict[str, str], float]:
    with av.open(str(path)) as container:
        chapters = []
        for chapter in container.chapters():
            time_base = chapter["time_base"] or Fraction(1, 1)
            chapters.append((
                chapter["metadata"].get("title", ""),
                round(float(chapter["start"] * time_base), 3),
                round(float(chapter["end"] * time_base), 3),
            ))
        metadata = {key: container.metadata.get(key, "") for key in ("title", "artist")}
        duration = float(container.duration or 0) / av.time_base
    return chapters, metadata, duration


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            self.failures += 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))


def check_file(checker: Checker, label: str, path: Path, expect_faststart: bool = True) -> tuple[Mp4Structure, list]:
    structure = parse_mp4(path)
    chapters, metadata, duration = read_back(path)
    expected = [(title, start, end) for title, start, end in CHAPTERS]
    checker.check(f"{label}: chpl atom", structure.has_chpl)
    checker.check(
        f"{label}: chapter text track via tref/chap",
        len(structure.chapter_refs) == 1 and structure.chapter_track_handlers == ["text"],
        f"refs={structure.chapter_refs} handlers={structure.track_handlers}",
    )
    checker.check(f"{label}: chapter titles/times", chapters == expected, f"{chapters}")
    checker.check(f"{label}: title/artist", metadata == {"title": TITLE, "artist": ARTIST}, f"{metadata}")
    if expect_faststart:
        checker.check(f"{label}: moov before mdat", structure.faststart, f"top={structure.top_level}")
    else:
        print(f"INFO  {label}: top-level boxes {structure.top_level}")
    checker.check(f"{label}: duration", abs(duration - DURATION_S) < 0.1, f"{duration:.3f}s")
    return structure, chapters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", type=Path, help="write the probe files into this directory and keep them")
    args = parser.parse_args()

    print(f"PyAV {av.__version__}, FFmpeg {av.ffmpeg_version_info}")
    checker = Checker()
    with tempfile.TemporaryDirectory() as temp:
        directory = args.keep or Path(temp)
        directory.mkdir(parents=True, exist_ok=True)

        at_close = directory / "chapters_at_close.m4b"
        up_front = directory / "chapters_up_front.m4b"
        write_pyav_m4b(at_close, chapters_before_first_packet=False)
        write_pyav_m4b(up_front, chapters_before_first_packet=True)
        close_structure, close_chapters = check_file(checker, "at-close", at_close)
        front_structure, front_chapters = check_file(checker, "up-front", up_front)
        checker.check(
            "at-close matches up-front",
            (close_structure, close_chapters) == (front_structure, front_chapters),
        )

        if shutil.which("ffmpeg") is None:
            print("SKIP  FFmpeg CLI baseline (no ffmpeg on PATH)")
        else:
            baseline = directory / "cli_baseline.m4b"
            source = directory / "cli_source.m4b"
            write_pyav_m4b(source, chapters_before_first_packet=False, with_chapters=False)
            write_cli_baseline(source, baseline)
            baseline_structure, baseline_chapters = check_file(
                checker, "cli-baseline", baseline, expect_faststart=False
            )
            checker.check(
                "at-close matches CLI baseline (ignoring box order)",
                (close_structure.has_chpl, close_structure.track_handlers, close_structure.chapter_refs, close_chapters)
                == (baseline_structure.has_chpl, baseline_structure.track_handlers, baseline_structure.chapter_refs,
                    baseline_chapters),
            )

    print("RESULT", "PASS" if checker.failures == 0 else f"FAIL ({checker.failures})")
    return 0 if checker.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
