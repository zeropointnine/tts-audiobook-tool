"""
Synthetic audio fixtures written with soundfile / PyAV — no FFmpeg executable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from av.stream import Disposition


def sine(frequency: float, sample_rate: int, duration_s: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(round(sample_rate * duration_s)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def write_sine_wav(path: Path, frequency: float = 440, sample_rate: int = 16000, duration_s: float = 1.0) -> None:
    """Mono 16-bit PCM WAV, written block by block so long fixtures stay cheap."""
    block = sample_rate * 10
    total = round(sample_rate * duration_s)
    with sf.SoundFile(str(path), "w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV") as file:
        for start in range(0, total, block):
            t = np.arange(start, min(start + block, total)) / sample_rate
            file.write((0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32))


@dataclass(frozen=True)
class AudioTrack:
    frequency: float
    channels: int = 1
    default: bool = False


def write_aac_tracks(
    path: Path,
    tracks: list[AudioTrack],
    sample_rate: int = 16000,
    duration_s: float = 1.0,
    container_format: str = "mp4",
) -> None:
    """One AAC stream per track, each a sine; `default` sets its default disposition."""
    signals = [sine(track.frequency, sample_rate, duration_s) for track in tracks]
    frame_size = 1024
    with av.open(str(path), "w", format=container_format) as container:
        streams = []
        for track in tracks:
            stream = container.add_stream("aac", rate=sample_rate, layout="stereo" if track.channels == 2 else "mono")
            stream.disposition = Disposition.default if track.default else Disposition(0)
            streams.append(stream)
        for start in range(0, len(signals[0]), frame_size):
            for track, stream, signal in zip(tracks, streams, signals):
                planar = np.ascontiguousarray(np.tile(signal[start:start + frame_size], (track.channels, 1)))
                frame = av.AudioFrame.from_ndarray(planar, format="fltp", layout=stream.layout.name)
                frame.sample_rate = sample_rate
                frame.pts = start
                container.mux(stream.encode(frame))
        for stream in streams:
            container.mux(stream.encode(None))


def write_encoded(
    path: Path,
    signal: np.ndarray,
    sample_rate: int,
    codec: str = "aac",
    container_format: str | None = None,
    metadata: dict[str, str] | None = None,
    chapters: list[tuple[float, float, str]] | None = None,
) -> None:
    """
    Encodes `signal` ((n,) mono or (n, channels)) with PyAV, eg codec "aac"
    (container "ipod"/"mp4") or "libmp3lame" (container "mp3"). Chapters are
    (start s, end s, title).
    """
    planar = signal.reshape(-1, 1).T if signal.ndim == 1 else signal.T
    channels = planar.shape[0]
    layout = {1: "mono", 2: "stereo", 6: "5.1"}[channels]
    container_format = container_format or {"aac": "ipod", "libmp3lame": "mp3", "flac": "flac"}[codec]
    with av.open(str(path), "w", format=container_format) as container:
        for key, value in (metadata or {}).items():
            container.metadata[key] = value
        stream = container.add_stream(codec, rate=sample_rate, layout=layout)
        sample_format = "s16" if codec == "flac" else "fltp"
        for start in range(0, planar.shape[1], 1152):
            block = np.ascontiguousarray(planar[:, start:start + 1152], dtype=np.float32)
            if sample_format == "s16":
                block = np.ascontiguousarray((np.clip(block, -1, 1) * 32767).astype(np.int16).T.reshape(1, -1))
            frame = av.AudioFrame.from_ndarray(block, format=sample_format, layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = start
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
        if chapters:
            from fractions import Fraction
            container.set_chapters([
                {"id": i + 1, "start": round(s * 1000), "end": round(e * 1000),
                 "time_base": Fraction(1, 1000), "metadata": {"title": title}}
                for i, (s, e, title) in enumerate(chapters)
            ])


def tiny_png() -> bytes:
    """A valid 1x1 PNG, for cover-art fixtures."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\0\0\0\0")) + chunk(b"IEND", b"")


def add_cover_art(path: Path) -> None:
    """Embeds a cover image (MP3: ID3 APIC, MP4: covr), which FFmpeg exposes as a video stream."""
    if path.suffix.lower() == ".mp3":
        from mutagen.id3 import APIC, ID3, ID3NoHeaderError

        try:
            tags = ID3(str(path))
        except ID3NoHeaderError:
            tags = ID3()
        tags.add(APIC(encoding=3, mime="image/png", type=3, desc="cover", data=tiny_png()))
        tags.save(str(path))
    else:
        from mutagen.mp4 import MP4, MP4Cover

        file = MP4(str(path))
        if file.tags is None:
            file.add_tags()
        assert file.tags is not None
        file.tags["covr"] = [MP4Cover(tiny_png(), imageformat=MP4Cover.FORMAT_PNG)]
        file.save()


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return float(frequencies[np.abs(np.fft.rfft(samples)).argmax()])


def open_fds_to(*paths: str | Path) -> list[str]:
    """Paths among `paths` this process holds open (Linux proxy for Windows file locking)."""
    targets = {os.path.realpath(path) for path in paths}
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target in targets or target.removesuffix(" (deleted)") in targets:
            found.append(target)
    return found


def assert_no_handles(*paths: str | Path) -> None:
    """
    Windows-safe rule check: no open descriptor to any of `paths`. Linux only
    (a no-op where /proc/self/fd doesn't exist).
    """
    if os.path.isdir("/proc/self/fd"):
        assert open_fds_to(*paths) == []
