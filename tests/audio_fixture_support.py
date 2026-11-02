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
