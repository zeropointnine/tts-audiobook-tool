"""
Pip-only audio stack check: PyAV + soundfile + mutagen with no FFmpeg executable.

Encodes and decodes a short AAC (.m4a) and FLAC file using only pip wheels,
reads the AAC tags back with mutagen, and confirms that every loaded FFmpeg
library comes from PyAV's bundled-library directory (one build).

Meant for a clean environment, e.g. a `python:3.11-slim` container:

    docker run --rm -v "$PWD":/src:ro -w /src python:3.11-slim sh -c '
        pip install -q --only-binary=av,soundfile av==18.1.0 soundfile==0.13.1 \
            mutagen==1.48.1 numpy psutil &&
        python -m testx.pip_only_audio_check'

Also runs from any app venv:

    ./venv-base/bin/python -m testx.pip_only_audio_check
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import mutagen.mp4
import numpy as np
import soundfile as sf

from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import ffmpeg_lib_report

SAMPLE_RATE = 44100
FREQUENCY = 440.0


def make_signal(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(round(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * FREQUENCY * t)).astype(np.float32)


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return float(frequencies[np.abs(np.fft.rfft(samples)).argmax()])


def aac_round_trip(path: Path, signal: np.ndarray) -> tuple[float, float]:
    with av.open(str(path), "w", format="ipod", options={"movflags": "+faststart"}) as container:
        container.metadata["title"] = "Check"
        stream = container.add_stream("aac", rate=SAMPLE_RATE, layout="mono")
        stream.bit_rate = 64000
        for start in range(0, len(signal), 4096):
            frame = av.AudioFrame.from_ndarray(signal[start:start + 4096].reshape(1, -1), format="fltp", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            frame.pts = start
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    with av.open(str(path)) as container:
        decoded = np.concatenate([
            frame.to_ndarray().reshape(-1)
            for frame in container.decode(container.streams.best("audio"))
            if isinstance(frame, av.AudioFrame)
        ])
    return len(decoded) / SAMPLE_RATE, dominant_frequency(decoded, SAMPLE_RATE)


def flac_round_trip(path: Path, signal: np.ndarray) -> tuple[float, float]:
    with sf.SoundFile(str(path), "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16", format="FLAC") as file:
        file.write(signal)
    decoded, sample_rate = sf.read(str(path), dtype="float32")
    return len(decoded) / sample_rate, dominant_frequency(decoded, sample_rate)


def main() -> int:
    failures = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))

    print(f"Python {sys.version.split()[0]}; PyAV {av.__version__} (FFmpeg {av.ffmpeg_version_info}); "
          f"soundfile {sf.__version__} (libsndfile {sf.__libsndfile_version__})")
    print(f"INFO  ffmpeg on PATH: {shutil.which('ffmpeg')}; ffprobe on PATH: {shutil.which('ffprobe')}")
    signal = make_signal()
    with tempfile.TemporaryDirectory() as temp:
        duration, frequency = aac_round_trip(Path(temp) / "check.m4a", signal)
        check("AAC encode/decode", abs(duration - 2.0) < 0.1 and abs(frequency - FREQUENCY) < 5,
              f"{duration:.3f}s, {frequency:.1f} Hz")
        title = (mutagen.mp4.MP4(str(Path(temp) / "check.m4a")).tags or {}).get("\xa9nam")
        check("AAC tags via mutagen", title == ["Check"], f"{title}")
        duration, frequency = flac_round_trip(Path(temp) / "check.flac", signal)
        check("FLAC encode/decode", duration == 2.0 and abs(frequency - FREQUENCY) < 5,
              f"{duration:.3f}s, {frequency:.1f} Hz")

    if sys.platform == "darwin":
        print("SKIP  loaded FFmpeg libraries (psutil memory_maps() is unavailable on macOS)")
    else:
        report = ffmpeg_lib_report()
        print(report.describe())
        check("FFmpeg libraries loaded", bool(report.libs))
        check("FFmpeg libraries only from PyAV bundle", report.only_bundled)
        check("single FFmpeg build", report.single_build)

    print("RESULT", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
