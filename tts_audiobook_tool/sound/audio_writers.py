"""
Streaming audio file writers for audiobook export.

Pip-only by design: FLAC goes through soundfile (libsndfile bundled in its
wheel) and AAC/MP4 through PyAV (FFmpeg bundled in its wheel). Neither ever
launches or loads a system FFmpeg.

Lifecycle: construct (opens the output), `write()` any number of rendered
chunks, then `close()` to finalize or `abort()` to discard. Output goes to a
staging file next to the destination, which replaces the destination only
after a successful close; on failure or abort, the staging file is removed and
any existing destination file is left untouched. Every native handle is
closed before the staging file is renamed or deleted (required on Windows).

Encoder/muxer/disk failures raise `AudioWriteError` after the partial output
has been removed. Programming errors (wrong channel layout, use after close)
raise `ValueError`/`RuntimeError` and leave the writer open, so the caller
still has to abort it; using the writer as a context manager does that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import NoReturn, Sequence

import numpy as np
from numpy import ndarray

from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, AAC_BITRATES, AAC_SUFFIXES


class AudioWriteError(Exception):
    """Writing the output file failed. Partial output has already been removed."""


@dataclass(frozen=True)
class Chapter:
    """An M4B chapter. Times are in seconds from the start of the file."""
    start: float
    end: float
    title: str


STAGING_SUFFIX = ".partial"


def staging_path_for(dest_path: str) -> str:
    return dest_path + STAGING_SUFFIX


def aac_bit_rate(bitrate: str) -> int:
    """Bits per second for one of AAC_BITRATES (eg "96k"); unknown values fall back to the default."""
    if bitrate not in AAC_BITRATES:
        bitrate = AAC_BITRATE_DEFAULT
    return int(bitrate.rstrip("k")) * 1000


class AudioWriter:
    """Base class; see the module docstring for the lifecycle."""

    def __init__(self, dest_path: str, sample_rate: int, channels: int) -> None:
        if channels not in (1, 2):
            raise ValueError(f"Unsupported channel count: {channels}")
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        self.dest_path = dest_path
        self.staging_path = staging_path_for(dest_path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_samples = 0
        self._state = "open"  # open | closed | aborted

    # --- public lifecycle

    @property
    def is_open(self) -> bool:
        return self._state == "open"

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate

    def write(self, data: ndarray) -> None:
        """
        Appends float audio in [-1, 1]: shape (n,) or (n, 1) for mono,
        (n, 2) interleaved for stereo. Out-of-range values are clipped.
        """
        self._require_open("write")
        samples = self._as_frames(data)
        if len(samples) == 0:
            return
        try:
            self._write(samples)
        except Exception as e:
            self._fail("write", e)
        self.num_samples += len(samples)

    def set_chapters(self, chapters: Sequence[Chapter]) -> None:
        raise NotImplementedError(f"{type(self).__name__} doesn't support chapters")

    def close(self) -> None:
        """Finalizes the file and moves it into place. Raises AudioWriteError on failure."""
        if self._state == "closed":
            return
        self._require_open("close")
        try:
            if self.num_samples == 0:
                # FLAC would be a zero-byte file, MP4 one with no audio track
                raise ValueError("no audio was written")
            self._finalize()
        except Exception as e:
            self._fail("finalize", e)
        try:
            os.replace(self.staging_path, self.dest_path)
        except OSError as e:
            self._fail("save", e)
        self._state = "closed"

    def abort(self) -> None:
        """Discards the output. Never raises; safe to call more than once or after close()."""
        if self._state != "open":
            return
        self._state = "aborted"
        try:
            self._release()
        except Exception:
            pass
        _delete_silently(self.staging_path)

    def __enter__(self) -> AudioWriter:
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    # --- subclass hooks

    def _write(self, samples: ndarray) -> None:
        """samples: float32 (n, channels), already validated."""
        raise NotImplementedError

    def _finalize(self) -> None:
        """Flushes and closes the native handle."""
        raise NotImplementedError

    def _release(self) -> None:
        """Closes the native handle without caring about the output."""
        raise NotImplementedError

    # --- helpers

    def _require_open(self, action: str) -> None:
        if self._state != "open":
            raise RuntimeError(f"Can't {action}: writer is {self._state}")

    def _as_frames(self, data: ndarray) -> ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != self.channels:
            raise ValueError(f"Expected audio with {self.channels} channel(s), got shape {np.shape(data)}")
        return array

    def _fail(self, action: str, error: Exception) -> NoReturn:
        self.abort()
        raise AudioWriteError(f"Couldn't {action} {Path(self.dest_path).name}: {error}") from error

    def _fail_on_open(self, error: Exception) -> NoReturn:
        """For subclass constructors: the native handle couldn't be created."""
        self._state = "aborted"
        _delete_silently(self.staging_path)
        raise AudioWriteError(f"Couldn't create {Path(self.dest_path).name}: {error}") from error


class FlacWriter(AudioWriter):
    """
    16-bit FLAC via soundfile. Float input is clipped and scaled by 32767
    before quantization, as the app always has (libsndfile's own float
    conversion scales differently and wraps instead of clipping).
    """

    # libsndfile maps 0..1 to FLAC levels 0..8 (lrint(x * 8)): 0.75 is level 6
    COMPRESSION_LEVEL = 0.75

    def __init__(self, dest_path: str, sample_rate: int, channels: int) -> None:
        super().__init__(dest_path, sample_rate, channels)
        import soundfile

        try:
            self._file: soundfile.SoundFile | None = soundfile.SoundFile(
                self.staging_path, mode="w", samplerate=sample_rate, channels=channels,
                format="FLAC", subtype="PCM_16", compression_level=self.COMPRESSION_LEVEL,
            )
        except Exception as e:
            self._fail_on_open(e)

    @staticmethod
    def quantize(samples: ndarray) -> ndarray:
        return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

    def _write(self, samples: ndarray) -> None:
        assert self._file is not None
        self._file.write(self.quantize(samples))

    def _finalize(self) -> None:
        file, self._file = self._file, None
        if file is not None:
            file.close()

    def _release(self) -> None:
        self._finalize()


class AacWriter(AudioWriter):
    """
    AAC in an MP4 container (M4B/M4A/MP4) via PyAV, always fast-start.

    Float input is clipped to [-1, 1] and fed to the encoder as planar float
    (its native format). Optional container metadata (eg title/artist) and
    chapters are written in the same pass; chapters are set after the last
    packet, just before the container is closed.
    """

    # Upper bound on samples per frame handed to the encoder (PyAV re-chunks
    # to the codec's 1024-sample frames internally)
    MAX_FRAME_SAMPLES = 1 << 16

    def __init__(
            self,
            dest_path: str,
            sample_rate: int,
            channels: int,
            bitrate: str = AAC_BITRATE_DEFAULT,
            metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__(dest_path, sample_rate, channels)
        import av

        suffix = Path(dest_path).suffix.lower()
        if suffix not in AAC_SUFFIXES:
            raise ValueError(f"Not an AAC/MP4 file name: {dest_path}")
        # What FFmpeg would pick from the extension (ipod for .m4a/.m4b)
        container_format = "mp4" if suffix == ".mp4" else "ipod"
        self._layout = "mono" if channels == 1 else "stereo"
        self._pts = 0
        self._chapters: list[Chapter] = []
        # The file handle is ours, not FFmpeg's: PyAV only closes a file it
        # opened itself if the header was written, so a failed header write
        # would otherwise leak the handle (and lock the file on Windows).
        # Opened for reading too, for the fast-start rewrite at close.
        try:
            self._file = _DiscardableFile(self.staging_path)
        except Exception as e:
            self._fail_on_open(e)
        try:
            self._container = av.open(
                self._file, mode="w", format=container_format,
                options={"movflags": "+faststart"},
            )
            for key, value in (metadata or {}).items():
                self._container.metadata[key] = value
            self._stream = self._container.add_stream("aac", rate=sample_rate, layout=self._layout)
            self._stream.bit_rate = aac_bit_rate(bitrate)
        except Exception as e:
            self._release_quietly()
            self._fail_on_open(e)

    def set_chapters(self, chapters: Sequence[Chapter]) -> None:
        """Chapters to write at close. Replaces any previously set."""
        self._require_open("set chapters")
        for chapter in chapters:
            if not (0 <= chapter.start <= chapter.end):
                raise ValueError(f"Invalid chapter times: {chapter}")
        self._chapters = list(chapters)

    @staticmethod
    def to_planar(samples: ndarray) -> ndarray:
        """(n, channels) float -> contiguous clipped float32 (channels, n), as `fltp` frames expect."""
        return np.ascontiguousarray(np.clip(samples, -1.0, 1.0).T, dtype=np.float32)

    def _write(self, samples: ndarray) -> None:
        import av

        planar = self.to_planar(samples)
        for offset in range(0, planar.shape[1], self.MAX_FRAME_SAMPLES):
            chunk = np.ascontiguousarray(planar[:, offset:offset + self.MAX_FRAME_SAMPLES])
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=self._layout)
            frame.sample_rate = self.sample_rate
            frame.time_base = Fraction(1, self.sample_rate)
            frame.pts = self._pts
            self._pts += chunk.shape[1]
            self._container.mux(self._stream.encode(frame))

    def _finalize(self) -> None:
        try:
            # Flush the encoder's delayed packets
            self._container.mux(self._stream.encode(None))
            if self._chapters:
                time_base = Fraction(1, 1000)
                self._container.set_chapters([
                    {
                        "id": index + 1,
                        "start": round(chapter.start * 1000),
                        "end": round(chapter.end * 1000),
                        "time_base": time_base,
                        "metadata": {"title": chapter.title},
                    }
                    for index, chapter in enumerate(self._chapters)
                ])
            self._container.close()
        finally:
            # Flushes Python's buffer; raises if that fails
            self._file.close()

    def _release(self) -> None:
        # Discarding the file first makes PyAV (18+) skip the trailer and the
        # fast-start rewrite of everything written so far; older PyAV still
        # writes them, into the discarded file
        try:
            self._file.discard()
        finally:
            self._container.close()

    def _release_quietly(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class _DiscardableFile:
    """
    The binary file an AacWriter hands to PyAV. `discard()` closes the real
    file and turns later I/O into no-ops.

    On abort, PyAV 18+ sees `closed` and skips the trailer. PyAV 16/17 (the
    vibevoice venv's range) writes it anyway; against a plain closed file,
    every write/seek raises inside PyAV's I/O callbacks, which prints "PyAV
    library exception being dropped" tracebacks to the console.
    """

    def __init__(self, path: str) -> None:
        self._file = open(path, "w+b")
        self._discarded = False
        self._position = 0

    @property
    def name(self) -> str:
        # PyAV's URL for the container; the fast-start rewrite reopens it
        return str(self._file.name)

    @property
    def closed(self) -> bool:
        return self._file.closed

    def read(self, size: int = -1) -> bytes:
        return b"" if self._discarded else self._file.read(size)

    def write(self, data) -> int:
        if self._discarded:
            size = len(memoryview(data))
            self._position += size
            return size
        return self._file.write(data)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if not self._discarded:
            return self._file.seek(offset, whence)
        # Nothing is stored, so the end is position 0
        self._position = max(0, offset + (self._position if whence == os.SEEK_CUR else 0))
        return self._position

    def tell(self) -> int:
        return self._position if self._discarded else self._file.tell()

    def flush(self) -> None:
        if not self._discarded:
            self._file.flush()

    def close(self) -> None:
        self._file.close()

    def discard(self) -> None:
        self._discarded = True
        self._file.close()


def open_writer(
        dest_path: str,
        sample_rate: int,
        channels: int,
        aac_bitrate: str = AAC_BITRATE_DEFAULT,
        metadata: dict[str, str] | None = None,
) -> AudioWriter:
    """
    AacWriter for AAC_SUFFIXES file names, else FlacWriter (".flac").
    `aac_bitrate` and `metadata` apply to AAC only.
    """
    suffix = Path(dest_path).suffix.lower()
    if suffix in AAC_SUFFIXES:
        return AacWriter(dest_path, sample_rate, channels, bitrate=aac_bitrate, metadata=metadata)
    if suffix == ".flac":
        return FlacWriter(dest_path, sample_rate, channels)
    raise ValueError(f"Unsupported output file type: {dest_path}")


def _delete_silently(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass
