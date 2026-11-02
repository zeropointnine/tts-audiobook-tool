"""
Streaming audio decoding for app-owned audio I/O.

Pip-only by design: decoding goes through PyAV (FFmpeg bundled in its wheel)
and never launches or loads a system FFmpeg. Unsupported or unreadable input
raises `AudioDecodeError`; there is no fallback to another decoder.

Behavior (see plans/pip-only-audio-dependencies.md, "Audio I/O rules"):

- Stream: `container.streams.best("audio")`, ie the default audio track, as
  the FFmpeg CLI picked it. Video/cover-art/subtitle/data streams are ignored.
- Output: float32 chunks, shape (n,) for mono or (n, channels) interleaved,
  at an explicit output sample rate and channel count.
- Timeline: seconds from the start of the audio stream (the stream's
  `start_time` is time zero, as for the FFmpeg CLI's `-ss`). Positions are
  counted in output samples from the first decoded frame on, so there is no
  per-chunk rounding drift; timestamp gaps in the input are not filled.
- Resampling: one stateful libswresample converter per run, flushed at EOF.
  It is only rebuilt if the input's sample format/layout/rate changes
  mid-stream (the old one is flushed first).
- Channels: stereo -> mono is the mean of the channels (as librosa does);
  mono -> stereo duplicates the channel at full level. More than two input
  channels are first folded to stereo with FFmpeg's standard downmix matrix.
- Ranges: `chunks(start, duration)` seeks to before `start` (leaving room for
  the AAC/MP3 decoder pre-roll after a seek), then discards samples before
  `start` and trims after `start + duration`, sample-accurately at the output
  rate.
- Cancellation: `cancel_check` is polled between packets; when it returns
  True the generator just stops.

`AudioDecoder` holds the file open between construction and `close()` (or
the end of a `with` block); callers must not keep one open across operations
that may replace the file (required on Windows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterator

import numpy as np
from numpy import ndarray

from tts_audiobook_tool.l import L


class AudioDecodeError(Exception):
    """The input couldn't be opened or decoded."""


@dataclass(frozen=True)
class SourceChapter:
    """A chapter of the input, in seconds on the decoder's timeline."""
    start: float
    end: float
    title: str


# Seek this far before the requested start: the first frames decoded after a
# seek may be incomplete (AAC/MP3 overlap-add needs the preceding frame)
SEEK_PREROLL_SECONDS = 1.0

# If a seek lands after the requested start (inexact seeking, eg VBR MP3
# without a seek table), retry from further back, finally from the beginning
_SEEK_FALLBACK_SECONDS = (30.0, math.inf)


class AudioDecoder:
    """
    Decodes the default audio track of `path`.

    `sample_rate` / `channels`: output format; None keeps the source's
    (channels: at most 2).
    Opening fails with AudioDecodeError if the file can't be read or has no
    audio stream.
    """

    def __init__(self, path: str, sample_rate: int | None = None, channels: int | None = None) -> None:
        import av

        if sample_rate is not None and sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        if channels is not None and channels not in (1, 2):
            raise ValueError(f"Unsupported channel count: {channels}")
        self.path = path
        try:
            self._container: Any = av.open(path, mode="r")
        except Exception as e:
            raise AudioDecodeError(f"Couldn't open audio file: {_describe(e)}") from e
        try:
            stream = self._container.streams.best("audio")
            if stream is None:
                raise AudioDecodeError("No audio stream found")
            self._stream: Any = stream
            codec_context = stream.codec_context
            self.source_sample_rate = int(codec_context.sample_rate or stream.rate or 0)
            self.source_channels = int(codec_context.channels or 0)
            if self.source_sample_rate <= 0 or self.source_channels <= 0:
                raise AudioDecodeError("Audio stream has no sample rate or channel layout")
        except BaseException as e:
            self._container.close()
            if isinstance(e, AudioDecodeError):
                raise
            if isinstance(e, Exception):
                raise AudioDecodeError(f"Couldn't read audio file: {_describe(e)}") from e
            raise
        self.sample_rate = sample_rate or self.source_sample_rate
        # More than two source channels are always folded to stereo
        self.channels = channels or min(self.source_channels, 2)
        self._time_base: Fraction = stream.time_base or Fraction(1, self.source_sample_rate)
        self._origin_pts: int = stream.start_time if stream.start_time is not None else 0
        self._used = False
        self._closed = False

    # --- info

    @property
    def duration(self) -> float | None:
        """Nominal duration in seconds from the container, if known."""
        stream = self._stream
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if self._container.duration is not None:
            return self._container.duration / 1_000_000
        return None

    @property
    def metadata(self) -> dict[str, str]:
        """Container-level tags (title, artist, ...) with FFmpeg's generic key names."""
        return {str(key): str(value) for key, value in self._container.metadata.items()}

    def chapters(self) -> list[SourceChapter]:
        result = []
        origin = self._container.start_time / 1_000_000 if self._container.start_time else 0.0
        for chapter in self._container.chapters():
            time_base = chapter["time_base"]
            start = max(0.0, float(chapter["start"] * time_base) - origin)
            end = max(start, float(chapter["end"] * time_base) - origin)
            title = str((chapter.get("metadata") or {}).get("title", ""))
            result.append(SourceChapter(start, end, title))
        return result

    # --- decoding

    def chunks(
            self,
            start: float = 0.0,
            duration: float | None = None,
            cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[ndarray]:
        """
        Yields float32 audio from `start` seconds for `duration` seconds
        (None: to the end). Raises AudioDecodeError on decoding failures.
        """
        if self._closed:
            raise RuntimeError("Decoder is closed")
        if not math.isfinite(start) or start < 0:
            raise ValueError(f"Invalid start: {start}")
        if duration is not None and (not math.isfinite(duration) or duration < 0):
            raise ValueError(f"Invalid duration: {duration}")

        rate = self.sample_rate
        total = None if duration is None else round(duration * rate)
        if total == 0:
            return

        seek_targets = [max(0.0, start - SEEK_PREROLL_SECONDS)] if start > 0 else [0.0]
        seek_targets += [max(0.0, start - back) for back in _SEEK_FALLBACK_SECONDS if start > 0]
        for attempt, seek_to in enumerate(seek_targets):
            is_last = attempt == len(seek_targets) - 1
            run = _DecodeRun(self, seek_to, start, total, cancel_check, may_retry=not is_last)
            yield from run.samples()
            if not run.landed_late:
                return
            L.w(f"Seek landed after {start:.3f}s in {self.path}; retrying from further back")

    def read(
            self,
            start: float = 0.0,
            duration: float | None = None,
            cancel_check: Callable[[], bool] | None = None,
    ) -> ndarray:
        """All of `chunks()` as one array; for bounded ranges and short files."""
        parts = list(self.chunks(start, duration, cancel_check))
        if not parts:
            shape = (0,) if self.channels == 1 else (0, self.channels)
            return np.zeros(shape, dtype=np.float32)
        return np.concatenate(parts)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._container.close()

    def __enter__(self) -> AudioDecoder:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # --- internal, used by _DecodeRun

    def _seek(self, seconds: float) -> None:
        """Positions the demuxer at or before `seconds` and resets the decoder."""
        if self._used or seconds > 0:
            target = self._origin_pts + int(seconds / self._time_base)
            try:
                self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
            except Exception as e:
                if seconds > 0:
                    raise AudioDecodeError(f"Couldn't seek in audio file: {_describe(e)}") from e
                # Unseekable input positioned elsewhere: can't rewind
                raise AudioDecodeError(f"Couldn't rewind audio file: {_describe(e)}") from e
            self._stream.codec_context.flush_buffers()
        self._used = True

    def _frame_time(self, frame: Any) -> float | None:
        if frame.pts is None:
            return None
        time_base = frame.time_base or self._time_base
        origin = self._origin_pts * self._time_base
        return float(frame.pts * time_base - origin)


class _DecodeRun:
    """One pass over the input from a seek point."""

    def __init__(
            self,
            decoder: AudioDecoder,
            seek_to: float,
            start: float,
            total: int | None,
            cancel_check: Callable[[], bool] | None,
            may_retry: bool,
    ) -> None:
        self.decoder = decoder
        self.seek_to = seek_to
        self.start = start
        self.remaining = total
        self.cancel_check = cancel_check
        self.may_retry = may_retry
        self.landed_late = False
        self._skip: int | None = None  # output samples still to discard; None until the first frame
        self._converter = _Converter(decoder.sample_rate, decoder.channels)
        self._bad_packets = 0

    def samples(self) -> Iterator[ndarray]:
        from av.error import FFmpegError, InvalidDataError

        decoder = self.decoder
        decoder._seek(self.seek_to)
        stream = decoder._stream
        try:
            for packet in decoder._container.demux(stream):
                if self.cancel_check is not None and self.cancel_check():
                    return
                try:
                    frames = packet.decode()
                except InvalidDataError as e:
                    # As the FFmpeg CLI did: skip a corrupt packet and go on
                    self._bad_packets += 1
                    if self._bad_packets <= 3:
                        L.w(f"Skipping undecodable audio packet in {decoder.path}: {e}")
                    continue
                for frame in frames:
                    if self._skip is None and not self._begin(frame):
                        return
                    for data in self._converter.convert(frame):
                        yield from self._emit(data)
                        if self.remaining == 0:
                            return
            for data in self._converter.flush():
                yield from self._emit(data)
                if self.remaining == 0:
                    return
        except FFmpegError as e:
            raise AudioDecodeError(f"Couldn't decode audio: {_describe(e)}") from e

    def _begin(self, frame: Any) -> bool:
        """Sets up trimming from the first decoded frame. False: seek landed too late, retry."""
        rate = self.decoder.sample_rate
        frame_time = self.decoder._frame_time(frame)
        if frame_time is None:
            frame_time = self.seek_to
        if frame_time > self.start + 1 / rate and self.may_retry and self.seek_to > 0:
            self.landed_late = True
            return False
        self._skip = max(0, round((self.start - frame_time) * rate))
        return True

    def _emit(self, data: ndarray) -> Iterator[ndarray]:
        if self._skip:
            if len(data) <= self._skip:
                self._skip -= len(data)
                return
            data = data[self._skip:]
            self._skip = 0
        if self.remaining is not None:
            data = data[:self.remaining]
            self.remaining -= len(data)
        if len(data):
            yield data


class _Converter:
    """Stateful format/rate conversion to float32 at the output rate, then explicit channel mapping."""

    def __init__(self, rate: int, channels: int) -> None:
        self.rate = rate
        self.channels = channels
        self._resampler: Any = None
        self._key: tuple | None = None

    def convert(self, frame: Any) -> Iterator[ndarray]:
        import av

        key = (frame.format.name, frame.layout.name, frame.sample_rate)
        if key != self._key:
            if self._resampler is not None:
                yield from self.flush()
            # More than two channels: FFmpeg's standard downmix to stereo
            layout = "stereo" if frame.layout.nb_channels > 2 else frame.layout
            self._resampler = av.AudioResampler(format="fltp", layout=layout, rate=self.rate)
            self._key = key
        for out in self._resampler.resample(frame):
            yield self._map_channels(out.to_ndarray())

    def flush(self) -> Iterator[ndarray]:
        if self._resampler is None:
            return
        resampler, self._resampler, self._key = self._resampler, None, None
        for out in resampler.resample(None):
            yield self._map_channels(out.to_ndarray())

    def _map_channels(self, planar: ndarray) -> ndarray:
        """(source channels, n) planar float -> (n,) or (n, channels)."""
        planar = planar.astype(np.float32, copy=False)
        if self.channels == 1:
            mono = planar[0] if planar.shape[0] == 1 else planar.mean(axis=0, dtype=np.float32)
            return np.ascontiguousarray(mono)
        if planar.shape[0] == 1:
            planar = np.repeat(planar, self.channels, axis=0)
        return np.ascontiguousarray(planar.T)


def _describe(error: Exception) -> str:
    return str(error) or type(error).__name__


def read_audio(
        path: str,
        sample_rate: int | None = None,
        channels: int | None = 1,
        start: float = 0.0,
        duration: float | None = None,
) -> tuple[ndarray, int]:
    """Decodes (a range of) a file in one go. Returns (samples, sample rate)."""
    with AudioDecoder(path, sample_rate=sample_rate, channels=channels) as decoder:
        return decoder.read(start, duration), decoder.sample_rate
