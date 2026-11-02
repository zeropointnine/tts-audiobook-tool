"""
File-to-file audio conversion (enhancement output, Tools commands) built on
the shared decoder and the export writers. Pip-only: PyAV and soundfile, no
FFmpeg executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, AAC_SUFFIXES
from tts_audiobook_tool.sound.audio_decoder import AudioDecodeError, AudioDecoder
from tts_audiobook_tool.sound.audio_writers import AacWriter, Chapter, open_writer


class AudioConvertCancelled(Exception):
    """`cancel_check` returned True; partial output has been removed."""


# Sample rates FFmpeg's native AAC encoder accepts. Other rates are converted
# to AAC_FALLBACK_SAMPLE_RATE.
AAC_SAMPLE_RATES = frozenset({
    7350, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000,
})
AAC_FALLBACK_SAMPLE_RATE = 48000

# Container tags not to carry over: the muxer writes its own encoder tag
_SKIPPED_METADATA_KEYS = frozenset({"encoder"})


@dataclass(frozen=True)
class OutputFormat:
    sample_rate: int
    channels: int


def choose_output_format(sources: Sequence[tuple[int, int]], is_aac: bool) -> OutputFormat:
    """
    Output format for combining inputs given as (sample rate, channels):
    the highest input rate (made AAC-compatible for AAC output), stereo if
    any input has more than one channel, else mono.
    """
    if not sources:
        raise ValueError("No inputs")
    rate = max(source_rate for source_rate, _ in sources)
    if is_aac and rate not in AAC_SAMPLE_RATES:
        rate = AAC_FALLBACK_SAMPLE_RATE
    channels = 2 if any(source_channels > 1 for _, source_channels in sources) else 1
    return OutputFormat(rate, channels)


def convert_audio_files(
        sources: Sequence[str],
        dest_path: str,
        aac_bitrate: str = AAC_BITRATE_DEFAULT,
        copy_tags: bool = False,
        cancel_check: Callable[[], bool] | None = None,
        on_source: Callable[[int, str], None] | None = None,
) -> None:
    """
    Decodes the default audio track of each source in turn and writes them
    back to back to `dest_path` (AAC/MP4 or FLAC, by suffix). Inputs with
    different sample rates/channel counts are converted to one format (see
    `choose_output_format`).

    `copy_tags`: carry the first source's container tags (title, artist, ...)
    and chapters over to AAC output, as the FFmpeg CLI did by default.

    The destination is replaced only on success; an existing file is left
    untouched otherwise. No file handle stays open once this returns.

    Raises AudioDecodeError, AudioWriteError, or AudioConvertCancelled.
    """
    if not sources:
        raise ValueError("No inputs")
    is_aac = _is_aac_path(dest_path)

    formats = []
    tags: dict[str, str] = {}
    chapters: list[Chapter] = []
    for index, source in enumerate(sources):
        with _open(source) as decoder:
            formats.append((decoder.source_sample_rate, decoder.source_channels))
            if index == 0 and copy_tags and is_aac:
                tags = {
                    key: value for key, value in decoder.metadata.items()
                    if key.lower() not in _SKIPPED_METADATA_KEYS
                }
                chapters = [Chapter(item.start, item.end, item.title) for item in decoder.chapters()]
    output = choose_output_format(formats, is_aac)

    writer = open_writer(
        dest_path, output.sample_rate, output.channels,
        aac_bitrate=aac_bitrate, metadata=tags or None,
    )
    try:
        for index, source in enumerate(sources):
            if on_source is not None:
                on_source(index, source)
            with _open(source, output.sample_rate, output.channels) as decoder:
                for block in decoder.chunks(cancel_check=cancel_check):
                    writer.write(block)
            if cancel_check is not None and cancel_check():
                raise AudioConvertCancelled()
        if chapters and isinstance(writer, AacWriter):
            # Source chapters can overshoot the re-encoded duration slightly
            end = writer.duration
            writer.set_chapters([
                Chapter(min(item.start, end), min(item.end, end), item.title) for item in chapters
            ])
        writer.close()
    finally:
        writer.abort()  # no-op after a successful close


def transcode_to_aac(
        source_path: str,
        dest_path: str,
        bitrate: str = AAC_BITRATE_DEFAULT,
        cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Re-encodes one file to AAC/MP4, keeping its tags and chapters. Raises like `convert_audio_files`."""
    if not _is_aac_path(dest_path):
        raise ValueError(f"Not an AAC/MP4 file name: {dest_path}")
    convert_audio_files(
        [source_path], dest_path, aac_bitrate=bitrate, copy_tags=True, cancel_check=cancel_check,
    )


def describe_error(error: Exception) -> str:
    if isinstance(error, AudioConvertCancelled):
        return "Interrupted"
    return str(error) or type(error).__name__


def _open(path: str, sample_rate: int | None = None, channels: int | None = None) -> AudioDecoder:
    try:
        return AudioDecoder(path, sample_rate=sample_rate, channels=channels)
    except AudioDecodeError as e:
        if path in str(e):
            raise
        raise AudioDecodeError(f"{e}: {path}") from e


def _is_aac_path(path: str) -> bool:
    return Path(path).suffix.lower() in AAC_SUFFIXES

