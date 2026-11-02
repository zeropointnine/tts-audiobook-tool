"""Export writers (FLAC via soundfile, AAC/M4B via PyAV), with real files.

Everything here runs under the no-FFmpeg guard: the writers must never launch
an FFmpeg executable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import av
import mutagen.mp4
import numpy as np
import pytest
import soundfile

from audio_fixture_support import assert_no_handles, open_fds_to
from testx.pyav_m4b_chapters import parse_mp4
from tts_audiobook_tool.constants import APP_META_MP4_MEAN, APP_META_MP4_TAG
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.audio_writers import (
    AacWriter, AudioWriteError, Chapter, FlacWriter, aac_bit_rate, open_writer, staging_path_for,
)

pytestmark = pytest.mark.no_ffmpeg_exe

SR = 48000
# Chunk sizes that don't line up with AAC's 1024-sample frames or FLAC blocks,
# ending with a very short final chunk
CHUNKS = [4096, 1000, 48000, 777, 7]


def tone(n: int, freq: float, offset: int = 0, level: float = 0.3) -> np.ndarray:
    t = (np.arange(n) + offset) / SR
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def chirp(n: int, f0: float, f1: float, level: float) -> np.ndarray:
    t = np.arange(n) / SR
    duration = n / SR
    return (level * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * duration)))).astype(np.float32)


def signal(channels: int) -> np.ndarray:
    """Chirps (not periodic, so cross-correlation pins down the alignment)."""
    n = sum(CHUNKS)
    left = chirp(n, 200, 4000, 0.3)
    if channels == 1:
        return left
    return np.stack([left, chirp(n, 3000, 300, 0.2)], axis=1)


def write_chunks(writer, data: np.ndarray) -> None:
    position = 0
    for size in CHUNKS:
        writer.write(data[position:position + size])
        position += size


def decode(path: Path) -> tuple[np.ndarray, int, int]:
    """Returns (channels, samples) float audio, sample rate, channel count."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        frames = [frame.to_ndarray() for frame in container.decode(stream)]
        return np.concatenate(frames, axis=1), stream.rate, stream.channels


def best_lag(reference: np.ndarray, decoded: np.ndarray, max_lag: int = 2048) -> int:
    """Lag (in samples) at which `decoded` best matches `reference`."""
    n = min(len(reference), len(decoded)) - 2 * max_lag
    middle = reference[max_lag:max_lag + n]
    scores = [float(np.dot(middle, decoded[max_lag + lag:max_lag + lag + n])) for lag in range(-max_lag, max_lag + 1)]
    return int(np.argmax(scores)) - max_lag


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    # Non-ASCII directory names everywhere (Windows-safe rule)
    directory = tmp_path / "Ausgabe – ü 書"
    directory.mkdir()
    return directory


class TestFlacWriter:

    @pytest.mark.parametrize("channels", [1, 2])
    def test_round_trip_is_exact_int16(self, out_dir: Path, channels: int) -> None:
        # FLAC is lossless: what reads back is exactly the clipped x32767 int16
        # quantization of every chunk, short final chunk included
        path = out_dir / "kapitel ü.flac"
        data = signal(channels)
        writer = FlacWriter(str(path), SR, channels)
        write_chunks(writer, data)
        writer.close()

        info = soundfile.info(str(path))
        assert (info.samplerate, info.channels, info.subtype, info.frames) == (SR, channels, "PCM_16", len(data))
        read, _ = soundfile.read(str(path), dtype="int16", always_2d=True)
        np.testing.assert_array_equal(read, FlacWriter.quantize(data.reshape(len(data), channels)))
        assert writer.num_samples == len(data)
        assert not os.path.exists(staging_path_for(str(path)))
        assert_no_handles(path, staging_path_for(str(path)))


    def test_quantization_clips(self) -> None:
        pcm = FlacWriter.quantize(np.array([2.0, -2.0, 1.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(pcm, [32767, -32767, 32767, -32767, 0])


class TestAacWriter:

    @pytest.mark.parametrize("channels", [1, 2])
    def test_round_trip_keeps_timing_and_channels(self, out_dir: Path, channels: int) -> None:
        # The decoded stream starts exactly where the input did (encoder
        # priming is hidden by the edit list) and only runs over by the final
        # frame's padding. Stereo keeps the left/right signals apart.
        path = out_dir / "kapitel ü.m4b"
        data = signal(channels)
        writer = AacWriter(str(path), SR, channels, bitrate="128k")
        write_chunks(writer, data)
        writer.close()

        decoded, rate, decoded_channels = decode(path)
        assert (rate, decoded_channels) == (SR, channels)
        assert len(data) <= decoded.shape[1] < len(data) + 1024
        with av.open(str(path)) as container:
            assert container.duration is not None
            assert container.duration / av.time_base == pytest.approx(len(data) / SR, abs=1e-3)

        reference = data.reshape(len(data), channels)
        for channel in range(channels):
            assert best_lag(reference[:, channel], decoded[channel]) == 0
            # Waveform match only roughly: FFmpeg's AAC encoder uses stereo
            # tools (intensity/M-S coding) that preserve how it sounds, not samples
            start, end = SR // 10, len(data) - SR // 10
            assert np.corrcoef(decoded[channel, start:end], reference[start:end, channel])[0, 1] > 0.95
        if channels == 2:
            assert np.corrcoef(decoded[0, :SR], decoded[1, :SR])[0, 1] < 0.2
        assert_no_handles(path, staging_path_for(str(path)))

    def test_bitrate_choice_is_applied(self, out_dir: Path) -> None:
        sizes = {}
        rng = np.random.default_rng(0)
        noise = (0.2 * rng.standard_normal(SR * 4)).astype(np.float32)
        for bitrate in ("64k", "128k"):
            path = out_dir / f"{bitrate}.m4a"
            with AacWriter(str(path), SR, 1, bitrate=bitrate) as writer:
                writer.write(noise)
            with av.open(str(path)) as container:
                assert container.streams.audio[0].codec_context.name == "aac"
            sizes[bitrate] = path.stat().st_size
        assert sizes["128k"] > 1.6 * sizes["64k"]
        assert aac_bit_rate("bogus") == aac_bit_rate("96k") == 96000


    def test_chapters_metadata_and_faststart_survive_the_abr_tag(self, out_dir: Path) -> None:
        # Single pass: chapters set at close, title/artist as iTunes atoms,
        # moov before mdat. Mutagen's ABR tag rewrite must keep all of it.
        path = out_dir / "buch [concat].m4b"
        final = out_dir / "buch.abr.m4b"
        chapters = [Chapter(0.0, 0.5, "Eins"), Chapter(0.5, 1.25, "Zwei — ü"), Chapter(1.25, 2.0, "書")]
        with AacWriter(str(path), SR, 2, metadata={"title": "Buch", "artist": "App"}) as writer:
            writer.write(np.stack([tone(SR * 2, 300)] * 2, axis=1))
            writer.set_chapters(chapters)

        error = AudioMetaUtil.set_mp4_metadata_tag(str(path), APP_META_MP4_MEAN, APP_META_MP4_TAG, "{\"a\": \"ü\"}", str(final))
        assert error == ""

        for file in (path, final):
            structure = parse_mp4(file)
            assert structure.has_chpl
            assert structure.chapter_track_handlers == ["text"]
            assert structure.faststart, structure.top_level
            assert {"©nam", "©ART"} <= set(structure.tag_atoms)
            with av.open(str(file)) as container:
                read = [
                    (c["metadata"]["title"], float(c["start"] * c["time_base"]), float(c["end"] * c["time_base"]))
                    for c in container.chapters()
                ]
                assert read == [(c.title, c.start, c.end) for c in chapters]
                assert (container.metadata["title"], container.metadata["artist"]) == ("Buch", "App")
        assert parse_mp4(final).tag_atoms[-1] == f"----:{APP_META_MP4_MEAN}:{APP_META_MP4_TAG}"
        assert AudioMetaUtil.get_mp4_metadata_tag(str(final), APP_META_MP4_MEAN, APP_META_MP4_TAG) == ("{\"a\": \"ü\"}", "")
        assert mutagen.mp4.MP4(str(final)).tags["©nam"] == ["Buch"]  # type: ignore[index]
        assert_no_handles(path, final)

    def test_without_chapters_there_is_no_chapter_track(self, out_dir: Path) -> None:
        path = out_dir / "plain.m4b"
        with AacWriter(str(path), SR, 1) as writer:
            writer.write(tone(SR, 300))
        structure = parse_mp4(path)
        assert not structure.has_chpl and structure.chapter_refs == []
        assert structure.faststart

    def test_planar_conversion_transposes_and_clips(self) -> None:
        interleaved = np.array([[0.1, -0.2], [1.5, -1.5], [0.3, 0.4]], dtype=np.float32)
        planar = AacWriter.to_planar(interleaved)
        assert planar.flags["C_CONTIGUOUS"] and planar.dtype == np.float32
        np.testing.assert_array_equal(planar, np.array([[0.1, 1.0, 0.3], [-0.2, -1.0, 0.4]], dtype=np.float32))


class TestLifecycle:

    @pytest.mark.parametrize("name", ["empty.flac", "empty.m4b"])
    def test_empty_input_is_an_error_not_a_broken_file(self, out_dir: Path, name: str) -> None:
        # libsndfile would leave a zero-byte FLAC, FFmpeg an MP4 with no track
        path = out_dir / name
        writer = open_writer(str(path), SR, 1)
        writer.write(np.zeros(0, dtype=np.float32))
        with pytest.raises(AudioWriteError, match="no audio"):
            writer.close()
        assert not os.path.exists(path)
        assert not os.path.exists(staging_path_for(str(path)))
        assert_no_handles(path, staging_path_for(str(path)))

    @pytest.mark.parametrize("name", ["abort ü.flac", "abort ü.m4b"])
    def test_abort_removes_partial_output_and_keeps_existing_destination(self, out_dir: Path, name: str) -> None:
        path = out_dir / name
        path.write_bytes(b"previous export")
        writer = open_writer(str(path), SR, 1)
        writer.write(tone(SR, 300))
        writer.abort()
        writer.abort()  # idempotent

        assert path.read_bytes() == b"previous export"
        assert not os.path.exists(staging_path_for(str(path)))
        assert_no_handles(path, staging_path_for(str(path)))
        with pytest.raises(RuntimeError):
            writer.write(tone(10, 300))

    @pytest.mark.parametrize("name", ["replace.flac", "replace.m4b"])
    def test_close_replaces_existing_destination(self, out_dir: Path, name: str) -> None:
        path = out_dir / name
        path.write_bytes(b"previous export")
        with open_writer(str(path), SR, 1) as writer:
            writer.write(tone(SR, 300))
        assert path.read_bytes() != b"previous export"
        assert decode(path)[0].shape[1] >= SR
        assert_no_handles(path, staging_path_for(str(path)))

    def test_exception_inside_context_aborts(self, out_dir: Path) -> None:
        path = out_dir / "ctx.m4b"
        with pytest.raises(KeyError):
            with open_writer(str(path), SR, 1) as writer:
                writer.write(tone(SR, 300))
                raise KeyError("cancelled")
        assert not os.path.exists(path) and not os.path.exists(staging_path_for(str(path)))
        assert_no_handles(path, staging_path_for(str(path)))

    def test_wrong_channel_layout_is_rejected(self, out_dir: Path) -> None:
        with open_writer(str(out_dir / "x.flac"), SR, 2) as writer:
            with pytest.raises(ValueError, match="2 channel"):
                writer.write(tone(100, 300))
            writer.write(np.zeros((100, 2), dtype=np.float32))

    def test_unsupported_type_and_missing_directory(self, out_dir: Path) -> None:
        with pytest.raises(ValueError):
            open_writer(str(out_dir / "x.wav"), SR, 1)
        for name in ("x.flac", "x.m4b"):
            with pytest.raises(AudioWriteError):
                open_writer(str(out_dir / "missing" / name), SR, 1)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /dev/full")
    @pytest.mark.parametrize("name", ["full.flac", "full.m4b"])
    def test_disk_errors_are_surfaced_and_cleaned_up(self, out_dir: Path, name: str) -> None:
        # The staging file is a symlink to /dev/full, so every write fails with
        # ENOSPC. The failure must raise (never a silent success) and leave
        # neither staging nor destination behind.
        path = out_dir / name
        path.write_bytes(b"previous export")
        staging = staging_path_for(str(path))
        os.symlink("/dev/full", staging)
        with pytest.raises(AudioWriteError):
            writer = open_writer(str(path), SR, 1)
            for _ in range(10):
                writer.write(tone(SR, 300))
            writer.close()
        assert not os.path.lexists(staging)
        assert path.read_bytes() == b"previous export"
        assert_no_handles(path, "/dev/full")


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd")
def test_open_handle_check_detects_open_files(out_dir: Path) -> None:
    # Guards the guard: the descriptor check above must see a handle that is open
    path = out_dir / "held ü.flac"
    with open(path, "wb"):
        assert open_fds_to(path) == [str(path)]
    assert open_fds_to(path) == []
