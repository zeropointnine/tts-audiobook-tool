"""The shared PyAV decoder: stream choice, resampling, channels, ranges, cancellation, handles."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf

from audio_fixture_support import (
    AudioTrack,
    add_cover_art,
    assert_no_handles,
    dominant_frequency,
    sine,
    write_aac_tracks,
    write_encoded,
)
from tts_audiobook_tool.sound.audio_decoder import AudioDecodeError, AudioDecoder, read_audio

pytestmark = pytest.mark.no_ffmpeg_exe

SR = 44100


def chirp(duration_s: float, sample_rate: int = SR) -> np.ndarray:
    """Rising tone: every position has a distinct waveform, so misplaced samples show up."""
    t = np.arange(round(duration_s * sample_rate)) / sample_rate
    return (0.3 * np.sin(2 * np.pi * (200 + 20 * t) * t)).astype(np.float32)


def make_source(directory: Path, kind: str, signal: np.ndarray, sample_rate: int = SR) -> Path:
    path = directory / f"source.{kind}"
    if kind in ("wav", "flac"):
        sf.write(str(path), signal, sample_rate, subtype="PCM_16")
    elif kind == "mp3":
        write_encoded(path, signal, sample_rate, codec="libmp3lame")
    else:
        write_encoded(path, signal, sample_rate, codec="aac")
    return path


@pytest.mark.parametrize("kind", ["wav", "flac", "mp3", "m4a"])
def test_full_decode_has_source_duration_and_content(tmp_path: Path, kind: str) -> None:
    # Encoder delay/priming is trimmed (MP3 gapless info, MP4 edit list), so
    # every format decodes to the source length, give or take AAC's last frame
    path = make_source(tmp_path, kind, sine(1000, SR, 3.0))
    data, rate = read_audio(str(path), sample_rate=16000, channels=1)
    assert rate == 16000
    assert data.dtype == np.float32 and data.ndim == 1
    assert len(data) == pytest.approx(48000, abs=1024 * 16000 / SR + 1)
    assert dominant_frequency(data, 16000) == pytest.approx(1000, abs=2)
    assert_no_handles(path)


def test_resampler_is_continuous_across_frames(tmp_path: Path) -> None:
    # One stateful resampler for the whole run: 44.1k -> 16k decoded frame by
    # frame matches the ideal 16k sine everywhere, with no seams at frame
    # boundaries and no accumulated drift at the end
    path = make_source(tmp_path, "wav", sine(440, SR, 20.0))
    with AudioDecoder(str(path), sample_rate=16000, channels=1) as decoder:
        blocks = list(decoder.chunks())
    assert len(blocks) > 100  # really streamed in pieces
    data = np.concatenate(blocks)
    assert len(data) == 20 * 16000
    expected = sine(440, 16000, 20.0)
    # Ignore the resampler's edge transients
    assert np.max(np.abs(data[100:-100] - expected[100:-100])) < 2e-3


def test_default_track_is_selected(tmp_path: Path) -> None:
    source = tmp_path / "dual.mp4"
    # The default narration is mono; the first (nondefault) track is stereo
    write_aac_tracks(source, [AudioTrack(440, channels=2), AudioTrack(880, channels=1, default=True)])
    with AudioDecoder(str(source)) as decoder:
        assert decoder.source_channels == 1
        assert dominant_frequency(decoder.read(), decoder.sample_rate) == pytest.approx(880, abs=5)


def test_without_default_disposition_the_container_best_stream_is_used(tmp_path: Path) -> None:
    # No track flagged default: the decoder follows FFmpeg's best-stream choice
    # (the same call every app path uses), never an arbitrary or mixed track
    source = tmp_path / "nodefault.mkv"
    write_aac_tracks(source, [AudioTrack(440, channels=1), AudioTrack(880, channels=2)], container_format="matroska")
    with av.open(str(source)) as container:
        assert not any(stream.disposition & av.stream.Disposition.default for stream in container.streams.audio)
        best = container.streams.best("audio")
        assert best is not None
        expected = 440 if best.index == 0 else 880
    data, rate = read_audio(str(source))
    assert dominant_frequency(data, rate) == pytest.approx(expected, abs=5)


@pytest.mark.parametrize("kind", ["mp3", "m4a"])
def test_cover_art_stream_is_ignored(tmp_path: Path, kind: str) -> None:
    path = make_source(tmp_path, kind, sine(660, SR, 2.0))
    add_cover_art(path)
    with av.open(str(path)) as container:
        assert len(container.streams.video) == 1  # the cover is a (video) stream
    data, rate = read_audio(str(path), sample_rate=16000)
    assert len(data) == pytest.approx(32000, abs=1100)
    assert dominant_frequency(data, rate) == pytest.approx(660, abs=2)


def test_channel_mapping_is_explicit(tmp_path: Path) -> None:
    left, right = np.full(SR, 0.4, np.float32), np.full(SR, 0.2, np.float32)
    stereo = tmp_path / "stereo.wav"
    sf.write(str(stereo), np.stack([left, right], axis=1), SR, subtype="FLOAT")
    mono = tmp_path / "mono.wav"
    sf.write(str(mono), left, SR, subtype="FLOAT")

    # Stereo -> mono is the channel mean (not FFmpeg's unnormalized 0.707 sum)
    data, _ = read_audio(str(stereo), channels=1)
    np.testing.assert_allclose(data, 0.3, atol=1e-6)
    # Mono -> stereo duplicates at full level (not -3 dB)
    data, _ = read_audio(str(mono), channels=2)
    assert data.shape == (SR, 2)
    np.testing.assert_allclose(data, 0.4, atol=1e-6)
    # Default: the source's own channels
    data, _ = read_audio(str(stereo), channels=None)
    assert data.shape == (SR, 2)


def test_more_than_two_channels_fold_to_stereo(tmp_path: Path) -> None:
    path = tmp_path / "surround.wav"
    sf.write(str(path), np.tile(sine(500, SR, 1.0)[:, None], (1, 6)) * 0.3, SR, subtype="FLOAT")
    with AudioDecoder(str(path)) as decoder:
        assert decoder.source_channels == 6
        assert decoder.channels == 2
        data = decoder.read()
    assert data.shape == (SR, 2)
    assert dominant_frequency(data, SR) == pytest.approx(500, abs=2)


@pytest.mark.parametrize("kind", ["wav", "flac", "mp3", "m4a"])
def test_range_decode_matches_the_full_decode(tmp_path: Path, kind: str) -> None:
    # Seeking + trimming lands on the same samples as decoding from the start,
    # including right after a seek, where AAC/MP3 need decoder pre-roll. The
    # residual is the sub-sample resampling phase (< 1 output sample).
    path = make_source(tmp_path, kind, chirp(20.0))
    rate = 48000
    full, _ = read_audio(str(path), sample_rate=rate)
    with AudioDecoder(str(path), sample_rate=rate, channels=1) as decoder:
        for start in (0.0, 0.3, 5.123, 12.5, 19.0):
            part = decoder.read(start, 0.5)
            assert len(part) == 0.5 * rate
            offset = round(start * rate)
            difference = np.max(np.abs(part - full[offset:offset + len(part)]))
            # 0.3 amplitude, <= 600 Hz: half a sample at 48k is at most ~0.012
            assert difference < 0.02, (start, difference)


def test_range_is_trimmed_to_available_audio(tmp_path: Path) -> None:
    path = make_source(tmp_path, "wav", sine(440, SR, 2.0))
    with AudioDecoder(str(path), sample_rate=SR) as decoder:
        assert len(decoder.read(1.5, 10.0)) == round(0.5 * SR)
        assert len(decoder.read(5.0, 1.0)) == 0
        assert len(decoder.read(0.0, 0.0)) == 0
        # The decoder can be reused for any number of ranges
        assert len(decoder.read(0.0, 1.0)) == SR
    assert_no_handles(path)


def test_cancel_check_stops_between_packets(tmp_path: Path) -> None:
    path = make_source(tmp_path, "wav", sine(440, SR, 10.0))
    with AudioDecoder(str(path)) as decoder:
        assert list(decoder.chunks(cancel_check=lambda: True)) == []

        calls = 0

        def cancel_after_some() -> bool:
            nonlocal calls
            calls += 1
            return calls > 5

        received = sum(len(block) for block in decoder.chunks(cancel_check=cancel_after_some))
        assert 0 < received < 10 * SR


def test_abandoned_generator_and_close_release_the_file(tmp_path: Path) -> None:
    folder = tmp_path / "Hörbuch ü"
    folder.mkdir()
    path = make_source(folder, "m4a", sine(440, SR, 5.0))
    decoder = AudioDecoder(str(path))
    stream = decoder.chunks()
    next(stream)
    stream.close()
    decoder.close()
    assert_no_handles(path)
    path.unlink()  # nothing holds it


def test_unreadable_inputs_raise_decode_errors_without_leaking_handles(tmp_path: Path) -> None:
    text = tmp_path / "notes.mp3"
    text.write_text("not audio at all " * 100)
    with pytest.raises(AudioDecodeError):
        AudioDecoder(str(text))
    with pytest.raises(AudioDecodeError, match="No such file"):
        AudioDecoder(str(tmp_path / "missing.m4a"))
    assert_no_handles(text)


def test_input_without_audio_stream_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
            frame.pts = index
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    with pytest.raises(AudioDecodeError, match="No audio stream"):
        AudioDecoder(str(path))
    assert_no_handles(path)


def test_metadata_and_chapters_are_exposed(tmp_path: Path) -> None:
    path = tmp_path / "tagged.m4a"
    write_encoded(
        path, sine(440, SR, 3.0), SR, metadata={"title": "Buch", "artist": "Autorin"},
        chapters=[(0.0, 1.5, "Eins"), (1.5, 3.0, "Zwei")],
    )
    with AudioDecoder(str(path)) as decoder:
        assert decoder.metadata["title"] == "Buch"
        assert decoder.metadata["artist"] == "Autorin"
        chapters = decoder.chapters()
        assert [chapter.title for chapter in chapters] == ["Eins", "Zwei"]
        assert chapters[1].start == pytest.approx(1.5, abs=1e-3)
        assert decoder.duration == pytest.approx(3.0, abs=0.05)
