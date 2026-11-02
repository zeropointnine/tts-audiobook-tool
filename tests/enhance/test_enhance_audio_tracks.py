"""Use the same default audio track for enhancement and review."""

from pathlib import Path

import av
import numpy as np
import pytest

from audio_fixture_support import (
    AudioTrack,
    add_cover_art,
    assert_no_handles,
    dominant_frequency,
    sine,
    write_aac_tracks,
    write_encoded,
)
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.enhance.enhance_alignment import _stream_audio_with_overlap
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

pytestmark = pytest.mark.no_ffmpeg_exe


@pytest.mark.parametrize("container_format", ["mp4", "matroska"])
@pytest.mark.parametrize("default_index", [1, None])
def test_enhance_uses_one_audio_track_for_transcription_output_and_preview(
    tmp_path: Path, container_format: str, default_index: int | None
) -> None:
    # Transcription, the re-encoded output, and the review preview must all
    # hear the same track, with or without a default disposition
    source = tmp_path / ("dual.mp4" if container_format == "mp4" else "dual.mkv")
    tracks = [
        AudioTrack(440, channels=2, default=default_index == 0),
        AudioTrack(880, channels=1, default=default_index == 1),
    ]
    write_aac_tracks(source, tracks, container_format=container_format)

    transcription_audio = next(_stream_audio_with_overlap(str(source)))
    expected = dominant_frequency(transcription_audio, 16000)
    if default_index == 1:
        assert expected == pytest.approx(880, abs=5)

    output = tmp_path / "enhanced.m4a"
    assert SoundFileUtil.transcode_to_aac_at(str(source), str(output)) == ""
    output_audio = next(_stream_audio_with_overlap(str(output)))
    assert dominant_frequency(output_audio, 16000) == pytest.approx(expected, abs=5)

    preview = SoundFileUtil.load_range(str(source), 0, 1)
    assert isinstance(preview, Sound)
    assert dominant_frequency(preview.data, preview.sr) == pytest.approx(expected, abs=5)
    assert_no_handles(source, output)


def test_transcoded_output_is_audio_only_with_bitrate_tags_and_chapters(tmp_path: Path) -> None:
    # Like the FFmpeg CLI defaults it replaces: cover art dropped, tags and
    # chapters carried over, requested bitrate, fast-start
    source = tmp_path / "Hörbuch.mp3"
    write_encoded(
        source, np.stack([sine(440, 44100, 6.0)] * 2, axis=1), 44100, codec="libmp3lame",
        metadata={"title": "Ein Buch", "artist": "Jemand"},
    )
    add_cover_art(source)
    output = tmp_path / "staging.m4b"
    assert SoundFileUtil.transcode_to_aac_at(str(source), str(output), "128k") == ""

    with av.open(str(output)) as container:
        assert [stream.type for stream in container.streams] == ["audio"]
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "aac"
        assert stream.rate == 44100 and stream.channels == 2
        assert stream.bit_rate == pytest.approx(128000, rel=0.15)
        assert container.metadata.get("title") == "Ein Buch"
        assert container.metadata.get("artist") == "Jemand"
        assert float(container.duration / 1_000_000) == pytest.approx(6.0, abs=0.05)
    data = output.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")
    assert_no_handles(source, output)


def test_transcode_keeps_source_chapters(tmp_path: Path) -> None:
    source = tmp_path / "chapters.mp4"
    write_encoded(
        source, sine(440, 48000, 4.0), 48000, container_format="mp4",
        chapters=[(0.0, 2.0, "One"), (2.0, 4.0, "Two")],
    )
    output = tmp_path / "out.m4a"
    assert SoundFileUtil.transcode_to_aac_at(str(source), str(output)) == ""
    with av.open(str(output)) as container:
        chapters = container.chapters()
    assert [chapter["metadata"]["title"] for chapter in chapters] == ["One", "Two"]


def test_transcode_failure_leaves_no_output(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"\xff" * 4000)
    output = tmp_path / "out.m4a"
    error = SoundFileUtil.transcode_to_aac_at(str(source), str(output))
    assert error
    assert list(tmp_path.iterdir()) == [source]
