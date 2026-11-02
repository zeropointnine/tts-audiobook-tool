"""Tools menu commands (directory concatenation, ABR FLAC to AAC) without the FFmpeg executable."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf

from audio_fixture_support import assert_no_handles, dominant_frequency, sine, write_encoded
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.constants import APP_META_FLAC_FIELD, APP_META_MP4_MEAN, APP_META_MP4_TAG
from tts_audiobook_tool.sound import audio_convert
from tts_audiobook_tool.sound.audio_convert import choose_output_format
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.mp3_concat import SoundConcatTranscodeUtil
from tts_audiobook_tool.sound.transcode_util import TranscodeUtil

pytestmark = pytest.mark.no_ffmpeg_exe

# (file name, frequency, sample rate, channels, seconds)
MIXED_INPUTS = [
    ("01 intro.wav", 300, 22050, 1, 1.0),
    ("02 kapitel ü.flac", 500, 44100, 2, 1.5),
    ("03 part.mp3", 700, 32000, 1, 1.0),
    ("04 end.m4a", 900, 48000, 2, 1.0),
]


def write_mixed_inputs(folder: Path) -> list[str]:
    for name, frequency, rate, channels, seconds in MIXED_INPUTS:
        tone = sine(frequency, rate, seconds)
        signal = tone if channels == 1 else np.stack([tone, tone], axis=1)
        path = folder / name
        if path.suffix in (".wav", ".flac"):
            sf.write(str(path), signal, rate, subtype="PCM_16")
        else:
            write_encoded(path, signal, rate, codec="libmp3lame" if path.suffix == ".mp3" else "aac")
    return [name for name, *_ in MIXED_INPUTS]


def test_output_format_takes_highest_rate_and_any_stereo() -> None:
    assert choose_output_format([(22050, 1), (44100, 1)], is_aac=False) == audio_convert.OutputFormat(44100, 1)
    assert choose_output_format([(22050, 1), (16000, 2)], is_aac=True) == audio_convert.OutputFormat(22050, 2)
    # AAC can't encode every rate FLAC can hold
    assert choose_output_format([(192000, 2)], is_aac=True).sample_rate == 48000
    assert choose_output_format([(192000, 2)], is_aac=False).sample_rate == 192000


@pytest.mark.parametrize("is_flac", [True, False])
def test_concatenates_mixed_formats_rates_and_layouts_in_order(tmp_path: Path, is_flac: bool) -> None:
    folder = tmp_path / "Hörbücher"
    folder.mkdir()
    names = write_mixed_inputs(folder)
    dest_name = "concatenated.flac" if is_flac else "concatenated.m4a"

    assert SoundConcatTranscodeUtil.concatenate_audio_files(names, str(folder), dest_name, is_flac) == ""

    output = folder / dest_name
    data, rate = sf.read(str(output)) if is_flac else _decode(output)
    assert rate == 48000
    assert data.ndim == 2 and data.shape[1] == 2
    total = sum(seconds for *_, seconds in MIXED_INPUTS)
    assert len(data) / rate == pytest.approx(total, abs=0.1)
    # Each input's tone appears in order, at its own place on the timeline
    position = 0.0
    for _, frequency, _, _, seconds in MIXED_INPUTS:
        middle = data[int((position + 0.2) * rate):int((position + seconds - 0.2) * rate)]
        assert dominant_frequency(middle, rate) == pytest.approx(frequency, abs=5)
        position += seconds
    assert sorted(path.name for path in folder.iterdir()) == sorted(names + [dest_name])
    assert_no_handles(*folder.iterdir())


def test_concatenation_error_keeps_existing_output_and_removes_staging(tmp_path: Path) -> None:
    names = write_mixed_inputs(tmp_path)
    (tmp_path / "05 broken.mp3").write_bytes(b"\xff" * 2000)
    existing = tmp_path / "concatenated.flac"
    existing.write_bytes(b"previous")

    error = SoundConcatTranscodeUtil.concatenate_audio_files(
        names + ["05 broken.mp3"], str(tmp_path), "concatenated.flac", True
    )

    assert error
    assert existing.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.partial"))
    assert_no_handles(*tmp_path.iterdir())


def test_concatenation_cancel_removes_partial_output(tmp_path: Path, monkeypatch) -> None:
    names = write_mixed_inputs(tmp_path)
    original = audio_convert.AudioDecoder.chunks

    def interrupt_while_decoding(self, *args, **kwargs):
        for block in original(self, *args, **kwargs):
            Interrupts()._flag = True
            yield block

    monkeypatch.setattr(audio_convert.AudioDecoder, "chunks", interrupt_while_decoding)

    error = SoundConcatTranscodeUtil.concatenate_audio_files(names, str(tmp_path), "concatenated.m4a", False)

    assert error == "Interrupted"
    assert not (tmp_path / "concatenated.m4a").exists()
    assert not list(tmp_path.glob("*.partial"))
    assert not Interrupts().did_interrupt  # cleared for the next operation
    assert_no_handles(*tmp_path.iterdir())


def test_abr_flac_to_aac_keeps_app_metadata(tmp_path: Path) -> None:
    folder = tmp_path / "Ausgabe ü"
    folder.mkdir()
    flac = folder / "book.abr.flac"
    sf.write(str(flac), np.stack([sine(440, 48000, 2.0)] * 2, axis=1), 48000, subtype="PCM_16")
    app_data = '{"version": 1, "text": "' + "x" * 5000 + '"}'
    assert AudioMetaUtil.set_flac_custom_metadata_field(str(flac), APP_META_FLAC_FIELD, app_data) == ""

    m4a_path, error = TranscodeUtil.transcode_abr_flac_to_aac(str(flac), kbps=64)

    assert error == ""
    assert m4a_path == str(flac.with_suffix(".m4a"))
    value, tag_error = AudioMetaUtil.get_mp4_metadata_tag(m4a_path, APP_META_MP4_MEAN, APP_META_MP4_TAG)
    assert tag_error == "" and value == app_data
    with av.open(m4a_path) as container:
        stream = container.streams.audio[0]
        assert (stream.codec_context.name, stream.rate, stream.channels) == ("aac", 48000, 2)
        assert stream.bit_rate == pytest.approx(64000, rel=0.15)
    assert AudioMetaUtil.get_audio_duration(m4a_path) == pytest.approx(2.0, abs=0.05)
    assert_no_handles(flac, m4a_path)


def test_abr_flac_to_aac_rejects_plain_flac_and_existing_output(tmp_path: Path) -> None:
    flac = tmp_path / "plain.flac"
    sf.write(str(flac), sine(440, 48000, 0.5), 48000, subtype="PCM_16")
    assert TranscodeUtil.transcode_abr_flac_to_aac(str(flac)) == ("", "FLAC file has no tts-audiobook-tool metadata")
    flac.with_suffix(".m4a").write_bytes(b"x")
    assert TranscodeUtil.transcode_abr_flac_to_aac(str(flac))[1] == "M4A file already exists with same file stem"


def _decode(path: Path) -> tuple[np.ndarray, int]:
    from tts_audiobook_tool.sound.audio_decoder import read_audio

    return read_audio(str(path), channels=None)
