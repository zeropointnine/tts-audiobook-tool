"""Preview range decoding and the common file loader, both without the FFmpeg executable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_fixture_support import (
    AudioTrack,
    assert_no_handles,
    dominant_frequency,
    sine,
    write_aac_tracks,
    write_encoded,
)
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

pytestmark = pytest.mark.no_ffmpeg_exe


def tone_steps(path: Path, sample_rate: int, step_s: float, count: int) -> None:
    """AAC file whose frequency changes every `step_s` seconds (step i: 300 + 100 i Hz)."""
    signal = np.concatenate([sine(300 + 100 * i, sample_rate, step_s) for i in range(count)])
    write_encoded(path, signal, sample_rate)


def test_load_range_seeks_to_the_requested_interval(tmp_path: Path) -> None:
    path = tmp_path / "book.m4b"
    tone_steps(path, 44100, 2.0, 10)
    result = SoundFileUtil.load_range(str(path), 12.5, 13.5)
    assert isinstance(result, Sound)
    assert result.sr == APP_SAMPLE_RATE
    assert result.data.ndim == 1
    assert len(result.data) == APP_SAMPLE_RATE
    # 12.5 s lies in step 6
    assert dominant_frequency(result.data, result.sr) == pytest.approx(900, abs=2)
    assert_no_handles(path)


def test_load_range_caps_decoding_to_one_minute(tmp_path: Path) -> None:
    path = tmp_path / "long.mp3"
    write_encoded(path, sine(440, 16000, 75.0), 16000, codec="libmp3lame")
    result = SoundFileUtil.load_range(str(path), 2.0, 500.0)
    assert isinstance(result, Sound)
    assert len(result.data) == SoundFileUtil.MAX_PREVIEW_SECONDS * APP_SAMPLE_RATE


def test_load_range_follows_the_default_audio_track(tmp_path: Path) -> None:
    path = tmp_path / "dual.mp4"
    write_aac_tracks(path, [AudioTrack(440, channels=2), AudioTrack(880, channels=1, default=True)])
    result = SoundFileUtil.load_range(str(path), 0, 1)
    assert isinstance(result, Sound)
    assert dominant_frequency(result.data, result.sr) == pytest.approx(880, abs=5)


def test_load_range_reports_errors_and_skips_invalid_intervals(tmp_path: Path) -> None:
    path = tmp_path / "short.m4a"
    write_encoded(path, sine(440, 44100, 1.0), 44100)
    assert SoundFileUtil.load_range(str(path), 5.0, 6.0) == "No audio was found in the selected interval"
    for start, end in ((1, 1), (-1, 5), (float("nan"), 2)):
        assert SoundFileUtil.load_range(str(path), start, end) == "No playable audio interval is available"
    error = SoundFileUtil.load_range(str(tmp_path / "missing.mp3"), 1, 2)
    assert isinstance(error, str) and "No such file" in error


def test_load_reads_libsndfile_formats_like_librosa_did(tmp_path: Path) -> None:
    # WAV/FLAC still go through soundfile, so samples are unchanged from the
    # librosa.load era: float32 channel mean at the file's own rate
    stereo = np.stack([sine(440, 22050, 1.0), sine(660, 22050, 1.0, amplitude=0.1)], axis=1)
    path = tmp_path / "voice.flac"
    sf.write(str(path), stereo, 22050, subtype="PCM_16")
    expected = sf.read(str(path), dtype="float32")[0].mean(axis=1)
    result = SoundFileUtil.load(str(path))
    assert isinstance(result, Sound)
    assert result.sr == 22050
    assert result.data.dtype == np.float32
    np.testing.assert_array_equal(result.data, expected.astype(np.float32))
    assert_no_handles(path)


def test_load_decodes_aac_with_pyav_and_resamples(tmp_path: Path) -> None:
    # libsndfile can't read AAC; this used to fall back to audioread/FFmpeg
    path = tmp_path / "Stimme ü.m4a"
    write_encoded(path, sine(550, 44100, 2.0), 44100)
    result = SoundFileUtil.load(str(path), target_sr=24000)
    assert isinstance(result, Sound)
    assert result.sr == 24000
    assert len(result.data) == pytest.approx(48000, abs=600)
    assert dominant_frequency(result.data, 24000) == pytest.approx(550, abs=2)
    assert_no_handles(path)


def test_load_errors_are_strings_not_ffmpeg_launches(tmp_path: Path) -> None:
    # Missing and unreadable files used to reach audioread's FFmpeg probe
    garbage = tmp_path / "voice.wav"
    garbage.write_bytes(b"RIFF nonsense" * 10)
    for path in (tmp_path / "missing.wav", garbage):
        result = SoundFileUtil.load(str(path))
        assert isinstance(result, str) and result
