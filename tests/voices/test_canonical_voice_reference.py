"""
Canonical voice references: local models are handed only PCM WAV/FLAC files
that libsndfile reads, so their own loaders (librosa, torchaudio) never need
system FFmpeg. App-imported samples pass through unchanged; other files are
decoded once by the app's PyAV decoder into a cached FLAC.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_fixture_support import dominant_frequency, sine, write_encoded
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound import canonical_reference
from tts_audiobook_tool.sound.canonical_reference import (
    VoiceReferenceError,
    canonical_reference_path,
    is_canonical_reference,
)
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

pytestmark = pytest.mark.no_ffmpeg_exe


@pytest.fixture
def cache_dir(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "cache"
    monkeypatch.setattr(canonical_reference, "default_cache_dir", lambda: str(path))
    return path


def cache_files(cache_dir: Path) -> list[str]:
    return sorted(os.listdir(cache_dir)) if cache_dir.exists() else []


@pytest.mark.parametrize("format, subtype", [("FLAC", "PCM_16"), ("WAV", "PCM_16"), ("WAV", "FLOAT"), ("FLAC", "PCM_24")])
def test_pcm_wav_and_flac_pass_through_unchanged(tmp_path, cache_dir, format, subtype):
    path = tmp_path / f"voice.{format.lower()}"
    sf.write(path, sine(220, 24000, 1.0), 24000, format=format, subtype=subtype)
    assert is_canonical_reference(str(path))
    assert canonical_reference_path(str(path)) == str(path)
    assert cache_files(cache_dir) == []


@pytest.mark.parametrize("codec, suffix", [("libmp3lame", ".mp3"), ("aac", ".m4a")])
@pytest.mark.parametrize("channels", [1, 2])
def test_compressed_source_becomes_cached_flac_with_same_audio(tmp_path, cache_dir, codec, suffix, channels):
    # MP3 is readable by libsndfile but is not canonical: torchaudio and
    # librosa versions differ in whether they read it without FFmpeg.
    source = tmp_path / f"voice ü{suffix}"
    tone = sine(330, 44100, 2.0)
    write_encoded(source, tone if channels == 1 else np.stack([tone, tone], axis=1), 44100, codec)
    assert not is_canonical_reference(str(source))

    result = canonical_reference_path(str(source))

    assert Path(result).parent == cache_dir
    assert result.endswith(".flac") and is_canonical_reference(result)
    info = sf.info(result)
    assert (info.samplerate, info.channels, info.subtype) == (44100, channels, "PCM_16")
    data, sr = sf.read(result, dtype="float32", always_2d=True)
    assert abs(len(data) / sr - 2.0) < 0.1
    assert abs(dominant_frequency(data[:, 0], sr) - 330) < 5
    assert cache_files(cache_dir) == [Path(result).name]


def test_unchanged_source_reuses_cached_file_and_clone_cache_key(tmp_path, cache_dir):
    # A model's voice-clone cache keys on (path, mtime, size): reusing the
    # cached file untouched keeps a prepared clone valid across generations.
    source = tmp_path / "voice.mp3"
    write_encoded(source, sine(330, 24000, 1.0), 24000, "libmp3lame")

    first = canonical_reference_path(str(source))
    key = TtsBaseModel._make_voice_clone_cache_key(first, "words")
    second = canonical_reference_path(str(source))

    assert second == first
    assert TtsBaseModel._make_voice_clone_cache_key(second, "words") == key


def test_changed_source_gets_new_cached_file_and_drops_old(tmp_path, cache_dir):
    # Editing the source invalidates the canonical copy, and so the model's
    # prepared clone, exactly as editing an app-imported FLAC would.
    source = tmp_path / "voice.m4a"
    write_encoded(source, sine(330, 24000, 1.0), 24000)
    old = canonical_reference_path(str(source))

    write_encoded(source, sine(550, 24000, 1.5), 24000)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    new = canonical_reference_path(str(source))

    assert new != old
    assert not os.path.exists(old)
    data, sr = sf.read(new, dtype="float32")
    assert abs(dominant_frequency(data, sr) - 550) < 5
    assert cache_files(cache_dir) == [Path(new).name]


def test_two_sources_get_separate_cached_files(tmp_path, cache_dir):
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    write_encoded(a, sine(330, 24000, 1.0), 24000, "libmp3lame")
    write_encoded(b, sine(550, 24000, 1.0), 24000, "libmp3lame")
    assert canonical_reference_path(str(a)) != canonical_reference_path(str(b))
    assert len(cache_files(cache_dir)) == 2


def test_undecodable_source_raises_and_leaves_nothing(tmp_path, cache_dir):
    source = tmp_path / "voice.mp3"
    source.write_bytes(b"not audio at all" * 100)
    with pytest.raises(VoiceReferenceError, match="Couldn't decode voice sample"):
        canonical_reference_path(str(source))
    assert cache_files(cache_dir) == []


def test_missing_source_raises_file_not_found(tmp_path, cache_dir):
    with pytest.raises(FileNotFoundError):
        canonical_reference_path(str(tmp_path / "missing.mp3"))


def make_project(tmp_path) -> Project:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return Project.model_validate({"dir_path": str(project_dir)})


def test_model_path_of_imported_sample_is_the_sample_itself(tmp_path, cache_dir):
    # The normal case: import already saved a canonical FLAC.
    project = make_project(tmp_path)
    sound = Sound(sine(220, APP_SAMPLE_RATE, 1.0), APP_SAMPLE_RATE)
    assert not ProjectVoiceUtil.set_voice_and_save(project, sound, "voice", "", TtsModelType.require_by_id("dots_local"))

    resolved = ProjectVoiceUtil.resolve_voice_file_path(project, "voice.flac")
    assert ProjectVoiceUtil.resolve_model_voice_file_path(project, "voice.flac") == resolved
    assert cache_files(cache_dir) == []


def test_model_path_of_legacy_compressed_sample_is_canonical(tmp_path, cache_dir):
    # Pre-FLAC versions of the app copied the user's file into the project
    # root as is; the model must still get a canonical file.
    project = make_project(tmp_path)
    legacy = Path(project.dir_path) / "narrator.m4a"
    write_encoded(legacy, sine(220, 24000, 1.0), 24000)

    path = ProjectVoiceUtil.resolve_model_voice_file_path(project, "narrator.m4a")

    assert Path(path).parent == cache_dir and is_canonical_reference(path)
    assert legacy.exists()


def test_model_path_of_missing_sample_is_left_to_the_model(tmp_path, cache_dir):
    project = make_project(tmp_path)
    expected = ProjectVoiceUtil.resolve_voice_file_path(project, "gone.mp3")
    assert ProjectVoiceUtil.resolve_model_voice_file_path(project, "gone.mp3") == expected


def test_default_cache_dir_is_under_app_temp_dir():
    from tts_audiobook_tool.app_support.app_paths import get_app_temp_dir
    assert canonical_reference.default_cache_dir() == os.path.join(get_app_temp_dir(), "voice_references")
