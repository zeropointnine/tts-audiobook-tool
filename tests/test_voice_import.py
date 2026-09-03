"""
Tests for voice clone import treatment: app-native 48 kHz resampling,
storage in the project's voice subdir with undecorated file names, and
voice/first path resolution with legacy project-root fallback.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE, PROJECT_VOICE_SUBDIR
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

L.init("test-voice-import")


def make_project(tmp_path) -> Project:
    return Project.model_validate({"dir_path": str(tmp_path)})


def make_sound(sr: int, amplitude: float = 0.1) -> Sound:
    return Sound(np.full(1000, amplitude, dtype=np.float32), sr)


def test_post_processing_resamples_to_app_rate():
    sound = SoundPipeline.apply_voice_clone_post_processing(make_sound(44_100, amplitude=0.5))
    assert sound.sr == APP_SAMPLE_RATE
    # Peak normalization with headroom applied
    assert np.max(np.abs(sound.data)) > 0.1


def test_post_processing_passes_app_rate_through():
    sound = make_sound(APP_SAMPLE_RATE, amplitude=0.5)
    result = SoundPipeline.apply_voice_clone_post_processing(sound)
    assert result.sr == APP_SAMPLE_RATE
    assert len(result.data) == len(sound.data)


def test_post_processing_trims_leading_and_trailing_silence():
    # 1s silence, 0.5s voice, 1s silence at app rate
    sr = APP_SAMPLE_RATE
    data = np.concatenate([
        np.zeros(sr, dtype=np.float32),
        np.full(sr // 2, 0.5, dtype=np.float32),
        np.zeros(sr, dtype=np.float32),
    ])
    result = SoundPipeline.apply_voice_clone_post_processing(Sound(data, sr))
    assert result.sr == sr
    # Tolerance for the trimmer's detection window
    assert abs(len(result.data) - sr // 2) < sr // 10


def test_post_processing_entirely_silent_input_returns_empty():
    result = SoundPipeline.apply_voice_clone_post_processing(make_sound(44_100, amplitude=0.0))
    assert result.data.size == 0


def test_set_voice_and_save_writes_untagged_file_in_voice_subdir(tmp_path):
    project = make_project(tmp_path)
    err = ProjectVoiceUtil.set_voice_and_save(
        project, make_sound(APP_SAMPLE_RATE), "myvoice", "hello", TtsModelType.DOTS
    )
    assert not err

    expected = tmp_path / PROJECT_VOICE_SUBDIR / "myvoice.flac"
    assert expected.exists()
    assert not (tmp_path / "myvoice.flac").exists()
    assert project.dots_voice_file_name == ["myvoice.flac"]
    assert project.dots_voice_transcript == ["hello"]


def test_set_voice_and_save_reimport_same_model_overwrites(tmp_path):
    project = make_project(tmp_path)
    ProjectVoiceUtil.set_voice_and_save(project, make_sound(APP_SAMPLE_RATE), "myvoice", "", TtsModelType.DOTS)
    ProjectVoiceUtil.set_voice_and_save(project, make_sound(APP_SAMPLE_RATE), "myvoice", "", TtsModelType.DOTS)
    assert project.dots_voice_file_name == ["myvoice.flac"]


def test_set_voice_and_save_disambiguates_cross_model_stem_collision(tmp_path):
    project = make_project(tmp_path)
    ProjectVoiceUtil.set_voice_and_save(project, make_sound(APP_SAMPLE_RATE), "myvoice", "", TtsModelType.DOTS)
    ProjectVoiceUtil.set_voice_and_save(project, make_sound(APP_SAMPLE_RATE), "myvoice", "", TtsModelType.MIRA)

    assert project.dots_voice_file_name == ["myvoice.flac"]
    assert project.mira_voice_file_name == ["myvoice_2.flac"]
    assert (tmp_path / PROJECT_VOICE_SUBDIR / "myvoice.flac").exists()
    assert (tmp_path / PROJECT_VOICE_SUBDIR / "myvoice_2.flac").exists()


def test_set_voice_and_save_append_same_stem_gets_distinct_name(tmp_path):
    project = make_project(tmp_path)
    ProjectVoiceUtil.set_voice_and_save(project, make_sound(APP_SAMPLE_RATE), "myvoice", "", TtsModelType.DOTS)
    ProjectVoiceUtil.set_voice_and_save(
        project, make_sound(APP_SAMPLE_RATE), "myvoice", "", TtsModelType.DOTS, append=True
    )
    assert project.dots_voice_file_name == ["myvoice.flac", "myvoice_2.flac"]


def test_set_voice_and_save_indextts2_emo_voice_untagged(tmp_path):
    project = make_project(tmp_path)
    ProjectVoiceUtil.set_voice_and_save(
        project, make_sound(APP_SAMPLE_RATE), "emo", "", TtsModelType.INDEXTTS2, is_secondary=True
    )
    assert project.indextts2_emo_voice_file_name == "emo.flac"
    assert (tmp_path / PROJECT_VOICE_SUBDIR / "emo.flac").exists()


def test_resolve_voice_file_path_prefers_subdir(tmp_path):
    project = make_project(tmp_path)
    voice_dir = tmp_path / PROJECT_VOICE_SUBDIR
    voice_dir.mkdir()
    sf.write(str(voice_dir / "a.flac"), np.zeros(100, dtype=np.float32), APP_SAMPLE_RATE)
    # Legacy root copy also exists; subdir must win
    sf.write(str(tmp_path / "a.flac"), np.zeros(100, dtype=np.float32), APP_SAMPLE_RATE)

    path = ProjectVoiceUtil.resolve_voice_file_path(project, "a.flac")
    assert path == os.path.join(str(voice_dir), "a.flac")


def test_resolve_voice_file_path_falls_back_to_project_root(tmp_path):
    project = make_project(tmp_path)
    sf.write(str(tmp_path / "legacy_dots.flac"), np.zeros(100, dtype=np.float32), APP_SAMPLE_RATE)

    path = ProjectVoiceUtil.resolve_voice_file_path(project, "legacy_dots.flac")
    assert path == str(tmp_path / "legacy_dots.flac")


def test_verify_voice_files_exist_accepts_legacy_root_placement(tmp_path, capsys):
    project = make_project(tmp_path)
    sf.write(str(tmp_path / "legacy_dots.flac"), np.zeros(100, dtype=np.float32), APP_SAMPLE_RATE)
    project.dots_voice_file_name = ["legacy_dots.flac"]

    warned = ProjectVoiceUtil.verify_voice_files_exist(project)
    assert not warned
    assert project.dots_voice_file_name == ["legacy_dots.flac"]
