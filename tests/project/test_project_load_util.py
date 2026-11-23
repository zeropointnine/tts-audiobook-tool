"""
End-to-end project loading, for paths that cross machines.

`project.json` is written by whichever operating system saved it last. Loading
must reduce those values to the app's portable form, persist that upgrade, and
still never destroy a saved reference just because its file is not here yet.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME, PROJECT_SOUND_SEGMENTS_SUBDIR, PROJECT_VOICE_SUBDIR
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from project_settings_test_support import get_setting


def make_project_dir(tmp_path: Path, settings: dict[str, Any]) -> str:
    dir_path = tmp_path / "project"
    dir_path.mkdir()
    (dir_path / PROJECT_JSON_FILE_NAME).write_text(json.dumps(settings), encoding="utf-8")
    return str(dir_path)


def make_chatterbox_project_dir(tmp_path: Path, settings: dict[str, Any]) -> str:
    """
    The voice-file check resolves the model from the *project's* saved selection,
    not the app's active type, so the project has to name it.
    """
    return make_project_dir(tmp_path, {"tts_model_type": "chatterbox_local", **settings})


def make_voice_file(dir_path: str, file_name: str) -> None:
    voice_dir = Path(dir_path) / PROJECT_VOICE_SUBDIR
    voice_dir.mkdir(exist_ok=True)
    sf.write(str(voice_dir / file_name), np.zeros(100, dtype=np.float32), 16000)


def read_saved(dir_path: str) -> dict[str, Any]:
    with open(os.path.join(dir_path, PROJECT_JSON_FILE_NAME), "r", encoding="utf-8") as f:
        return json.load(f)


def load(dir_path: str) -> Any:
    project = ProjectLoadUtil.load_using_dir_path(dir_path, prompt_on_warnings=False)
    assert not isinstance(project, str), project
    return project


def test_load_reduces_paths_saved_by_another_os_and_persists_the_upgrade(
        tmp_path: Path,
) -> None:
    dir_path = make_chatterbox_project_dir(tmp_path, {
        "chatterbox_voice_file_name": [
            "C:\\Users\\lee\\mybook\\voice\\narrator.flac",
            "/home/lee/mybook/voice/second.flac",
        ],
    })
    make_voice_file(dir_path, "narrator.flac")
    make_voice_file(dir_path, "second.flac")

    project = load(dir_path)

    assert get_setting(project, "chatterbox_voice_file_name") == ["narrator.flac", "second.flac"]
    saved = read_saved(dir_path)
    assert saved["voice_references"] == [
        {"file_name": "narrator.flac", "transcript": "", "room_reverb": "off"},
        {"file_name": "second.flac", "transcript": "", "room_reverb": "off"},
    ]
    project.kill()


def test_load_keeps_a_reference_whose_file_has_not_arrived_yet(
        tmp_path: Path,
) -> None:
    """
    The file may simply not have been copied over, which is the normal state
    right after a project's settings travel from another computer. Clearing the
    reference and saving would destroy it permanently.
    """
    dir_path = make_chatterbox_project_dir(tmp_path, {"chatterbox_voice_file_name": ["gone.flac"]})

    project = load(dir_path)

    assert get_setting(project, "chatterbox_voice_file_name") == ["gone.flac"]
    saved = read_saved(dir_path)
    assert saved["voice_references"] == [{"file_name": "gone.flac", "transcript": "", "room_reverb": "off"}]
    project.kill()


def test_load_keeps_a_corrupt_reference_for_lazy_validation(
        tmp_path: Path,
) -> None:
    """
    Load-time voice verification is gone: a reference to an undecodable file
    survives loading untouched and surfaces through the pre-feature
    validate_voices flow instead of being dropped at open time.
    """
    dir_path = make_chatterbox_project_dir(tmp_path, {"chatterbox_voice_file_name": ["bad.flac"]})
    voice_dir = Path(dir_path) / PROJECT_VOICE_SUBDIR
    voice_dir.mkdir()
    (voice_dir / "bad.flac").write_bytes(b"not audio at all")

    project = load(dir_path)

    assert get_setting(project, "chatterbox_voice_file_name") == ["bad.flac"]
    assert "bad.flac" in json.dumps(read_saved(dir_path))
    project.kill()


def test_load_leaves_a_foreign_model_target_intact_but_says_something(
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
) -> None:
    """
    A `*_target` may hold a model repository id rather than a path, so it is
    never rewritten or cleared. The user still needs to be told it cannot work.
    """
    saved_target = "C:\\Users\\lee\\models\\VibeVoice"
    dir_path = make_project_dir(tmp_path, {"vibevoice_target": saved_target})

    project = load(dir_path)

    assert get_setting(project, "vibevoice_target") == saved_target
    saved = read_saved(dir_path)["model_settings"]["models"]["vibevoice_local"]
    assert saved["parameters"]["target"] == saved_target
    assert "vibevoice_local_target" in capsys.readouterr().out
    project.kill()


def test_load_ignores_a_dir_path_written_by_another_os(tmp_path: Path) -> None:
    r"""
    The saved `dir_path` is only a record; the directory being opened wins.
    Creating the saved one instead would litter the working directory with a
    folder literally named `C:\Users\lee\mybook`.
    """
    dir_path = make_project_dir(tmp_path, {"dir_path": "C:\\Users\\lee\\mybook"})

    project = load(dir_path)

    assert project.dir_path == dir_path
    assert not os.path.exists("C:\\Users\\lee\\mybook")
    assert sorted(item.name for item in Path(dir_path).iterdir()) == [
        PROJECT_JSON_FILE_NAME,
        # Settings without a version are v1.
        PROJECT_JSON_FILE_NAME + '.pre-v5.bak',
        PROJECT_SOUND_SEGMENTS_SUBDIR,
    ]
    project.kill()


@pytest.mark.parametrize("target", ["/runs/fw-leda/lora/checkpoint-step-000500", ""])
def test_load_drops_a_project_wide_breeze_lora_with_a_warning(
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        target: str,
) -> None:
    """
    Breeze TTS 2 LoRAs moved from the model settings to the voices. The old
    value isn't migrated: it's dropped (and the drop saved), and the user is
    told that LoRAs are now set per voice. An empty old value says nothing.
    """
    dir_path = make_project_dir(tmp_path, {
        "version": 5,
        "tts_model_type": "breeze_tts_2_local",
        "voice_references": [{"file_name": "", "transcript": "", "room_reverb": "off"}],
        "model_settings": {"models": {"breeze_tts_2_local": {"parameters": {
            "lora_target": target, "lora_strength": 0.5, "temperature": 0.7,
        }}}, "shared": {}},
    })

    project = load(dir_path)

    out = capsys.readouterr().out
    assert ("per-voice" in out and target in out) == bool(target)
    saved = read_saved(dir_path)["model_settings"]["models"].get("breeze_tts_2_local", {})
    parameters = saved.get("parameters", {})
    assert "lora_target" not in parameters and "lora_strength" not in parameters
    assert parameters["temperature"] == 0.7
    assert project.voice_references[0].loras == {}
    project.kill()
