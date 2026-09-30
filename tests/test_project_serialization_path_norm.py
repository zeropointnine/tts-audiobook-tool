"""
Path handling in project serialization.

Saved paths are written by one operating system and read back by another, so
the load side must reduce them to the app's portable form and the write side
must not persist anything machine-local. These tests pin both directions, and
pin the completeness of the field list that drives them.

Version 3 stores model-scoped values in `model_settings`; the flat voice fields
are accepted on load (the legacy migration) and never written back.
"""

from pathlib import Path
from typing import Any, cast

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings_declarations import (
    BUILTIN_LEGACY_FIELDS,
)
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from project_settings_test_support import set_setting

PATH_WARNING_MARKER = "portable form"


def path_warnings(warnings: list[str]) -> list[str]:
    """Only the warnings raised for rewritten paths; the load pass reports others."""
    return [warning for warning in warnings if PATH_WARNING_MARKER in warning]


def test_every_project_voice_file_name_field_is_normalized() -> None:
    """
    Guards the field list that both the load and save passes walk. A legacy
    `*_voice_file_name` field that is not registered would silently keep
    whatever grammar wrote it, and no flat voice field may remain on `Project`.
    """
    covered = set(ProjectSerializationUtil.get_project_local_path_field_names())
    declared = {
        name for name in BUILTIN_LEGACY_FIELDS if name.endswith("_voice_file_name")
    }

    assert declared <= covered
    # Only the project-wide placeholder stays at the project level.
    assert not [
        name for name in Project.model_fields
        if name.endswith("_voice_file_name") and name != "none_voice_file_name"
    ]


def test_normalize_loaded_project_dict_reduces_a_foreign_voice_list() -> None:
    warnings: list[str] = []
    d: dict[str, Any] = {
        "chatterbox_voice_file_name": [
            "C:\\Users\\lee\\mybook\\voice\\narrator.flac",
            "voice/second.flac",
        ],
    }

    ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=warnings)

    references = d["model_settings"]["models"]["chatterbox_local"]["voice_references"]
    assert references == [
        {"file_name": "narrator.flac", "transcript": ""},
        {"file_name": "voice/second.flac", "transcript": ""},
    ]
    assert "chatterbox_voice_file_name" not in d
    assert len(path_warnings(warnings)) == 1
    assert "chatterbox_voice_file_name" in path_warnings(warnings)[0]


def test_normalize_loaded_project_dict_reduces_scalar_path_fields() -> None:
    warnings: list[str] = []
    d: dict[str, Any] = {
        "none_voice_file_name": "/home/lee/mybook/voice/none.flac",
        "indextts2_emo_voice_file_name": "C:\\Users\\lee\\voice\\emo.flac",
    }

    ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=warnings)

    assert d["none_voice_file_name"] == "none.flac"
    assert d["model_settings"]["models"]["indextts2_local"]["files"]["emo_voice"] == "emo.flac"
    assert "indextts2_emo_voice_file_name" not in d
    assert len(path_warnings(warnings)) == 1
    assert "none_voice_file_name" in path_warnings(warnings)[0]
    assert "indextts2_emo_voice_file_name" in path_warnings(warnings)[0]


def test_normalize_loaded_project_dict_is_idempotent() -> None:
    d: dict[str, Any] = {"chatterbox_voice_file_name": ["C:\\Users\\lee\\voice\\a.flac"]}

    ProjectSerializationUtil.normalize_loaded_project_dict(d)
    once = dict(d)

    second_warnings: list[str] = []
    ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=second_warnings)

    assert d == once
    assert path_warnings(second_warnings) == []


def test_normalize_loaded_project_dict_leaves_transcript_text_alone() -> None:
    """
    Transcript fields share the voice-list serialization helper but hold text,
    not paths, so they must survive untouched into the paired reference.
    """
    text = "C:\\Users\\lee said \\hello\\ to the voice/actor."
    d: dict[str, Any] = {
        "fish_s1_voice_file_name": ["voice.flac"],
        "fish_s1_voice_transcript": [text],
    }

    ProjectSerializationUtil.normalize_loaded_project_dict(d)

    references = d["model_settings"]["models"]["fish_s1_local"]["voice_references"]
    assert references == [{"file_name": "voice.flac", "transcript": text}]


def test_normalize_loaded_project_dict_leaves_aliased_transcript_text_alone() -> None:
    """
    `glm_voice_text` is an older key for the transcript, not for a file name.
    The alias handling must pair the text with the voice file without treating
    the text as a path.
    """
    text = "C:\\Users\\lee said \\hello\\ to the voice/actor."
    d: dict[str, Any] = {
        "glm_voice_file_name": ["voice.flac"],
        "glm_voice_text": [text],
    }

    ProjectSerializationUtil.normalize_loaded_project_dict(d)

    references = d["model_settings"]["models"]["glm_local"]["voice_references"]
    assert references == [{"file_name": "voice.flac", "transcript": text}]
    assert "glm_voice_text" not in d
    assert "glm_voice_transcript" not in d


def test_to_project_json_dict_canonicalizes_values_never_loaded() -> None:
    """
    A `Project` can hold a machine-local path that never passed through the load
    funnel — set directly in memory by an older build. Saving must not persist
    it in that form.
    """
    project = Project.model_validate({})
    set_setting(project, "chatterbox_voice_file_name", cast(Any, [
        "C:\\Users\\lee\\mybook\\voice\\narrator.flac",
        "/home/lee/mybook/voice/other.flac",
        "voice/kept.flac",
    ]))

    result = ProjectSerializationUtil.to_project_json_dict(project)

    references = result["model_settings"]["models"]["chatterbox_local"]["voice_references"]
    assert [reference["file_name"] for reference in references] == [
        "narrator.flac",
        "other.flac",
        "voice/kept.flac",
    ]


def test_to_project_json_dict_keeps_dir_path_for_older_builds(tmp_path: Path) -> None:
    project = Project(dir_path=str(tmp_path))

    result = ProjectSerializationUtil.to_project_json_dict(project)

    assert result["dir_path"] == str(tmp_path)


def test_normalize_stored_path_values_drops_values_that_reduce_to_nothing() -> None:
    values, changed = ProjectSerializationUtil.normalize_stored_path_values(["C:\\", "a.flac"])

    assert values == ["a.flac"]
    assert changed
