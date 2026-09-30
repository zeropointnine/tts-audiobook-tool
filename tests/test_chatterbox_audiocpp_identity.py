"""The audio.cpp Chatterbox rename preserves saved selections and overrides."""
from copy import deepcopy

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil


OLD_ID = "chatterbox_multilingual_audiocpp"
MODEL_ID = "chatterbox_audiocpp"


@pytest.mark.parametrize("selected_model", [OLD_ID, "echo_tts_audiocpp"])
def test_old_chatterbox_id_and_settings_migrate_and_round_trip(selected_model):
    overrides = {
        "parameters": {"temperature": 0.7, "top_p": 0.9, "seed": 123},
        "voice_references": [{"file_name": "voices/sample.wav"}],
    }
    source = {
        "version": 3,
        "tts_model_type": selected_model,
        "model_settings": {"models": {OLD_ID: deepcopy(overrides)}, "shared": {}},
    }

    project = Project.model_validate(source)

    expected_selection = MODEL_ID if selected_model == OLD_ID else selected_model
    assert project.tts_model_type == expected_selection
    assert project.get_tts_model_type().id == expected_selection
    assert OLD_ID not in project.model_settings.models
    assert project.get_model_setting(MODEL_ID, "temperature") == 0.7
    assert project.get_model_setting(MODEL_ID, "top_p") == 0.9
    assert project.get_model_setting(MODEL_ID, "seed") == 123
    assert project.get_model_setting(MODEL_ID, "file_name") == ["voices/sample.wav"]

    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["tts_model_type"] == expected_selection
    assert OLD_ID not in saved["model_settings"]["models"]
    restored = Project.model_validate(saved)
    assert restored.tts_model_type == expected_selection
    assert OLD_ID not in restored.model_settings.models
    assert restored.model_settings.models[MODEL_ID] == project.model_settings.models[MODEL_ID]


def test_current_chatterbox_settings_win_without_losing_old_overrides():
    source = {
        "tts_model_type": OLD_ID,
        "model_settings": {"models": {
            OLD_ID: {
                "parameters": {"temperature": 0.7, "seed": 123},
                "voice_references": [{"file_name": "voices/sample.wav"}],
            },
            MODEL_ID: {"parameters": {"temperature": 0.9}},
        }},
    }

    ProjectLoadUtil.remap_legacy_keys(source)

    assert source["tts_model_type"] == MODEL_ID
    assert source["model_settings"]["models"] == {MODEL_ID: {
        "parameters": {"temperature": 0.9, "seed": 123},
        "voice_references": [{"file_name": "voices/sample.wav"}],
    }}
    migrated = deepcopy(source)
    ProjectLoadUtil.remap_legacy_keys(source)
    assert source == migrated


def test_current_chatterbox_voice_references_take_precedence():
    source = {"model_settings": {"models": {
        OLD_ID: {"voice_references": [{"file_name": "old.wav"}]},
        MODEL_ID: {"voice_references": [{"file_name": "new.wav"}]},
    }}}

    project = Project.model_validate(source)

    assert OLD_ID not in project.model_settings.models
    assert project.get_model_setting(MODEL_ID, "file_name") == ["new.wav"]
