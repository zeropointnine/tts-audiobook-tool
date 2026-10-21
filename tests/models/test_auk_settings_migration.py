"""Retire historical AuK / AuK-Flash sharing without changing effective settings."""
from copy import deepcopy
import json

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.model_settings_compat import SHARED_MEMBERS
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions


IDS = SHARED_MEMBERS["auk"]
AUK, FLASH = "auk_sglomni", "auk_flash_sglomni"
REFERENCES = [{"file_name": "one.flac", "transcript": "one"},
              {"file_name": "two.flac", "transcript": ""}]
VOICES = [Voice(**ref) for ref in REFERENCES]


def old_source():
    return {"models": {}, "shared": {"auk": {
        "model_ids": list(IDS), "parameters": {"speed": 1.3, "seed": 7},
        "orchestration": {"concurrent_requests": 3},
        "voice_references": deepcopy(REFERENCES),
    }}}


def test_group_is_retired_and_both_members_own_their_settings():
    assert IDS == (AUK, FLASH)
    for model_id in IDS:
        for name in ("speed", "seed", "concurrent_requests"):
            assert REGISTRY.get(model_id, name).group == ""


def test_shared_auk_forks_every_value_to_both_members():
    source = old_source()
    before = deepcopy(source)
    settings = REGISTRY.reconcile(source)
    assert source == before
    assert "auk" not in settings.shared
    for model_id in IDS:
        assert settings.models[model_id]["parameters"] == {"speed": 1.3, "seed": 7}
        assert settings.models[model_id]["orchestration"] == {"concurrent_requests": 3}
    # The fork is idempotent: reconciling migrated output changes nothing.
    assert REGISTRY.reconcile(settings).to_dict() == settings.to_dict()


def test_migrated_values_resolve_identically_for_both_variants():
    project = Project.model_validate({"version": 3, "model_settings": old_source()})
    definitions = load_definitions().models
    for model_id in IDS:
        assert ConfiguredSettings.get(project, definitions[model_id].parameters["speed"], model_id) == 1.3
        assert project.get_model_setting(model_id, "seed") == 7
        assert project.get_model_setting(model_id, "concurrent_requests") == 3
    assert project.voice_references == VOICES


@pytest.mark.parametrize("reset", [None, 1.0, 0.7])
def test_raw_private_parameter_keys_win_including_default_intent(reset):
    # A private key present before the fork (even null or the default) is the
    # user's newer intent and must not be overwritten by the old shared value.
    source = old_source()
    source["models"][FLASH] = {"parameters": {"speed": reset}}
    project = Project.model_validate({"version": 3, "model_settings": source})
    expected = 1.0 if reset is None else reset
    assert project.get_model_setting(FLASH, "speed") == expected
    assert project.get_model_setting(FLASH, "seed") == 7
    assert project.get_model_setting(AUK, "speed") == 1.3
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "auk" not in saved["model_settings"]["shared"]
    restored = Project.model_validate(saved)
    assert restored.get_model_setting(FLASH, "speed") == expected


@pytest.mark.parametrize("version", [1, 2])
def test_flat_auk_fields_fork_to_both_variants(version):
    project = Project.model_validate({
        "version": version, "tts_model_type": FLASH,
        "auk_speed": 1.4, "auk_seed": 9, "auk_server_concurrent_requests": 2,
        "auk_voice_file_name": ["one.flac", "two.flac"], "auk_voice_transcript": ["one"],
    })
    for model_id in IDS:
        assert project.get_model_setting(model_id, "speed") == 1.4
        assert project.get_model_setting(model_id, "seed") == 9
        assert project.get_model_setting(model_id, "concurrent_requests") == 2
    # The flat list is one source even though two models historically read it.
    assert project.voice_references == VOICES
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "auk" not in saved["model_settings"]["shared"]
    assert not any(key.startswith("auk_") for key in saved)


def test_flat_default_speed_stays_unset_but_random_seed_is_preserved():
    settings = REGISTRY.reconcile(None, legacy={"auk_speed": 1.0, "auk_seed": -1})
    for model_id in IDS:
        assert settings.models[model_id] == {"parameters": {"seed": -1}}
    saved = REGISTRY.serialize(settings)
    assert saved["models"][AUK]["parameters"]["speed"] is None


def test_shared_source_wins_over_flat_remnants():
    source = {"shared": {"auk": {"model_ids": list(IDS), "parameters": {}}}}
    settings = REGISTRY.reconcile(source, legacy={"auk_speed": 1.7, "auk_seed": 42})
    project = Project(model_settings=settings, voice_references=[])
    for model_id in IDS:
        assert project.get_model_setting(model_id, "speed") == 1.0
        assert project.get_model_setting(model_id, "seed") == -1


def test_edits_after_fork_are_independent_and_do_not_resurrect():
    project = Project.model_validate({"version": 3, "model_settings": old_source()})
    project.set_model_setting(FLASH, "speed", 0.8)
    project.set_model_setting(FLASH, "seed", 11)
    project.set_model_setting(AUK, "speed", None, reset=True)
    assert project.get_model_setting(AUK, "seed") == 7
    assert project.get_model_setting(FLASH, "concurrent_requests") == 3
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    for _ in range(2):
        project = Project.model_validate(deepcopy(saved))
        assert project.get_model_setting(AUK, "speed") == 1.0
        assert project.get_model_setting(FLASH, "speed") == 0.8
        assert project.get_model_setting(FLASH, "seed") == 11
        assert "auk" not in project.model_settings.shared
        assert ProjectSerializationUtil.to_project_json_dict(project) == saved


def test_out_of_bounds_shared_value_resets_for_each_member(caplog):
    # Each variant validates its private copy with its own catalog bounds.
    source = old_source()
    source["shared"]["auk"]["parameters"]["speed"] = 3.0
    project = Project.model_validate({"version": 3, "model_settings": source})
    for model_id in IDS:
        assert project.get_model_setting(model_id, "speed") == 1.0
        assert project.get_model_setting(model_id, "seed") == 7
        assert f"models.{model_id}.parameters.speed" in caplog.text
    assert "resetting to default" in caplog.text


@pytest.mark.parametrize("member_ids", [None, [*IDS, "unauthorized"], [IDS[0]] * 2])
def test_bad_historical_membership_is_non_destructive_on_load_and_save(member_ids):
    source = old_source()
    source["shared"]["auk"].pop("voice_references")
    source["shared"]["auk"]["model_ids"] = member_ids
    before = deepcopy(source)
    with pytest.raises(ValueError, match="shared.auk.model_ids"):
        REGISTRY.reconcile(source)
    assert source == before
    warnings = []
    project = Project.model_validate({"version": 3, "model_settings": source}, context={"warnings": warnings})
    assert any("shared.auk.model_ids" in warning for warning in warnings)
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["model_settings"]["shared"]["auk"] == before["shared"]["auk"]


def test_disk_load_of_current_project_forks_in_memory_without_backup(tmp_path):
    # A current-version project already has its voice list, so loading does not
    # force a rewrite; the fork happens in memory and lands on disk at next save.
    payload = ProjectSerializationUtil.to_project_json_dict(Project(tts_model_type=FLASH))
    for model_id in IDS:
        payload["model_settings"]["models"].pop(model_id, None)
    old = old_source()["shared"]["auk"]
    old.pop("voice_references")
    payload["model_settings"]["shared"]["auk"] = old
    path = tmp_path / "project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    project = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(project, Project)
    assert project.get_model_setting(FLASH, "speed") == 1.3
    assert project.get_model_setting(AUK, "seed") == project.get_model_setting(FLASH, "seed") == 7
    assert "auk" not in project.model_settings.shared
    assert path.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.bak"))
