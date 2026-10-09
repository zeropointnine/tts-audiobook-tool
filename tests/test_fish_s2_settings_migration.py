"""Retire historical Fish S2 local/server sharing without changing effective settings."""
from copy import deepcopy
import json

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.model_settings_compat import RETIRED_GROUPS, SHARED_MEMBERS
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions


IDS = SHARED_MEMBERS["fish_s2"]
SAMPLING = {"temperature": 0.8, "top_p": 0.85, "top_k": 50}
REFERENCES = [{"file_name": "one.flac", "transcript": "one"},
              {"file_name": "two.flac", "transcript": ""}]


def old_source():
    return {"models": {}, "shared": {"fish_s2": {
        "model_ids": list(IDS), "parameters": {**SAMPLING, "seed": 7},
        "voice_references": deepcopy(REFERENCES),
    }}}


def test_group_is_retired_and_server_has_no_seed():
    assert "fish_s2" in RETIRED_GROUPS
    assert "fish_s2" not in REGISTRY.members
    for model_id in IDS:
        for name in SAMPLING:
            assert REGISTRY.get(model_id, name).group == ""
    # Server Fish S2 never sends a seed, so it no longer stores one.
    assert ("fish_s2_local", "seed") in REGISTRY.bindings
    assert ("fish_s2_sglomni", "seed") not in REGISTRY.bindings


def test_shared_fish_s2_forks_to_both_members_and_seed_only_to_local():
    source = old_source()
    source["models"]["fish_s2_local"] = {"parameters": {"compile_enabled": False, "rolling_cont": 2}}
    source["models"]["fish_s2_sglomni"] = {"orchestration": {"concurrent_requests": 3}}
    before = deepcopy(source)
    settings = REGISTRY.reconcile(source)
    assert source == before
    assert "fish_s2" not in settings.shared
    assert settings.models["fish_s2_local"]["parameters"] == {
        **SAMPLING, "seed": 7, "compile_enabled": False, "rolling_cont": 2}
    assert settings.models["fish_s2_sglomni"]["parameters"] == SAMPLING
    assert settings.models["fish_s2_sglomni"]["orchestration"] == {"concurrent_requests": 3}
    # The fork is idempotent: reconciling migrated output changes nothing.
    assert REGISTRY.reconcile(settings).to_dict() == settings.to_dict()


def test_migrated_values_resolve_identically_for_local_and_server():
    project = Project.model_validate({"version": 3, "model_settings": old_source()})
    definition = load_definitions().models["fish_s2_sglomni"]
    for name, value in SAMPLING.items():
        assert project.get_model_setting("fish_s2_local", name) == value
        assert ConfiguredSettings.get(project, definition.parameters[name], "fish_s2_sglomni") == value
    assert project.get_model_setting("fish_s2_local", "seed") == 7
    assert project.voice_references == REFERENCES


@pytest.mark.parametrize("reset", [None, -1.0, 0.5])
def test_raw_private_parameter_keys_win_including_default_intent(reset):
    # A private key present before the fork (even null or the sentinel) is the
    # user's newer intent and must not be overwritten by the old shared value.
    source = old_source()
    source["models"]["fish_s2_sglomni"] = {"parameters": {"temperature": reset}}
    project = Project.model_validate({"version": 3, "model_settings": source})
    expected = -1.0 if reset is None else reset
    assert project.get_model_setting("fish_s2_sglomni", "temperature") == expected
    assert project.get_model_setting("fish_s2_sglomni", "top_k") == 50
    assert project.get_model_setting("fish_s2_local", "temperature") == 0.8
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "fish_s2" not in saved["model_settings"]["shared"]
    restored = Project.model_validate(saved)
    assert restored.get_model_setting("fish_s2_sglomni", "temperature") == expected


@pytest.mark.parametrize("version", [1, 2])
def test_flat_fish_s2_fields_fork_with_private_local_and_server_fields(version):
    project = Project.model_validate({
        "version": version, "tts_model_type": "fish_s2_sglomni",
        "fish_s2_temperature": 0.8, "fish_s2_top_p": 0.85, "fish_s2_top_k": 50,
        "fish_s2_seed": -1, "fish_s2_compile_enabled": False, "fish_s2_rolling_cont": 2,
        "fish_s2_server_concurrent_requests": 3,
        "fish_s2_voice_file_name": ["one.flac", "two.flac"], "fish_s2_voice_transcript": ["one"],
    })
    for model_id in IDS:
        for name, value in SAMPLING.items():
            assert project.get_model_setting(model_id, name) == value
    assert project.get_model_setting("fish_s2_local", "seed") == -1
    assert project.get_model_setting("fish_s2_local", "compile_enabled") is False
    assert project.get_model_setting("fish_s2_local", "rolling_cont") == 2
    assert project.get_model_setting("fish_s2_sglomni", "concurrent_requests") == 3
    # The flat list is one source even though two models historically read it.
    assert project.voice_references == REFERENCES
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "fish_s2" not in saved["model_settings"]["shared"]
    assert not any(key.startswith("fish_s2_") for key in saved)


def test_flat_default_sampling_stays_unset_but_random_seed_is_preserved():
    settings = REGISTRY.reconcile(None, legacy={
        "fish_s2_temperature": -1.0, "fish_s2_top_k": -1, "fish_s2_seed": -1,
    })
    assert settings.models["fish_s2_local"] == {"parameters": {"seed": -1}}
    assert "fish_s2_sglomni" not in settings.models or settings.models["fish_s2_sglomni"] == {}
    saved = REGISTRY.serialize(settings)
    assert saved["models"]["fish_s2_sglomni"]["parameters"]["temperature"] is None
    assert saved["models"]["fish_s2_local"]["parameters"]["top_k"] is None


def test_shared_source_wins_over_flat_remnants():
    source = {"shared": {"fish_s2": {"model_ids": list(IDS), "parameters": {}}}}
    settings = REGISTRY.reconcile(source, legacy={"fish_s2_top_k": 77, "fish_s2_seed": 42})
    project = Project(model_settings=settings, voice_references=[])
    for model_id in IDS:
        assert project.get_model_setting(model_id, "top_k") == -1
    assert project.get_model_setting("fish_s2_local", "seed") == -1


def test_edits_after_fork_are_independent_and_do_not_resurrect():
    project = Project.model_validate({"version": 3, "model_settings": old_source()})
    project.set_model_setting("fish_s2_sglomni", "top_k", 20)
    project.set_model_setting("fish_s2_local", "temperature", None, reset=True)
    assert project.get_model_setting("fish_s2_local", "top_k") == 50
    assert project.get_model_setting("fish_s2_sglomni", "temperature") == 0.8
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    for _ in range(2):
        project = Project.model_validate(deepcopy(saved))
        assert project.get_model_setting("fish_s2_local", "temperature") == -1.0
        assert project.get_model_setting("fish_s2_sglomni", "top_k") == 20
        assert "fish_s2" not in project.model_settings.shared
        assert ProjectSerializationUtil.to_project_json_dict(project) == saved


def test_out_of_bounds_shared_value_resets_for_each_member(caplog):
    # Local Fish S2 borrows the server's catalog bounds, so a value invalid for
    # the old shared group stays invalid for both private copies.
    source = old_source()
    source["shared"]["fish_s2"]["parameters"]["top_k"] = 101
    project = Project.model_validate({"version": 3, "model_settings": source})
    for model_id in IDS:
        assert project.get_model_setting(model_id, "top_k") == -1
        assert project.get_model_setting(model_id, "temperature") == 0.8
    assert "resetting to default" in caplog.text


@pytest.mark.parametrize("member_ids", [None, [*IDS, "unauthorized"], [IDS[0]] * 2])
def test_bad_historical_membership_is_non_destructive_on_load_and_save(member_ids):
    source = old_source()
    source["shared"]["fish_s2"].pop("voice_references")
    source["shared"]["fish_s2"]["model_ids"] = member_ids
    before = deepcopy(source)
    with pytest.raises(ValueError, match="shared.fish_s2.model_ids"):
        REGISTRY.reconcile(source)
    assert source == before
    warnings = []
    project = Project.model_validate({"version": 3, "model_settings": source}, context={"warnings": warnings})
    assert any("shared.fish_s2.model_ids" in warning for warning in warnings)
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["model_settings"]["shared"]["fish_s2"] == before["shared"]["fish_s2"]


def test_disk_load_of_v4_project_forks_in_memory_without_backup(tmp_path):
    # A v4 project already has its voice list, so loading does not force a
    # rewrite; the fork happens in memory and lands on disk at the next save.
    payload = ProjectSerializationUtil.to_project_json_dict(Project(tts_model_type="fish_s2_sglomni"))
    for model_id in IDS:
        payload["model_settings"]["models"].pop(model_id, None)
    old = old_source()["shared"]["fish_s2"]
    old.pop("voice_references")
    payload["model_settings"]["shared"]["fish_s2"] = old
    path = tmp_path / "project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    project = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(project, Project)
    assert project.get_model_setting("fish_s2_sglomni", "top_k") == 50
    assert project.get_model_setting("fish_s2_local", "seed") == 7
    assert "fish_s2" not in project.model_settings.shared
    assert path.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.bak"))
