"""Retire historical MOSS sharing without changing effective user settings."""
from copy import deepcopy
import json

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import ModelSettingsRegistry, REGISTRY
from tts_audiobook_tool.project_support.model_settings_compat import SHARED_MEMBERS
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions


IDS = SHARED_MEMBERS["moss"]
SAMPLING = {
    "delay_temperature": 1.9, "delay_top_p": 0.77, "delay_top_k": 32,
    "local_temperature": 1.2, "local_top_p": 0.91, "local_top_k": 58,
}
REFERENCES = [{"file_name": "one.flac", "transcript": "one"},
              {"file_name": "two.flac", "transcript": ""}]


def old_source():
    return {"models": {}, "shared": {"moss": {
        "model_ids": list(IDS), "parameters": {**SAMPLING, "seed": 0},
        "orchestration": {"batch_size": 3}, "voice_references": deepcopy(REFERENCES),
    }}}


def test_shared_moss_forks_only_applicable_settings_and_preserves_source():
    source = old_source()
    source["models"]["moss_local"] = {"parameters": {
        "target": "OpenMOSS-Team/MOSS-TTS-Local-Transformer", "rolling_cont": 2,
    }}
    # Historical shared parameters never owned these local-only fields.
    source["shared"]["moss"]["parameters"].update(target="ignored", rolling_cont=1)
    before = deepcopy(source)
    settings = REGISTRY.reconcile(source)
    assert source == before
    assert "moss" not in settings.shared
    for model_id in IDS:
        obj = settings.models[model_id]
        expected = {name: value for name, value in SAMPLING.items()
                    if model_id == "moss_local" or name.startswith(
                        "delay_" if model_id == "moss_delay_sglomni" else "local_")}
        if model_id == "moss_local":
            expected.update(delay_seed=0, local_seed=0, local_v15_seed=0)
        else:
            expected["seed"] = 0
        if model_id == "moss_local":
            expected.update(target="OpenMOSS-Team/MOSS-TTS-Local-Transformer", rolling_cont=2)
        assert obj["parameters"] == expected
        assert obj["orchestration"] == {"batch_size": 3}
        assert obj["voice_references"] == REFERENCES
    assert REGISTRY.reconcile(settings).to_dict() == settings.to_dict()


def test_migrated_values_resolve_identically_in_local_and_server_generation():
    project = Project.model_validate({"version": 3, "model_settings": old_source()})
    definitions = load_definitions().models
    for config, model_id, expected in (
            (MossConfigs.DELAY, "moss_delay_sglomni", (1.9, 0.77, 32)),
            (MossConfigs.LOCAL, "moss_local_sglomni", (1.2, 0.91, 58))):
        assert MossBaseModel.get_generation_params(project, config) == expected
        assert tuple(ConfiguredSettings.get(project, parameter, model_id)
                     for parameter in definitions[model_id].parameters.values()) == expected
    # The historical shared seed had no v1.5-specific tuning; its forked
    # v1.5 sampling values resolve to the preset's model-card defaults.
    assert MossBaseModel.get_generation_params(project, MossConfigs.LOCAL_V15) == (1.7, 0.8, 25)


@pytest.mark.parametrize("reset", [None, -1.0, 1.7])
def test_raw_private_parameter_keys_win_including_default_intent(reset):
    source = old_source()
    source["models"]["moss_delay_sglomni"] = {"parameters": {"delay_temperature": reset}}
    project = Project.model_validate({"version": 3, "model_settings": source})
    assert project.get_model_setting("moss_delay_sglomni", "delay_temperature") == (
        -1.0 if reset is None else reset)
    assert project.get_model_setting("moss_delay_sglomni", "delay_top_k") == 32
    assert project.get_model_setting("moss_local", "delay_temperature") == 1.9
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "moss" not in saved["model_settings"]["shared"]
    restored = Project.model_validate(saved)
    assert restored.get_model_setting("moss_delay_sglomni", "delay_temperature") == (
        -1.0 if reset is None else reset)


def test_private_voice_list_seed_and_orchestration_win_without_merging():
    source = old_source()
    source["models"]["moss_delay_sglomni"] = {
        "voice_references": [], "parameters": {"seed": -1}, "orchestration": {"batch_size": 1},
    }
    new_references = [{"file_name": "new.flac", "transcript": "new"}]
    source["models"]["moss_local_sglomni"] = {"voice_references": new_references}
    # V4 counts each independent legacy scoped list as its own source, so this
    # combination is an ambiguity rather than something to merge.
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="model-list choice"):
        Project.model_validate({"version": 3, "model_settings": deepcopy(source)})
    # With the historical shared list out of the picture, the remaining private
    # list is adopted as the single project list, and private non-voice
    # settings still win over the shared group's.
    source["shared"]["moss"].pop("voice_references")
    project = Project.model_validate({"version": 3, "model_settings": source})
    assert project.voice_references == new_references
    assert project.get_model_setting("moss_delay_sglomni", "seed") == -1
    assert project.get_model_setting("moss_delay_sglomni", "batch_size") == 1


@pytest.mark.parametrize("version", [1, 2])
def test_flat_moss_fields_and_aliases_fork_with_private_target_and_continuation(version):
    project = Project.model_validate({
        "version": version, "tts_model_type": "moss_local_sglomni",
        "moss_temperature": 1.9, "moss_top_p": 0.77, "moss_top_k": 32,
        "moss_local_temperature": 1.2, "moss_local_top_p": 0.91, "moss_local_top_k": 58,
        "moss_voice_file_name": ["one.flac", "two.flac"], "moss_voice_transcript": ["one"],
        "moss_seed": -1, "moss_batch_size": 3,
        "moss_target": "OpenMOSS-Team/MOSS-TTS-Local-Transformer", "moss_rolling_cont": 2,
    })
    assert project.tts_model_type == "moss_local_sglomni"
    assert project.get_model_setting("moss_local", "target") == "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
    assert project.get_model_setting("moss_local", "rolling_cont") == 2
    for model_id in IDS:
        if model_id == "moss_local":
            for name in ("delay_seed", "local_seed", "local_v15_seed"):
                assert project.get_model_setting(model_id, name) == -1
        else:
            assert project.get_model_setting(model_id, "seed") == -1
        assert project.get_model_setting(model_id, "batch_size") == 3
        for name, value in SAMPLING.items():
            if (model_id, name) in REGISTRY.bindings:
                assert project.get_model_setting(model_id, name) == value
    # The flat list is one source even though three models historically read it.
    assert project.voice_references == REFERENCES
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "moss" not in saved["model_settings"]["shared"]
    assert not any(key.startswith("moss_") for key in saved)


def test_flat_default_sampling_stays_unset_but_random_seed_is_preserved():
    settings = REGISTRY.reconcile(None, legacy={
        "moss_delay_temperature": -1.0, "moss_local_top_k": -1,
        "moss_batch_size": 1, "moss_seed": -1,
    })
    for model_id in IDS:
        expected_seeds = ({"delay_seed": -1, "local_seed": -1, "local_v15_seed": -1}
                          if model_id == "moss_local" else {"seed": -1})
        assert settings.models[model_id] == {"parameters": expected_seeds}
    saved = REGISTRY.serialize(settings)
    assert saved["models"]["moss_local"]["parameters"]["delay_temperature"] is None
    assert saved["models"]["moss_local_sglomni"]["parameters"]["local_top_k"] is None


def test_shared_source_wins_over_flat_remnants_even_when_empty():
    source = {"shared": {"moss": {"model_ids": list(IDS), "parameters": {}, "voice_references": []}}}
    settings = REGISTRY.reconcile(source, legacy={
        "moss_delay_top_k": 77, "moss_voice_file_name": ["stale.flac"], "moss_seed": 42,
        "moss_target": "local-checkpoint", "moss_rolling_cont": 1,
    })
    project = Project(model_settings=settings)
    for model_id in IDS:
        assert project.get_model_setting(model_id, "file_name") == []
        if model_id == "moss_local":
            for name in ("delay_seed", "local_seed", "local_v15_seed"):
                assert project.get_model_setting(model_id, name) == -1
        else:
            assert project.get_model_setting(model_id, "seed") == -1
    assert project.get_model_setting("moss_delay_sglomni", "delay_top_k") == -1
    assert project.get_model_setting("moss_local", "target") == "local-checkpoint"
    assert project.get_model_setting("moss_local", "rolling_cont") == 1


def test_flat_fallback_fills_only_missing_private_keys_without_mutation():
    source = {"models": {"moss_local": {"parameters": {"target": "new-checkpoint"}},
                         "moss_delay_sglomni": {"parameters": {"delay_top_k": None}, "voice_references": []}}}
    legacy = {"moss_delay_top_k": 32, "moss_seed": 0, "moss_voice_file_name": "old.flac",
              "moss_voice_transcript": "old", "moss_target": "stale-checkpoint", "moss_rolling_cont": 1}
    before = deepcopy((source, legacy))
    settings = REGISTRY.reconcile(source, legacy=legacy)
    # The scoped fork this direct call performs is legacy storage, so supply
    # the authoritative list explicitly to avoid a migration ambiguity.
    project = Project(model_settings=settings, voice_references=[])
    assert project.get_model_setting("moss_local", "delay_top_k") == 32
    assert project.get_model_setting("moss_local", "target") == "new-checkpoint"
    # Preserve existing whole-private-object precedence for legacy local-only fields.
    assert project.get_model_setting("moss_local", "rolling_cont") == 0
    assert project.get_model_setting("moss_delay_sglomni", "delay_top_k") == -1
    # V4 project voice access reads only the project-wide list; the scoped
    # fork that reconcile still performs for this direct call is not
    # authoritative project storage.
    assert project.get_model_setting("moss_delay_sglomni", "file_name") == []
    assert project.get_model_setting("moss_local_sglomni", "file_name") == []
    assert project.voice_references == []
    assert (source, legacy) == before


def test_stored_private_seed_forks_to_per_preset_seeds():
    project = Project.model_validate({"version": 3, "model_settings": {"models": {
        "moss_local": {"parameters": {"seed": 77, "local_temperature": 1.2}},
    }}})
    for name in ("delay_seed", "local_seed", "local_v15_seed"):
        assert project.get_model_setting("moss_local", name) == 77
    assert project.get_model_setting("moss_local", "local_temperature") == 1.2
    # Explicit per-preset seeds win over the retired shared value.
    project = Project.model_validate({"version": 3, "model_settings": {"models": {
        "moss_local": {"parameters": {"seed": 77, "local_seed": 5}},
    }}})
    assert project.get_model_setting("moss_local", "delay_seed") == 77
    assert project.get_model_setting("moss_local", "local_seed") == 5
    # The retired name is consumed and does not survive a save round trip.
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "seed" not in saved["model_settings"]["models"]["moss_local"]["parameters"]


def test_edits_and_resets_after_fork_are_independent_and_do_not_resurrect():
    project = Project.model_validate({"version": 3, "model_settings": old_source()})
    assert project.voice_references == REFERENCES
    assert project.get_model_setting("moss_delay_sglomni", "transcript") == ["one", ""]
    project.set_model_setting("moss_local", "local_seed", 123)
    project.set_model_setting("moss_local", "batch_size", 1)
    project.set_model_setting("moss_local", "delay_top_k", 80)
    project.set_model_setting("moss_delay_sglomni", "delay_temperature", None, reset=True)
    # The shared voice list is project-wide: clearing it clears transcripts too.
    project.set_model_setting("moss_delay_sglomni", "file_name", [])
    assert project.voice_references == []
    assert project.get_model_setting("moss_delay_sglomni", "transcript") == []
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    for _ in range(2):
        project = Project.model_validate(deepcopy(saved))
        assert project.get_model_setting("moss_delay_sglomni", "delay_temperature") == -1.0
        assert project.voice_references == []
        assert "moss" not in project.model_settings.shared
        assert ProjectSerializationUtil.to_project_json_dict(project) == saved


def test_other_groups_and_unknown_whole_objects_are_unchanged():
    source = old_source()
    source["shared"]["fish_s2"] = {"model_ids": list(REGISTRY.members["fish_s2"]),
                                     "parameters": {"temperature": 0.8}}
    source["shared"]["future_group"] = {"model_ids": ["future"], "future": {"x": [1, 2]}}
    source["models"]["future"] = {"future": "untouched"}
    project = Project.model_validate({"version": 3, "model_settings": source})
    assert project.model_settings.shared == {name: obj for name, obj in source["shared"].items() if name != "moss"}
    assert project.model_settings.models["future"] == source["models"]["future"]
    project.set_model_setting("fish_s2_sglomni", "temperature", 0.9)
    assert project.get_model_setting("fish_s2_local", "temperature") == 0.9


def test_registry_without_moss_keeps_unknown_historical_group():
    registry = ModelSettingsRegistry()
    source = old_source()
    assert registry.reconcile(source).to_dict() == source


@pytest.mark.parametrize("member_ids", [None, [*IDS, "unauthorized"], [IDS[0]] * 3])
def test_bad_historical_membership_is_non_destructive_on_load_and_save(member_ids):
    source = old_source()
    source["shared"]["moss"]["model_ids"] = member_ids
    before = deepcopy(source)
    with pytest.raises(ValueError, match="shared.moss.model_ids"):
        REGISTRY.reconcile(source)
    assert source == before
    warnings = []
    project = Project.model_validate({"version": 3, "model_settings": source}, context={"warnings": warnings})
    assert any("shared.moss.model_ids" in warning for warning in warnings)
    assert project.model_settings.to_dict() == before
    assert ProjectSerializationUtil.to_project_json_dict(project)["model_settings"]["shared"]["moss"] == before["shared"]["moss"]


@pytest.mark.parametrize("section,value", [
    ("voice_references", {}), ("voice_references", [{"file_name": "one.flac", "transcript": 3}]),
    ("orchestration", []),
])
def test_bad_historical_structure_does_not_mutate_or_consume_source(section, value):
    source = old_source()
    source["shared"]["moss"][section] = value
    before = deepcopy(source)
    with pytest.raises(ValueError, match=f"shared.moss.{section}"):
        REGISTRY.reconcile(source)
    assert source == before
    warnings = []
    project = Project.model_validate({"version": 3, "model_settings": source}, context={"warnings": warnings})
    assert warnings
    assert ProjectSerializationUtil.to_project_json_dict(project)["model_settings"]["shared"]["moss"] == before["shared"]["moss"]


def test_migrated_and_new_private_invalid_values_reset_without_losing_other_overrides(caplog):
    source = old_source()
    source["shared"]["moss"]["parameters"]["local_temperature"] = 99
    source["models"]["moss_delay_sglomni"] = {"parameters": {"delay_top_k": 1}}
    project = Project.model_validate({"version": 3, "model_settings": source})
    assert project.get_model_setting("moss_local", "local_temperature") == -1.0
    assert project.get_model_setting("moss_local_sglomni", "local_temperature") == -1.0
    assert project.get_model_setting("moss_delay_sglomni", "delay_top_k") == -1
    assert project.get_model_setting("moss_local", "delay_top_k") == 32
    assert "resetting to default" in caplog.text
    assert "moss" not in project.model_settings.shared
    caplog.clear()
    private_only = Project.model_validate({"version": 3, "model_settings": {"models": {
        "moss_local": {"parameters": {"delay_top_p": 0.1}},
    }}})
    assert private_only.get_model_setting("moss_local", "delay_top_p") == -1.0
    assert "resetting to default" in caplog.text


def test_snapshot_migration_round_trip_and_portable_voice_paths():
    source = old_source()
    source["shared"]["moss"]["voice_references"][0]["file_name"] = "C:\\books\\voice\\one.flac"
    project = ProjectTransferUtil.validate_abr_snapshot({"version": 3, "tts_model_type": "moss_local", "model_settings": source})
    for model_id in IDS:
        assert project.get_model_setting(model_id, "file_name") == ["one.flac", "two.flac"]
    snapshot = ProjectSerializationUtil.to_snapshot_dict(project)
    assert "moss" not in snapshot["model_settings"]["shared"]
    restored = ProjectTransferUtil.make_project_from_snapshot("", snapshot)
    assert restored.tts_model_type == "moss_local"
    assert ProjectSerializationUtil.to_snapshot_dict(restored) == snapshot


def test_disk_worker_load_migrates_and_saves_v4_with_backup(tmp_path):
    # Workers and interactive loading use this same on-disk load funnel, which
    # now migrates the retired MOSS group and the voice list and saves v4.
    payload = ProjectSerializationUtil.to_project_json_dict(Project(tts_model_type="moss_delay_sglomni"))
    for model_id in IDS:
        payload["model_settings"]["models"].pop(model_id)
    payload["model_settings"]["shared"]["moss"] = old_source()["shared"]["moss"]
    payload["version"] = 3
    payload.pop("voice_references", None)
    path = tmp_path / "project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    project = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(project, Project)
    assert project.tts_model_type == "moss_delay_sglomni"
    assert project.get_model_setting("moss_delay_sglomni", "delay_top_k") == 32
    assert project.voice_references == REFERENCES
    assert (tmp_path / "project.json.pre-v4.bak").read_text(encoding="utf-8") == before
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["version"] == 4
    assert "moss" not in saved["model_settings"]["shared"]
    assert saved["voice_references"] == REFERENCES
    restored = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(restored, Project)
    assert restored.model_settings.to_dict() == project.model_settings.to_dict()
