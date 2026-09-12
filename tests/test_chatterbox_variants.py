from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import (
    ProjectSerializationUtil,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.chatterbox_base_model import (
    ChatterboxBaseModel,
    ChatterboxType,
)


def test_new_projects_default_to_multilingual_v3():
    assert Project().chatterbox_type == ChatterboxType.MULTILINGUAL_V3
    assert list(ChatterboxType)[0] == ChatterboxType.MULTILINGUAL_V3


def test_legacy_multilingual_project_id_remains_v2():
    project = Project.model_validate({"chatterbox_type": "multilingual"})

    assert ChatterboxType.MULTILINGUAL is ChatterboxType.MULTILINGUAL_V2
    assert project.chatterbox_type == ChatterboxType.MULTILINGUAL_V2
    assert ProjectSerializationUtil.to_project_json_dict(project)["chatterbox_type"] == "multilingual"


def test_multilingual_v3_project_id_round_trips():
    project = Project.model_validate({"chatterbox_type": "multilingual-v3"})

    assert project.chatterbox_type == ChatterboxType.MULTILINGUAL_V3
    assert ProjectSerializationUtil.to_project_json_dict(project)["chatterbox_type"] == "multilingual-v3"


def test_chatterbox_variant_metadata_and_defaults():
    assert ChatterboxType.MULTILINGUAL_V2.is_multilingual
    assert ChatterboxType.MULTILINGUAL_V2.multilingual_t3_model == "v2"
    assert ChatterboxType.MULTILINGUAL_V3.is_multilingual
    assert ChatterboxType.MULTILINGUAL_V3.multilingual_t3_model == "v3"
    assert not ChatterboxType.TURBO.is_multilingual
    assert ChatterboxType.TURBO.multilingual_t3_model is None

    assert ChatterboxBaseModel.default_repetition_penalty(
        ChatterboxType.MULTILINGUAL_V2
    ) == 2.0
    assert ChatterboxBaseModel.default_repetition_penalty(
        ChatterboxType.MULTILINGUAL_V3
    ) == 1.2
    assert ChatterboxBaseModel.default_repetition_penalty(ChatterboxType.TURBO) == 1.2


def test_chatterbox_variant_is_part_of_model_lifecycle_params():
    v2 = Project.model_validate({"chatterbox_type": "multilingual"})
    v3 = Project.model_validate({"chatterbox_type": "multilingual-v3"})

    assert Tts.get_model_params_using_project(v2)["chatterbox_type"] == ChatterboxType.MULTILINGUAL_V2
    assert Tts.get_model_params_using_project(v3)["chatterbox_type"] == ChatterboxType.MULTILINGUAL_V3


def test_multilingual_repetition_penalty_values_are_independent():
    project = Project.model_validate({
        "chatterbox_ml_v2_repetition_penalty": 1.7,
        "chatterbox_ml_v3_repetition_penalty": 1.3,
        "chatterbox_turbo_repetition_penalty": 1.4,
    })

    assert project.chatterbox_ml_v2_repetition_penalty == 1.7
    assert project.chatterbox_ml_v3_repetition_penalty == 1.3
    assert project.chatterbox_turbo_repetition_penalty == 1.4

    d = ProjectSerializationUtil.to_project_json_dict(project)
    assert d["chatterbox_ml_v2_repetition_penalty"] == 1.7
    assert d["chatterbox_ml_v3_repetition_penalty"] == 1.3
    assert d["chatterbox_turbo_repetition_penalty"] == 1.4
    assert "chatterbox_ml_repetition_penalty" not in d


def test_legacy_shared_repetition_penalty_key_maps_to_v2_only():
    d = ProjectSerializationUtil.normalize_loaded_project_dict({
        "chatterbox_type": "multilingual-v3",
        "chatterbox_ml_repetition_penalty": 1.7,
    })
    project = Project.model_validate(d)

    assert project.chatterbox_ml_v2_repetition_penalty == 1.7
    assert project.chatterbox_ml_v3_repetition_penalty == -1

    red = ProjectSerializationUtil.to_project_json_dict(project)
    assert red["chatterbox_ml_v2_repetition_penalty"] == 1.7
    assert red["chatterbox_ml_v3_repetition_penalty"] == -1
    assert "chatterbox_ml_repetition_penalty" not in red


def test_new_repetition_penalty_keys_survive_load():
    d = ProjectSerializationUtil.normalize_loaded_project_dict({
        "chatterbox_type": "multilingual",
        "chatterbox_ml_v2_repetition_penalty": 1.9,
        "chatterbox_ml_v3_repetition_penalty": 1.25,
    })
    project = Project.model_validate(d)

    assert project.chatterbox_ml_v2_repetition_penalty == 1.9
    assert project.chatterbox_ml_v3_repetition_penalty == 1.25
