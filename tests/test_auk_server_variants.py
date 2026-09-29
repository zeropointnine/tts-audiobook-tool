import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from project_settings_test_support import get_setting
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


@pytest.mark.parametrize(
    ("model_type", "substring", "proper_name"),
    [
        (TtsModelType.AUK_SERVER, "auk", "AuK"),
        (TtsModelType.AUK_FLASH_SERVER, "auk-flash", "AuK-Flash"),
    ],
)
def test_auk_variants_have_distinct_catalog_identity(
    model_type, substring, proper_name
):
    info = model_type.value

    assert info.backend_kind is TtsBackendKind.SGL_OMNI
    assert info.sgl_omni_model_id_substring == substring
    assert info.ui["proper_name"] == proper_name
    assert info.default_output_sample_rate == 24_000
    assert REGISTRY.voice_binding(model_type.id).group == "auk"
    assert REGISTRY.transcript_binding(model_type.id).group == "auk"
    assert REGISTRY.orchestration_binding(model_type.id).name == "concurrent_requests"
    assert info.requires_voice
    assert not info.can_stream
    assert info.requirements_file_name == "requirements-sgl-omni.txt"


def test_find_auk_variant_using_sgl_omni_model_id():
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK")
        is TtsModelType.AUK_SERVER
    )
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK-Flash")
        is TtsModelType.AUK_FLASH_SERVER
    )
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id(
            "/models/TENCENT/AUK-FLASH"
        )
        is TtsModelType.AUK_FLASH_SERVER
    )


def test_auk_project_fields_are_shared_normalized_and_serialized():
    project = Project.model_validate(
        {
            "auk_voice_file_name": "voice.flac",
            "auk_voice_transcript": "reference transcript",
            "auk_server_concurrent_requests": 0,
            "auk_speed": "invalid",
            "auk_seed": "invalid",
        }
    )

    assert get_setting(project, "auk_voice_file_name") == ["voice.flac"]
    assert get_setting(project, "auk_voice_transcript") == ["reference transcript"]
    assert get_setting(project, "auk_server_concurrent_requests") == 1
    assert get_setting(project, "auk_speed") == 1.0
    assert get_setting(project, "auk_seed") == -1

    # AuK and AuK-Flash share one storage group.
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    shared = payload["model_settings"]["shared"]["auk"]
    assert shared["model_ids"] == ["server_auk", "server_auk_flash"]
    assert shared["voice_references"] == [
        {"file_name": "voice.flac", "transcript": "reference transcript"}
    ]
    # The invalid speed, concurrency and seed normalized to their defaults; a
    # value equal to the default is absent (seeds stay explicit by design).
    assert shared["parameters"] == {"seed": -1}
    assert "orchestration" not in shared


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_auk_non_finite_integer_fields_are_normalized(invalid_value):
    project = Project.model_validate(
        {
            "auk_server_concurrent_requests": invalid_value,
            "auk_seed": invalid_value,
        }
    )

    assert get_setting(project, "auk_server_concurrent_requests") == 1
    assert get_setting(project, "auk_seed") == -1


def test_auk_boolean_integer_fields_are_normalized():
    project = Project.model_validate(
        {
            "auk_server_concurrent_requests": True,
            "auk_seed": True,
        }
    )

    assert get_setting(project, "auk_server_concurrent_requests") == 1
    assert get_setting(project, "auk_seed") == -1
