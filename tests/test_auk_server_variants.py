import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from project_settings_test_support import get_setting, set_setting
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.sgl_omni_detection import detect_sgl_omni_models
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


@pytest.mark.parametrize(
    ("model_type", "substring", "proper_name"),
    [
        (TtsModelType.require_by_id("auk_sglomni"), "auk", "AuK"),
        (TtsModelType.require_by_id("auk_flash_sglomni"), "auk-flash", "AuK-Flash"),
    ],
)
def test_auk_variants_have_distinct_catalog_identity(
    model_type, substring, proper_name
):
    info = model_type.value

    assert info.backend_kind is TtsBackendKind.SGL_OMNI
    assert detect_sgl_omni_models([{"id": f"tencent/{substring}"}]) == [
        (model_type, f"tencent/{substring}")
    ]
    assert info.ui["proper_name"] == proper_name
    assert info.default_output_sample_rate == 24_000
    assert REGISTRY.voice_binding(model_type.id).group == "auk"
    assert REGISTRY.transcript_binding(model_type.id).group == "auk"
    assert REGISTRY.orchestration_binding(model_type.id).name == "concurrent_requests"
    assert info.requires_voice
    assert not info.can_stream
    assert info.requirements_file_name == "requirements-remote.txt"


@pytest.mark.parametrize(
    "model_type",
    [TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("auk_flash_sglomni")],
)
def test_auk_variants_declare_no_concurrency(model_type):
    definition = Tts._configured_definitions[model_type.id]
    assert definition.can_batch is False
    assert not Tts.can_batch(model_type)

    # Storage ownership stays with the registry (projects keep round-tripping),
    # but the effective value never exceeds one.
    assert REGISTRY.orchestration_binding(model_type.id).name == "concurrent_requests"
    project = Project(tts_model_type=model_type.id)
    set_setting(project, "auk_server_concurrent_requests", 4)
    assert get_setting(project, "auk_server_concurrent_requests") == 4
    assert ProjectVoiceUtil.get_batch_size(project) == 1

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["model_settings"]["shared"]["auk"]["orchestration"] == {
        "concurrent_requests": 4
    }


@pytest.mark.parametrize(
    "model_type",
    [TtsModelType.require_by_id("fish_s2_sglomni"), TtsModelType.require_by_id("qwen3tts_sglomni"), TtsModelType.require_by_id("zonos2_sglomni")],
)
def test_other_server_variants_keep_declared_concurrency(model_type):
    project = Project(tts_model_type=model_type.id)
    project.set_model_setting(model_type.id, "concurrent_requests", 4)
    assert Tts.can_batch(model_type)
    assert ProjectVoiceUtil.get_batch_size(project) == 4


def test_find_auk_variant_using_sgl_omni_model_id():
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK")
        is TtsModelType.require_by_id("auk_sglomni")
    )
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK-Flash")
        is TtsModelType.require_by_id("auk_flash_sglomni")
    )
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id(
            "/models/TENCENT/AUK-FLASH"
        )
        is TtsModelType.require_by_id("auk_flash_sglomni")
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
    # V4 keeps one project-wide voice list; scoped clone storage is not emitted.
    assert payload["voice_references"] == [
        {"file_name": "voice.flac", "transcript": "reference transcript"}
    ]
    shared = payload["model_settings"]["shared"]["auk"]
    assert shared["model_ids"] == ["auk_sglomni", "auk_flash_sglomni"]
    assert "voice_references" not in shared
    # Unset speed is documented as null; seeds stay explicit by design.
    # Default concurrency remains absent from the orchestration section.
    assert shared["parameters"] == {"speed": None, "seed": -1}
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
