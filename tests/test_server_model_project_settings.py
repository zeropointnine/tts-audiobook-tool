"""Spec/storage migration checks for configured SGL-Omni built-ins.

The legacy per-server generation classes are gone; these tests keep the
project-format and catalog guarantees that outlive them.
"""
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from project_settings_test_support import get_setting
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts_models.sgl_omni_detection import detect_sgl_omni_models
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def test_higgs_v3_server_spec_and_project_voice_fields():
    info = TtsModelType.require_by_id("higgs_v3_sglomni").value
    project = Project.model_validate({
        "higgs_v3_voice_file_name": "voice.flac",
        "higgs_v3_voice_transcript": "reference transcript",
        "higgs_v3_voice_target": "https://legacy.example/voice.flac",
    })

    assert info.backend_kind == TtsBackendKind.SGL_OMNI
    assert REGISTRY.voice_binding(info.id).name == "file_name"
    assert REGISTRY.transcript_binding(info.id).name == "transcript"
    assert not info.requires_voice
    assert get_setting(project, "higgs_v3_voice_file_name") == ["voice.flac"]
    assert get_setting(project, "higgs_v3_voice_transcript") == ["reference transcript"]

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    model = payload["model_settings"]["models"]["higgs_v3_sglomni"]
    assert model["voice_references"] == [
        {"file_name": "voice.flac", "transcript": "reference transcript"}
    ]
    assert "higgs_v3_voice_target" not in payload["model_settings"]
    assert "higgs_v3_voice_file_path" not in payload["model_settings"]


def test_fish_s2_server_spec_shares_local_voice_fields():
    local_info = TtsModelType.require_by_id("fish_s2_local").value
    server_info = TtsModelType.require_by_id("fish_s2_sglomni").value
    project = Project.model_validate({
        "fish_s2_voice_file_name": "voice.flac",
        "fish_s2_voice_transcript": "reference transcript",
        "fish_s2_server_voice_target": "https://legacy.example/voice.flac",
        "fish_s2_server_voice_transcript": "legacy transcript",
    })

    assert server_info.backend_kind == TtsBackendKind.SGL_OMNI
    assert REGISTRY.voice_binding(local_info.id).group == REGISTRY.voice_binding(server_info.id).group == "fish_s2"
    assert REGISTRY.transcript_binding(local_info.id).group == REGISTRY.transcript_binding(server_info.id).group == "fish_s2"
    assert not server_info.requires_voice
    assert get_setting(project, "fish_s2_voice_file_name") == ["voice.flac"]
    assert get_setting(project, "fish_s2_voice_transcript") == ["reference transcript"]

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    shared = payload["model_settings"]["shared"]["fish_s2"]
    assert shared["voice_references"] == [
        {"file_name": "voice.flac", "transcript": "reference transcript"}
    ]
    assert "fish_s2_server_voice_target" not in payload["model_settings"]
    assert "fish_s2_server_voice_transcript" not in payload["model_settings"]


def test_zonos2_server_spec_and_registration():
    info = TtsModelType.require_by_id("zonos2_sglomni").value

    assert info.backend_kind == TtsBackendKind.SGL_OMNI
    assert detect_sgl_omni_models([{"id": "Zyphra/ZONOS2-0.5B"}]) == [
        (TtsModelType.require_by_id("zonos2_sglomni"), "Zyphra/ZONOS2-0.5B")
    ]
    assert REGISTRY.voice_binding(info.id).name == "file_name"
    assert REGISTRY.transcript_binding(info.id) is None
    assert REGISTRY.orchestration_binding(info.id).name == "concurrent_requests"
    assert not info.requires_voice
    assert info.can_stream
    assert info.requirements_file_name == "requirements-remote.txt"
    model = TtsModelType.find_tts_type_using_sgl_omni_model_id("Zyphra/ZONOS2-0.5B")
    assert model is not None
    assert model.id == "zonos2_sglomni"


def test_zonos2_project_fields_normalize_and_serialize():
    project = Project.model_validate({
        "zonos2_server_voice_file_name": "voice.flac",
        "zonos2_server_concurrent_requests": 0,
        "zonos2_top_k": 150.0,
        "zonos2_temperature": 1.25,
        "zonos2_repetition_penalty": 1.35,
    })

    assert get_setting(project, "zonos2_server_voice_file_name") == ["voice.flac"]
    assert get_setting(project, "zonos2_server_concurrent_requests") == 1
    assert get_setting(project, "zonos2_top_k") == 150
    assert get_setting(project, "zonos2_temperature") == 1.25
    assert get_setting(project, "zonos2_repetition_penalty") == 1.35

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    model = payload["model_settings"]["models"]["zonos2_sglomni"]
    assert model["voice_references"] == [{"file_name": "voice.flac", "transcript": ""}]
    # Zonos2 declares no transcript storage.
    assert "zonos2_voice_transcript" not in payload["model_settings"]
    # The concurrency value normalized to its default, so nothing is stored.
    assert "orchestration" not in model
    assert model["parameters"]["top_k"] == 150
    assert model["parameters"]["temperature"] == 1.25
    assert model["parameters"]["repetition_penalty"] == 1.35


def test_zonos2_invalid_sampling_values_normalize_to_default_sentinel():
    project = Project.model_validate({
        "zonos2_top_k": 201,
        "zonos2_temperature": 2.01,
        "zonos2_repetition_penalty": 0.99,
    })

    assert get_setting(project, "zonos2_top_k") == -1
    assert get_setting(project, "zonos2_temperature") == -1.0
    assert get_setting(project, "zonos2_repetition_penalty") == -1.0
