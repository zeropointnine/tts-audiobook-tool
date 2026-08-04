import numpy as np

from tts_audiobook_tool.app_types import ReadinessIssue, Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.zonos2_server_base_model import Zonos2ServerBaseModel
from tts_audiobook_tool.tts_models.zonos2_server_model import Zonos2ServerModel


def test_zonos2_server_spec_and_registration(monkeypatch):
    info = TtsModelType.ZONOS2_SERVER.value

    assert info.is_sgl_omni
    assert info.server_model_id_substring == "zonos2"
    assert info.voice_target_attr == "zonos2_server_voice_file_name"
    assert info.voice_transcript_attr == ""
    assert info.batch_size_attr == "zonos2_server_concurrent_requests"
    assert not info.requires_voice
    assert info.can_stream
    assert info.requirements_file_name == "requirements-sgl-omni.txt"
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id(
        "Zyphra/ZONOS2-0.5B"
    ) == TtsModelType.ZONOS2_SERVER
    monkeypatch.setattr(Tts, "_type", TtsModelType.ZONOS2_SERVER)
    assert Tts.get_class() is Zonos2ServerBaseModel


def test_zonos2_project_fields_normalize_and_serialize():
    project = Project.model_validate({
        "zonos2_server_voice_file_name": "voice.flac",
        "zonos2_server_concurrent_requests": 0,
        "zonos2_top_k": 150.0,
        "zonos2_temperature": 1.25,
        "zonos2_repetition_penalty": 1.35,
    })

    assert project.zonos2_server_voice_file_name == ["voice.flac"]
    assert project.zonos2_server_concurrent_requests == 1
    assert project.zonos2_top_k == 150
    assert project.zonos2_temperature == 1.25
    assert project.zonos2_repetition_penalty == 1.35

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["zonos2_server_voice_file_name"] == "voice.flac"
    assert "zonos2_voice_transcript" not in payload
    assert payload["zonos2_server_concurrent_requests"] == 1
    assert payload["zonos2_top_k"] == 150
    assert payload["zonos2_temperature"] == 1.25
    assert payload["zonos2_repetition_penalty"] == 1.35


def test_zonos2_sampling_constants_match_reference_ui():
    assert Zonos2ServerBaseModel.TOP_K_DEFAULT == 100
    assert Zonos2ServerBaseModel.TOP_K_MIN == 1
    assert Zonos2ServerBaseModel.TOP_K_MAX == 200
    assert Zonos2ServerBaseModel.TEMPERATURE_DEFAULT == 1.15
    assert Zonos2ServerBaseModel.TEMPERATURE_MIN == 0.05
    assert Zonos2ServerBaseModel.TEMPERATURE_MAX == 2.0
    assert Zonos2ServerBaseModel.REPETITION_PENALTY_DEFAULT == 1.2
    assert Zonos2ServerBaseModel.REPETITION_PENALTY_MIN == 1.0
    assert Zonos2ServerBaseModel.REPETITION_PENALTY_MAX == 2.0


def test_zonos2_invalid_sampling_values_normalize_to_default_sentinel():
    project = Project.model_validate({
        "zonos2_top_k": 201,
        "zonos2_temperature": 2.01,
        "zonos2_repetition_penalty": 0.99,
    })

    assert project.zonos2_top_k == -1
    assert project.zonos2_temperature == -1
    assert project.zonos2_repetition_penalty == -1


def test_zonos2_readiness_does_not_require_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_base_model.SglOmniUtil.check_readiness",
        lambda _: None,
    )

    no_voice_issues = Zonos2ServerBaseModel.get_blocking_issues(
        Project.model_validate({"dir_path": str(tmp_path)}), None
    )
    assert no_voice_issues == []

    (tmp_path / "voice.flac").write_bytes(b"audio")
    ready_issues = Zonos2ServerBaseModel.get_blocking_issues(
        Project.model_validate({
            "dir_path": str(tmp_path),
            "zonos2_server_voice_file_name": "voice.flac",
        }),
        None,
    )
    assert ready_issues == []


def test_generate_using_project_omits_reference_without_voice(monkeypatch, tmp_path):
    calls = []

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append((base_url, payloads, print_request))
        return []

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )

    project = Project.model_validate({"dir_path": str(tmp_path)})
    result = Zonos2ServerModel().generate_using_project(project, ["hello"])

    assert result == []
    assert "references" not in calls[0][1][0]


def test_zonos2_readiness_includes_server_issue(monkeypatch, tmp_path):
    expected = ReadinessIssue("server", "unavailable")
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_base_model.SglOmniUtil.check_readiness",
        lambda _: expected,
    )

    issues = Zonos2ServerBaseModel.get_blocking_issues(
        Project.model_validate({"dir_path": str(tmp_path)}), None
    )
    assert expected in issues


def test_generate_using_project_builds_zonos2_payload(monkeypatch, tmp_path):
    calls = []
    expected = Sound(np.asarray([], dtype=np.float32), 44_100)

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append((base_url, payloads, print_request))
        return [expected]

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SoundUtil.make_audio_data_uri",
        lambda path: f"data:audio/flac;base64,{path}",
    )

    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "zonos2_server_voice_file_name": "voice.flac",
    })
    long_prompt = " ".join(["word"] * 100)
    result = Zonos2ServerModel().generate_using_project(
        project, ["hello", long_prompt], print_generation_request=True
    )

    assert result == [expected]
    assert calls == [(
        "http://example.test",
        [{
            "input": "hello",
            "stream": False,
            "max_new_tokens": 240,
            "top_k": Zonos2ServerBaseModel.TOP_K_DEFAULT,
            "temperature": Zonos2ServerBaseModel.TEMPERATURE_DEFAULT,
            "repetition_penalty": Zonos2ServerBaseModel.REPETITION_PENALTY_DEFAULT,
            "references": [{
                "audio_path": f"data:audio/flac;base64,{tmp_path}/voice.flac",
            }],
        }, {
            "input": long_prompt,
            "stream": False,
            "max_new_tokens": 4096,
            "top_k": Zonos2ServerBaseModel.TOP_K_DEFAULT,
            "temperature": Zonos2ServerBaseModel.TEMPERATURE_DEFAULT,
            "repetition_penalty": Zonos2ServerBaseModel.REPETITION_PENALTY_DEFAULT,
            "references": [{
                "audio_path": f"data:audio/flac;base64,{tmp_path}/voice.flac",
            }],
        }],
        True,
    )]


def test_generate_using_project_uses_custom_zonos2_sampling_values(monkeypatch, tmp_path):
    calls = []

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append((base_url, payloads, print_request))
        return []

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.zonos2_server_model.SoundUtil.make_audio_data_uri",
        lambda path: f"data:audio/flac;base64,{path}",
    )

    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "zonos2_server_voice_file_name": "voice.flac",
        "zonos2_top_k": 175,
        "zonos2_temperature": 0.85,
        "zonos2_repetition_penalty": 1.45,
    })
    result = Zonos2ServerModel().generate_using_project(project, ["hello"])

    assert result == []
    assert calls[0][1][0]["top_k"] == 175
    assert calls[0][1][0]["temperature"] == 0.85
    assert calls[0][1][0]["repetition_penalty"] == 1.45
