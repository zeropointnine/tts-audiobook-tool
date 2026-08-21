import numpy as np

from tts_audiobook_tool.app_types import ReadinessIssue, Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts_models.higgs_v3_server_base_model import HiggsV3ServerBaseModel
from tts_audiobook_tool.tts_models.higgs_v3_server_model import HiggsV3ServerModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def test_higgs_v3_server_spec_and_project_voice_fields():
    info = TtsModelType.HIGGS_V3_SERVER.value
    project = Project.model_validate({
        "higgs_v3_voice_file_name": "voice.flac",
        "higgs_v3_voice_transcript": "reference transcript",
        "higgs_v3_voice_target": "https://legacy.example/voice.flac",
    })

    assert info.backend_kind == TtsBackendKind.SGL_OMNI
    assert info.voice_target_attr == "higgs_v3_voice_file_name"
    assert info.voice_transcript_attr == "higgs_v3_voice_transcript"
    assert not info.requires_voice
    assert project.higgs_v3_voice_file_name == ["voice.flac"]
    assert project.higgs_v3_voice_transcript == ["reference transcript"]

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["higgs_v3_voice_file_name"] == "voice.flac"
    assert payload["higgs_v3_voice_transcript"] == "reference transcript"
    assert "higgs_v3_voice_target" not in payload
    assert "higgs_v3_voice_file_path" not in payload


def test_higgs_v3_readiness_checks_local_voice_and_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_base_model.SglOmniUtil.check_readiness",
        lambda _: None,
    )

    missing_file_issues = HiggsV3ServerBaseModel.get_blocking_issues(
        Project.model_validate({
            "dir_path": str(tmp_path),
            "higgs_v3_voice_file_name": "missing.flac",
            "higgs_v3_voice_transcript": "reference transcript",
        }),
        None,
    )
    assert any(issue.short == "voice sample" for issue in missing_file_issues)

    (tmp_path / "voice.flac").write_bytes(b"audio")
    missing_transcript_issues = HiggsV3ServerBaseModel.get_blocking_issues(
        Project.model_validate({
            "dir_path": str(tmp_path),
            "higgs_v3_voice_file_name": ["voice.flac", "second.flac"],
            "higgs_v3_voice_transcript": ["first transcript"],
        }),
        None,
    )
    assert any(issue.short == "voice clone transcript" for issue in missing_transcript_issues)


def test_higgs_v3_readiness_includes_server_issue(monkeypatch, tmp_path):
    expected = ReadinessIssue("server", "unavailable")
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_base_model.SglOmniUtil.check_readiness",
        lambda _: expected,
    )

    issues = HiggsV3ServerBaseModel.get_blocking_issues(
        Project.model_validate({"dir_path": str(tmp_path)}), None
    )

    assert issues == [expected]


def test_generate_using_project_sends_selected_higgs_voice_as_data_uri(monkeypatch, tmp_path):
    calls = []
    encoded_paths = []
    expected = Sound(np.asarray([], dtype=np.float32), 24_000)

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append((base_url, payloads, print_request))
        return [expected]

    def fake_make_audio_data_uri(path):
        encoded_paths.append(path)
        return f"data:audio/flac;base64,{path}"

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_model.SoundUtil.make_audio_data_uri",
        fake_make_audio_data_uri,
    )

    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "higgs_v3_voice_file_name": ["first.flac", "second.flac"],
        "higgs_v3_voice_transcript": ["first transcript", "second transcript"],
        "higgs_v3_temperature": 0.7,
        "higgs_v3_top_p": 0.8,
        "higgs_v3_top_k": 12,
    })

    result = HiggsV3ServerModel().generate_using_project(
        project,
        ["hello"],
        voice_selection_index=1,
        print_generation_request=True,
    )

    second_path = str(tmp_path / "second.flac")
    assert result == [expected]
    assert encoded_paths == [second_path]
    assert calls == [(
        "http://example.test",
        [{
            "input": "hello",
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 12,
            "max_tokens": HiggsV3ServerBaseModel.MAX_TOKENS,
            "references": [{
                "audio_path": f"data:audio/flac;base64,{second_path}",
                "text": "second transcript",
            }],
        }],
        True,
    )]


def test_generate_using_project_streams_higgs_voice_data_uri(monkeypatch, tmp_path):
    calls = []
    expected = Sound(np.asarray([0.1, 0.2], dtype=np.float32), 24_000)

    def fake_generate_streaming(
        base_url,
        payload,
        on_stream_chunk=None,
        on_stream_end=None,
        should_print=False,
    ):
        calls.append((base_url, payload, on_stream_chunk, on_stream_end, should_print))
        return expected

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_model.SglOmniUtil.generate_streaming",
        fake_generate_streaming,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_model.SoundUtil.make_audio_data_uri",
        lambda path: f"data:audio/flac;base64,{path}",
    )

    on_stream_chunk = lambda _: None
    on_stream_end = lambda: None
    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "higgs_v3_voice_file_name": "voice.flac",
        "higgs_v3_voice_transcript": "reference transcript",
    })

    result = HiggsV3ServerModel().generate_using_project(
        project,
        ["hello"],
        on_stream_chunk=on_stream_chunk,
        on_stream_end=on_stream_end,
        print_generation_request=True,
    )

    assert result == [expected]
    assert calls[0][0] == "http://example.test"
    assert calls[0][1]["references"] == [{
        "audio_path": f"data:audio/flac;base64,{tmp_path}/voice.flac",
        "text": "reference transcript",
    }]
    assert calls[0][1]["stream"] is True
    assert calls[0][2:] == (on_stream_chunk, on_stream_end, True)


def test_generate_using_project_omits_reference_without_higgs_voice(monkeypatch):
    calls = []

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append(payloads)
        return []

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.higgs_v3_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )

    result = HiggsV3ServerModel().generate_using_project(Project(), ["hello"])

    assert result == []
    assert "references" not in calls[0][0]
