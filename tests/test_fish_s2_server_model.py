import numpy as np

from tts_audiobook_tool.app_types import ReadinessIssue, Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.fish_s2_server_base_model import FishS2ServerBaseModel
from tts_audiobook_tool.tts_models.fish_s2_server_model import FishS2ServerModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def test_fish_s2_server_spec_shares_local_voice_fields():
    local_info = TtsModelType.FISH_S2.value
    server_info = TtsModelType.FISH_S2_SERVER.value
    project = Project.model_validate({
        "fish_s2_voice_file_name": "voice.flac",
        "fish_s2_voice_transcript": "reference transcript",
        "fish_s2_server_voice_target": "https://legacy.example/voice.flac",
        "fish_s2_server_voice_transcript": "legacy transcript",
    })

    assert server_info.backend_kind == TtsBackendKind.SGL_OMNI
    assert server_info.voice_target_attr == local_info.voice_target_attr == "fish_s2_voice_file_name"
    assert server_info.voice_transcript_attr == local_info.voice_transcript_attr == "fish_s2_voice_transcript"
    assert not server_info.requires_voice
    assert project.fish_s2_voice_file_name == ["voice.flac"]
    assert project.fish_s2_voice_transcript == ["reference transcript"]

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["fish_s2_voice_file_name"] == "voice.flac"
    assert payload["fish_s2_voice_transcript"] == "reference transcript"
    assert "fish_s2_server_voice_target" not in payload
    assert "fish_s2_server_voice_transcript" not in payload


def test_fish_s2_server_readiness_checks_local_voice_and_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_base_model.SglOmniUtil.check_readiness",
        lambda _: None,
    )

    missing_file_issues = FishS2ServerBaseModel.get_blocking_issues(
        Project.model_validate({
            "dir_path": str(tmp_path),
            "fish_s2_voice_file_name": "missing.flac",
            "fish_s2_voice_transcript": "reference transcript",
        }),
        None,
    )
    assert any(issue.short == "voice sample" for issue in missing_file_issues)

    (tmp_path / "voice.flac").write_bytes(b"audio")
    missing_transcript_issues = FishS2ServerBaseModel.get_blocking_issues(
        Project.model_validate({
            "dir_path": str(tmp_path),
            "fish_s2_voice_file_name": ["voice.flac", "second.flac"],
            "fish_s2_voice_transcript": ["first transcript"],
        }),
        None,
    )
    assert any(issue.short == "voice clone transcript" for issue in missing_transcript_issues)


def test_fish_s2_server_readiness_includes_server_issue(monkeypatch, tmp_path):
    expected = ReadinessIssue("server", "unavailable")
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_base_model.SglOmniUtil.check_readiness",
        lambda _: expected,
    )

    issues = FishS2ServerBaseModel.get_blocking_issues(
        Project.model_validate({"dir_path": str(tmp_path)}), None
    )

    assert issues == [expected]


def test_generate_using_project_sends_selected_fish_voice_as_data_uri(monkeypatch, tmp_path):
    calls = []
    encoded_paths = []
    expected = Sound(np.asarray([], dtype=np.float32), 44_100)

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append((base_url, payloads, print_request))
        return [expected]

    def fake_make_audio_data_uri(path):
        encoded_paths.append(path)
        return f"data:audio/flac;base64,{path}"

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_model.SoundUtil.make_audio_data_uri",
        fake_make_audio_data_uri,
    )

    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "fish_s2_voice_file_name": ["first.flac", "second.flac"],
        "fish_s2_voice_transcript": ["first transcript", "second transcript"],
        "fish_s2_temperature": 0.6,
        "fish_s2_top_p": 0.8,
        "fish_s2_top_k": 12,
    })

    result = FishS2ServerModel().generate_using_project(
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
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 12,
            "references": [{
                "audio_path": f"data:audio/flac;base64,{second_path}",
                "text": "second transcript",
            }],
        }],
        True,
    )]


def test_generate_using_project_streams_fish_voice_data_uri(monkeypatch, tmp_path):
    calls = []
    expected = Sound(np.asarray([0.1, 0.2], dtype=np.float32), 44_100)

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
        "tts_audiobook_tool.tts_models.fish_s2_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_model.SglOmniUtil.generate_streaming",
        fake_generate_streaming,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_model.SoundUtil.make_audio_data_uri",
        lambda path: f"data:audio/flac;base64,{path}",
    )

    on_stream_chunk = lambda _: None
    on_stream_end = lambda: None
    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "fish_s2_voice_file_name": "voice.flac",
        "fish_s2_voice_transcript": "reference transcript",
    })

    result = FishS2ServerModel().generate_using_project(
        project,
        ["hello"],
        on_stream_chunk=on_stream_chunk,
        on_stream_end=on_stream_end,
        print_generation_request=True,
    )

    assert result == [expected]
    assert calls[0][0] == "http://example.test"
    assert calls[0][1] == {
        "input": "hello",
        "stream": True,
        "temperature": FishS2BaseModel.TEMPERATURE_DEFAULT,
        "top_p": FishS2BaseModel.TOP_P_DEFAULT,
        "top_k": FishS2BaseModel.TOP_K_DEFAULT,
        "references": [{
            "audio_path": f"data:audio/flac;base64,{tmp_path}/voice.flac",
            "text": "reference transcript",
        }],
    }
    assert calls[0][2:] == (on_stream_chunk, on_stream_end, True)


def test_generate_using_project_omits_reference_without_fish_voice(monkeypatch):
    calls = []

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append(payloads)
        return []

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.fish_s2_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )

    result = FishS2ServerModel().generate_using_project(Project(), ["hello"])

    assert result == []
    assert "references" not in calls[0][0]
