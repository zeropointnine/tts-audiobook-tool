import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.qwen3_server_model import Qwen3ServerModel


def test_generate_using_project_builds_qwen3_server_payload(monkeypatch, tmp_path):
    calls = []
    expected = Sound(np.asarray([], dtype=np.float32), 24000)

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append((base_url, payloads, print_request))
        return [expected]

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.qwen3_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.qwen3_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.qwen3_server_model.SoundUtil.make_audio_data_uri",
        lambda path: f"data:audio/wav;base64,{path}",
    )

    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "qwen3_voice_file_name": "ref.wav",
        "qwen3_voice_transcript": "reference transcript",
        "qwen3_temperature": 0.7,
        "qwen3_top_p": 0.8,
        "qwen3_top_k": 12,
        "qwen3_repetition_penalty": 1.1,
        "qwen3_seed": 123,
    })

    result = Qwen3ServerModel().generate_using_project(
        project,
        ["hello"],
        print_generation_request=True,
    )

    assert result == [expected]
    assert calls == [(
        "http://example.test",
        [{
            "input": "hello",
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 12,
            "repetition_penalty": 1.1,
            "references": [{
                "audio_path": f"data:audio/wav;base64,{tmp_path}/ref.wav",
                "text": "reference transcript",
            }],
        }],
        True,
    )]


def test_generate_using_project_streams_qwen3_server_payload(monkeypatch):
    calls = []
    expected = Sound(np.asarray([0.1, 0.2], dtype=np.float32), 24000)

    def fake_generate_streaming(base_url, payload, on_stream_chunk=None, on_stream_end=None, should_print=False):
        calls.append((base_url, payload, on_stream_chunk, on_stream_end, should_print))
        return expected

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.qwen3_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.qwen3_server_model.SglOmniUtil.generate_streaming",
        fake_generate_streaming,
    )

    on_stream_chunk = lambda _: None
    on_stream_end = lambda: None

    result = Qwen3ServerModel().generate_using_project(
        Project.model_validate({}),
        ["hello"],
        on_stream_chunk=on_stream_chunk,
        on_stream_end=on_stream_end,
        print_generation_request=True,
    )

    assert result == [expected]
    assert calls == [(
        "http://example.test",
        {
            "input": "hello",
            "stream": True,
            "response_format": "pcm",
            "temperature": 0.9,
            "top_p": 1.0,
            "top_k": 50,
        },
        on_stream_chunk,
        on_stream_end,
        True,
    )]


def test_generate_using_project_rejects_multi_prompt_qwen3_server_streaming():
    result = Qwen3ServerModel().generate_using_project(
        Project.model_validate({}),
        ["hello", "world"],
        on_stream_chunk=lambda _: None,
    )

    assert result == "Streaming generation supports exactly one prompt"
