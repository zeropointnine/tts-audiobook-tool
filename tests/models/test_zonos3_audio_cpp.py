"""ZONOS3-Mini via audio.cpp: explicit speaker mode, guidance and per-voice rate."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.audio_cpp_behavior_zonos3 import FMAX_UNKNOWN_BUCKET, LOUDNESS_UNKNOWN_BUCKET
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions

MODEL_ID = "zonos3_mini_audiocpp"


def definition():
    return load_audio_cpp_definitions().models[MODEL_ID]


@pytest.fixture
def capture(monkeypatch):
    payloads: list[dict] = []

    def fake_generate(url, payload, print_request=False):
        payloads.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=44100)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    return payloads


def make_project(tmp_path: Path, voices: list[Voice], values: dict | None = None) -> Project:
    sf.write(tmp_path / "sample.flac", np.full(100, .2), 44100, format="FLAC")
    project = Project(dir_path=str(tmp_path), tts_model_type=MODEL_ID, voice_references=voices)
    for name, value in (values or {}).items():
        project.set_model_setting(MODEL_ID, name, value)
    return project


def generate(project: Project, index: int = 0) -> None:
    item = definition()
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "zonos3")
    result = adapter.generate_using_project(project, ["hello"], voice_selection_index=index)
    assert not isinstance(result, str), result


def test_definition_leaves_text_handling_to_the_app():
    # The app owns segmentation and normalization; the server's text_normalization
    # also validates `language`, so the field is never sent.
    item = definition()
    assert item.family == "zonos3" and set(item.tasks) == {"tts", "clon"} and item.mode == "offline"
    assert not item.voice_required and not item.reference_transcript
    assert item.language_policy == "omit"
    assert item.request_options["text_normalization"] is False
    assert item.request_options["text_chunk_size"] >= 100000
    # Choice values are the server's speaker_mode enum.
    mode = item.parameters["speaker_mode"]
    assert mode.default == "embedding" and {choice.value for choice in mode.choices} == {"embedding", "reference"}


@pytest.mark.parametrize("mode", ["embedding", "reference"])
def test_sampled_voice_sends_mode_guidance_and_pinned_options(tmp_path, capture, mode):
    # The mode is always explicit (never the server's default), and booleans
    # travel as JSON booleans inside `options`.
    project = make_project(tmp_path, [Voice("sample.flac")], {"speaker_mode": mode, "guidance_scale": 1.5})
    generate(project)
    payload = capture[0]
    assert payload["voice_ref"]["data"].startswith("data:audio/wav;base64,")
    assert payload["guidance_scale"] == 1.5
    assert payload["options"]["speaker_mode"] == mode
    assert payload["options"]["text_normalization"] is False
    assert "language" not in payload and "language" not in payload["options"]
    assert "reference_text" not in payload and "condition_buckets" not in payload["options"]


def test_sampleless_voice_omits_guidance_but_keeps_mode(tmp_path, capture):
    # The server rejects guidance_scale != 1 without a voice source.
    project = make_project(tmp_path, [Voice()], {"guidance_scale": 2.0})
    generate(project)
    payload = capture[0]
    assert "voice_ref" not in payload and "guidance_scale" not in payload
    assert payload["options"]["speaker_mode"] == "embedding"


@pytest.mark.parametrize("index,expected", [(0, 0), (1, 29), (2, None)])
def test_selected_voice_rate_bucket_fills_only_the_rate_row(tmp_path, capture, index, expected):
    # Storage is 1-based, the wire is zero-based; loudness/fmax stay at their
    # "unknown" IDs, and another model's override never leaks in.
    project = make_project(tmp_path, [
        Voice(speaking_rate_buckets={MODEL_ID: 1}),
        Voice("sample.flac", speaking_rate_buckets={MODEL_ID: 30, "zonos3_mini_sglomni": 7}),
        Voice(speaking_rate_buckets={"zonos3_mini_sglomni": 4}),
    ])
    generate(project, index)
    options = capture[0]["options"]
    if expected is None:
        assert "condition_buckets" not in options
    else:
        assert options["condition_buckets"] == f"{LOUDNESS_UNKNOWN_BUCKET},{FMAX_UNKNOWN_BUCKET},{expected}"
