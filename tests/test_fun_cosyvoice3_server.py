"""Fun-CosyVoice3's configured SGL-Omni catalog entry and request shape."""
from __future__ import annotations

import numpy as np
import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

MODEL_ID = "cosyvoice3_sglomni"
HF_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


@pytest.fixture
def cosyvoice(monkeypatch):
    previous = (Tts._backend_mode, getattr(Tts, "_type", None), Tts._config_fingerprint,
                Tts._configured_definitions, Tts._configured_runtime, Tts._catalog_initialized,
                 Tts._selected_server_model_id)
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.REMOTE_CLIENT))
    try:
        Tts.init_local_model_type()
        model_type = TtsModelType.require_by_id(MODEL_ID)
        Tts._type = model_type
        Tts._selected_server_model_id = HF_ID
        monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
            backend_kind=TtsBackendKind.SGL_OMNI, candidates=((model_type, HF_ID),)))
        yield model_type
    finally:
        TtsModelType.reset_catalog()
        REGISTRY.reset_to_builtins()
        (Tts._backend_mode, old_type, Tts._config_fingerprint,
         Tts._configured_definitions, Tts._configured_runtime, Tts._catalog_initialized,
                 Tts._selected_server_model_id) = previous
        if old_type is None:
            if hasattr(Tts, "_type"):
                delattr(Tts, "_type")
        else:
            Tts._type = old_type


def test_cosyvoice_definition_registers_and_detects(cosyvoice):
    definition = Tts.get_configured_definition(cosyvoice)
    assert definition is not None
    assert cosyvoice.value == definition.spec
    assert cosyvoice.value.default_output_sample_rate == 24_000
    assert cosyvoice.value.requires_voice and cosyvoice.value.can_stream
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id(HF_ID) is cosyvoice
    assert set(definition.parameters) == {"temperature", "top_p", "top_k", "repetition_penalty"}
    assert all(item.parameter != "speed" for item in definition.menu)
    assert definition.request_stream_defaults == {"response_format": "pcm"}
    assert definition.transcript_policy == "optional"
    assert not [name for name in Project.model_fields if name.startswith("fun_cosyvoice3_")]


def test_cosyvoice_readiness_ignores_voice_state(cosyvoice, monkeypatch, tmp_path):
    monkeypatch.setattr(SglOmniUtil, "check_readiness", staticmethod(lambda _: None))
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": MODEL_ID})
    project.tts_model_type = cosyvoice.id
    support = Tts.get_model_support(project)
    # Voice/transcript state is validated lazily (pre-flight, generation), not
    # by readiness. Generation itself still requires a reference.
    assert support.get_blocking_issues(project) == []

    project.set_model_setting(MODEL_ID, "file_name", ["missing.wav"])
    assert support.get_blocking_issues(project) == []
    (tmp_path / "reference.wav").write_bytes(b"audio")
    project.set_model_setting(MODEL_ID, "file_name", ["reference.wav"])
    assert not support.get_blocking_issues(project)


def test_cosyvoice_buffered_and_streamed_payloads(cosyvoice, monkeypatch, tmp_path):
    (tmp_path / "reference.wav").write_bytes(b"audio")
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": MODEL_ID})
    instance = Tts.get_instance()
    assert instance.generate_using_project(project, ["hello"]) == "A voice clone sample is required"
    project.set_model_setting(MODEL_ID, "file_name", ["reference.wav"])
    project.set_model_setting(MODEL_ID, "temperature", 0.6)
    monkeypatch.setattr(SglOmniUtil, "get_base_url", staticmethod(lambda: "http://example.test"))
    monkeypatch.setattr(SoundUtil, "make_audio_data_uri", staticmethod(lambda path: "data:reference"))
    sound = Sound(np.array([0.2], dtype=np.float32), 24_000)
    buffered = []
    streamed = []

    def concurrent(base_url, payloads, **kwargs):
        buffered.extend(payloads)
        assert kwargs["fallback_sample_rate"] == 24_000
        return [sound for _ in payloads]

    def streaming(base_url, payload, **kwargs):
        streamed.append(payload)
        assert kwargs["fallback_sample_rate"] == 24_000
        return sound

    monkeypatch.setattr(SglOmniUtil, "generate_concurrent", concurrent)
    monkeypatch.setattr(SglOmniUtil, "generate_streaming", streaming)
    assert instance.generate_using_project(project, ["hello"], force_random_seed=True) == [sound]
    expected = {"temperature": 0.6, "top_p": 0.8,
                "top_k": 20, "repetition_penalty": 1.1,
                "input": "hello", "stream": False,
                "references": [{"audio_path": "data:reference"}]}
    assert buffered == [expected]
    assert "seed" not in buffered[0] and "max_new_tokens" not in buffered[0]

    project.set_model_setting(MODEL_ID, "transcript", ["Reference transcript"])
    project.set_model_setting(MODEL_ID, "top_k", 32)
    assert instance.generate_using_project(project, ["world"], on_stream_chunk=lambda _: None) == [sound]
    assert streamed == [{**expected, "input": "world", "stream": True, "top_k": 32,
                         "response_format": "pcm",
                         "references": [{"audio_path": "data:reference", "text": "Reference transcript"}]}]

    saved = ProjectSerializationUtil.to_project_json_dict(project)
    settings = saved["model_settings"]["models"][MODEL_ID]
    assert settings["parameters"] == {"temperature": 0.6, "top_p": None, "top_k": 32, "repetition_penalty": None}
    # V4: the pair lives in the project-wide list, not the model object.
    assert "voice_references" not in settings
    assert saved["voice_references"] == [{"file_name": "reference.wav", "transcript": "Reference transcript"}]
    restored = Project.model_validate(saved)
    assert restored.get_model_setting(MODEL_ID, "top_k") == 32
    assert restored.get_model_setting(MODEL_ID, "transcript") == ["Reference transcript"]

    # Older projects may still have a CosyVoice speed override; discard it on load.
    settings["parameters"]["speed"] = 1.3
    restored = Project.model_validate(saved)
    assert "speed" not in ProjectSerializationUtil.to_project_json_dict(restored)["model_settings"]["models"][MODEL_ID]["parameters"]
