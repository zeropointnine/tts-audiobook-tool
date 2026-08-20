import os

import pytest
import torch

pytest.importorskip("glm_tts")

import tts_audiobook_tool.tts_models.glm_model as glm_model
from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.glm_model import GlmModel


class FakeTextFrontend:
    def text_normalize(self, text: str) -> str:
        return (text or "").strip()

    def split_by_len(self, text: str):
        return [text]


class FakeFrontend:
    def __init__(self):
        self.text_token_calls: list[str] = []
        self.speech_token_calls: list[str] = []
        self.speech_feat_calls: list[str] = []
        self.spk_embedding_calls: list[str] = []
        self.last_speech_token: torch.Tensor | None = None
        self.last_spk_embedding: torch.Tensor | None = None

    def _extract_text_token(self, text: str) -> torch.Tensor:
        self.text_token_calls.append(text)
        return torch.randint(0, 1000, (1, max(1, len(text) // 2)), dtype=torch.int64)

    def _extract_speech_token(self, paths: list) -> torch.Tensor:
        self.speech_token_calls.append(paths[0])
        self.last_speech_token = torch.randint(0, 1000, (1, 10), dtype=torch.int32)
        return self.last_speech_token

    def _extract_speech_feat(self, path: str, sample_rate: int) -> torch.Tensor:
        self.speech_feat_calls.append(path)
        return torch.randn(80, 10)

    def _extract_spk_embedding(self, path: str) -> torch.Tensor:
        self.spk_embedding_calls.append(path)
        self.last_spk_embedding = torch.randn(192, requires_grad=True)
        return self.last_spk_embedding


def make_model() -> GlmModel:
    model = GlmModel.__new__(GlmModel)
    model._device_type = DeviceType.CPU
    model.sample_rate = 24000
    model.use_phoneme = False
    model.use_cache = True
    model.uttid_counter = 0
    model.frontend = FakeFrontend()
    model.text_frontend = FakeTextFrontend()
    model.llm = object()
    model.flow = object()
    return model


def stub_generate_long(monkeypatch, calls: list) -> None:
    def fake_generate_long(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 4000), None, [], {}

    monkeypatch.setattr(glm_model, "generate_long", fake_generate_long)


def generate(model: GlmModel, voice_path: str, transcript: str):
    return model.generate(
        prompt_text=transcript,
        prompt_speech=voice_path,
        syn_text="Test sentence.",
        seed=1,
    )


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_glm_reuses_prepared_voice_prompt_for_a_b_a(tmp_path, monkeypatch):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()
    calls: list = []
    stub_generate_long(monkeypatch, calls)

    assert not isinstance(generate(model, str(voice_a), "words a"), str)
    assert not isinstance(generate(model, str(voice_b), "words b"), str)
    assert not isinstance(generate(model, str(voice_a), "words a"), str)

    frontend = model.frontend
    assert frontend.speech_token_calls == [str(voice_a), str(voice_b)]
    assert frontend.spk_embedding_calls == [str(voice_a), str(voice_b)]
    assert len(calls) == 3
    assert len(model._voice_clone_cache) == 2

    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    assert key_a[1] == "words a"
    voice_a_clone = model._voice_clone_cache[key_a]

    # Cached prompt is a CPU clone with fresh storage, no grad
    assert voice_a_clone.prompt_speech_token.device.type == "cpu"
    assert not voice_a_clone.prompt_speech_token.requires_grad
    assert voice_a_clone.prompt_speech_token.dtype == torch.int32
    assert voice_a_clone.speech_feat.device.type == "cpu"
    assert voice_a_clone.embedding.device.type == "cpu"
    assert not voice_a_clone.embedding.requires_grad
    assert voice_a_clone.prompt_text_token.device.type == "cpu"

    # The per-call objects handed to generation are built from the cache:
    # the last generate_long call (voice A again) used voice A's prompt data
    last_call = calls[-1]
    assert last_call["cache"]["cache_text"] == [voice_a_clone.prompt_text]
    assert last_call["cache"]["cache_speech_token"][0] == voice_a_clone.prompt_speech_token.squeeze().tolist()
    assert last_call["embedding"].device.type == "cpu"
    assert torch.equal(last_call["embedding"], voice_a_clone.embedding)
    assert last_call["flow_prompt_token"].dtype == torch.int32
    assert last_call["speech_feat"].device.type == "cpu"


def test_glm_rebuilds_changed_voice_and_clears_cache(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()
    calls: list = []
    stub_generate_long(monkeypatch, calls)

    assert not isinstance(generate(model, str(voice_path), "first transcript"), str)
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)
    assert len(model._voice_clone_cache) == 1

    voice_path.write_bytes(b"different file contents")
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)

    assert len(model.frontend.speech_token_calls) == 3
    assert len(model._voice_clone_cache) == 1

    model.kill()
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.frontend is None
    assert model.text_frontend is None
    assert model.llm is None
    assert model.flow is None


def test_glm_error_string_on_extraction_failure(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()
    calls: list = []
    stub_generate_long(monkeypatch, calls)

    def boom(path):
        raise RuntimeError("boom")

    model.frontend._extract_spk_embedding = boom

    assert generate(model, str(voice_path), "t") == (
        f"Couldn't create voice clone for {voice_path} - RuntimeError: boom"
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert calls == []  # generation never ran


def test_glm_requires_voice_path(tmp_path, monkeypatch):
    model = make_model()
    calls: list = []
    stub_generate_long(monkeypatch, calls)

    assert model.generate(prompt_text="t", prompt_speech="", syn_text="x", seed=1) == (
        "Voice clone path is required"
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert calls == []
