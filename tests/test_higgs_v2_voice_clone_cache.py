import os

import numpy as np
import pytest
import torch

pytest.importorskip("boson_multimodal")

try:
    from tts_audiobook_tool.tts_models.higgs_v2_model import HiggsV2Model
except Exception as e:  # e.g. pinned model snapshots not in the local HF cache
    pytest.skip(f"Cannot import Higgs V2 model module: {e}", allow_module_level=True)

from tts_audiobook_tool.app_types import DeviceType


class FakeAudioTokenizer:
    def __init__(self):
        self.encode_calls: list[str] = []
        self.last_tokens: torch.Tensor | None = None

    def encode(self, path: str) -> torch.Tensor:
        self.encode_calls.append(path)
        self.last_tokens = torch.randint(0, 1000, (1, 40), dtype=torch.long)
        return self.last_tokens


class FakeClient:
    def __init__(self):
        self.generate_calls: list[tuple] = []
        self.killed = False

    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        temperature,
        top_k,
        top_p,
        ras_win_len,
        ras_win_max_num_repeat,
        seed,
    ):
        self.generate_calls.append((messages, audio_ids))
        return np.zeros(48000, dtype=np.float32), 24000, "ok"

    def kill(self) -> None:
        self.killed = True


def make_model() -> HiggsV2Model:
    model = HiggsV2Model.__new__(HiggsV2Model)
    model._device_type = DeviceType.CPU
    model.audio_tokenizer = FakeAudioTokenizer()
    model.model_client = FakeClient()
    return model


def generate(model: HiggsV2Model, voice_path: str, transcript: str = ""):
    return model.generate(
        p_voice_path=voice_path,
        p_voice_transcript=transcript,
        text="Test sentence.",
        seed=1,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    )


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_higgs_reuses_encoded_tokens_for_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    assert not isinstance(generate(model, str(voice_a), "transcript a"), str)
    assert not isinstance(generate(model, str(voice_b), "transcript b"), str)
    assert not isinstance(generate(model, str(voice_a), "transcript a"), str)

    assert model.audio_tokenizer.encode_calls == [str(voice_a), str(voice_b)]
    assert len(model._voice_clone_cache) == 2

    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    cached_a = model._voice_clone_cache[key_a]
    assert cached_a.device.type == "cpu"
    assert not cached_a.requires_grad

    # Generation consumed the cached tensor directly (identity)
    _, last_audio_ids = model.model_client.generate_calls[-1]
    assert last_audio_ids[0] is cached_a


def test_higgs_rebuilds_on_transcript_and_file_change(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert not isinstance(generate(model, str(voice_path), "first transcript"), str)
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)
    assert len(model._voice_clone_cache) == 1

    voice_path.write_bytes(b"different file contents")
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)

    assert len(model.audio_tokenizer.encode_calls) == 3
    assert len(model._voice_clone_cache) == 1

    client = model.model_client
    model.kill()
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.audio_tokenizer is None
    assert client.killed
    assert model.model_client is None


def test_higgs_error_string_on_encode_failure(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    def boom(path):
        raise ValueError("boom")

    model.audio_tokenizer.encode = boom

    assert generate(model, str(voice_path)) == (
        f"Couldn't create voice clone for {voice_path} - ValueError: boom"
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.model_client.generate_calls == []  # generation never ran


def test_higgs_no_voice_bypasses_cache(tmp_path):
    model = make_model()

    assert not isinstance(generate(model, ""), str)

    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.audio_tokenizer.encode_calls == []
    _, audio_ids = model.model_client.generate_calls[0]
    assert audio_ids == []
