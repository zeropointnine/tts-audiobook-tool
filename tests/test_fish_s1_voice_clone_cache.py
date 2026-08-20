import math
import os
import struct
import wave

import pytest

pytest.importorskip("torchaudio")

import torch

from tts_audiobook_tool.app_types import DeviceType, Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.fish_s1_model import FishS1Model


def make_wav(path, seconds=0.2, sr=16000) -> None:
    """Write a small 16-bit PCM mono wav readable by torchaudio."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(int(sr * seconds)):
            frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / sr)))
        w.writeframes(bytes(frames))


class FakeDac:
    sample_rate = 24000

    def __init__(self):
        self.encode_calls: list[tuple] = []
        self.last_tokens: torch.Tensor | None = None

    def encode(self, audios, audio_lengths):
        # audios arrives as (1, 1, T)
        self.encode_calls.append(tuple(audios.shape))
        self.last_tokens = torch.randint(
            0, 1000, (1, max(1, audios.shape[2] // 5)), dtype=torch.int64
        )
        return self.last_tokens, None


def make_model() -> FishS1Model:
    model = FishS1Model.__new__(FishS1Model)
    model._device_type = DeviceType.CPU
    model.dac_model = FakeDac()
    model._voice_clone = None
    return model


def make_project(tmp_path, voice_file_name: str, transcript: str = "") -> Project:
    project = Project.model_validate({"dir_path": str(tmp_path)})
    project.fish_s1_voice_file_name = [voice_file_name] if voice_file_name else []
    project.fish_s1_voice_transcript = [transcript] if transcript else []
    return project


def stub_generate(model, monkeypatch, generated: list) -> None:
    monkeypatch.setattr(
        model,
        "generate",
        lambda **kwargs: (generated.append(kwargs), Sound([0.0] * 10, 24000))[1],
    )


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_fish_s1_reuses_prompt_tokens_for_a_b_a(tmp_path, monkeypatch):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    make_wav(voice_a)
    make_wav(voice_b)
    model = make_model()
    generated: list = []
    stub_generate(model, monkeypatch, generated)

    project_a = make_project(tmp_path, "a.wav", "transcript a")
    project_b = make_project(tmp_path, "b.wav", "transcript b")

    assert isinstance(model.generate_using_project(project_a, ["one"]), list)
    assert isinstance(model.generate_using_project(project_b, ["two"]), list)
    assert isinstance(model.generate_using_project(project_a, ["three"]), list)

    assert len(model.dac_model.encode_calls) == 2
    assert len(model._voice_clone_cache) == 2
    assert len(generated) == 3

    # The active clone is the cached object for voice A
    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    assert key_a[1] == "transcript a"
    assert model._voice_clone is model._voice_clone_cache[key_a]

    # The prompt tokens are a CPU clone (fresh storage, same values)
    tokens = model._voice_clone.prompt_tokens
    assert tokens.device.type == "cpu"
    assert not tokens.requires_grad


def test_fish_s1_rebuilds_on_transcript_and_file_change(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    make_wav(voice_path)
    model = make_model()
    stub_generate(model, monkeypatch, [])

    project = make_project(tmp_path, "voice.wav", "first transcript")
    assert isinstance(model.generate_using_project(project, ["one"]), list)

    project.fish_s1_voice_transcript = ["second transcript"]
    assert isinstance(model.generate_using_project(project, ["two"]), list)

    # In-place modification: new valid content (longer clip => new size/mtime)
    make_wav(voice_path, seconds=0.35)
    assert isinstance(model.generate_using_project(project, ["three"]), list)

    assert len(model.dac_model.encode_calls) == 3
    assert len(model._voice_clone_cache) == 1

    model.kill()
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.dac_model is None
    assert model._voice_clone is None


def test_fish_s1_error_string_on_encode_failure(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    make_wav(voice_path)
    model = make_model()
    stub_generate(model, monkeypatch, [])

    def boom(audios, audio_lengths):
        raise ValueError("boom")

    model.dac_model.encode = boom

    project = make_project(tmp_path, "voice.wav", "t")
    result = model.generate_using_project(project, ["one"])

    expected_path = os.path.join(str(tmp_path), "voice.wav")
    assert result == f"Couldn't create voice clone for {expected_path} - ValueError: boom"
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model._voice_clone is None


def test_fish_s1_no_voice_clears_state(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    make_wav(voice_path)
    model = make_model()
    generated: list = []
    stub_generate(model, monkeypatch, generated)

    project = make_project(tmp_path, "voice.wav", "t")
    assert isinstance(model.generate_using_project(project, ["one"]), list)
    assert len(model._voice_clone_cache) == 1

    assert isinstance(model.generate_using_project(make_project(tmp_path, ""), ["two"]), list)

    assert model._voice_clone is None
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert len(generated) == 2  # generation still ran without a voice


def test_fish_s1_clear_voice_clone_api(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    make_wav(voice_path)
    model = make_model()
    stub_generate(model, monkeypatch, [])

    project = make_project(tmp_path, "voice.wav", "t")
    assert isinstance(model.generate_using_project(project, ["one"]), list)
    assert len(model._voice_clone_cache) == 1

    model.clear_voice_clone()
    assert model._voice_clone is None
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
