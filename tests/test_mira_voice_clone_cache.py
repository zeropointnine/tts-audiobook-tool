import os
from types import SimpleNamespace

import pytest

pytest.importorskip("mira")

import torch

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.mira_model import MiraModel


class FakeMiraTTS:
    def __init__(self):
        self.encode_calls: list[str] = []
        self.gen_config = SimpleNamespace(random_seed=None)
        self.set_params_calls: list[dict] = []
        self.generate_calls: list[tuple] = []

    def set_params(self, **kwargs) -> None:
        self.set_params_calls.append(kwargs)

    def encode_audio(self, path: str) -> str:
        self.encode_calls.append(path)
        with open(path, "rb") as f:
            return "ctx(" + f.read(4).hex() + ")"

    def generate(self, prompt: str, context_tokens: str) -> torch.Tensor:
        self.generate_calls.append((prompt, context_tokens))
        return torch.zeros(1000, dtype=torch.float16)


def make_model() -> MiraModel:
    model = MiraModel.__new__(MiraModel)
    model.mira_tts = FakeMiraTTS()
    model.context_tokens = None
    model._device_type = None
    return model


def make_project(tmp_path, voice_file_name: str) -> Project:
    project = Project.model_validate({"dir_path": str(tmp_path)})
    project.mira_voice_file_name = [voice_file_name] if voice_file_name else []
    return project


def generate(model: MiraModel, tmp_path, voice_name: str = "voice.wav"):
    return model.generate_using_project(make_project(tmp_path, voice_name), ["Test sentence."])


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_mira_reuses_context_tokens_for_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"alpha alpha alpha")
    voice_b.write_bytes(b"beta beta beta beta")
    model = make_model()

    assert isinstance(generate(model, tmp_path, "a.wav"), list)
    assert isinstance(generate(model, tmp_path, "b.wav"), list)
    assert isinstance(generate(model, tmp_path, "a.wav"), list)

    assert model.mira_tts.encode_calls == [str(voice_a), str(voice_b)]
    assert len(model._voice_clone_cache) == 2

    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    assert key_a[1] == ""
    assert model.context_tokens == model._voice_clone_cache[key_a]

    # The library received the retained (cached) context token string
    assert model.mira_tts.generate_calls[-1][1] == model._voice_clone_cache[key_a]


def test_mira_rebuilds_changed_voice_and_clears_cache(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert isinstance(generate(model, tmp_path), list)
    voice_path.write_bytes(b"different file contents")
    assert isinstance(generate(model, tmp_path), list)

    assert len(model.mira_tts.encode_calls) == 2
    assert len(model._voice_clone_cache) == 1

    model.kill()
    assert model.mira_tts is None
    assert model.context_tokens is None
    assert getattr(model, "_voice_clone_cache", None) in (None, {})


def test_mira_error_string_on_encode_failure(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    def boom(path):
        raise ValueError("boom")

    model.mira_tts.encode_audio = boom

    assert generate(model, tmp_path) == (
        f"Couldn't create voice clone for {voice_path} - ValueError: boom"
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.context_tokens is None


def test_mira_no_voice_clears_state(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"some bytes here")
    model = make_model()

    assert isinstance(generate(model, tmp_path), list)
    assert model.context_tokens is not None
    assert len(model._voice_clone_cache) == 1

    # Without a voice, the clone is cleared and generation reports the
    # missing clone
    assert generate(model, tmp_path, "") == "Logic error - voice clone not set"
    assert model.context_tokens is None
    assert getattr(model, "_voice_clone_cache", None) in (None, {})


def test_mira_clear_voice_clone_api(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"some bytes here")
    model = make_model()

    assert isinstance(generate(model, tmp_path), list)
    assert len(model._voice_clone_cache) == 1

    model.clear_voice_clone()
    assert model.context_tokens is None
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
