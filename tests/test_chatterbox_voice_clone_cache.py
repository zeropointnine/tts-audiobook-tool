import os

import pytest
import torch

pytest.importorskip("chatterbox")

from chatterbox.mtl_tts import Conditionals
from chatterbox.models.t3.modules.cond_enc import T3Cond

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel


class FakeChatterbox:
    """
    Stand-in for the library's TTS class. `prepare_conditionals` mimics the
    library by producing a *fresh* Conditionals object (with gradient-carrying
    tensors) on each call, the way the real model builds on-device objects.
    """

    def __init__(self):
        self.device = "cpu"
        self.conds = None
        self.prepare_calls: list[str] = []
        self.created_conds: list[Conditionals] = []
        self.generate_calls: list[tuple] = []

    def prepare_conditionals(self, path: str) -> None:
        self.prepare_calls.append(path)
        self.conds = Conditionals(
            t3=T3Cond(
                speaker_emb=torch.randn(3, 4, requires_grad=True),
                clap_emb=torch.randn(5, requires_grad=True),
                cond_prompt_speech_tokens=torch.randint(0, 1000, (2, 6)),
                cond_prompt_speech_emb=torch.randn(2, 7, requires_grad=True),
            ),
            gen={
                "prompt_token": torch.randint(0, 100, (3,)),
                "prompt_feat": torch.randn(3, 9, requires_grad=True),
                "prompt_token_len": 3,
            },
        )
        self.created_conds.append(self.conds)

    def generate(self, text: str, **dic) -> torch.Tensor:
        self.generate_calls.append((text, dict(dic)))
        return torch.zeros(4800)


def make_model() -> ChatterboxModel:
    model = ChatterboxModel.__new__(ChatterboxModel)
    model._device_type = DeviceType.CPU
    model._model_type = ChatterboxType.MULTILINGUAL
    model._chatterbox = FakeChatterbox()
    return model


def generate(model: ChatterboxModel, voice_path: str):
    return model.generate(text="Test sentence.", voice_path=voice_path, seed=1)


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_chatterbox_reuses_cpu_conditionals_for_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    assert not isinstance(generate(model, str(voice_a)), str)
    assert not isinstance(generate(model, str(voice_b)), str)
    assert not isinstance(generate(model, str(voice_a)), str)

    wrapper = model._chatterbox
    assert wrapper.prepare_calls == [str(voice_a), str(voice_b)]
    assert len(wrapper.generate_calls) == 3
    assert len(model._voice_clone_cache) == 2

    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    assert key_a[1] == ""

    cached_a = model._voice_clone_cache[key_a]
    assert isinstance(cached_a, Conditionals)

    # The cache value is a CPU clone: no grad, fresh storage, same values as
    # the library-created object
    created_a = wrapper.created_conds[0]
    for name in ("speaker_emb", "clap_emb", "cond_prompt_speech_tokens", "cond_prompt_speech_emb"):
        cached = getattr(cached_a.t3, name)
        original = getattr(created_a.t3, name)
        assert cached is not original
        assert cached.device.type == "cpu"
        assert not cached.requires_grad
        assert cached.data_ptr() != original.data_ptr()
        assert torch.equal(cached, original)
    assert cached_a.gen["prompt_feat"].data_ptr() != created_a.gen["prompt_feat"].data_ptr()
    assert not cached_a.gen["prompt_feat"].requires_grad
    assert cached_a.gen["prompt_token_len"] == 3

    # The library's current conds is a *separate* fresh copy (not the cached
    # one, not the object prepare_conditionals made), so library-side
    # mutation of it cannot touch the cache
    assert wrapper.conds is not cached_a
    assert wrapper.conds is not created_a
    assert wrapper.conds.t3.speaker_emb.data_ptr() != cached_a.t3.speaker_emb.data_ptr()
    assert torch.equal(wrapper.conds.t3.speaker_emb, cached_a.t3.speaker_emb)


def test_chatterbox_rebuilds_changed_voice_and_clears_cache(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert not isinstance(generate(model, str(voice_path)), str)
    voice_path.write_bytes(b"different file contents")
    assert not isinstance(generate(model, str(voice_path)), str)

    assert len(model._chatterbox.prepare_calls) == 2
    assert len(model._voice_clone_cache) == 1

    model.kill()
    assert model._chatterbox is None
    assert model._voice_clone_cache == {}

    # A killed model refuses to generate
    assert generate(model, str(voice_path)) == "Logic error: Model is not initialized"


def test_chatterbox_error_string_on_prepare_failure(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    def boom(path):
        raise ValueError("boom")

    model._chatterbox.prepare_conditionals = boom

    assert generate(model, str(voice_path)) == (
        f"Couldn't create voice clone for {voice_path} - ValueError: boom"
    )
    assert model._voice_clone_cache == {}


def test_chatterbox_no_voice_path_bypasses_cache(tmp_path):
    model = make_model()

    assert not isinstance(model.generate(text="hi", voice_path="", seed=1), str)

    # The lazy cache attribute is only created on first use
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model._chatterbox.conds is None
