import os

import numpy as np
import pytest
import torch

pytest.importorskip("omnivoice")

from omnivoice.models.omnivoice import VoiceClonePrompt  # type: ignore

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.omnivoice_model import OmniVoiceModel


class FakeOmniVoice:
    def __init__(self):
        self.create_calls: list[tuple] = []
        self.created_prompts: list[VoiceClonePrompt] = []
        self.generate_calls: list[dict] = []
        self.cpu_calls = 0
        # kill() tolerates any of these being present
        self._asr_pipe = None
        self.audio_tokenizer = None
        self.text_tokenizer = None
        self.feature_extractor = None
        self.duration_estimator = None
        self.sampling_rate = None
        self.llm = None
        self.audio_embeddings = None
        self.audio_heads = None
        self.codebook_layer_offsets = None
        self.normalized_audio_codebook_weights = None

    def cpu(self) -> None:
        self.cpu_calls += 1

    def create_voice_clone_prompt(self, ref_audio: str, ref_text: str | None) -> VoiceClonePrompt:
        self.create_calls.append((ref_audio, ref_text))
        # Float dtype on purpose: the real prompt carries float tokens, and
        # float tensors (unlike int ones) can hold gradients, which the
        # CPU-cloning step must strip
        prompt = VoiceClonePrompt(
            ref_audio_tokens=torch.randn(8, 40, requires_grad=True),
            ref_text=ref_text or "",
            ref_rms=0.5,
        )
        self.created_prompts.append(prompt)
        return prompt

    def generate(self, text, voice_clone_prompt=None, instruct=None, speed=1.0, generation_config=None):
        self.generate_calls.append(
            dict(
                text=text,
                voice_clone_prompt=voice_clone_prompt,
                instruct=instruct,
                speed=speed,
                generation_config=generation_config,
            )
        )
        # Mimic the library: consume the prompt without mutating it
        _ = voice_clone_prompt
        return [np.zeros(24000, dtype=np.float32)]


def make_model() -> OmniVoiceModel:
    model = OmniVoiceModel.__new__(OmniVoiceModel)
    model._model_target = "fake"
    model._device_type = DeviceType.CPU
    model._model = FakeOmniVoice()
    return model


def generate_clone(model: OmniVoiceModel, voice_path: str, ref_text: str = ""):
    return model._generate_voice_clone(
        prompts=["Test sentence."],
        voice_path=voice_path,
        ref_text=ref_text,
        instruct="",
        cfg=1.5,
        speed=1.0,
        steps=10,
        seed=1,
    )


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_omnivoice_reuses_voice_clone_prompt_for_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    assert isinstance(generate_clone(model, str(voice_a), "words a"), list)
    assert isinstance(generate_clone(model, str(voice_b), "words b"), list)
    assert isinstance(generate_clone(model, str(voice_a), "words a"), list)

    assert model._model.create_calls == [(str(voice_a), "words a"), (str(voice_b), "words b")]
    assert len(model._voice_clone_cache) == 2

    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    assert key_a[1] == "words a"
    cached_a: VoiceClonePrompt = model._voice_clone_cache[key_a]

    # The cached prompt is a CPU clone: no grad, fresh storage, same values,
    # with ref text/rms carried through
    created_a = model._model.created_prompts[0]
    assert cached_a.ref_audio_tokens is not created_a.ref_audio_tokens
    assert cached_a.ref_audio_tokens.device.type == "cpu"
    assert not cached_a.ref_audio_tokens.requires_grad
    assert cached_a.ref_audio_tokens.data_ptr() != created_a.ref_audio_tokens.data_ptr()
    assert torch.equal(cached_a.ref_audio_tokens, created_a.ref_audio_tokens)
    assert cached_a.ref_text == "words a"
    assert cached_a.ref_rms == 0.5

    # Generation consumed the cached object directly (identity)
    assert model._model.generate_calls[-1]["voice_clone_prompt"] is cached_a


def test_omnivoice_rebuilds_on_transcript_and_file_change(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert isinstance(generate_clone(model, str(voice_path), "first transcript"), list)
    assert isinstance(generate_clone(model, str(voice_path), "second transcript"), list)
    assert len(model._voice_clone_cache) == 1

    voice_path.write_bytes(b"different file contents")
    assert isinstance(generate_clone(model, str(voice_path), "second transcript"), list)

    assert len(model._model.create_calls) == 3
    assert len(model._voice_clone_cache) == 1


def test_omnivoice_error_string_on_prompt_failure(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    def boom(ref_audio, ref_text):
        raise ValueError("boom")

    model._model.create_voice_clone_prompt = boom

    assert generate_clone(model, str(voice_path)) == (
        f"Couldn't create voice clone for {voice_path} - ValueError: boom"
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model._model.generate_calls == []  # generation never ran


def test_omnivoice_non_clone_modes_do_not_touch_cache(tmp_path):
    model = make_model()

    assert isinstance(
        model._generate_auto_voice(prompts=["hi"], cfg=1.0, speed=1.0, steps=5, seed=1), list
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model._model.generate_calls[-1]["voice_clone_prompt"] is None

    assert isinstance(
        model._generate_voice_design(
            prompts=["hi"], instruct="warm and low", cfg=1.0, speed=1.0, steps=5, seed=1
        ),
        list,
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model._model.generate_calls[-1]["instruct"] == "warm and low"


def test_omnivoice_kill_clears_everything(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    assert isinstance(generate_clone(model, str(voice_path)), list)
    assert len(model._voice_clone_cache) == 1

    fake_model = model._model
    model.kill()

    assert fake_model.cpu_calls == 1
    assert fake_model.audio_tokenizer is None
    assert fake_model.llm is None
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model._model is None
