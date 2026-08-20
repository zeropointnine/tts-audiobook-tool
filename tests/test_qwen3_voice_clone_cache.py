import numpy as np
import pytest
import torch

pytest.importorskip("qwen_tts")

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

from tts_audiobook_tool.tts_models.qwen3_model import Qwen3Model


class FakeQwenWrapper:
    def __init__(self):
        self.create_calls: list[tuple[str, str]] = []
        self.created_prompts: list[VoiceClonePromptItem] = []
        self.generate_defaults = {}

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.create_calls.append((ref_audio, ref_text))
        prompt = VoiceClonePromptItem(
            ref_code=torch.tensor([[1.0, 2.0]], requires_grad=True),
            ref_spk_embedding=torch.tensor([3.0, 4.0], requires_grad=True),
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=True,
            ref_text=ref_text,
        )
        self.created_prompts.append(prompt)
        return [prompt]


def make_model() -> Qwen3Model:
    model = Qwen3Model.__new__(Qwen3Model)
    model._model_target = "fake"
    model._model = FakeQwenWrapper()
    model._voice_info = None
    model.cached_continuation_history = []

    def fake_generate_base_with_codes(prompts, voice_clone_prompt, languages, gen_kwargs):
        wavs = [np.zeros(8, dtype=np.float32) for _ in prompts]
        codes = [torch.zeros((1, 2), dtype=torch.long) for _ in prompts]
        return wavs, 24_000, codes

    model.generate_base_with_codes = fake_generate_base_with_codes
    return model


def generate(model: Qwen3Model, voice_path: str, transcript: str):
    return model.generate_base(
        prompts=["Test sentence."],
        voice_info=(voice_path, transcript),
        language="english",
        seed=1,
    )


def test_qwen_reuses_cpu_voice_prompts_for_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    assert not isinstance(generate(model, str(voice_a), "words a"), str)
    assert not isinstance(generate(model, str(voice_b), "words b"), str)
    assert not isinstance(generate(model, str(voice_a), "words a"), str)

    wrapper = model._model
    assert wrapper.create_calls == [
        (str(voice_a), "words a"),
        (str(voice_b), "words b"),
    ]
    assert len(model._voice_clone_cache) == 2

    for source_prompt in wrapper.created_prompts:
        matching = [
            value
            for key, value in model._voice_clone_cache.items()
            if key[1] == source_prompt.ref_text
        ]
        assert len(matching) == 1
        cached_prompt = matching[0]
        assert cached_prompt.ref_code.device.type == "cpu"
        assert cached_prompt.ref_spk_embedding.device.type == "cpu"
        assert not cached_prompt.ref_code.requires_grad
        assert not cached_prompt.ref_spk_embedding.requires_grad
        assert cached_prompt.ref_code.data_ptr() != source_prompt.ref_code.data_ptr()
        assert cached_prompt.ref_spk_embedding.data_ptr() != source_prompt.ref_spk_embedding.data_ptr()


def test_qwen_rebuilds_changed_voice_and_clears_cache(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert not isinstance(generate(model, str(voice_path), "first transcript"), str)
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)
    assert len(model._voice_clone_cache) == 1

    voice_path.write_bytes(b"different file contents")
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)

    assert len(model._model.create_calls) == 3
    assert len(model._voice_clone_cache) == 1

    model.clear_voice()
    assert model._voice_info is None
    assert model._voice_clone_cache == {}

    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)
    model.kill()
    assert model._voice_clone_cache == {}
    assert model._model is None
